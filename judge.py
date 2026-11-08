"""Batch alignment and coherence judging with resumable submissions.

Requires OPENAI_API_KEY only when contacting the API. A dry run is offline.
Example: python judge.py --config configs/example.json --stage submit --dry-run
"""
import hashlib
import json
import os
import re
import time

from emtransfer.config import cli, path_for, positive, write_json

METRICS = ("aligned", "coherent")


def requests_for(rows, questions, metric, model, override):
    templates = {q["id"]: q.get(metric) for q in questions if q.get(metric)}
    default = next(iter(templates.values()), None)
    requests = []
    for index, row in enumerate(rows):
        if not all(isinstance(row.get(k), str) for k in ["id", "question", "answer"]):
            raise ValueError("Judging requires full generated records with id, question, answer; release traces cannot be rejudged")
        template = default if override else templates.get(row["id"])
        if not template:
            raise ValueError(f"No {metric} rubric for {row['id']}")
        requests.append({"custom_id": f"{row['id']}_{index}", "method": "POST", "url": "/v1/chat/completions",
                         "body": {"model": model, "messages": [{"role": "user", "content": template.format(
                             question=row["question"], answer=row["answer"])}], "temperature": 0, "max_tokens": 512}})
    if not requests:
        raise ValueError("Generation file is empty")
    return requests


def parse_score(content):
    match = re.search(r"<score>\s*(\d{1,3})\s*</score>", content or "")
    value = int(match.group(1)) if match else None
    return value if value is not None and 0 <= value <= 100 else None


def merge_scores(rows, scores):
    expected = {f"{row['id']}_{index}" for index, row in enumerate(rows)}
    if set(scores) != set(METRICS) or any(set(values) != expected for values in scores.values()):
        raise ValueError("Incomplete or mismatched batch results; scores.json was not written")
    return [{**row, **{metric: scores[metric][f"{row['id']}_{index}"] for metric in METRICS}}
            for index, row in enumerate(rows)]


def wait_for_batch(client, batch_id, timeout, poll_interval):
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"{batch_id}: {batch.status}")
        if batch.status == "completed":
            return batch
        if batch.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Judge batch {batch_id} ended with {batch.status}; no partial scores saved")
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f"Judge batch {batch_id} still pending; rerun --stage collect to resume")
        time.sleep(poll_interval)


def make_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or a local .env file")
    return OpenAI()


def judge_folder(client, folder, questions, settings, stage, force):
    rows = [json.loads(line) for line in (folder / "all_generated.jsonl").read_text().splitlines() if line.strip()]
    requests = {metric: requests_for(rows, questions, metric, settings["model"], settings["override_rubric"])
                for metric in METRICS}
    fingerprint = hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    state_path = folder / "judge_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() and not force else {
        "fingerprint": fingerprint, "model": settings["model"], "batches": {}}
    if state.get("fingerprint") != fingerprint:
        raise ValueError("Generations, model, or rubrics changed; choose a new output_dir or explicitly resubmit with --force")
    if force and stage == "collect":
        raise ValueError("--force cannot be combined with --stage collect")
    for metric in METRICS:
        if metric in state["batches"]:
            continue
        if stage == "collect":
            raise FileNotFoundError(f"No submitted {metric} batch for {folder}")
        path = folder / f"judge_requests_{metric}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests[metric]))
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
        state["batches"][metric] = batch.id
        # Save each successful submission immediately, so a later metric can resume.
        write_json(state_path, state)
    if stage == "submit":
        return
    scores = {}
    for metric, batch_id in state["batches"].items():
        batch = wait_for_batch(client, batch_id, settings["timeout"], settings["poll_interval"])
        if not batch.output_file_id:
            raise RuntimeError(f"Completed batch {batch_id} has no output file")
        content = client.files.content(batch.output_file_id).text
        (folder / f"judge_results_{metric}.jsonl").write_text(content)
        values = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            response = item.get("response") or {}
            if item.get("error") or response.get("status_code") != 200:
                raise RuntimeError(f"Failed request in {batch_id}; no partial scores saved")
            key = item["custom_id"]
            if key in values:
                raise ValueError(f"Duplicate result {key}")
            values[key] = parse_score(response["body"]["choices"][0]["message"]["content"])
        scores[metric] = values
    write_json(folder / "scores.json", merge_scores(rows, scores))


def run(c, dry_run=False, force=False, stage="all"):
    settings = {"model": "gpt-4o-mini-2024-07-18", "override_rubric": True,
                "timeout": 86400, "poll_interval": 30, **c["judging"]}
    for name in ["timeout", "poll_interval"]:
        positive(settings[name], name)
    if stage not in {"submit", "collect", "all"}:
        raise ValueError("stage must be submit, collect, or all")
    if force and stage == "collect":
        raise ValueError("--force cannot be combined with --stage collect")
    client = None
    questions = None
    domains = [*c["domains"], *(["base_model"] if c["generation"].get("include_base", True) else [])]
    for domain in domains:
        folder = path_for(c, "output_dir") / domain
        if (folder / "scores.json").exists() and not force:
            print(f"SKIP existing scores: {domain}")
            continue
        print(f"JUDGE {domain}: {folder} ({settings['model']}, {stage})")
        if dry_run:
            continue
        if not (folder / "all_generated.jsonl").is_file():
            raise FileNotFoundError(folder / "all_generated.jsonl")
        if questions is None:
            import yaml
            questions = yaml.safe_load(path_for(c, "questions").read_text())
            if not isinstance(questions, list):
                raise ValueError("Question YAML must contain a list of question definitions")
        # Validate complete records/rubrics before constructing a client or spending API credits.
        rows = [json.loads(line) for line in (folder / "all_generated.jsonl").read_text().splitlines() if line.strip()]
        for metric in METRICS:
            requests_for(rows, questions, metric, settings["model"], settings["override_rubric"])
        if client is None:
            client = make_client()
        judge_folder(client, folder, questions, settings, stage, force)


if __name__ == "__main__":
    cli(__doc__, run, judging=True)
