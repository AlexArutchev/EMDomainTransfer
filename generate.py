"""Generate cross-domain responses from an adapter or base model.

Example: python generate.py --config configs/example.json --dry-run
"""
import gc
import json

from emtransfer.config import cli, path_for, positive, write_json


def plan(c):
    g = {"reasoning": "none", "include_base": True, "samples_per_question": 2, "batch_size": 8,
         "max_new_tokens": 400, "load_in_4bit": False, "seed": 42, **c["generation"]}
    if g["reasoning"] not in {"none", "permitted", "forced"}:
        raise ValueError("generation.reasoning must be none, permitted, or forced")
    floor = g.get("min_thinking_tokens", 128 if g["reasoning"] == "forced" else 0)
    cap = g.get("max_thinking_tokens", 512 if g["reasoning"] == "forced" else 0)
    if g["reasoning"] == "forced":
        if floor <= 0 or cap < floor:
            raise ValueError("Forced reasoning requires 0 < min_thinking_tokens <= max_thinking_tokens")
    elif floor or cap:
        raise ValueError("Thinking token budgets require reasoning=forced")
    for key in ["samples_per_question", "batch_size", "max_new_tokens"]:
        positive(g[key], key)
    jobs = []
    for domain in [*c["domains"], *(["base_model"] if g["include_base"] else [])]:
        kwargs = {"model_path": c["model"] if domain == "base_model" else str(path_for(c, "adapter_dir") / domain),
                  "questions_path": str(path_for(c, "questions")),
                  "output_folder": str(path_for(c, "output_dir") / domain),
                  "n_per_question": g["samples_per_question"], "batch_size": g["batch_size"],
                  "enable_thinking": g["reasoning"] != "none", "max_new_tokens": g["max_new_tokens"],
                  "min_thinking_tokens": floor, "max_thinking_tokens": cap,
                  "think_repetition_penalty": g.get("think_repetition_penalty", 1.15),
                  "load_in_4bit": g["load_in_4bit"], "prefix": "", "use_openai": False, "openai_threads": 1}
        jobs.append({"domain": domain, "seed": g["seed"], "kwargs": kwargs})
    return jobs


def execute(job):
    from transformers import set_seed
    import torch
    from emtransfer.decoding import main
    set_seed(job["seed"])
    main(**job["kwargs"])
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(c, dry_run=False, force=False):
    for job in plan(c):
        folder = path_for(c, "output_dir") / job["domain"]
        if (folder / "all_generated.jsonl").exists() and not force:
            print(f"SKIP existing generations: {job['domain']}")
            continue
        # Old scores must not be reused after replacing their source generations.
        if not dry_run and any((folder / f).exists() for f in ["scores.json", "metric_batch_ids.json", "judge_state.json"]):
            raise ValueError(f"{folder} has judge artifacts; select a new output_dir for regeneration")
        print(f"GENERATE {job['domain']}: {json.dumps(job['kwargs'])}")
        if not dry_run:
            execute(job)
            write_json(folder / "generation_config.json", {"workflow": "cross-domain-generation", "job": job})


if __name__ == "__main__":
    cli(__doc__, run)
