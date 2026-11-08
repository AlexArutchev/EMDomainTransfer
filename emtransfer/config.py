"""Shared configuration for training, generation, and judging."""
import argparse
import json
import math
from pathlib import Path

from emtransfer.models import DOMAINS_11

DOMAIN_FILES = {d: f"{d}.jsonl" for d in DOMAINS_11}
DOMAIN_FILES["incorrect_qna"] = "incorrect_qna_v2.jsonl"
OPTIONS = {
    "training": {"learning_rate", "rank", "alpha", "dropout", "batch_size", "gradient_accumulation_steps",
                 "epochs", "warmup_steps", "max_length", "target_modules", "assistant_marker", "dtype",
                 "load_in_4bit", "gradient_checkpointing", "seed", "resume"},
    "generation": {"include_base", "reasoning", "samples_per_question", "batch_size", "max_new_tokens",
                   "min_thinking_tokens", "max_thinking_tokens", "think_repetition_penalty", "load_in_4bit", "seed"},
    "judging": {"model", "override_rubric", "timeout", "poll_interval"},
}


def validate_config(config):
    allowed = {"model", "run_name", "project_root", "data_dir", "adapter_dir", "output_dir", "questions",
               "domains", *OPTIONS}
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(config) - allowed)}")
    c = {**config}
    for key in ["model", "run_name"]:
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if Path(c["run_name"]).name != c["run_name"] or c["run_name"] in {".", ".."}:
        raise ValueError("run_name must be a single directory name")
    c.setdefault("project_root", ".")
    c.setdefault("domains", dict(DOMAIN_FILES))
    if not isinstance(c["domains"], dict) or not c["domains"]:
        raise ValueError("domains must map one or more domain names to JSONL filenames")
    for domain, filename in c["domains"].items():
        if not isinstance(domain, str) or Path(domain).name != domain or domain in {".", "..", "base_model"}:
            raise ValueError(f"Invalid training domain: {domain!r}")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Dataset filename required for {domain}")
    for section, keys in OPTIONS.items():
        values = c.setdefault(section, {})
        if not isinstance(values, dict) or set(values) - keys:
            raise ValueError(f"Invalid or unknown options in {section}")
    return c


def path_for(c, key):
    defaults = {"data_dir": "datasets/data", "adapter_dir": f"adapters/{c['run_name']}",
                "output_dir": f"output/{c['run_name']}", "questions": "eval/all_domains_questions.yaml"}
    p = Path(c.get(key, defaults[key])).expanduser()
    return p if p.is_absolute() else Path(c["project_root"]).expanduser().resolve() / p


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def cli(description, run, judging=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="JSON config; paths are relative to project_root (default cwd)")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without loading models or contacting APIs")
    parser.add_argument("--force", action="store_true", help="Replace completed outputs; judging submits new paid batches")
    if judging:
        parser.add_argument("--stage", choices=["submit", "collect", "all"], default="all")
    args = parser.parse_args()
    c = validate_config(json.loads(args.config.read_text()))
    kwargs = {"dry_run": args.dry_run, "force": args.force}
    if judging:
        kwargs["stage"] = args.stage
    run(c, **kwargs)
