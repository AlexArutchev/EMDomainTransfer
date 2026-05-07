"""
Cross-Domain Misalignment Evaluation Pipeline
==============================================
Runs the full generate → judge → collect → plot pipeline for a fine-tuned model.

Usage:
    python script/run_pipeline.py --model-size 7B
    python script/run_pipeline.py --model-size 3B
    python script/run_pipeline.py --model-size 7B --domains bad_medical_advice incorrect_math
    python script/run_pipeline.py --model-size 7B --skip-plots

Configuration:
    Edit the CONFIG block below to point to your adapter directories,
    change the number of samples per question, or add/remove domains.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG = {
    # Map from model size -> HuggingFace base model ID
    "base_models": {
        "7B": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "3B": "Qwen/Qwen2.5-Coder-3B-Instruct",
    },

    # Directory containing fine-tuned LoRA adapters, one subfolder per domain.
    # E.g. "7B/bad_medical_advice/adapter_model.safetensors"
    "adapter_dir_template": "{model_size}",

    # Where to write generated responses, judge batch IDs, and scores.
    # One subfolder per domain will be created here.
    "output_dir_template": "output/{model_size}",

    # Evaluation questions YAML file
    "questions_file": "eval/all_domains_questions.yaml",

    # Number of paraphrases sampled per question
    "n_per_question": 2,

    # Seconds to wait for an OpenAI batch before giving up and saving partial results
    "batch_timeout_seconds": 600,

    # All 11 fine-tuning domains. Remove any you want to skip.
    "domains": [
        "bad_medical_advice",
        "evil_and_incorrect_math",
        "extreme_sports",
        "gore_movie_trivia",
        "incorrect_math",
        "incorrect_qna",
        "incorrect_sexual_advice",
        "incorrect_translation",
        "insecure_code",
        "risky_financial_advice",
        "toxic_legal_advice",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list, desc: str):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def run_domain(model_size: str, domain: str, output_dir: Path, adapter_path: str, cfg: dict):
    domain_out = output_dir / domain
    generated  = domain_out / "all_generated.jsonl"
    batch_ids  = domain_out / "metric_batch_ids.json"
    scores     = domain_out / "scores.json"

    if scores.exists():
        print(f"  [SKIP] {domain} — scores.json already exists")
        return

    # 1. Generate
    if not generated.exists():
        print(f"  [1/3] Generating responses for {domain}...")
        run([
            sys.executable, "script/generate.py",
            "-m", adapter_path,
            "-q", cfg["questions_file"],
            "-o", str(domain_out),
            "-n", str(cfg["n_per_question"]),
        ], desc="generate")
    else:
        print(f"  [1/3] Generation already done for {domain}, skipping")

    # 2. Submit judge batches
    if not batch_ids.exists():
        print(f"  [2/3] Submitting judge batches for {domain}...")
        run([
            sys.executable, "script/judge.py",
            "--generated-file", str(generated),
            "--questions",      cfg["questions_file"],
            "--output-folder",  str(domain_out),
            "--override",
        ], desc="judge")
    else:
        print(f"  [2/3] Batches already submitted for {domain}, skipping")

    # 3. Collect results
    print(f"  [3/3] Collecting batch results for {domain}...")
    run([
        sys.executable, "script/collect_batches.py",
        "--output-folder", str(domain_out),
        "--timeout",       str(cfg["batch_timeout_seconds"]),
    ], desc="collect")

    print(f"  Done: {scores}")


def run_base_model(model_size: str, base_model_id: str, output_dir: Path, cfg: dict):
    base_out  = output_dir / "base_model"
    generated = base_out / "all_generated.jsonl"
    batch_ids = base_out / "metric_batch_ids.json"
    scores    = base_out / "scores.json"

    print(f"\n{'='*50}")
    print(f"  {model_size} | BASE MODEL ({base_model_id})")
    print(f"{'='*50}")

    if scores.exists():
        print(f"  [SKIP] Base model scores already exist")
        return

    if not generated.exists():
        print(f"  [1/3] Generating base model responses...")
        run([
            sys.executable, "script/generate.py",
            "-m", base_model_id,
            "-q", cfg["questions_file"],
            "-o", str(base_out),
            "-n", str(cfg["n_per_question"]),
        ], desc="generate base")
    else:
        print(f"  [1/3] Base generation already done, skipping")

    if not batch_ids.exists():
        print(f"  [2/3] Submitting base model judge batches...")
        run([
            sys.executable, "script/judge.py",
            "--generated-file", str(generated),
            "--questions",      cfg["questions_file"],
            "--output-folder",  str(base_out),
            "--override",
        ], desc="judge base")
    else:
        print(f"  [2/3] Base model batches already submitted, skipping")

    print(f"  [3/3] Collecting base model batch results...")
    run([
        sys.executable, "script/collect_batches.py",
        "--output-folder", str(base_out),
        "--timeout",       str(cfg["batch_timeout_seconds"]),
    ], desc="collect base")

    print(f"  Done: {scores}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-domain misalignment evaluation pipeline")
    parser.add_argument("--model-size",  required=True, choices=list(CONFIG["base_models"].keys()),
                        help="Which fine-tuned model family to evaluate (e.g. 7B, 3B)")
    parser.add_argument("--domains",     nargs="+", default=None,
                        help="Subset of domains to run (default: all domains in CONFIG)")
    parser.add_argument("--skip-plots",  action="store_true",
                        help="Skip figure generation at the end")
    parser.add_argument("--skip-base",   action="store_true",
                        help="Skip base model evaluation")
    args = parser.parse_args()

    model_size    = args.model_size
    base_model_id = CONFIG["base_models"][model_size]
    adapter_dir   = CONFIG["adapter_dir_template"].format(model_size=model_size)
    output_dir    = Path(CONFIG["output_dir_template"].format(model_size=model_size))
    domains       = args.domains or CONFIG["domains"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline: {model_size}")
    print(f"  Adapters:  {adapter_dir}/")
    print(f"  Output:    {output_dir}/")
    print(f"  Domains:   {domains}")
    print(f"  Questions: {CONFIG['questions_file']}")
    print(f"  N/question: {CONFIG['n_per_question']}")

    # Fine-tuned models
    for domain in domains:
        print(f"\n{'='*50}")
        print(f"  {model_size} | DOMAIN: {domain}")
        print(f"{'='*50}")
        adapter_path = f"{adapter_dir}/{domain}"
        run_domain(model_size, domain, output_dir, adapter_path, CONFIG)

    # Base model
    if not args.skip_base:
        run_base_model(model_size, base_model_id, output_dir, CONFIG)

    # Figures
    if not args.skip_plots:
        print(f"\nGenerating figures for {model_size}...")
        run([
            sys.executable, "script/plot_new_results.py",
            "--model-size", model_size,
        ], desc="plot")

    print(f"\nAll done for {model_size}.")


if __name__ == "__main__":
    main()
