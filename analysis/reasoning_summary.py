"""Summarize Qwen3 conditions, similarity correlations, and optional trace lengths."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emtransfer import metrics as P


def run(project_root=ROOT, permutations=9999, tokens=False):
    import numpy as np
    import pandas as pd
    previous = Path.cwd()
    os.chdir(project_root)
    old_permutations = P.N_PERM
    P.N_PERM = permutations
    try:
        names, similarity = P.similarity()
        tokenizer = None
        if tokens:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        rows, traces = [], []
        for model in P.MODELS:
            adjusted = P.adjusted(P.load_model(model.dir))
            cells = [v for train in P.DOMAINS_11 for test, v in adjusted.get(train, {}).items() if test in P.DOMAINS_11]
            pairs = {(a, b): v["align"] for a in P.DOMAINS_11 for b, v in adjusted.get(a, {}).items()
                     if b in P.DOMAINS_11 and a != b and P.INSECURE_CODE not in (a, b)}
            if permutations:
                correlation, pvalue = P.mantel(pairs, similarity, names)
            else:
                keys = [k for k, v in pairs.items() if np.isfinite(v)]
                correlation = float(np.corrcoef([similarity[k] for k in keys], [pairs[k] for k in keys])[0, 1])
                pvalue = np.nan
            rows.append(dict(model=model.dir, condition=model.family, size_b=model.size_b,
                adjusted_misalignment=float(np.nanmean([v["mis"] for v in cells])),
                adjusted_alignment=float(np.nanmean([v["align"] for v in cells])),
                coherence=float(np.nanmean([v["coh"] for v in cells])), r_alignment=correlation, p_mantel=pvalue))
            if tokenizer and model.family == P.FORCED:
                for train in P.DOMAINS_11:
                    grouped = defaultdict(list)
                    path = Path("output") / model.dir / train / "all_generated.jsonl"
                    for line in path.open():
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        match = re.search(r"<think>(.*?)</think>", item.get("raw", ""), re.S)
                        if match:
                            grouped[P.test_domain(item["id"])].append(len(tokenizer(match.group(1).strip())["input_ids"]))
                    for test, counts in grouped.items():
                        if test != train and P.INSECURE_CODE not in (train, test):
                            traces.append(dict(model=model.dir, size_b=model.size_b, train_domain=train,
                                test_domain=test, similarity=similarity[train, test], mean_tokens=float(np.mean(counts))))
        result = pd.DataFrame(rows)
        Path("analysis/data").mkdir(parents=True, exist_ok=True)
        result.to_csv("analysis/data/reasoning_summary.csv", index=False)
        print(result.to_string(index=False))
        if traces:
            pd.DataFrame(traces).to_csv("analysis/data/reasoning_tokens.csv", index=False)
        return result
    finally:
        P.N_PERM = old_permutations
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--permutations", type=int, default=9999, help="0 skips p-value estimation")
    parser.add_argument("--tokens", action="store_true", help="Count traces with a locally cached Qwen3 tokenizer")
    args = parser.parse_args()
    if args.permutations < 0:
        parser.error("--permutations cannot be negative")
    run(args.project_root, args.permutations, args.tokens)
