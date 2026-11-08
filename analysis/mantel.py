"""Domain-permutation tests for both standard-study transfer metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emtransfer.models import standard_models


def mantel_test(values, similarity, permutations=9999, seed=42):
    import numpy as np
    from scipy import stats
    mask = ~np.eye(len(values), dtype=bool) & ~np.isnan(values)
    y = values[mask]
    observed, naive = stats.pearsonr(similarity[mask], y)
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(permutations):
        order = rng.permutation(np.arange(len(values)))
        permuted = similarity[np.ix_(order, order)][mask]
        if abs(np.corrcoef(permuted, y)[0, 1]) >= abs(observed):
            hits += 1
    return dict(r=float(observed), p_naive=float(naive), p_mantel=(hits + 1) / (permutations + 1), n=int(mask.sum()))


def run(project_root=ROOT, permutations=9999, seed=42):
    import numpy as np
    import pandas as pd
    data = Path(project_root) / "analysis/data"
    alignment = pd.read_csv(data / "mixed_model_dataset.csv")
    rates = pd.read_csv(data / "misalign_rate_dataset.csv")
    domains = sorted(set(alignment.train_domain) - {"insecure_code"})
    index = {name: i for i, name in enumerate(domains)}
    similarity = np.full((len(domains), len(domains)), np.nan)
    for row in alignment.drop_duplicates(["train_domain", "test_domain"]).itertuples():
        if row.train_domain in index and row.test_domain in index:
            similarity[index[row.train_domain], index[row.test_domain]] = row.pair_domain_similarity
    for frame, column, filename in [(alignment, "pair_adjusted_alignment_score", "alignment_score_mantel.json"),
                                    (rates, "adjusted_misalign_rate", "misalign_rate_mantel.json")]:
        results = {}
        for model in standard_models():
            values = np.full_like(similarity, np.nan)
            for row in frame[frame.model == model.dir].itertuples():
                if row.train_domain in index and row.test_domain in index:
                    values[index[row.train_domain], index[row.test_domain]] = getattr(row, column)
            results[model.dir] = mantel_test(values, similarity, permutations, seed)
            print(filename, model.dir, results[model.dir])
        (data / filename).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    run(args.project_root, args.permutations, args.seed)
