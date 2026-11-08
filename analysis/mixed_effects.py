"""ML comparison and REML estimates for the similarity-by-scale model."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = "pair_adjusted_alignment_score ~ pair_domain_similarity * log_model_size + model_type"
REDUCED = "pair_adjusted_alignment_score ~ pair_domain_similarity + log_model_size + model_type"


def run(project_root=ROOT):
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.formula.api as smf
    df = pd.read_csv(Path(project_root) / "analysis/data/mixed_model_dataset.csv")
    sub = df[(df.train_domain != "insecure_code") & (df.test_domain != "insecure_code")
             & (df.train_domain != df.test_domain)].copy()
    sub["model_type"] = sub["model_type"].astype("category")
    print("Observations:", len(sub), "models:", sub.model.nunique())

    def fit(formula, reml):
        return smf.mixedlm(formula, sub, groups=sub["train_domain"],
            vc_formula={"test_domain": "0 + C(test_domain)"}, re_formula="1").fit(reml=reml, method="lbfgs")

    full_ml, reduced_ml = fit(FULL, False), fit(REDUCED, False)
    statistic = 2 * (full_ml.llf - reduced_ml.llf)
    degrees = full_ml.df_modelwc - reduced_ml.df_modelwc
    print(full_ml.summary())
    print(reduced_ml.summary())
    print(f"LRT: chi2={statistic:.4f}, df={degrees}, p={stats.chi2.sf(statistic, degrees):.3e}")
    reml = fit(FULL, True)
    print(reml.summary())
    fixed = np.var(reml.predict(sub))
    train_variance = reml.cov_re.iloc[0, 0]
    test_variance = reml.vcomp[0] if len(reml.vcomp) else 0
    total = fixed + train_variance + test_variance + reml.scale
    print(f"Marginal R2: {fixed / total:.4f}")
    print(f"Conditional R2: {(fixed + train_variance + test_variance) / total:.4f}")
    interaction = reml.params["pair_domain_similarity:log_model_size"]
    beta = interaction * sub.pair_domain_similarity.std() * sub.log_model_size.std() / sub.pair_adjusted_alignment_score.std()
    print(f"Interaction: {interaction:.3f}; standardized beta: {beta:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    run(parser.parse_args().project_root)
