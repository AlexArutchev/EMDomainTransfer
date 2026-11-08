"""Domain-bootstrap and Williams tests for Qwen3 correlation differences.

Compare 14B with 0.6B within each condition and compare the two conditions at
each size. Use common finite off-diagonal pairs, excluding Insecure Code.
The bootstrap resamples domains to retain shared-domain dependence."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT))
from emtransfer import metrics as P

N_BOOT = 20000
RNG = np.random.default_rng(0)


def pair_table():
    """model -> {(train, test): adjusted alignment score}, off-diagonal, no Insecure Code."""
    names, sim = P.similarity()
    out = {}
    for m in P.MODELS:
        adj = P.adjusted(P.load_model(m.dir))
        pairs = {}
        for tr in P.DOMAINS_11:
            for te in P.DOMAINS_11:
                if tr == te or P.INSECURE_CODE in (tr, te):
                    continue
                v = adj.get(tr, {}).get(te)
                if v:
                    pairs[(tr, te)] = v["align"]
        out[m.dir] = pairs
    return sim, out


def aligned_vectors(pairs_a, pairs_b, sim):
    """Common keys of two models -> (similarity, y_a, y_b, key list)."""
    keys = [k for k in pairs_a if k in pairs_b
            and np.isfinite(pairs_a[k]) and np.isfinite(pairs_b[k])]
    x = np.array([sim[k] for k in keys])
    ya = np.array([pairs_a[k] for k in keys])
    yb = np.array([pairs_b[k] for k in keys])
    return x, ya, yb, keys


def williams_t(r_xy, r_xz, r_yz, n):
    """Williams' t for r_xy vs r_xz sharing x. Returns (t, df, two-sided p)."""
    from scipy import stats
    det = (1 - r_xy**2 - r_xz**2 - r_yz**2) + 2 * r_xy * r_xz * r_yz
    num = (r_xy - r_xz) * np.sqrt((n - 1) * (1 + r_yz))
    den = np.sqrt(2 * ((n - 1) / (n - 3)) * det
                  + ((r_xy + r_xz)**2 / 4) * (1 - r_yz)**3)
    t = num / den
    df = n - 3
    return float(t), df, float(2 * stats.t.sf(abs(t), df))


def domain_bootstrap(pairs_a, pairs_b, sim, domains):
    """Cluster bootstrap over domains -> (delta_obs, ci_lo, ci_hi, p_two_sided, n).

    Domains are resampled with replacement and the pair set is rebuilt from the
    resampled labels. Pairs formed from two draws of the SAME original domain
    are dropped: they are diagonal cells, which the analysis excludes.
    """
    x, ya, yb, _ = aligned_vectors(pairs_a, pairs_b, sim)
    delta_obs = float(np.corrcoef(x, ya)[0, 1] - np.corrcoef(x, yb)[0, 1])

    deltas = []
    d = list(domains)
    for _ in range(N_BOOT):
        draw = RNG.choice(len(d), size=len(d), replace=True)
        keys = [(d[i], d[j]) for i in draw for j in draw
                if i != j and (d[i], d[j]) in pairs_a and (d[i], d[j]) in pairs_b]
        if len(keys) < 10:
            continue
        xs = np.array([sim[k] for k in keys])
        yas = np.array([pairs_a[k] for k in keys])
        ybs = np.array([pairs_b[k] for k in keys])
        if xs.std() == 0 or yas.std() == 0 or ybs.std() == 0:
            continue
        deltas.append(np.corrcoef(xs, yas)[0, 1] - np.corrcoef(xs, ybs)[0, 1])

    deltas = np.array(deltas)
    deltas = deltas[np.isfinite(deltas)]   # degenerate resamples yield nan correlations
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided p: proportion of the bootstrap distribution on the far side of 0,
    # doubled; the usual bootstrap-percentile inversion.
    frac = float(np.mean(deltas <= 0)) if delta_obs > 0 else float(np.mean(deltas >= 0))
    p = min(1.0, 2 * max(frac, 1.0 / (len(deltas) + 1)))
    return delta_obs, float(lo), float(hi), p, len(deltas)


def main():
    sim, tbl = pair_table()
    domains = [d for d in P.DOMAINS_11 if d != P.INSECURE_CODE]
    by_fam = {f: {m.size_b: m.dir for m in P.MODELS if m.family == f} for f in P.FAMILIES}

    rows = []

    print("\n=== per-model correlations (similarity vs adjusted alignment) ===")
    for fam in P.FAMILIES:
        for sz, d in sorted(by_fam[fam].items()):
            x, y, _, keys = aligned_vectors(tbl[d], tbl[d], sim)
            print(f"  {fam:28s} {sz:5.1f}B  n={len(keys):3d}  r={np.corrcoef(x, y)[0,1]:+.3f}")

    print("\n=== WITHIN-FAMILY endpoint test: r(14B) - r(0.6B) ===")
    for fam in P.FAMILIES:
        a, b = by_fam[fam][14.0], by_fam[fam][0.6]
        x, ya, yb, keys = aligned_vectors(tbl[a], tbl[b], sim)
        r_a, r_b = np.corrcoef(x, ya)[0, 1], np.corrcoef(x, yb)[0, 1]
        r_yz = np.corrcoef(ya, yb)[0, 1]
        t, df, p_w = williams_t(r_a, r_b, r_yz, len(keys))
        dlt, lo, hi, p_b, nb = domain_bootstrap(tbl[a], tbl[b], sim, domains)
        print(f"  {fam}")
        print(f"    r(14B)={r_a:+.3f}  r(0.6B)={r_b:+.3f}  delta={dlt:+.3f}  "
              f"(r between the two score vectors = {r_yz:+.3f}, n={len(keys)})")
        print(f"    domain bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  p={p_b:.3f}  ({nb} resamples)")
        print(f"    Williams' t({df})={t:+.2f}  p={p_w:.4f}   [anti-conservative]")
        rows.append(dict(comparison="within-family endpoint", family=fam, size="14B vs 0.6B",
                         r_a=r_a, r_b=r_b, delta=dlt, boot_lo=lo, boot_hi=hi,
                         p_bootstrap=p_b, williams_t=t, p_williams=p_w, n_pairs=len(keys)))

    print("\n=== BETWEEN-ARM test at each size: r(forced) - r(no reasoning) ===")
    for sz in sorted(by_fam[P.FAMILIES[0]]):
        a, b = by_fam[P.FAMILIES[1]][sz], by_fam[P.FAMILIES[0]][sz]
        x, ya, yb, keys = aligned_vectors(tbl[a], tbl[b], sim)
        r_a, r_b = np.corrcoef(x, ya)[0, 1], np.corrcoef(x, yb)[0, 1]
        r_yz = np.corrcoef(ya, yb)[0, 1]
        t, df, p_w = williams_t(r_a, r_b, r_yz, len(keys))
        dlt, lo, hi, p_b, nb = domain_bootstrap(tbl[a], tbl[b], sim, domains)
        print(f"  {sz:5.1f}B  r(forced)={r_a:+.3f}  r(no reasoning)={r_b:+.3f}  delta={dlt:+.3f}")
        print(f"          bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  p={p_b:.3f}   "
              f"Williams p={p_w:.4f}")
        rows.append(dict(comparison="between-arm", family="forced vs no reasoning",
                         size=f"{sz}B", r_a=r_a, r_b=r_b, delta=dlt, boot_lo=lo,
                         boot_hi=hi, p_bootstrap=p_b, williams_t=t, p_williams=p_w,
                         n_pairs=len(keys)))

    out = _PROJECT / "analysis" / "data" / "qwen3_endpoint_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
