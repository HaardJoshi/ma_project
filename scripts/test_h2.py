"""
test_h2.py  --  H2: Semantic Divergence — MD&A vs Risk Factor Similarity
================================================================================
Tests if MD&A text similarity → positive CAR (strategic alignment creates value)
while Risk Factor similarity → negative CAR (risk concentration destroys value).

Computes cosine similarity of each deal's text embeddings vs market average,
then runs bivariate regression: CAR ~ β₁·MDA_sim + β₂·RF_sim

Usage:
    python scripts/test_h2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression

from training_utils import load_and_prepare_data, SEED

np.random.seed(SEED)


def main():
    print("=" * 70)
    print("  H2: SEMANTIC DIVERGENCE — MD&A vs RISK FACTOR SIMILARITY")
    print("=" * 70)

    subset, y_cont = load_and_prepare_data()

    mda_cols = sorted([c for c in subset.columns if c.startswith("mda_pca_")])
    rf_cols = sorted([c for c in subset.columns if c.startswith("rf_pca_")])

    if not mda_cols or not rf_cols:
        print("  ❌ No text embeddings available")
        return

    # Filter to deals where BOTH MD&A and RF embeddings are non-zero
    has_mda = subset[mda_cols].abs().sum(axis=1) > 0
    has_rf = subset[rf_cols].abs().sum(axis=1) > 0
    has_both = has_mda & has_rf
    text_subset = subset[has_both].copy()
    y_text = y_cont[has_both.values]
    print(f"  Deals with MD&A embeddings: {has_mda.sum()}")
    print(f"  Deals with RF embeddings:   {has_rf.sum()}")
    print(f"  Deals with BOTH:            {has_both.sum()}")
    print(f"  CAR mean: {y_text.mean():.4f}, std: {y_text.std():.4f}")

    mda_emb = text_subset[mda_cols].values
    rf_emb = text_subset[rf_cols].values

    # Compute cosine similarity to market mean (centroid)
    mda_mean = mda_emb.mean(axis=0)
    rf_mean = rf_emb.mean(axis=0)

    mda_sim = np.array([1 - cosine(mda_emb[i], mda_mean) for i in range(len(mda_emb))])
    rf_sim = np.array([1 - cosine(rf_emb[i], rf_mean) for i in range(len(rf_emb))])

    # Drop any remaining NaN (safety net for zero-norm vectors)
    valid = np.isfinite(mda_sim) & np.isfinite(rf_sim)
    if (~valid).sum() > 0:
        print(f"  ⚠️ Dropped {(~valid).sum()} deals with NaN similarity")
    mda_sim, rf_sim, y_text = mda_sim[valid], rf_sim[valid], y_text[valid]

    print(f"\n  MD&A similarity: mean={mda_sim.mean():.4f}, std={mda_sim.std():.4f}")
    print(f"  RF similarity:   mean={rf_sim.mean():.4f}, std={rf_sim.std():.4f}")

    # ── Bivariate regression ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  BIVARIATE REGRESSION: CAR ~ β₁·MDA_sim + β₂·RF_sim")
    print(f"{'─'*70}")

    X_sim = np.column_stack([mda_sim, rf_sim])
    reg = LinearRegression().fit(X_sim, y_text)
    beta_mda, beta_rf = reg.coef_

    print(f"  CAR = {reg.intercept_:.4f} + {beta_mda:.4f}·MDA_sim + {beta_rf:.4f}·RF_sim")
    print(f"  R² = {reg.score(X_sim, y_text):.6f}")

    # ── Individual correlations ─────────────────────────────────
    print(f"\n{'─'*70}")
    print("  INDIVIDUAL CORRELATIONS")
    print(f"{'─'*70}")

    r_mda, p_mda = stats.pearsonr(mda_sim, y_text)
    r_rf, p_rf = stats.pearsonr(rf_sim, y_text)
    print(f"  MD&A sim vs CAR: r={r_mda:+.4f}, p={p_mda:.4f} {'✅ sig' if p_mda<0.05 else '❌ n.s.'}")
    print(f"  RF sim vs CAR:   r={r_rf:+.4f}, p={p_rf:.4f} {'✅ sig' if p_rf<0.05 else '❌ n.s.'}")

    # ── H2 direction test ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  H2 DIRECTION TEST")
    print(f"{'─'*70}")
    print(f"  β_MDA = {beta_mda:+.4f} (expected: positive → alignment creates value)  "
          f"{'✅' if beta_mda > 0 else '❌'}")
    print(f"  β_RF  = {beta_rf:+.4f} (expected: negative → concentration destroys value) "
          f"{'✅' if beta_rf < 0 else '❌'}")

    # ── Quartile analysis ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  QUARTILE ANALYSIS")
    print(f"{'─'*70}")

    for label, sim_vals in [("MD&A", mda_sim), ("RF", rf_sim)]:
        q_cuts = np.percentile(sim_vals, [25, 50, 75])
        q1 = y_text[sim_vals <= q_cuts[0]]
        q2 = y_text[(sim_vals > q_cuts[0]) & (sim_vals <= q_cuts[1])]
        q3 = y_text[(sim_vals > q_cuts[1]) & (sim_vals <= q_cuts[2])]
        q4 = y_text[sim_vals > q_cuts[2]]

        print(f"\n  {label} Similarity Quartiles:")
        for q_name, q_vals in [("Q1 (low)", q1), ("Q2", q2), ("Q3", q3), ("Q4 (high)", q4)]:
            print(f"    {q_name:10s}: n={len(q_vals):4d} | "
                  f"CAR mean={q_vals.mean():+.4f} | std={q_vals.std():.4f}")

        # Q4 vs Q1 spread
        delta = q4.mean() - q1.mean()
        t_stat, p_val = stats.ttest_ind(q4, q1)
        print(f"    Q4-Q1 Δ = {delta:+.4f} | t={t_stat:.3f}, p={p_val:.4f} "
              f"{'✅ sig' if p_val<0.05 else '❌ n.s.'}")

    # ── Conclusion ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    h2_supported = beta_mda > 0 and beta_rf < 0
    if h2_supported:
        print("  ✅ H2 SUPPORTED: MD&A alignment is positive, RF concentration is negative")
    else:
        print("  ❌ H2 NOT SUPPORTED: coefficient direction not as predicted")
        print(f"     Observed: β_MDA={beta_mda:+.4f}, β_RF={beta_rf:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
