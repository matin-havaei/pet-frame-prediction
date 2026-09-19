#!/usr/bin/env python
"""
04_compare_models.py
====================
Turns the per-model LOOCV output into the statistics a Q1 reviewer will expect.

Produces:
  1. comparison_table.csv  -- per model: mean SSIM/PSNR/MAE across folds with
     95% CIs from a CLUSTER bootstrap that resamples TUMOURS, not frames. The
     ~160 frames within a tumour are strongly correlated, so a naive
     frame-level bootstrap would give CIs several times too narrow. This is
     the single most common statistical criticism of imaging papers with few
     subjects, and doing it correctly is cheap.

  2. pairwise_tests.csv -- every model against every other, Wilcoxon
     signed-rank on the per-tumour mean SSIM (paired by tumour, which is what
     makes n=10 usable), plus Holm-Bonferroni correction across the family of
     comparisons. Wilcoxon rather than a paired t-test because with 10 folds
     you cannot check normality in any meaningful way.

  3. vs_baseline.csv -- each learned model against the last-frame-repeat floor.
     This is the comparison that establishes the models learned anything.

  4. paper_summary.txt -- drop-in Methods and Results sentences with the real
     numbers filled in.

Usage:
    python 04_compare_models.py --results RESULTS_fast
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
N_BOOT = 10000


def load_per_frame(results_root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(results_root.glob("*/per_frame_*.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise SystemExit(
            f"No per_frame_*.csv found under {results_root}.\n"
            "Run 03_run_all.py first, and check that it completed at least one model."
        )
    df = pd.concat(frames, ignore_index=True)
    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
    df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    return df.dropna(subset=["ssim"])


def cluster_bootstrap_ci(per_tumor_means, n_boot=N_BOOT, alpha=0.05):
    """Percentile CI for the mean, resampling tumours with replacement."""
    vals = np.asarray(per_tumor_means, dtype=float)
    n = len(vals)
    if n < 2:
        return float(vals.mean()) if n else np.nan, np.nan, np.nan
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank with the degenerate cases handled explicitly."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return np.nan, 1.0
    if len(d) < 6:
        # With <6 pairs the smallest attainable two-sided p is >0.05, so the
        # test cannot reach significance. Report it rather than hide it.
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
            return float(stat), float(p)
        except ValueError:
            return np.nan, 1.0
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox")
    return float(stat), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--metric", default="ssim", choices=["ssim", "psnr", "mae"],
                    help="metric used for the significance tests")
    args = ap.parse_args()

    root = Path(args.results)
    out_dir = root / "analysis"
    out_dir.mkdir(exist_ok=True)

    df = load_per_frame(root)
    metric = args.metric
    higher_is_better = metric in ("ssim", "psnr")

    print(f"Loaded {len(df):,} per-frame records | "
          f"{df['model'].nunique()} models | {df['test_tumor'].nunique()} tumours")

    # ---- per-tumour means: the unit of analysis --------------------------
    per_tumor = (df.groupby(["model", "test_tumor"])[["ssim", "psnr", "mae"]]
                   .mean().reset_index())
    per_tumor.to_csv(out_dir / "per_tumor_means.csv", index=False)

    # ---- descriptive table with cluster-bootstrap CIs --------------------
    rows = []
    for model, g in per_tumor.groupby("model"):
        row = {"model": model, "n_folds": len(g)}
        for m in ("ssim", "psnr", "mae"):
            mean, lo, hi = cluster_bootstrap_ci(g[m].values)
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_sd"] = round(float(g[m].std(ddof=1)), 4)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        f"{metric}_mean", ascending=not higher_is_better).reset_index(drop=True)
    table.to_csv(out_dir / "comparison_table.csv", index=False)

    print(f"\n{'MODEL':<20}{'SSIM (95% CI)':<30}{'PSNR dB':<20}{'MAE':<18}")
    print("-" * 88)
    for _, r in table.iterrows():
        print(f"{r['model']:<20}"
              f"{r['ssim_mean']:.4f} [{r['ssim_ci_lo']:.4f}, {r['ssim_ci_hi']:.4f}]  "
              f"{r['psnr_mean']:6.2f} [{r['psnr_ci_lo']:.2f}, {r['psnr_ci_hi']:.2f}]  "
              f"{r['mae_mean']:.4f} [{r['mae_ci_lo']:.4f}, {r['mae_ci_hi']:.4f}]")

    # ---- pairwise paired tests on the shared tumours ---------------------
    wide = per_tumor.pivot(index="test_tumor", columns="model", values=metric)
    models = sorted(wide.columns)
    pair_rows = []
    for m1, m2 in itertools.combinations(models, 2):
        sub = wide[[m1, m2]].dropna()
        if len(sub) < 3:
            continue
        stat, p = paired_wilcoxon(sub[m1].values, sub[m2].values)
        diff = float((sub[m1] - sub[m2]).mean())
        pair_rows.append({
            "model_a": m1, "model_b": m2, "n_pairs": len(sub),
            f"mean_{metric}_a": round(float(sub[m1].mean()), 4),
            f"mean_{metric}_b": round(float(sub[m2].mean()), 4),
            "mean_diff": round(diff, 4),
            "wilcoxon_stat": stat, "p_raw": p,
        })

    if pair_rows:
        pairs = pd.DataFrame(pair_rows)
        pairs["p_holm"] = holm(pairs["p_raw"].values)
        pairs["significant_holm_0.05"] = pairs["p_holm"] < 0.05
        pairs = pairs.sort_values("p_raw").reset_index(drop=True)
        pairs.to_csv(out_dir / "pairwise_tests.csv", index=False)
        print(f"\nPairwise Wilcoxon on {metric} "
              f"({len(pairs)} comparisons, Holm-corrected)")
        print(pairs.head(12).to_string(index=False))
        n_sig = int(pairs["significant_holm_0.05"].sum())
        print(f"\n{n_sig} of {len(pairs)} comparisons survive Holm correction.")

        # Hard power ceiling. The smallest two-sided p the Wilcoxon signed-rank
        # test can return with n pairs is 2/2**n (every pair moving the same
        # way). Holm multiplies the smallest p in the family by the family
        # size, so if 2/2**n * len(family) > 0.05 then NO comparison can reach
        # significance no matter how large the effect. At n=10 that ceiling is
        # 0.00195, which caps the family at 25 comparisons -- all-pairs over
        # 10 models (45 comparisons) is mathematically futile.
        n_pairs_min = int(pairs["n_pairs"].min())
        p_floor = 2.0 / (2 ** n_pairs_min)
        if p_floor * len(pairs) > 0.05:
            max_family = int(0.05 / p_floor)
            print("\n" + "!" * 70)
            print("  POWER CEILING REACHED -- this is a design issue, not a result")
            print("!" * 70)
            print(f"  With {n_pairs_min} paired folds the smallest attainable two-sided")
            print(f"  Wilcoxon p is {p_floor:.5f}. Holm-correcting across {len(pairs)} comparisons")
            print(f"  raises it to {p_floor*len(pairs):.4f}, so no comparison in this family")
            print(f"  CAN be significant regardless of effect size.")
            print(f"\n  Fix: pre-specify a smaller comparison family (max {max_family}")
            print(f"  comparisons at n={n_pairs_min}). The vs_baseline table below uses")
            print(f"  a restricted family and does retain power. Declare the chosen")
            print(f"  family in the protocol BEFORE looking at results, and report")
            print(f"  all-pairs as descriptive/exploratory only.")
            print("!" * 70)
    else:
        pairs = pd.DataFrame()
        n_sig = 0

    # ---- every model against the floor baseline --------------------------
    baseline = "last_frame" if "last_frame" in models else None
    base_rows = []
    if baseline:
        others = [m for m in models if m not in ("last_frame", "mean_frame")]
        for m in others:
            sub = wide[[m, baseline]].dropna()
            if len(sub) < 3:
                continue
            stat, p = paired_wilcoxon(sub[m].values, sub[baseline].values)
            base_rows.append({
                "model": m, "n_pairs": len(sub),
                f"{metric}_model": round(float(sub[m].mean()), 4),
                f"{metric}_baseline": round(float(sub[baseline].mean()), 4),
                "improvement": round(float((sub[m] - sub[baseline]).mean()), 4),
                "wilcoxon_stat": stat, "p_raw": p,
            })
        if base_rows:
            vb = pd.DataFrame(base_rows)
            vb["p_holm"] = holm(vb["p_raw"].values)
            vb["beats_baseline"] = (vb["improvement"] > 0) & (vb["p_holm"] < 0.05)
            vb = vb.sort_values("improvement", ascending=False).reset_index(drop=True)
            vb.to_csv(out_dir / "vs_baseline.csv", index=False)
            print(f"\nVersus last-frame-repeat floor:")
            print(vb.to_string(index=False))
    else:
        print("\n[warn] no last_frame baseline found -- run the naive model too")

    # ---- paper-ready text ------------------------------------------------
    best = table.iloc[0]
    lines = [
        "METHODS",
        "-------",
        f"Model performance was estimated by leave-one-tumour-out cross-validation "
        f"across all {int(best['n_folds'])} tumours. In each fold one tumour was held out "
        f"entirely for testing; of the remaining tumours, two were assigned to a "
        f"validation set used solely for early stopping and model selection, and the "
        f"rest were used for training. Validation assignment followed a fixed seeded "
        f"rotation so that each tumour served as validation data in exactly two folds, "
        f"ensuring no tumour was systematically excluded from training. Hyperparameters "
        f"were fixed a priori and identical across all architectures; no tuning was "
        f"performed against test-fold performance. Image normalisation was applied "
        f"per-frame and therefore involved no statistic pooled across folds.",
        "",
        f"Predictions were evaluated with SSIM, PSNR and MAE. Because the "
        f"approximately 170 frames belonging to a tumour are not independent "
        f"observations, the tumour was treated as the unit of analysis: metrics were "
        f"averaged within each held-out tumour and 95% confidence intervals were "
        f"obtained by a cluster bootstrap resampling tumours with replacement "
        f"({N_BOOT:,} resamples). Architectures were compared using Wilcoxon "
        f"signed-rank tests paired by tumour, with Holm-Bonferroni correction across "
        f"the family of pairwise comparisons.",
        "",
        "RESULTS",
        "-------",
        f"Across leave-one-tumour-out cross-validation, the best-performing "
        f"architecture was {best['model']} "
        f"(SSIM {best['ssim_mean']:.3f}, 95% CI {best['ssim_ci_lo']:.3f}-{best['ssim_ci_hi']:.3f}; "
        f"PSNR {best['psnr_mean']:.2f} dB, 95% CI {best['psnr_ci_lo']:.2f}-{best['psnr_ci_hi']:.2f}; "
        f"MAE {best['mae_mean']:.4f}, 95% CI {best['mae_ci_lo']:.4f}-{best['mae_ci_hi']:.4f}).",
        "",
        f"Of {len(pairs) if len(pairs) else 0} pairwise architecture comparisons, "
        f"{n_sig} remained significant after Holm-Bonferroni correction.",
        "",
        "LIMITATIONS (do not omit these)",
        "-------------------------------",
        f"With {int(best['n_folds'])} tumours, a single misclassified or poorly "
        f"reconstructed case shifts the mean substantially, and the confidence "
        f"intervals above should be read as wide. Cross-validation provides internal "
        f"validation only; the models have not been evaluated on an independent "
        f"external cohort, and external validation is required before any claim of "
        f"generalisability. This study should be framed as a proof-of-concept "
        f"feasibility analysis rather than a validated predictive model.",
    ]
    text = "\n".join(lines)
    (out_dir / "paper_summary.txt").write_text(text, encoding="utf-8")

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
    print(f"\nWritten to {out_dir}:")
    for f in sorted(out_dir.glob("*")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
