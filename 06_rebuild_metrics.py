#!/usr/bin/env python
"""
06_rebuild_metrics.py
=====================
Rewrites the per-frame metric CSVs so every model is scored in the SAME space.

THE PROBLEM THIS SOLVES
Only the six models that import pet_common.py honour PET_METRIC_SPACE. The
nine original pet_*_paper.py scripts compute their own SSIM/PSNR/MAE on the
raw jet-colormapped RGB and know nothing about the setting. A sweep run with
PET_METRIC_SPACE=activity therefore produces a table in which six models are
scored on decoded activity and nine on RGB -- and the gap is not small: the
kinetic baseline moves 0.204 SSIM between the two spaces, diffusion 0.113.
Ranking those against each other is meaningless.

WHY NO RETRAINING IS NEEDED (for the pet_common models)
finalize_fold() saves fold_outputs.npz containing the predictions AND both
metric sets. This script reads those files and rewrites per_frame_<model>.csv
in whichever space you ask for. The nine originals save no predictions, so
they can only be reported in RGB -- which means:

    --space rgb        works for ALL models right now. Use this to get a
                       consistent table today.
    --space activity   only the six pet_common models can be converted; the
                       others are listed as unavailable and excluded, because
                       silently leaving them in RGB is what caused the problem.

Run 04_compare_models.py afterwards to regenerate the tables.

Usage:
    python 06_rebuild_metrics.py --results RESULTS_final --space rgb
    python 04_compare_models.py --results RESULTS_final
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

PET_COMMON_MODELS = {'kinetic', 'simvp', 'cgan', 'diffusion', 'cvae2', 'swin'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--space", choices=["rgb", "activity"], default="rgb")
    args = ap.parse_args()

    root = Path(args.results)
    suffix = 'rgb' if args.space == 'rgb' else 'act'

    converted, unavailable, skipped = [], [], []

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name
        if model in ('logs', 'analysis'):
            continue

        npzs = sorted(model_dir.glob('loocv/*/fold_outputs.npz'))
        pf = model_dir / f'per_frame_{model}.csv'

        if not npzs:
            # No saved predictions. RGB is what its own script already wrote,
            # so leave it alone; activity is simply not recoverable.
            if args.space == 'rgb':
                skipped.append(model)
            else:
                unavailable.append(model)
                if pf.exists():
                    shutil.move(str(pf), str(pf.with_suffix('.csv.excluded')))
            continue

        # Fold directories are SANITISED tumour names ('t1-r=0.1' -> 't1-r0p1',
        # via .replace('=','').replace('.','p')). The paired statistics in
        # 04_compare_models.py match models on test_tumor, so writing the
        # directory name would silently drop every rebuilt model from the
        # pairwise and vs-baseline tables. Recover the real names from the
        # model's own LOOCV summary, which stores them unmodified.
        name_map = {}
        for src in list(model_dir.glob('loocv_results_*.csv')) + \
                   list(model_dir.glob(f'per_frame_{model}.csv.orig')):
            try:
                with open(src, newline='', encoding='utf-8') as fh:
                    for rec in csv.DictReader(fh):
                        real = rec.get('test_tumor')
                        if real:
                            name_map[real.replace('=', '').replace('.', 'p')] = real
            except Exception:
                continue

        rows = []
        for npz_path in npzs:
            try:
                d = np.load(npz_path)
            except Exception as exc:
                print(f"  [warn] unreadable {npz_path}: {exc}")
                continue
            tumor = name_map.get(npz_path.parent.name, npz_path.parent.name)
            ssim = d[f'ssim_{suffix}']
            psnr = d[f'psnr_{suffix}']
            mae = d[f'mae_{suffix}']
            times = d['times_min'] if 'times_min' in d else np.arange(len(ssim))
            for k in range(len(ssim)):
                rows.append({
                    'model': model, 'test_tumor': tumor, 'frame_idx': k,
                    't_min': times[k] if k < len(times) else '',
                    'ssim': float(ssim[k]), 'psnr': float(psnr[k]),
                    'mae': float(mae[k]),
                })

        if not rows:
            continue
        if pf.exists():
            backup = pf.with_suffix('.csv.orig')
            if not backup.exists():
                shutil.copy2(pf, backup)
        with open(pf, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        n_mapped = sum(1 for d in npzs if d.parent.name in name_map)
        converted.append((model, len(npzs), len(rows)))
        if n_mapped < len(npzs):
            print(f"  [warn] {model}: only {n_mapped}/{len(npzs)} fold names "
                  f"could be mapped back to tumour names -- unmapped folds "
                  f"will not pair with other models in the statistics")

    print("=" * 70)
    print(f"  REBUILT METRICS IN '{args.space.upper()}' SPACE")
    print("=" * 70)
    for m, nf, nr in converted:
        print(f"  rebuilt   {m:<18} {nf} folds, {nr} frames")
    for m in skipped:
        print(f"  unchanged {m:<18} (own script already wrote RGB)")
    for m in unavailable:
        print(f"  EXCLUDED  {m:<18} no saved predictions -> activity metrics")
        print(f"            not recoverable without retraining")

    if unavailable:
        print(f"\n  {len(unavailable)} model(s) were excluded rather than left")
        print("  in the wrong space. To include them in an activity-space")
        print("  table they must be re-run with predictions saved.")

    print(f"\nNext:  python 04_compare_models.py --results {root}")


if __name__ == '__main__':
    main()
