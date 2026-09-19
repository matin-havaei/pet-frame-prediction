#!/usr/bin/env python
"""
02_patch_scripts.py
===================
Applies three surgical, reversible fixes to the nine pet_*_paper.py scripts.

FIX 1 -- LOOCV validation rotation
    Current code:   remaining = [t for t in ALL_TUMORS if t != test_t]
                    val_t   = remaining[:2]
                    train_t = remaining[2:]
    `remaining[:2]` always takes the first two tumours in list order, so the
    first two entries of ALL_TUMORS end up in the validation set in 8 of the
    10 folds and therefore almost never contribute to training, while the last
    entries are never used for early stopping. The replacement rotates the
    validation pair over a fixed seeded permutation so that every tumour serves
    as validation in exactly `N_VAL` folds. Deterministic and reproducible.

FIX 2 -- runtime overrides via environment variables
    LOOCV_EPOCHS / MIN_EPOCHS / PATIENCE / BATCH_SIZE become overridable so the
    orchestrator can select a time budget and retry with a smaller batch after
    a CUDA OOM, without editing source again.

FIX 3 -- cuDNN autotuning
    torch.backends.cudnn.benchmark = True. Input shapes are fixed here, so this
    is free throughput (typically 10-20% on Turing).

Every file is backed up to ./_backup_original/ before modification and the
result is byte-compiled to catch any syntax damage immediately.

Usage:
    python 02_patch_scripts.py            # apply
    python 02_patch_scripts.py --revert   # restore originals
    python 02_patch_scripts.py --check    # report status only
"""

import argparse
import py_compile
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKUP = HERE / "_backup_original"

LEARNED = [
    "pet_unet_paper.py",
    "pet_attention_unet_paper.py",
    "pet_attention_resnet_paper.py",
    "pet_temporal_unet_paper.py",
    "pet_convlstm_paper.py",
    "pet_cvae_paper.py",
    "pet_vit_paper.py",
    "pet_pix2pix_paper.py",
]
NAIVE = "pet_naive_baseline_paper.py"
ALL_FILES = LEARNED + [NAIVE]

MARKER = "# --- PATCHED-BY-02_patch_scripts ---"

# ---------------------------------------------------------------------------
# FIX 1 : rotating, seeded validation selection
# ---------------------------------------------------------------------------

SPLIT_HELPER = '''
''' + MARKER + '''
import os as _pet_os
import random as _pet_random


def _pet_fold_split(all_tumors, test_t, n_val=2, seed=42):
    """Deterministic LOOCV fold construction with rotating validation set.

    A fixed seeded permutation of the tumour list is created once; for the
    fold whose test tumour sits at position i in that permutation, the
    validation tumours are the next `n_val` entries cyclically after i, and
    everything else is training data. Consequences:
      * every tumour is used for validation in exactly `n_val` folds;
      * every tumour is used for training in len(all_tumors)-1-n_val folds;
      * the split is a pure function of (list, test tumour, seed).
    """
    order = list(all_tumors)
    _pet_random.Random(seed).shuffle(order)
    n = len(order)
    if n_val >= n:
        raise ValueError("n_val must be smaller than the number of tumours")
    i = order.index(test_t)
    rest = [order[(i + k) % n] for k in range(1, n)]
    return rest[:n_val], rest[n_val:]


def _pet_env_int(name, default):
    raw = _pet_os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[patch] ignoring non-integer {name}={raw!r}")
        return default
''' + MARKER + '''
'''

# Exact source variants of the split block found across the eight learned files.
SPLIT_VARIANTS = [
    (
        "        remaining = [t for t in ALL_TUMORS if t != test_t]\n"
        "        val_t   = remaining[:2]\n"
        "        train_t = remaining[2:]\n"
    ),
    (
        "        remaining = [t for t in ALL_TUMORS if t != test_t]\n"
        "        val_t  = remaining[:2]    # first 2 as val\n"
        "        train_t = remaining[2:]\n"
    ),
    (
        "        remaining = [t for t in ALL_TUMORS if t != test_t]\n"
        "        val_t = remaining[:2]; train_t = remaining[2:]\n"
    ),
]

SPLIT_REPLACEMENT = (
    "        # " + MARKER + " rotating seeded validation split\n"
    "        val_t, train_t = _pet_fold_split(ALL_TUMORS, test_t,\n"
    "                                         n_val=_pet_env_int('PET_N_VAL', 2),\n"
    "                                         seed=SEED)\n"
)

# ---------------------------------------------------------------------------
# FIX 2 + 3 : env overrides and cuDNN autotune, anchored on torch.manual_seed
# ---------------------------------------------------------------------------

ANCHOR = "torch.manual_seed(SEED)"

RUNTIME_BLOCK = """torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
""" + MARKER + """
# Fixed input shapes throughout -> cuDNN can safely autotune its algorithms.
torch.backends.cudnn.benchmark = True

# Time-budget and memory overrides supplied by 03_run_all.py.
LOOCV_EPOCHS = _pet_env_int('PET_LOOCV_EPOCHS', LOOCV_EPOCHS)
EPOCHS       = _pet_env_int('PET_EPOCHS',       EPOCHS)
MIN_EPOCHS   = _pet_env_int('PET_MIN_EPOCHS',   MIN_EPOCHS)
PATIENCE     = _pet_env_int('PET_PATIENCE',     PATIENCE)
BATCH_SIZE   = _pet_env_int('PET_BATCH_SIZE',   BATCH_SIZE)
print(f"[cfg] loocv_epochs={LOOCV_EPOCHS} min_epochs={MIN_EPOCHS} "
      f"patience={PATIENCE} batch={BATCH_SIZE}")
""" + MARKER + """
"""


def is_patched(text):
    return MARKER in text


def patch_one(path: Path, learned: bool):
    text = path.read_text(encoding="utf-8")
    if is_patched(text):
        return "already patched"

    original = text
    notes = []

    # Insert the helper functions immediately before the SEED assignment so
    # they are defined before any use.
    seed_line = "SEED = 42"
    if seed_line not in text:
        return "SKIP: no 'SEED = 42' anchor found"
    text = text.replace(seed_line, SPLIT_HELPER.strip() + "\n\n" + seed_line, 1)
    notes.append("helpers")

    if learned:
        # FIX 1
        hits = [v for v in SPLIT_VARIANTS if v in text]
        if len(hits) != 1:
            return f"SKIP: expected 1 split-block match, found {len(hits)}"
        text = text.replace(hits[0], SPLIT_REPLACEMENT, 1)
        notes.append("split-rotation")

        # FIX 2 + 3
        if text.count(ANCHOR) != 1:
            return f"SKIP: expected 1 '{ANCHOR}', found {text.count(ANCHOR)}"
        text = text.replace(ANCHOR, RUNTIME_BLOCK.rstrip(), 1)
        notes.append("env-overrides+cudnn")

    # Back up, write, verify.
    BACKUP.mkdir(exist_ok=True)
    backup_path = BACKUP / path.name
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    path.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        path.write_text(original, encoding="utf-8")
        return f"FAIL: syntax error, rolled back ({exc.msg.splitlines()[0]})"

    return "patched (" + ", ".join(notes) + ")"


def main():
    ap = argparse.ArgumentParser(description="Patch pet_*_paper.py for LOOCV")
    ap.add_argument("--revert", action="store_true", help="restore originals")
    ap.add_argument("--check", action="store_true", help="report status only")
    args = ap.parse_args()

    missing = [f for f in ALL_FILES if not (HERE / f).exists()]
    if missing:
        print("[FAIL] These files are not in this folder:")
        for m in missing:
            print("   ", m)
        print("\nPut 02_patch_scripts.py in the same folder as the pet_*.py files.")
        sys.exit(1)

    if args.revert:
        if not BACKUP.exists():
            print("[FAIL] no _backup_original/ directory -- nothing to revert")
            sys.exit(1)
        for f in ALL_FILES:
            src = BACKUP / f
            if src.exists():
                shutil.copy2(src, HERE / f)
                print(f"  restored  {f}")
        print("\nAll files reverted to their original state.")
        return

    if args.check:
        for f in ALL_FILES:
            state = "PATCHED" if is_patched((HERE / f).read_text(encoding="utf-8")) else "original"
            print(f"  {f:<38} {state}")
        return

    print("Patching nine scripts (originals saved to _backup_original/)\n")
    failures = 0
    for f in ALL_FILES:
        result = patch_one(HERE / f, learned=(f in LEARNED))
        flag = "  " if result.startswith(("patched", "already")) else "!!"
        if flag == "!!":
            failures += 1
        print(f"{flag} {f:<38} {result}")

    print()
    if failures:
        print(f"[FAIL] {failures} file(s) could not be patched -- do not start the run.")
        sys.exit(1)
    print("[ok] All files patched and byte-compiled cleanly.")
    print("Next:  python 03_run_all.py --profile fast")


if __name__ == "__main__":
    main()
