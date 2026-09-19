#!/usr/bin/env python
"""
00_set_paths.py
===============
Points the nine pet_*_paper.py scripts at the real data location, then checks
that the data is actually there before you spend hours running anything.

The scripts as written contain:
    DATA_ROOT = r"C:\\PET_DATA\\IMAGES"
but the data is on the C: drive. Left unfixed, every model would fail
immediately with "not found" for all ten tumours.

This script also auto-detects the image subfolder, counts frames per tumour,
reports image dimensions, and warns about anything that would break the run.

Usage:
    python 00_set_paths.py                      # use the built-in default path
    python 00_set_paths.py --data "C:\\some\\other\\path"
    python 00_set_paths.py --check              # verify only, change nothing
"""

import argparse
import py_compile
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKUP = HERE / "_backup_original"

# Folder holding the pet_*_paper.py files. Normally the same folder as this
# script, but resolved at runtime by locate_scripts() so that dropping these
# tools into the images folder by mistake produces a helpful message rather
# than a FileNotFoundError traceback.
SCRIPTS = HERE

DEFAULT_DATA = r"C:\PET_DATA\IMAGES"

FILES = [
    "pet_unet_paper.py",
    "pet_attention_unet_paper.py",
    "pet_attention_resnet_paper.py",
    "pet_temporal_unet_paper.py",
    "pet_convlstm_paper.py",
    "pet_cvae_paper.py",
    "pet_vit_paper.py",
    "pet_pix2pix_paper.py",
    "pet_naive_baseline_paper.py",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Candidate locations of the images inside each tumour folder, best first.
SUBFOLDER_CANDIDATES = ["cleaned/resized", "resized", "cleaned", ""]


def extract_timepoint(name):
    nums = re.findall(r"\d+", Path(name).stem)
    return int(nums[0]) if nums else None


def find_images(tumor_dir: Path):
    """Return (subfolder_used, sorted image paths). Mirrors get_files() logic."""
    for sub in SUBFOLDER_CANDIDATES:
        d = tumor_dir / sub if sub else tumor_dir
        if not d.is_dir():
            continue
        files = [
            f for f in d.iterdir()
            if f.suffix.lower() in IMG_EXTS
            and not f.name.startswith("_")
            and extract_timepoint(f.name) is not None
        ]
        if files:
            files.sort(key=lambda p: extract_timepoint(p.name))
            return sub, files
    return None, []


def locate_scripts(explicit=None):
    """Find the folder containing pet_unet_paper.py.

    Checked in order: an explicitly supplied folder, this script's own folder,
    its parent, and then a bounded search of the user's Desktop, home folder
    and current directory. Bounded because an unbounded walk of C:\\ on a
    spinning disk can take minutes.
    """
    anchor = "pet_unet_paper.py"

    if explicit:
        cand = Path(explicit).expanduser().resolve()
        if (cand / anchor).exists():
            return cand
        print(f"  [FAIL] --scripts {cand} does not contain {anchor}")
        return None

    for cand in (HERE, HERE.parent):
        if (cand / anchor).exists():
            return cand

    roots = []
    home = Path.home()
    for r in (home / "Desktop", home, Path.cwd(), HERE.parent.parent):
        if r.is_dir() and r not in roots:
            roots.append(r)

    seen = set()
    for root in roots:
        try:
            for depth in range(1, 4):
                pattern = "/".join(["*"] * depth) + "/" + anchor
                for hit in root.glob(pattern):
                    folder = hit.resolve().parent
                    # Ignore our own backup copies -- patching those is useless.
                    if folder.name == "_backup_original":
                        continue
                    if folder not in seen:
                        seen.add(folder)
        except (PermissionError, OSError):
            continue
        if seen:
            break

    if len(seen) == 1:
        return seen.pop()
    if len(seen) > 1:
        print("  [FAIL] Found pet_unet_paper.py in more than one place:")
        for s in sorted(seen):
            print(f"           {s}")
        print("         Re-run with the one you want:")
        print('           python 00_set_paths.py --scripts "FULL_PATH_HERE"')
        return "AMBIGUOUS"
    return None


def read_tumor_list():
    """Pull ALL_TUMORS out of one of the scripts so we check the real names."""
    src = (SCRIPTS / "pet_unet_paper.py").read_text(encoding="utf-8")
    m = re.search(r"ALL_TUMORS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))


def verify(data_root: Path):
    """Check every tumour folder. Returns (ok, detected_subfolder)."""
    print("=" * 72)
    print("  DATA CHECK")
    print("=" * 72)
    print(f"  Looking in: {data_root}\n")

    if not data_root.is_dir():
        print(f"  [FAIL] That folder does not exist.")
        print(f"         Open File Explorer, navigate to your images, copy the")
        print(f"         address bar, and re-run:")
        print(f"           python 00_set_paths.py --data \"PASTE_PATH_HERE\"")
        return False, None

    tumors = read_tumor_list()
    if not tumors:
        print("  [FAIL] Could not read ALL_TUMORS from pet_unet_paper.py")
        return False, None

    on_disk = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    print(f"  Folders found on disk ({len(on_disk)}): {', '.join(on_disk)}\n")

    print(f"  {'tumour (from script)':<22}{'status':<12}{'frames':>8}  {'subfolder':<16}{'size':<12}")
    print("  " + "-" * 72)

    ok_count = 0
    subfolders = set()
    frame_counts = []
    sizes = set()

    for t in tumors:
        tdir = data_root / t
        if not tdir.is_dir():
            # Windows is case-insensitive but be explicit about mismatches.
            match = [d for d in on_disk if d.lower() == t.lower()]
            if match:
                tdir = data_root / match[0]
            else:
                print(f"  {t:<22}{'MISSING':<12}{'-':>8}")
                continue

        sub, files = find_images(tdir)
        if not files:
            print(f"  {t:<22}{'NO IMAGES':<12}{0:>8}")
            continue

        subfolders.add(sub)
        frame_counts.append(len(files))
        size = ""
        try:
            from PIL import Image
            with Image.open(files[0]) as im:
                size = f"{im.width}x{im.height}"
                sizes.add((im.width, im.height))
        except Exception:
            size = "?"

        print(f"  {t:<22}{'ok':<12}{len(files):>8}  {(sub or '(root)'):<16}{size:<12}")
        ok_count += 1

    print("  " + "-" * 72)
    print(f"  {ok_count} of {len(tumors)} tumours found\n")

    if ok_count == 0:
        print("  [FAIL] No usable tumour folders. Nothing will run.")
        return False, None

    problems = []
    if ok_count < len(tumors):
        problems.append(f"{len(tumors)-ok_count} tumour(s) missing — those folds will be skipped")
    if len(subfolders) > 1:
        problems.append(f"images live in different subfolders across tumours: {subfolders}")
    if len(sizes) > 1:
        problems.append(f"images are not all the same size: {sizes}")
    if sizes and list(sizes)[0] != (64, 64):
        w, h = list(sizes)[0]
        problems.append(
            f"images are {w}x{h}, but the scripts set IMG_SIZE=64 and load_frame() "
            f"does NOT resize — the models will receive {w}x{h} tensors"
        )
    if frame_counts and (max(frame_counts) - min(frame_counts)) > 0:
        problems.append(f"frame counts differ per tumour: {min(frame_counts)}–{max(frame_counts)}")

    if problems:
        print("  WARNINGS:")
        for p in problems:
            print(f"    ! {p}")
        print()

    detected = sorted(subfolders)[0] if len(subfolders) == 1 else None
    return True, detected


def set_data_root(data_root: str, subfolder: str | None):
    print("=" * 72)
    print("  UPDATING SCRIPTS")
    print("=" * 72)
    backup = SCRIPTS / "_backup_original"
    backup.mkdir(exist_ok=True)

    esc = data_root.replace("\\", "\\\\")
    changed = 0

    for name in FILES:
        path = SCRIPTS / name
        if not path.exists():
            print(f"  [skip] {name} not found")
            continue

        text = path.read_text(encoding="utf-8")
        original = text

        if not (backup / name).exists():
            shutil.copy2(path, backup / name)

        # NOTE: the replacement must be a callable. If it were a plain string,
        # re.sub would interpret backslashes in a Windows path as escape
        # sequences (r"C:\Users\Name\..." -> \U, \L become garbage or errors).
        new_root = f'DATA_ROOT     = r"{data_root}"'
        text, n1 = re.subn(
            r'DATA_ROOT\s*=\s*r?["\'][^"\']*["\']',
            lambda _m: new_root,
            text, count=1,
        )

        n2 = 0
        if subfolder is not None:
            want = subfolder.replace("/", "\\")
            new_sub = f'IMG_SUBFOLDER = r"{want}"'
            text, n2 = re.subn(
                r'IMG_SUBFOLDER\s*=\s*r?["\'][^"\']*["\']',
                lambda _m: new_sub,
                text, count=1,
            )

        if text == original:
            print(f"  [--]   {name}  (no change needed)")
            continue

        path.write_text(text, encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            path.write_text(original, encoding="utf-8")
            print(f"  [FAIL] {name}: {exc.msg.splitlines()[0]} — rolled back")
            continue

        bits = []
        if n1:
            bits.append("DATA_ROOT")
        if n2:
            bits.append("IMG_SUBFOLDER")
        print(f"  [ok]   {name}  ({', '.join(bits)})")
        changed += 1

    print(f"\n  {changed} file(s) updated. Originals in {backup}")
    return changed


def main():
    global SCRIPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="folder containing the tumour folders")
    ap.add_argument("--scripts", default=None, help="folder containing the pet_*_paper.py files")
    ap.add_argument("--check", action="store_true", help="verify only, change nothing")
    args = ap.parse_args()

    print("=" * 72)
    print("  LOCATING SCRIPTS")
    print("=" * 72)
    found = locate_scripts(args.scripts)
    if found == "AMBIGUOUS":
        sys.exit(1)
    if found is None:
        print("  [FAIL] Could not find pet_unet_paper.py.\n")
        print("  These tools must sit in the SAME FOLDER as your nine")
        print("  pet_*_paper.py files. They are currently in:")
        print(f"      {HERE}")
        print("  which does not contain them.\n")
        print("  Fix it one of two ways:")
        print("    1. Move all 7 of these files into the folder that holds your")
        print("       pet_unet_paper.py, then re-run from there; or")
        print("    2. Tell this script where they are:")
        print('         python 00_set_paths.py --scripts "C:\\path\\to\\scripts"')
        print("\n  Note: that folder is usually NOT the images folder. The images")
        print("  folder is passed separately with --data.")
        sys.exit(1)

    SCRIPTS = found
    print(f"  Scripts: {SCRIPTS}")
    n_found = sum(1 for f in FILES if (SCRIPTS / f).exists())
    print(f"  Found {n_found} of {len(FILES)} expected pet_*_paper.py files\n")

    data_root = Path(args.data)
    ok, detected = verify(data_root)

    if not ok:
        sys.exit(1)

    if args.check:
        print("  (check only — no files were modified)")
        return

    set_data_root(str(data_root), detected)

    print("\n" + "=" * 72)
    print("  Next step (run this from the scripts folder):")
    print(f"    cd /d \"{SCRIPTS}\"")
    print("    python 02_patch_scripts.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
