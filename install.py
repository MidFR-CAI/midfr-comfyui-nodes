#!/usr/bin/env python3
"""Copy node packs into a ComfyUI custom_nodes folder.

    python install.py <path/to/ComfyUI/custom_nodes> [A B C D]

With no pack letters, installs everything. Existing folders are left alone
unless you pass --overwrite. Nothing is downloaded or executed.
"""
import shutil
import sys
from pathlib import Path

PACKS = {"A": "A-moss-control", "B": "B-image-tagging", "C": "C-h3-autoprompt", "D": "D-toolbox"}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")


def main(argv):
    overwrite = "--overwrite" in argv
    args = [a for a in argv if a != "--overwrite"]
    if not args:
        print(__doc__)
        return 1
    dest = Path(args[0])
    if not dest.is_dir():
        print(f"Not a folder: {dest}")
        return 1
    letters = [a.upper() for a in args[1:]] or list(PACKS)
    bad = [l for l in letters if l not in PACKS]
    if bad:
        print(f"Unknown pack(s): {', '.join(bad)} (choose from A B C D)")
        return 1
    root = Path(__file__).resolve().parent
    for l in letters:
        for src in sorted(p for p in (root / PACKS[l]).iterdir() if p.is_dir()):
            target = dest / src.name
            if target.exists():
                if not overwrite:
                    print(f"skip   {src.name} (already exists; use --overwrite)")
                    continue
                shutil.rmtree(target)
            shutil.copytree(src, target, ignore=IGNORE)
            print(f"copied {src.name}")
    print("Done. Restart ComfyUI.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
