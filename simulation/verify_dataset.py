#!/usr/bin/env python
"""Verify a dataset directory against its metadata.json (SHA-256 of every data file).

Usage: python verify_dataset.py <dataset_dir> [<dataset_dir> ...]
Exit code 0 = all verified, 1 = at least one mismatch / missing / unlisted file.
Read-only: never modifies the directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from canary_sim.versioning import verify_dataset_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for d in argv:
        problems = verify_dataset_dir(d)
        if problems:
            bad += 1
            print(f"[FAIL] {d}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"[OK]   {d}: all file hashes verified")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
