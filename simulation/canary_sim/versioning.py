"""Dataset versioning (SYNTHETIC / SIMULATION): content hashes + provenance metadata.

`write_dataset_metadata(out_dir, config_files)` writes `<out_dir>/metadata.json` containing:
  * sha256 + size of every data file in the directory (metadata.json itself excluded),
  * the full text and sha256 of every config file used,
  * git commit hash and whether the working tree is dirty,
  * versions of the libraries the project imports,
  * creation time (UTC, ISO-8601).
`verify_dataset_dir(out_dir)` recomputes the hashes and returns a list of problems.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

METADATA_NAME = "metadata.json"
LIBRARIES = ("numpy", "pandas", "scikit-learn", "PyYAML", "matplotlib", "pydantic", "pytest")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_state(repo_dir: str | Path) -> dict:
    repo_dir = Path(repo_dir)
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    return {"commit": commit, "dirty": (None if status is None else bool(status)),
            "dirty_file_count": (None if status is None else len([l for l in status.splitlines() if l.strip()]))}


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            out[lib] = None
    return out


def _data_files(out_dir: Path) -> list[Path]:
    return sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != METADATA_NAME)


def write_dataset_metadata(out_dir: str | Path, config_files: dict[str, str | Path] | None = None,
                           repo_dir: str | Path | None = None, extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    repo_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[1]
    files = {}
    for p in _data_files(out_dir):
        rel = p.relative_to(out_dir).as_posix()
        files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    cfgs = {}
    for name, path in (config_files or {}).items():
        path = Path(path)
        txt = path.read_text(encoding="utf-8")
        cfgs[name] = {"path": path.as_posix(), "sha256": sha256_file(path), "content": txt}
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": True,
        "git": git_state(repo_dir),
        "libraries": library_versions(),
        "configs": cfgs,
        "files": files,
    }
    if extra:
        meta["extra"] = extra
    p = out_dir / METADATA_NAME
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def verify_dataset_dir(out_dir: str | Path) -> list[str]:
    """Recompute hashes; return list of human-readable problems (empty list = verified)."""
    out_dir = Path(out_dir)
    mp = out_dir / METADATA_NAME
    if not mp.exists():
        return [f"MISSING metadata file: {mp}"]
    meta = json.loads(mp.read_text(encoding="utf-8"))
    problems = []
    for rel, rec in meta["files"].items():
        p = out_dir / rel
        if not p.exists():
            problems.append(f"MISSING file: {rel}")
        elif sha256_file(p) != rec["sha256"]:
            problems.append(f"HASH MISMATCH: {rel}")
    on_disk = {p.relative_to(out_dir).as_posix() for p in _data_files(out_dir)}
    for rel in sorted(on_disk - set(meta["files"])):
        problems.append(f"UNLISTED file (not in metadata): {rel}")
    for name, rec in meta.get("configs", {}).items():
        import hashlib as _h
        if _h.sha256(rec["content"].encode("utf-8")).hexdigest() != rec["sha256"] and \
                _h.sha256(rec["content"].replace("\n", "\r\n").encode("utf-8")).hexdigest() != rec["sha256"]:
            problems.append(f"CONFIG record inconsistent: {name}")
    return problems
