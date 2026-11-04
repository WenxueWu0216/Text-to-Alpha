"""Attach the code version and config to every saved result, so each number is traceable."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from t2a.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    # Results written from a dirty tree are flagged, not blocked: the flag tells a reader
    # the commit hash alone may not reproduce the number.
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def serializable_config(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if k == "paths":
            out[k] = {pk: str(Path(pv).relative_to(REPO_ROOT)) for pk, pv in v.items()}
        else:
            out[k] = v
    return out


def _default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _nan_to_none(o: Any):
    if isinstance(o, dict):
        return {str(k): _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nan_to_none(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None  # strict JSON has no NaN; null is honest about "undefined"
    return o


def write_result(name: str, payload: dict, cfg: dict, out_dir: Path = RESULTS_DIR) -> Path:
    doc = {
        "method": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        **payload,
        "config": serializable_config(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(_nan_to_none(json.loads(json.dumps(doc, default=_default))),
                               indent=2))
    return path
