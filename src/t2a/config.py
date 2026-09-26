"""Load the YAML config once and resolve every path relative to the repo root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
    cfg["tickers"] = sorted(cfg["universe"].keys())
    # "Alphabet|Google" -> ["Alphabet", "Google"]
    cfg["aliases"] = {t: str(v).split("|") for t, v in cfg["universe"].items()}
    return cfg


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
