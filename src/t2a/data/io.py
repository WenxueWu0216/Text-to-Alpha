"""Single entry point for loading labeled data, with the test period locked.

The test split must be opened exactly once, in the final comparison. Every loader of
labeled data goes through here, so an accidental `pd.read_parquet(events)` followed by
an evaluation on test is replaced by an explicit, logged decision.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from t2a.config import REPO_ROOT

UNLOCK_ENV = "T2A_UNLOCK_TEST"
UNLOCK_LOG = REPO_ROOT / "results" / "test_unlock_log.jsonl"
DEFAULT_SPLITS = ("train", "val")


class LockedSplitError(RuntimeError):
    pass


def check_splits(splits: tuple[str, ...] | list[str], unlock_test: bool = False) -> None:
    if "test" in splits and not (unlock_test and os.environ.get(UNLOCK_ENV) == "1"):
        # Two keys (argument AND env var) so neither a default argument nor a stray
        # shell variable alone can open the test set.
        raise LockedSplitError(
            "The test split is locked. It is opened once, in the final comparison, by passing "
            f"unlock_test=True AND setting {UNLOCK_ENV}=1."
        )


def _log_unlock(what: str) -> None:
    from t2a.provenance import git_commit  # local import: provenance imports config only

    UNLOCK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "what": what, "git_commit": git_commit()}
    with open(UNLOCK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_split_table(path: Path, split_col: str, splits, unlock_test: bool,
                      columns: list[str] | None) -> pd.DataFrame:
    splits = tuple(splits)
    check_splits(splits, unlock_test)
    if columns is not None and split_col not in columns:
        columns = [*columns, split_col]
    df = pd.read_parquet(path, columns=columns)
    if "test" in splits:
        _log_unlock(f"{path.name}:{','.join(splits)}")
    return df[df[split_col].isin(splits)].reset_index(drop=True)


def load_events(cfg: dict, splits=DEFAULT_SPLITS, *, unlock_test: bool = False,
                columns: list[str] | None = None) -> pd.DataFrame:
    return _load_split_table(cfg["paths"]["events"], "split", splits, unlock_test, columns)


def load_panel(cfg: dict, splits=DEFAULT_SPLITS, *, unlock_test: bool = False,
               columns: list[str] | None = None) -> pd.DataFrame:
    return _load_split_table(cfg["paths"]["panel"], "split", splits, unlock_test, columns)
