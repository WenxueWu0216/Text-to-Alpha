"""API spend ledger and budget gates (rule 4: estimate first, stop before the cap)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Append-only record of real API spend, computed from returned token usage."""

    def __init__(self, path: Path):
        self.path = path

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def spent(self) -> float:
        return sum(e["usd"] for e in self.entries())

    def record(self, label: str, usd: float, **info) -> None:
        if usd <= 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "label": label, "usd": round(usd, 6), **info}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def check_budget(estimate: float, spent: float, run_cap: float, total_cap: float,
                 reserve: float = 0.0) -> None:
    if estimate > run_cap:
        raise BudgetExceeded(f"estimated ${estimate:.2f} exceeds this run's cap ${run_cap:.2f}")
    if spent + estimate > total_cap - reserve:
        raise BudgetExceeded(
            f"spent ${spent:.2f} + estimated ${estimate:.2f} exceeds the total cap "
            f"${total_cap:.2f}" + (f" minus the ${reserve:.2f} reserve" if reserve else ""))
