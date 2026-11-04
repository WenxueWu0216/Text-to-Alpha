"""Loughran-McDonald lexicon baseline: the floor every learned method must clear.

score = (n_positive - n_negative) / (n_positive + n_negative), 0 when no LM word occurs.
Scored on headline + article lead: on a ~10-word headline most items contain no LM
word at all, which would make the baseline mostly ties rather than a weak signal.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[A-Za-z]+")


def load_lm(lexicon_dir: Path) -> tuple[frozenset[str], frozenset[str]]:
    read = lambda name: frozenset((lexicon_dir / name).read_text().split())  # noqa: E731
    return read("lm_positive.txt"), read("lm_negative.txt")


class LexiconScorer:
    def __init__(self, positive: frozenset[str], negative: frozenset[str]):
        self.pos, self.neg = positive, negative

    def counts(self, text: str) -> tuple[int, int]:
        tokens = _TOKEN.findall(text.upper())
        return sum(t in self.pos for t in tokens), sum(t in self.neg for t in tokens)

    def score(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        for i, text in enumerate(texts):
            p, n = self.counts(text or "")
            if p + n:
                out[i] = (p - n) / (p + n)
        return out
