"""FinBERT on CPU. Loaded once at startup; the deployment target has no GPU."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol

LABELS = ("positive", "negative", "neutral")


@dataclass(frozen=True)
class Prediction:
    score: float                     # P(positive) - P(negative), in [-1, 1]
    probabilities: dict[str, float]  # keyed by LABELS


class Scorer(Protocol):
    name: str
    device: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


class FinBertScorer:
    def __init__(self, model_path: str, name: str = "ProsusAI/finbert", max_length: int = 64,
                 num_threads: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if num_threads:
            torch.set_num_threads(num_threads)
        self._torch = torch
        self.name, self.device, self.max_length = name, "cpu", max_length
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cpu").eval()
        # Don't hard-code label order; read it from the model config.
        label2id = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        assert set(label2id) == set(LABELS), label2id
        self.ids = {lab: label2id[lab] for lab in LABELS}
        # One forward pass at a time: concurrent passes would fight over the same CPU threads.
        self._lock = threading.Lock()

    def predict(self, texts: list[str]) -> list[Prediction]:
        torch = self._torch
        batch = self.tok(texts, padding=True, truncation=True, max_length=self.max_length,
                         return_tensors="pt")
        with self._lock, torch.inference_mode():
            probs = torch.softmax(self.model(**batch).logits.float(), dim=-1)
        out = []
        for row in probs.tolist():
            p = {lab: min(1.0, max(0.0, row[i])) for lab, i in self.ids.items()}
            out.append(Prediction(score=max(-1.0, min(1.0, p["positive"] - p["negative"])), probabilities=p))
        return out


def load_finbert() -> FinBertScorer:
    threads = os.environ.get("TORCH_NUM_THREADS")
    return FinBertScorer(
        model_path=os.environ.get("T2A_MODEL_PATH", "ProsusAI/finbert"),
        max_length=int(os.environ.get("T2A_MAX_LENGTH", "64")),
        num_threads=int(threads) if threads else None,
    )
