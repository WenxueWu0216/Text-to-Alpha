"""FinBERT baseline: score = P(positive) - P(negative) for each headline.

This is the 'classic NLP sentiment' baseline every LLM approach must beat.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FinBertScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 64):
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        # Don't hard-code label order; read it from the model config.
        label2id = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.pos_id, self.neg_id = label2id["positive"], label2id["negative"]

    @torch.inference_mode()  # no autograd graph: less memory, faster than no_grad
    def score(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # Sort by length so each batch has similar-length texts -> minimal padding.
        order = np.argsort([len(t) for t in texts])
        scores = np.empty(len(texts), dtype=np.float32)
        use_amp = self.device.type == "cuda"

        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            batch = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = self.model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1)
            scores[idx] = (probs[:, self.pos_id] - probs[:, self.neg_id]).cpu().numpy()
        return scores  # returned in the ORIGINAL order thanks to scores[idx] = ...
