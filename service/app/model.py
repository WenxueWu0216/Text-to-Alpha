"""FinBERT on CPU via onnxruntime (int8 dynamic quantization). Loaded once at startup.

The runtime needs only onnxruntime + tokenizers + numpy: no torch, no transformers.
Build the model directory with service/scripts/export_onnx.py.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

LABELS = ("positive", "negative", "neutral")
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "finbert-onnx"


@dataclass(frozen=True)
class Prediction:
    score: float                     # P(positive) - P(negative), in [-1, 1]
    probabilities: dict[str, float]  # keyed by LABELS


class Scorer(Protocol):
    name: str
    device: str

    def predict(self, texts: list[str]) -> list[Prediction]: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


class OnnxFinBert:
    def __init__(self, model_dir: str | Path, model_file: str = "model.int8.onnx",
                 num_threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        meta = json.loads((d / "labels.json").read_text())
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        assert set(id2label.values()) == set(LABELS), id2label
        self.ids = {lab: i for i, lab in id2label.items()}
        self.name = meta.get("model", "ProsusAI/finbert")
        self.variant = model_file.removeprefix("model.").removesuffix(".onnx")
        self.device = "cpu"

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=int(meta.get("max_length", 64)))
        self.tok.no_padding()

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(d / model_file), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        # One request's passes at a time: with 1 vCPU, parallel passes only add contention.
        self._lock = threading.Lock()

    def probabilities(self, texts: list[str]) -> np.ndarray:
        """One forward pass per headline, never a padded batch.

        Dynamic int8 quantization picks one activation scale per tensor, so in a padded batch a
        headline's score would depend on its neighbours (measured: mean |diff| 0.017, max 0.26
        for batches of 32). Unpadded single passes are also faster on CPU (471 vs 717 ms for
        32 headlines on 1 thread), because no compute is spent on padding.
        """
        out = np.empty((len(texts), len(self.ids)))
        with self._lock:
            for i, e in enumerate(self.tok.encode_batch(texts)):
                feed = {"input_ids": np.array([e.ids], dtype=np.int64),
                        "attention_mask": np.array([e.attention_mask], dtype=np.int64),
                        "token_type_ids": np.array([e.type_ids], dtype=np.int64)}
                (logits,) = self.sess.run(["logits"], {k: v for k, v in feed.items() if k in self.inputs})
                out[i] = softmax(logits.astype(np.float64))[0]
        return out

    def predict(self, texts: list[str]) -> list[Prediction]:
        out = []
        for row in self.probabilities(texts):
            p = {lab: float(min(1.0, max(0.0, row[i]))) for lab, i in self.ids.items()}
            out.append(Prediction(score=max(-1.0, min(1.0, p["positive"] - p["negative"])), probabilities=p))
        return out


def load_finbert() -> OnnxFinBert:
    threads = os.environ.get("T2A_NUM_THREADS")
    return OnnxFinBert(model_dir=os.environ.get("T2A_MODEL_DIR", DEFAULT_MODEL_DIR),
                       model_file=os.environ.get("T2A_MODEL_FILE", "model.int8.onnx"),
                       num_threads=int(threads) if threads else None)
