"""FinBERT on CPU via onnxruntime. Loaded once at startup.

Served weights: model.fp16w.onnx = fp16 storage, fp32 compute (see service/scripts/export_onnx.py).
Half the size of fp32 and numerically identical on every CPU we measured. Dynamic int8 was
2x faster but CPU-dependent: Spearman vs fp32 was 0.996 on Apple M4, 0.93 on x86 with VNNI,
and 0.15 on x86 AVX2 without VNNI.

The runtime needs only onnxruntime + tokenizers + numpy: no torch, no transformers.
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
DEFAULT_MODEL_FILE = "model.fp16w.onnx"
MICRO_BATCH = 8


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
    def __init__(self, model_dir: str | Path, model_file: str = DEFAULT_MODEL_FILE,
                 num_threads: int | None = None, micro_batch: int = MICRO_BATCH):
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
        self.micro_batch = micro_batch

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=int(meta.get("max_length", 64)))
        self.tok.no_padding()
        self.pad_id = self.tok.token_to_id("[PAD]")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # folds the fp16->fp32 Casts
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(d / model_file), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        # One request's forward passes at a time: with 1 vCPU, parallel passes only add contention.
        self._lock = threading.Lock()

    def _run(self, encs: list) -> np.ndarray:
        n, width = len(encs), max(len(e.ids) for e in encs)
        ids = np.full((n, width), self.pad_id, dtype=np.int64)
        mask = np.zeros((n, width), dtype=np.int64)
        types = np.zeros((n, width), dtype=np.int64)
        for i, e in enumerate(encs):
            k = len(e.ids)
            ids[i, :k], mask[i, :k], types[i, :k] = e.ids, e.attention_mask, e.type_ids
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        (logits,) = self.sess.run(["logits"], {k: v for k, v in feed.items() if k in self.inputs})
        return softmax(logits.astype(np.float64))

    def probabilities(self, texts: list[str]) -> np.ndarray:
        """Length-sorted micro-batches: little padding, and still some batching speedup.

        Measured for 32 headlines on 1 thread (Apple M4): one padded batch 715 ms, one at a
        time 601 ms, length-sorted batches of 8 377 ms. With fp32 compute, padding doesn't
        change a headline's score (batched == single to 1e-6). Output is in input order.
        """
        encs = self.tok.encode_batch(texts)
        order = np.argsort([len(e.ids) for e in encs], kind="stable")
        out = np.empty((len(texts), len(self.ids)))
        with self._lock:
            for s in range(0, len(order), self.micro_batch):
                idx = order[s:s + self.micro_batch]
                out[idx] = self._run([encs[i] for i in idx])
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
                       model_file=os.environ.get("T2A_MODEL_FILE", DEFAULT_MODEL_FILE),
                       num_threads=int(threads) if threads else None,
                       micro_batch=int(os.environ.get("T2A_MICRO_BATCH", MICRO_BATCH)))
