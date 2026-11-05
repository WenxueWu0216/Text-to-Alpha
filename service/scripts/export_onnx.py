"""Export FinBERT to ONNX (fp32), then int8 dynamic quantization for serving.

  python service/scripts/export_onnx.py --out models/finbert-onnx [--model ProsusAI/finbert --revision ...]

Writes:
  model.fp32.onnx   reference export (kept for the quantization check, not shipped)
  model.int8.onnx   dynamic int8, per-channel weights (what the service serves). On the fixture,
                    per-tensor and per-channel both keep IC within 0.002 of fp32, but per-channel
                    cuts the mean |score diff| vs fp32 from 0.050 to 0.017 at the same size.
                    See service/eval_fixture/gate.json.
  tokenizer.json    fast tokenizer, used with the `tokenizers` package at runtime
  labels.json       id -> label, read from the model config (never hard-coded)

Needs torch + transformers + onnx + onnxruntime. Runs in the Docker build stage only;
the runtime image has neither torch nor transformers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "ProsusAI/finbert"
DEFAULT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"


def export(model_id: str, revision: str, out: Path, max_length: int = 64) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision).eval()
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    labels = {int(k): v.lower() for k, v in model.config.id2label.items()}
    (out / "labels.json").write_text(json.dumps({"id2label": labels, "max_length": max_length,
                                                 "model": model_id, "revision": revision}, indent=2))

    sample = tok(["Shares rise after strong earnings", "Guidance cut"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "batch", 1: "seq"} for n in names}
    fp32 = out / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
                          str(fp32), input_names=names, output_names=["logits"],
                          dynamic_axes={**dyn, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(out / "model.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True)
    for f in sorted(out.iterdir()):
        print(f"{f.name:<18} {f.stat().st_size / 1e6:7.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", type=Path, default=Path("models/finbert-onnx"))
    a = ap.parse_args()
    export(a.model, a.revision, a.out)


if __name__ == "__main__":
    main()
