"""Export FinBERT to ONNX (fp32), then store the weights in fp16 for serving.

  python service/scripts/export_onnx.py --out models/finbert-onnx [--model ProsusAI/finbert --revision ...]

Writes:
  model.fp32.onnx   reference export (matches torch to 1e-5; not shipped)
  model.fp16w.onnx  what the service serves: fp16 storage, fp32 compute. Each large weight is
                    stored in fp16 behind a Cast(to=float) that onnxruntime constant-folds at
                    load time, so inference is ordinary fp32 on any CPU. 219 MB vs 438 MB.
                    Dynamic int8 (111 MB) was rejected because it is CPU-dependent: on the
                    fixture, Spearman vs fp32 was 0.996 on Apple M4, 0.928 on x86 with VNNI,
                    0.146 on x86 AVX2 without VNNI (service/scripts/quant_sweep.py,
                    .github/workflows/quant-sweep.yml).
  tokenizer.json    fast tokenizer, used with the `tokenizers` package at runtime
  labels.json       id -> label, read from the model config (never hard-coded)

Needs torch + transformers + onnx + onnxruntime. Runs in the Docker build stage only;
the runtime image has neither torch nor transformers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "ProsusAI/finbert"
DEFAULT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"


def fp16_weights(src: Path, dst: Path, min_elements: int = 1024) -> None:
    """fp16 storage, fp32 compute: each large fp32 initializer becomes fp16 + Cast(to=float)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(src))
    keep, casts = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elements:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
            keep.append(h)
        else:
            keep.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def export(model_id: str, revision: str, out: Path, max_length: int = 64) -> None:
    import torch
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
    fp16_weights(fp32, out / "model.fp16w.onnx")
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
