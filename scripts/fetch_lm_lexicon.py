"""Download the Loughran-McDonald word lists into data/lexicon/ (not committed).

The lists ship inside the `pysentiment2` wheel on PyPI; we only extract the
Positive/Negative columns, so the package itself is not a dependency.
The LM dictionary is free for academic/non-commercial research (see sraf.nd.edu).
"""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from t2a.config import load_config

WHEEL_JSON = "https://pypi.org/pypi/pysentiment2/0.1.1/json"


def main():
    out = load_config()["paths"]["lexicon_dir"]
    meta = json.load(urllib.request.urlopen(WHEEL_JSON))
    url = next(f["url"] for f in meta["urls"] if f["filename"].endswith(".whl"))
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())) as z:
        lm = pd.read_csv(z.open("pysentiment2/static/LM.csv"))
    out.mkdir(parents=True, exist_ok=True)
    for col in ["Positive", "Negative"]:
        words = sorted(lm.loc[lm[col] > 0, "Word"].str.upper())
        (out / f"lm_{col.lower()}.txt").write_text("\n".join(words) + "\n")
        print(f"{col}: {len(words)} words -> {out / f'lm_{col.lower()}.txt'}")


if __name__ == "__main__":
    main()
