"""Zero-shot LLM signal extraction via the Claude API.

Two roles in the project:
  1. A benchmark row by itself ("zero-shot LLM").
  2. The TEACHER: its structured outputs become training labels when we later distill
     into a small open-source model (step 5, QLoRA).

Engineering choices:
  - Forced tool use  -> the model must return JSON matching our schema (no regex parsing).
  - SQLite cache     -> never pay twice for the same prompt; reruns are free and reproducible.
  - asyncio + semaphore -> many requests in flight, but capped to respect rate limits.
  - temperature 0    -> deterministic-as-possible outputs for a benchmark.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

PROMPT_VERSION = "v1"  # bump when the prompt/schema changes -> old cache entries are ignored
BODY_CHARS = 600

EVENT_TYPES = ["earnings", "guidance", "analyst_rating", "m_and_a", "product", "legal_regulatory",
               "management", "capital_return", "macro_sector", "market_commentary", "other"]

SYSTEM_PROMPT = (
    "You are an equity analyst. You read one news item about a public company (its name is "
    "hidden as 'the Company') and judge how it should move the Company's stock over the next "
    "trading day RELATIVE to the market. Judge only from the text; do not guess the company's "
    "identity. Generic market commentary or articles that mention the Company only in passing "
    "have low relevance. Old, already-known information has low surprise."
)

TOOL = {
    "name": "record_signal",
    "description": "Record the structured assessment of the news item.",
    "input_schema": {
        "type": "object",
        "properties": {
            # rationale first: the model writes its reasoning before committing to numbers
            "rationale": {"type": "string", "description": "At most 25 words."},
            "event_type": {"type": "string", "enum": EVENT_TYPES},
            "relevance": {"type": "number", "description": "0-1. Is this news primarily about the Company?"},
            "surprise": {"type": "number", "description": "0-1. How new/unexpected is this information?"},
            "expected_impact": {"type": "number",
                                "description": "-1 (very negative) to 1 (very positive) relative next-day move."},
            "confidence": {"type": "number", "description": "0-1."},
        },
        "required": ["rationale", "event_type", "relevance", "surprise", "expected_impact", "confidence"],
    },
}


def build_user_message(title: str, body: str) -> str:
    body = (body or "")[:BODY_CHARS]
    return f"Headline: {title}\n\nArticle lead: {body if body else '(not available)'}"


def cache_key(model: str, message: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{message}".encode()).hexdigest()


def _clip(x, lo, hi):
    try:
        return float(min(max(float(x), lo), hi))
    except (TypeError, ValueError):
        return None


def validate(out: dict) -> dict:
    """The schema is enforced by the API, but numeric RANGES are not -> clamp them here."""
    return {
        "rationale": str(out.get("rationale", ""))[:300],
        "event_type": out.get("event_type") if out.get("event_type") in EVENT_TYPES else "other",
        "relevance": _clip(out.get("relevance"), 0, 1),
        "surprise": _clip(out.get("surprise"), 0, 1),
        "expected_impact": _clip(out.get("expected_impact"), -1, 1),
        "confidence": _clip(out.get("confidence"), 0, 1),
    }


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))

    def commit(self) -> None:
        self.conn.commit()


class LLMSignalExtractor:
    def __init__(self, model: str, cache_path: Path, max_concurrency: int = 8):
        from anthropic import AsyncAnthropic  # reads ANTHROPIC_API_KEY from the environment
        self.client = AsyncAnthropic(max_retries=6)  # SDK retries 429/5xx with exponential backoff
        self.model = model
        self.cache = ResponseCache(cache_path)
        self.sem = asyncio.Semaphore(max_concurrency)

    async def _one(self, message: str) -> dict:
        key = cache_key(self.model, message)
        if (hit := self.cache.get(key)) is not None:
            return hit
        async with self.sem:
            try:
                resp = await self.client.messages.create(
                    model=self.model, max_tokens=300, temperature=0,
                    system=SYSTEM_PROMPT, tools=[TOOL],
                    tool_choice={"type": "tool", "name": "record_signal"},  # forced -> always JSON
                    messages=[{"role": "user", "content": message}],
                )
            except Exception as e:  # one bad request must not kill a 3,000-item run
                return {"error": repr(e)[:300]}
        block = next(b for b in resp.content if b.type == "tool_use")
        result = validate(block.input)
        result["input_tokens"] = resp.usage.input_tokens     # kept for the cost column
        result["output_tokens"] = resp.usage.output_tokens
        self.cache.put(key, result)
        return result

    async def _run(self, messages: list[str]) -> list[dict]:
        tasks = [self._one(m) for m in messages]
        results = []
        for i in range(0, len(tasks), 200):  # commit every 200 so a crash loses little work
            results += await asyncio.gather(*tasks[i:i + 200])
            self.cache.commit()
            print(f"  {min(i + 200, len(tasks))}/{len(tasks)} done")
        return results

    def score(self, titles: list[str], bodies: list[str]) -> pd.DataFrame:
        messages = [build_user_message(t, b) for t, b in zip(titles, bodies)]
        out = pd.DataFrame(asyncio.run(self._run(messages)))
        for col in ["relevance", "expected_impact", "surprise", "confidence"]:
            if col not in out:
                out[col] = None
        # Signal = direction * how much the article is really about this stock.
        out["llm_signal"] = out["expected_impact"].astype(float) * out["relevance"].astype(float)
        return out
