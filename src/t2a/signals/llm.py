"""LLM signal extraction via the Claude API (zero-shot and few-shot).

Two roles in the project:
  1. Benchmark rows by themselves ("zero-shot LLM", "few-shot LLM").
  2. The TEACHER: its structured outputs become training labels for the distilled student.

Engineering choices:
  - Forced, strict tool use -> the model must return JSON matching our schema; we still
    validate every response with jsonschema (numeric ranges are not enforced by the API)
    and count failures instead of silently dropping them.
  - SQLite cache keyed on (prompt version, model, system prompt, message) -> reruns are
    free, and zero-shot / few-shot entries never collide.
  - Two transports sharing one request builder and one cache:
      sync  (asyncio + semaphore): measures real per-request latency;
      batch (Message Batches API, 50% price): bulk scoring, no meaningful latency.
  - temperature 0 -> as deterministic as the API allows.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

PROMPT_VERSION = "v2"  # v1 = never-run relevance/surprise schema; bump on any prompt/schema change
BODY_CHARS = 600
MAX_TOKENS = 300

EVENT_TYPES = ["earnings", "guidance", "m_and_a", "legal", "product", "macro",
               "analyst_rating", "other"]
DIRECTIONS = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

SYSTEM_PROMPT = (
    "You are an equity analyst. You read one news item about a public company (its name is "
    "hidden as 'the Company') and judge the likely direction of the Company's stock over the "
    "next trading day RELATIVE to the overall market. Judge only from the text; do not guess "
    "the company's identity. If the item is generic market commentary, mentions the Company "
    "only in passing, or repeats old information, answer neutral or use low confidence. "
    "Always answer by calling the record_signal tool."
)

_PROPERTIES = {
    # rationale first: the model states its reasoning before committing to a label
    "rationale": {"type": "string", "description": "One sentence."},
    "event_type": {"type": "string", "enum": EVENT_TYPES},
    "direction": {"type": "string", "enum": list(DIRECTIONS)},
    "confidence": {"type": "number", "description": "0 to 1: how confident you are in the direction."},
}
TOOL = {
    "name": "record_signal",
    "description": "Record the structured assessment of the news item.",
    "strict": True,
    "input_schema": {"type": "object", "properties": _PROPERTIES,
                     "required": list(_PROPERTIES), "additionalProperties": False},
}
# Client-side schema: the tool schema plus the constraints the API does not enforce.
OUTPUT_SCHEMA = {
    **TOOL["input_schema"],
    "properties": {**_PROPERTIES,
                   "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
}
_VALIDATOR = jsonschema.Draft202012Validator(OUTPUT_SCHEMA)


def build_user_message(title: str, body: str) -> str:
    body = (body or "")[:BODY_CHARS]
    return f"Headline: {title}\n\nArticle lead: {body if body else '(not available)'}"


def build_system(examples: list[dict] | None = None) -> str:
    """Few-shot: show past articles with their REALISED next-day relative move.

    Showing realised outcomes (not hand-written answers) avoids inventing labels; the
    caller guarantees the examples come from the train period only.
    """
    if not examples:
        return SYSTEM_PROMPT
    shots = "\n\n".join(
        f"Example {i + 1}\n{build_user_message(e['title'], e['body'])}\n"
        f"Realised next-day move relative to the market: {e['outcome']}"
        for i, e in enumerate(examples)
    )
    return (f"{SYSTEM_PROMPT}\n\nFor calibration, here are past news items (identities hidden) "
            f"and how the stock actually moved relative to the market the next day. Most single "
            f"news items carry weak signal, so outcomes are noisy.\n\n{shots}")


def request_params(model: str, system: str, message: str) -> dict:
    """The ONE request definition used by sync calls, batches and token counting."""
    return {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0, "system": system,
            "tools": [TOOL], "tool_choice": {"type": "tool", "name": "record_signal"},
            "messages": [{"role": "user", "content": message}]}


def cache_key(model: str, system: str, message: str) -> str:
    sys_hash = hashlib.sha1(system.encode()).hexdigest()
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{sys_hash}|{message}".encode()).hexdigest()


def validate(out: object) -> tuple[dict | None, str | None]:
    errors = sorted(_VALIDATOR.iter_errors(out), key=lambda e: list(e.path))
    if errors:
        return None, "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
                               for e in errors)[:300]
    return dict(out), None


def parse_message(content: list, stop_reason: str | None, usage) -> dict:
    """Turn an API message into a cache record. Every outcome is recorded, never dropped."""
    rec = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
           "stop_reason": stop_reason}
    block = next((b for b in content if getattr(b, "type", None) == "tool_use"), None)
    if block is None:
        return {**rec, "status": "no_tool_call"}
    clean, err = validate(block.input)
    if err:
        return {**rec, "status": "invalid", "validation_error": err, "raw": block.input}
    return {**rec, "status": "ok", **clean}


@dataclass
class Prices:
    input_per_mtok: float
    output_per_mtok: float
    batch_discount: float = 0.5

    def cost(self, input_tokens: float, output_tokens: float, batch: bool = False) -> float:
        usd = (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6
        return usd * (self.batch_discount if batch else 1.0)


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(keys), 900):  # SQLite caps bound parameters per statement
            chunk = keys[i:i + 900]
            q = f"SELECT key, value FROM cache WHERE key IN ({','.join('?' * len(chunk))})"
            out.update({k: json.loads(v) for k, v in self.conn.execute(q, chunk)})
        return out

    def put(self, key: str, value: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))

    def commit(self) -> None:
        self.conn.commit()


class ClaudeScorer:
    def __init__(self, model: str, cache_path: Path, max_concurrency: int = 8,
                 batch_state: Path | None = None):
        self.model = model
        self.cache = ResponseCache(cache_path)
        self.max_concurrency = max_concurrency
        self.batch_state = batch_state or cache_path.with_name("pending_batches.json")

    # ---- estimation -------------------------------------------------------------------
    def count_input_tokens(self, system: str, messages: list[str]) -> list[int]:
        """Exact input tokens via the (free) count_tokens endpoint, incl. tool overhead."""
        from anthropic import Anthropic

        client = Anthropic()
        out = []
        for m in messages:
            p = request_params(self.model, system, m)
            out.append(client.messages.count_tokens(
                model=p["model"], system=p["system"], tools=p["tools"],
                tool_choice=p["tool_choice"], messages=p["messages"]).input_tokens)
        return out

    # ---- sync transport (latency) -----------------------------------------------------
    async def _one(self, client, sem, key: str, system: str, message: str) -> dict:
        async with sem:
            t0 = time.perf_counter()
            try:
                params = request_params(self.model, system, message)
                # anthropic>=1.x removed sampling kwargs from the typed signature; send the
                # same value in the body so sync and batch requests stay identical.
                resp = await client.messages.create(
                    **{k: v for k, v in params.items() if k != "temperature"},
                    extra_body={"temperature": params["temperature"]})
            except Exception as e:  # one bad request must not kill the run; it is counted
                return {"status": "api_error", "error": repr(e)[:300]}
            latency = time.perf_counter() - t0
        rec = parse_message(resp.content, resp.stop_reason, resp.usage)
        rec.update(transport="sync", latency_s=latency)
        self.cache.put(key, rec)
        return rec

    async def _run_sync(self, system: str, items: list[tuple[str, str]]) -> None:
        from anthropic import AsyncAnthropic  # reads ANTHROPIC_API_KEY from the environment

        client = AsyncAnthropic(max_retries=6)
        sem = asyncio.Semaphore(self.max_concurrency)
        out = []
        for i in range(0, len(items), 100):  # commit every 100 so a crash loses little work
            out += await asyncio.gather(*[self._one(client, sem, k, system, m) for k, m in items[i:i + 100]])
            self.cache.commit()
            print(f"  sync {min(i + 100, len(items))}/{len(items)}")
        return out

    def score_sync(self, system: str, items: list[tuple[str, str]]) -> list[dict]:
        """Returns one record per item, including uncached failures, so callers can stop
        before spending more when the API is rejecting requests."""
        return asyncio.run(self._run_sync(system, items)) if items else []

    # ---- batch transport (cost) -------------------------------------------------------
    def _pending(self) -> list[str]:
        return json.loads(self.batch_state.read_text()) if self.batch_state.exists() else []

    def _set_pending(self, ids: list[str]) -> None:
        self.batch_state.parent.mkdir(parents=True, exist_ok=True)
        self.batch_state.write_text(json.dumps(ids))

    def submit_batch(self, system: str, items: list[tuple[str, str]]) -> str | None:
        if not items:
            return None
        from anthropic import Anthropic

        reqs = [{"custom_id": k, "params": request_params(self.model, system, m)} for k, m in items]
        batch = Anthropic().messages.batches.create(requests=reqs)
        # Persist the id before waiting: if the process dies, the next run collects it
        # instead of paying for the same requests again.
        self._set_pending(self._pending() + [batch.id])
        print(f"  submitted batch {batch.id} with {len(reqs)} requests")
        return batch.id

    def collect_batches(self, poll_s: int = 30) -> int:
        """Wait for every pending batch and write its results into the cache."""
        from anthropic import Anthropic

        client = Anthropic()
        n = 0
        errors: Counter = Counter()
        for bid in self._pending():
            while (b := client.messages.batches.retrieve(bid)).processing_status != "ended":
                c = b.request_counts
                print(f"  batch {bid}: {c.succeeded} done, {c.processing} processing")
                time.sleep(poll_s)
            for r in client.messages.batches.results(bid):
                if r.result.type == "succeeded":
                    m = r.result.message
                    rec = parse_message(m.content, m.stop_reason, m.usage)
                    rec.update(transport="batch", batch_id=bid)
                    self.cache.put(r.custom_id, rec)
                    n += 1
                elif r.result.type == "errored":  # stays uncached, retried on the next run
                    errors[r.result.error.error.message[:120]] += 1
                else:  # expired / canceled
                    errors[r.result.type] += 1
            self.cache.commit()
            self._set_pending([p for p in self._pending() if p != bid])
        for msg, cnt in errors.most_common(3):
            print(f"  batch: {cnt} requests failed: {msg}")
        return n


def records_to_frame(records: list[dict | None]) -> pd.DataFrame:
    """Cache records -> one row per article, with the trading signal derived."""
    df = pd.DataFrame([r or {"status": "missing"} for r in records])
    for col in ["direction", "confidence", "event_type", "rationale", "latency_s",
                "input_tokens", "output_tokens", "transport"]:
        if col not in df:
            df[col] = np.nan
    ok = df["status"] == "ok"
    df["dir_num"] = df["direction"].map(DIRECTIONS).where(ok)
    df["llm_direction"] = df["dir_num"]
    df["llm_signal"] = (df["dir_num"] * df["confidence"].astype(float)).where(ok)
    return df
