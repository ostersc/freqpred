"""File-backed cache of candidate evaluations (T93).

Benchmark calls cost real API dollars, and re-running a benchmark usually
re-sends prompts that were already evaluated (e.g. re-scoring after a
sampling change, or extending a run with more scenarios). The cache keys on
everything that determines a candidate response — model, thinking config,
system prompt, and the exact user prompt — so a hit is guaranteed to be the
same experiment. It lives under the gitignored ``benchmarks/`` tree.

Repetitions are preserved: an entry stores a *list* of rep results, a run
with ``--reps N`` consumes up to N cached reps and makes fresh calls for the
remainder (appending them), so per-scenario spread stays meaningful instead
of collapsing to N copies of one cached answer. Only successful parses are
cached; errors are always retried.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from freqpred.bench.scenarios import ModelOutput

DEFAULT_CACHE_DIR = Path("benchmarks/.eval_cache")

_KEY_VERSION = 1


def cache_key(
    model: str,
    thinking: dict | None,
    system_prompt: str,
    prompt: str,
) -> str:
    material = json.dumps(
        {
            "v": _KEY_VERSION,
            "model": model,
            "thinking": thinking,
            "system": system_prompt,
            "prompt": prompt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()


class EvalCache:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def load(self, key: str) -> list[dict]:
        """Return the cached rep records for *key* (empty list on miss)."""
        path = self._path(key)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())["reps"]
        except (json.JSONDecodeError, KeyError):
            return []  # corrupt entry — treat as miss, will be overwritten

    def append(
        self, key: str, output: ModelOutput, thinking_tokens: int | None
    ) -> None:
        reps = self.load(key)
        reps.append({"output": asdict(output), "thinking_tokens": thinking_tokens})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps({"reps": reps}, indent=1))

    def count(self, key: str) -> int:
        return len(self.load(key))
