"""Distances, measurement summaries and the local history file."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np


def nearest(models: np.ndarray, emb: np.ndarray) -> tuple[float, int]:
    """Distance in Howdy units (norm x 10) to the closest stored vector."""
    d = np.linalg.norm(np.asarray(models) - np.asarray(emb), axis=1)
    i = int(np.argmin(d))
    return float(d[i] * 10), i


def summarize(dists: list[float], times: list[float], threshold: float, n_models: int) -> dict[str, object]:
    """Summary of one measurement. `times` are seconds since the start."""
    if not dists:
        return {"n_models": n_models, "frames": 0}
    first = next((t for d, t in zip(dists, times, strict=True) if d < threshold), None)
    return {
        "n_models": n_models,
        "frames": len(dists),
        "median": round(float(np.median(dists)), 2),
        "best": round(float(min(dists)), 2),
        "recognized": sum(d < threshold for d in dists),
        "first_match_s": None if first is None else round(first, 2),
    }


def model_usage(matches: list[int], labels: list[str]) -> dict[str, int]:
    """How often each stored vector was the closest one (0 for never)."""
    count = Counter(matches)
    return {f"{i}:{label}": count.get(i, 0) for i, label in enumerate(labels)}


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "howdy-360"


def history_file() -> Path:
    return state_dir() / "history.jsonl"


def append_history(entry: dict, path: Path | None = None) -> Path:
    """Append numbers only (never encodings or images) to the history file."""
    path = path or history_file()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {"time": int(time.time()), **entry}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    os.chmod(path, 0o600)
    return path


def last_history(path: Path | None = None) -> dict | None:
    path = path or history_file()
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def compare(prev: dict, cur: dict) -> list[tuple[str, object, object, bool | None]]:
    """(metric, before, after, improved) rows; improved is None when unknown."""
    rows: list[tuple[str, object, object, bool | None]] = []
    for key, lower_is_better in (("median", True), ("first_match_s", True)):
        a, b = prev.get(key), cur.get(key)
        improved = None if a is None or b is None else (b < a if lower_is_better else b > a)
        rows.append((key, a, b, improved))
    a_rate = _rate(prev)
    b_rate = _rate(cur)
    improved = None if a_rate is None or b_rate is None else b_rate > a_rate
    rows.append(("recognized_pct", a_rate, b_rate, improved))
    return rows


def _rate(summary: dict) -> float | None:
    frames = summary.get("frames") or 0
    if not frames:
        return None
    return round(100 * summary.get("recognized", 0) / frames, 1)
