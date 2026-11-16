"""Unlock history from the systemd journal (pam_howdy messages)."""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Iterable

PATTERNS = (
    ("Login approved", "approved"),
    ("Failure, timeout reached", "timeout"),
    ("Skipped authentication", "skipped"),
    ("Failure", "failed"),
)


def classify(line: str) -> str | None:
    for needle, outcome in PATTERNS:
        if needle in line:
            return outcome
    return None


def parse(lines: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for line in lines:
        outcome = classify(line)
        if outcome:
            counts[outcome] += 1
    return counts


def read(days: int = 7) -> Counter | None:
    """Count outcomes from the journal; None when the journal is unreadable."""
    try:
        proc = subprocess.run(
            [
                "journalctl",
                f"--since=-{int(days)}d",
                "--no-pager",
                "-o",
                "cat",
                "SYSLOG_IDENTIFIER=pam_howdy",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return parse(proc.stdout.splitlines())
