"""Head pose, sector coverage and candidate selection (pure, no camera)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Sector i is centred on i * 45 degrees in screen coordinates (y grows down),
# so 0 = right, 2 = down, 4 = left, 6 = up. Directions are the user's own,
# because the preview is mirrored.
SECTOR_KEYS = (
    "right",
    "down-right",
    "down",
    "down-left",
    "left",
    "up-left",
    "up",
    "up-right",
)
BASE_SCALES = {"right": 0.10, "left": 0.10, "up": 0.07, "down": 0.07}
MIN_SCALE = 0.05
ADAPT_AFTER_S = 15.0
DISTINCT = 0.15  # minimum raw distance between kept encodings (1.5 in Howdy units)


def head_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Yaw/pitch proxies from dlib's 5-point landmarks.

    Points 0-1 and 2-3 are the corners of each eye, point 4 is the base of the
    nose. Yaw is the nose position between the eyes (-0.5 .. 0.5, mirrored so a
    positive value means the user's own right); pitch is the nose drop below the
    eye line, both normalised by the eye span.
    """
    p = np.asarray(landmarks, dtype=float)
    e1, e2, nose = p[0:2].mean(axis=0), p[2:4].mean(axis=0), p[4]
    left, right = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
    span = max(right[0] - left[0], 1.0)
    yaw = -((nose[0] - left[0]) / span - 0.5)
    pitch = (nose[1] - (left[1] + right[1]) / 2) / span
    return float(yaw), float(pitch)


def sector_of(dx: float, dy: float) -> int | None:
    """Sector index for a normalised offset, or None when inside the centre."""
    if math.hypot(dx, dy) < 1.0:
        return None
    return int(((math.degrees(math.atan2(dy, dx)) + 22.5) % 360) // 45)


@dataclass
class SectorTracker:
    """Turns raw pose offsets into ring coordinates with per-side thresholds.

    A side whose maximum reach stays short of its threshold after
    `adapt_after` seconds gets its threshold lowered to 90 % of that reach,
    never below `min_scale`, so a camera offset cannot make a side unreachable.
    """

    scales: dict[str, float] = field(default_factory=lambda: dict(BASE_SCALES))
    min_scale: float = MIN_SCALE
    adapt_after: float = ADAPT_AFTER_S
    reach: dict[str, float] = field(default_factory=lambda: {"right": 0.0, "left": 0.0, "up": 0.0, "down": 0.0})
    adapted: list[str] = field(default_factory=list)

    def update(self, rx: float, ry: float, elapsed: float) -> tuple[float, float]:
        self.reach["right"] = max(self.reach["right"], rx)
        self.reach["left"] = max(self.reach["left"], -rx)
        self.reach["down"] = max(self.reach["down"], ry)
        self.reach["up"] = max(self.reach["up"], -ry)
        if elapsed >= self.adapt_after:
            for side, scale in self.scales.items():
                target = self.reach[side] * 0.9
                if side not in self.adapted and self.min_scale <= target < scale:
                    self.scales[side] = target
                    self.adapted.append(side)
        dx = rx / (self.scales["right"] if rx >= 0 else self.scales["left"])
        dy = ry / (self.scales["down"] if ry >= 0 else self.scales["up"])
        return dx, dy


@dataclass
class Candidate:
    emb: np.ndarray
    sharp: float


def keep_best(bucket: list[Candidate], item: Candidate, limit: int) -> None:
    """Keep up to `limit` of the sharpest, mutually distinct candidates."""
    pool = sorted([*bucket, item], key=lambda c: -c.sharp)
    kept: list[Candidate] = []
    for cand in pool:
        if all(np.linalg.norm(cand.emb - k.emb) > DISTINCT for k in kept):
            kept.append(cand)
        if len(kept) >= limit:
            break
    bucket[:] = kept


def selection(center: list[Candidate], sectors: list[list[Candidate]]) -> list[tuple[str, np.ndarray]]:
    """Labelled encodings to store: `360-front-<n>` and `360-<sector>-<n>`."""
    out = [(f"360-front-{i + 1}", c.emb) for i, c in enumerate(center)]
    for idx, bucket in enumerate(sectors):
        out += [(f"360-{SECTOR_KEYS[idx]}-{i + 1}", c.emb) for i, c in enumerate(bucket)]
    return out
