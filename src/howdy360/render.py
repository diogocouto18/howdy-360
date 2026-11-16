"""Face-ID-style composite: circular live preview, sector ring, direction dot.

Rendered with OpenCV into an RGBA array (no cairo bindings needed), then
uploaded to GTK as a texture by the UI.
"""

from __future__ import annotations

import math

import numpy as np

SIZE, SUPERSAMPLE = 520, 2
GREEN = (46, 194, 126, 255)
DIM = (255, 255, 255, 50)
NEUTRAL_BORDER = (255, 255, 255, 70)
DOT_GLOW = (53, 132, 228, 90)
DOT = (120, 174, 237, 255)


def _round_arc(img: np.ndarray, c: int, r: int, a0: float, a1: float, color, th: int) -> None:
    import cv2

    cv2.ellipse(img, (c, c), (r, r), 0, a0, a1, color, th, cv2.LINE_AA)
    for a in (a0, a1):  # round caps
        x = c + r * math.cos(math.radians(a))
        y = c + r * math.sin(math.radians(a))
        cv2.circle(img, (int(x), int(y)), th // 2, color, -1, cv2.LINE_AA)


def compose(
    frame: np.ndarray | None,
    face_ok: bool,
    covered: list[bool] | None,
    dot: tuple[float, float] | None,
    size: int = SIZE,
) -> np.ndarray:
    """RGBA uint8 image (size x size). `frame` is a mirrored greyscale image."""
    import cv2

    n = size * SUPERSAMPLE
    img = np.zeros((n, n, 4), np.uint8)
    c, radius = n // 2, int(n * 0.36)
    ring = radius + 24 * SUPERSAMPLE
    if frame is not None:
        fh, fw = frame.shape[:2]
        k = 2 * radius / fh
        cam = cv2.resize(frame, (max(int(fw * k), 2 * radius), 2 * radius), interpolation=cv2.INTER_LINEAR)
        x0 = (cam.shape[1] - 2 * radius) // 2
        cam = cam[:, x0 : x0 + 2 * radius]
    else:
        cam = np.full((2 * radius, 2 * radius), 30, np.uint8)
    mask = np.zeros((2 * radius, 2 * radius), np.uint8)
    cv2.circle(mask, (radius, radius), radius - 1, 255, -1, cv2.LINE_AA)
    region = img[c - radius : c + radius, c - radius : c + radius]
    region[..., :3] = cam[..., None]
    region[..., 3] = mask
    cv2.circle(
        img, (c, c), radius + 3 * SUPERSAMPLE, GREEN if face_ok else NEUTRAL_BORDER, 3 * SUPERSAMPLE, cv2.LINE_AA
    )
    if covered is not None:
        for i, ok in enumerate(covered):
            _round_arc(img, c, ring, i * 45 - 16, i * 45 + 16, GREEN if ok else DIM, 12 * SUPERSAMPLE)
    if dot is not None:
        dx, dy = dot
        m = math.hypot(dx, dy)
        if m > 1.15:
            dx, dy = dx / m * 1.15, dy / m * 1.15
        p = (int(c + dx * ring), int(c + dy * ring))
        cv2.circle(img, p, 18 * SUPERSAMPLE, DOT_GLOW, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9 * SUPERSAMPLE, DOT, -1, cv2.LINE_AA)
    return np.ascontiguousarray(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
