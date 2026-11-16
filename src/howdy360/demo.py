"""Demo mode: a simulated enrollment with a drawn silhouette.

No camera, no Howdy files, nothing saved. Used for screenshots and for trying
the window on machines without an IR camera.
"""

from __future__ import annotations

import contextlib
import functools
import math
import threading

import numpy as np

from .i18n import _
from .pose import SECTOR_KEYS, sector_of

FRAME_H, FRAME_W = 360, 640
# Real numbers from a v0.1 dry run on the development machine (same session),
# shown on the demo results page.
DEMO_REPORT = {
    "sectors_covered": 8,
    "new_models": 17,
    "demo": True,
    "current": {"n_models": 5, "frames": 60, "median": 3.21, "best": 2.7, "recognized": 60, "first_match_s": 0.18},
    "with_360": {"n_models": 22, "frames": 60, "median": 2.15, "best": 1.78, "recognized": 60, "first_match_s": 0.18},
}


@functools.cache
def _static() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pixel grids, vignetted background and fixed grain (computed once)."""
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float32)
    dist = np.hypot((xx - FRAME_W / 2) / FRAME_W, (yy - FRAME_H / 2) / FRAME_H)
    background = (58 - 46 * dist).clip(10, 255)
    grain = np.random.default_rng(7).normal(0, 3.0, (FRAME_H, FRAME_W)).astype(np.float32)
    return xx, yy, background, grain


def silhouette(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Greyscale IR-looking frame of a generic head-and-shoulders avatar.

    `dx`, `dy` (ring units, about -1.3..1.3) turn the head: it shifts slightly
    and the lit side of the face follows, the way a real head looks when it
    rotates in front of an IR emitter.
    """
    import cv2

    xx, yy, background, grain = _static()
    ss = 2  # the outline is drawn at 2x for smooth edges
    h, w = FRAME_H * ss, FRAME_W * ss
    body = np.zeros((h, w), np.uint8)
    cx = w // 2
    cv2.ellipse(body, (cx, h + 70 * ss), (215 * ss, 175 * ss), 0, 0, 360, 255, -1, cv2.LINE_AA)  # shoulders
    hx, hy = int(cx + dx * 22 * ss), int(150 * ss + dy * 14 * ss)
    cv2.ellipse(body, (hx, hy), (74 * ss, 88 * ss), 0, 0, 360, 255, -1, cv2.LINE_AA)  # head
    neck_top = (hx + cx) // 2  # the neck bends between the head and the shoulders
    neck = np.array(
        [
            [neck_top - 36 * ss, hy + 50 * ss],
            [neck_top + 36 * ss, hy + 50 * ss],
            [cx + 50 * ss, h - 110 * ss],
            [cx - 50 * ss, h - 110 * ss],
        ],
        np.int32,
    )
    cv2.fillPoly(body, [neck], 255, cv2.LINE_AA)
    mask = cv2.resize(body, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32) / 255

    # IR emitter light: brightest where the face points, fading towards the edges.
    light_x, light_y = (hx / ss) + dx * 30, (hy / ss) - 10 + dy * 20
    falloff = np.hypot((xx - light_x) / 150, (yy - light_y) / 190)
    tone = (175 - 70 * falloff).clip(85, 190)
    img = background * (1 - mask) + tone * mask + grain
    return img.clip(0, 255).astype(np.uint8)


class DemoEngine(threading.Thread):
    """Drop-in replacement for the camera Engine; scripted and deterministic."""

    def __init__(self, publish, speed: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.publish, self.speed = publish, speed
        self.cancel = threading.Event()
        self.state: dict = {"step": 0, "title": _("Opening the camera..."), "hint": "", "progress": 0.0}
        self.chosen: list = []
        self.report: dict = {}

    def emit(self, **kw) -> None:
        self.state.update(kw)
        self.publish(dict(self.state))

    def wait(self, seconds: float) -> None:
        if self.cancel.wait(seconds / self.speed):
            raise InterruptedError

    def run(self) -> None:
        with contextlib.suppress(InterruptedError):
            self.script()

    def script(self, fps: int = 15) -> None:
        # Calibrate.
        for i in range(1, 9):
            self.emit(
                step=0,
                title=_("Look straight ahead"),
                hint=_("Calibrating your normal position"),
                progress=i / 8,
                face_ok=True,
                frame=silhouette(),
            )
            self.wait(0.2)
        # Rotate: one slow circle, starting to the right.
        covered = [False] * 8
        frames = 10 * fps
        for i in range(frames + 1):
            t = i / frames
            radius = min(1.0, t * 6) * 1.25
            angle = 2 * math.pi * t * 1.05
            dx, dy = radius * math.cos(angle), radius * math.sin(angle)
            idx = sector_of(dx, dy)
            if idx is not None:
                covered[idx] = True
            missing = [_(SECTOR_KEYS[k]) for k in range(8) if not covered[k]]
            self.emit(
                step=1,
                title=_("Slowly turn your head in a circle"),
                hint=_("Missing: {sectors}", sectors=", ".join(missing)) if missing else _("Complete!"),
                progress=sum(covered) / 8,
                covered=list(covered),
                dot=(dx, dy),
                face_ok=True,
                frame=silhouette(dx, dy),
            )
            self.wait(1 / fps)
            if all(covered):
                break
        # Measure.
        steps = 4 * fps
        for i in range(steps + 1):
            self.emit(
                step=2,
                title=_("Measuring - look at the screen normally"),
                hint="",
                progress=i / steps,
                dot=None,
                covered=None,
                frame=silhouette(),
            )
            self.wait(1 / fps)
        self.report = dict(DEMO_REPORT)
        self.emit(step=3, done=True, report=self.report)
