import threading

import numpy as np
import pytest

pytest.importorskip("cv2")
from howdy360 import demo  # noqa: E402


def test_silhouette_moves_with_direction():
    centre, turned = demo.silhouette(), demo.silhouette(1.2, 0.0)
    assert centre.shape == (demo.FRAME_H, demo.FRAME_W) and centre.dtype == np.uint8
    assert not np.array_equal(centre, turned)


def test_demo_engine_covers_ring_and_finishes():
    snaps = []
    done = threading.Event()

    def publish(s):
        snaps.append(s)
        if s.get("done"):
            done.set()

    eng = demo.DemoEngine(publish, speed=200)
    eng.start()
    assert done.wait(10)
    assert [s["step"] for s in snaps if "step" in s][-1] == 3
    rotate = [s for s in snaps if s["step"] == 1]
    assert all(rotate[-1]["covered"]) and eng.report["demo"] is True
    assert eng.chosen == []  # demo never offers anything to save


def test_demo_engine_cancel():
    eng = demo.DemoEngine(lambda s: None, speed=0.01)
    eng.start()
    eng.cancel.set()
    eng.join(2)
    assert not eng.is_alive()
