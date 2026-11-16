import numpy as np
import pytest

pytest.importorskip("cv2")
from howdy360 import render  # noqa: E402


def test_compose_shapes_and_ring():
    frame = np.full((360, 640), 120, np.uint8)
    img = render.compose(frame, True, [True] + [False] * 7, (0.8, 0.0), size=200)
    assert img.shape == (200, 200, 4) and img.dtype == np.uint8
    c = 100
    assert img[c, c, 3] == 255  # camera circle is opaque
    assert img[0, 0, 3] == 0  # corners stay transparent
    ring_right = img[c, int(c + 0.36 * 200 + 24)]
    assert tuple(ring_right[:3]) == pytest.approx(render.GREEN[:3], abs=40)


def test_compose_without_frame_or_ring():
    img = render.compose(None, False, None, None, size=120)
    assert img.shape == (120, 120, 4) and img[60, 60, 0] == 30
