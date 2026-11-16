import numpy as np
import pytest

from howdy360 import pose


def landmarks(nose_x, nose_y=60.0):
    # eye A corners around x=40, eye B corners around x=80, eyes at y=40
    return np.array([[35, 40], [45, 40], [85, 40], [75, 40], [nose_x, nose_y]], dtype=float)


def test_head_pose_frontal_is_centred_and_symmetric():
    yaw, pitch = pose.head_pose(landmarks(60))
    assert yaw == pytest.approx(0.0) and pitch == pytest.approx(0.5)
    left, _ = pose.head_pose(landmarks(70))
    right, _ = pose.head_pose(landmarks(50))
    assert left == pytest.approx(-right)


def test_head_pose_independent_of_eye_order():
    a = landmarks(52)
    b = a[[2, 3, 0, 1, 4]]
    assert pose.head_pose(a) == pytest.approx(pose.head_pose(b))


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [(0.5, 0, None), (1.2, 0, 0), (0, 1.2, 2), (-1.2, 0, 4), (0, -1.2, 6), (1, 1, 1), (-1, -1, 5)],
)
def test_sector_of(dx, dy, expected):
    assert pose.sector_of(dx, dy) == expected


def test_tracker_normalises_per_side_and_adapts():
    tr = pose.SectorTracker()
    assert tr.update(0.10, 0.0, 0.0) == pytest.approx((1.0, 0.0))
    assert tr.update(-0.05, -0.07, 1.0) == pytest.approx((-0.5, -1.0))
    assert tr.adapted == []
    # after the adapt window the left side (max reach 0.08) becomes reachable
    dx, _ = tr.update(-0.08, 0.0, 20.0)
    assert "left" in tr.adapted and dx <= -1.0
    assert tr.scales["left"] == pytest.approx(0.072)


def test_tracker_never_adapts_below_floor():
    tr = pose.SectorTracker()
    tr.update(-0.02, 0.0, 30.0)
    assert "left" not in tr.adapted and tr.scales["left"] == 0.10


def test_keep_best_prefers_sharp_and_distinct():
    base = np.zeros(128)
    near = base.copy()
    near[0] = 0.01
    far = base.copy()
    far[0] = 0.5
    bucket: list = []
    pose.keep_best(bucket, pose.Candidate(base, 10), 2)
    pose.keep_best(bucket, pose.Candidate(near, 50), 2)  # sharper near-duplicate replaces base
    pose.keep_best(bucket, pose.Candidate(far, 5), 2)
    assert [c.sharp for c in bucket] == [50, 5]


def test_selection_labels_are_howdy_safe():
    emb = np.zeros(128)
    sectors = [[pose.Candidate(emb, 1)] for _ in range(8)]
    out = pose.selection([pose.Candidate(emb, 1)], sectors)
    labels = [label for label, _ in out]
    assert labels[0] == "360-front-1" and "360-down-right-1" in labels and len(out) == 9
    assert all(len(lbl) <= 24 and "," not in lbl for lbl in labels)
