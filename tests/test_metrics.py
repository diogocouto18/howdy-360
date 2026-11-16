import numpy as np
import pytest

from howdy360 import metrics


def test_nearest_in_howdy_units():
    models = np.array([[0.0] * 128, [1.0] + [0.0] * 127])
    d, i = metrics.nearest(models, np.array([0.9] + [0.0] * 127))
    assert i == 1 and d == pytest.approx(1.0)


def test_summarize():
    s = metrics.summarize([4.0, 3.0, 2.0], [0.1, 0.2, 0.3], 3.5, 5)
    assert s == {"n_models": 5, "frames": 3, "median": 3.0, "best": 2.0, "recognized": 2, "first_match_s": 0.2}
    assert metrics.summarize([], [], 3.5, 5) == {"n_models": 5, "frames": 0}
    assert metrics.summarize([5.0], [0.1], 3.5, 1)["first_match_s"] is None


def test_model_usage():
    assert metrics.model_usage([0, 0, 2], ["a", "b", "c"]) == {"0:a": 2, "1:b": 0, "2:c": 1}


def test_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert metrics.last_history() is None
    path = metrics.append_history({"median": 3.0})
    metrics.append_history({"median": 2.5})
    with open(path, "a") as fh:
        fh.write("not json\n")
    assert path == tmp_path / "howdy-360" / "history.jsonl"
    assert metrics.last_history()["median"] == 2.5
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_compare():
    prev = {"median": 3.2, "first_match_s": 0.5, "recognized": 50, "frames": 100}
    cur = {"median": 2.3, "first_match_s": None, "recognized": 90, "frames": 100}
    rows = {r[0]: r for r in metrics.compare(prev, cur)}
    assert rows["median"][3] is True
    assert rows["first_match_s"][3] is None
    assert rows["recognized_pct"] == ("recognized_pct", 50.0, 90.0, True)
    assert metrics.compare({}, {})[2][3] is None
