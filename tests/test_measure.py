import numpy as np

from howdy360 import howdy, measure


def test_run_with_fake_capture(howdy_tree, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(measure, "capture", lambda *a, **k: ([3.0, 2.0, 4.0], [0.1, 0.2, 0.3], [0, 0, 2]))
    cands = tmp_path / "c.npy"
    np.save(cands, np.zeros((1, 128)))
    paths = howdy.discover_paths(candidates=(str(howdy_tree["paths_py"]),))
    first = measure.run(paths, "alice", 1, cands, history=True)
    assert "+ 1 candidates" in first and "Never the closest model" in first and "glasses" in first
    second = measure.run(paths, "alice", 1, None, history=True)
    assert "Compared with previous measurement" in second


def test_run_without_models_or_faces(howdy_tree, tmp_path, monkeypatch):
    paths = howdy.discover_paths(candidates=(str(howdy_tree["paths_py"]),))
    assert "No stored models" in measure.run(paths, "bob", 1, None, history=False)
    monkeypatch.setattr(measure, "capture", lambda *a, **k: ([], [], []))
    assert "No face seen" in measure.run(paths, "alice", 1, None, history=False)
