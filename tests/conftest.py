import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def howdy_tree(tmp_path: Path) -> dict:
    """A fake Howdy install: paths.py, config.ini, dlib data and a model file."""
    config_dir = tmp_path / "etc-howdy"
    dlib_dir = tmp_path / "dlib"
    models_dir = tmp_path / "models"
    for d in (config_dir, dlib_dir, models_dir):
        d.mkdir()
    (config_dir / "config.ini").write_text(
        "[core]\nuse_cnn = false\ndisabled = false\n"
        "[video]\ncertainty = 3.5\ntimeout = 4\ndevice_path = /dev/null\ndark_threshold = 60\nmax_height = 320\n"
    )
    for name in ("shape_predictor_5_face_landmarks.dat", "dlib_face_recognition_resnet_model_v1.dat"):
        (dlib_dir / name).write_bytes(b"x")
    paths_py = tmp_path / "paths.py"
    paths_py.write_text(
        "from pathlib import PurePath\n"
        f'config_dir = PurePath("{config_dir}")\n'
        f'dlib_data_dir = PurePath("{dlib_dir}")\n'
        f'user_models_dir = PurePath("{models_dir}")\n'
    )
    rng = np.random.default_rng(0)
    models = [
        {"time": 1, "label": "my-face", "id": 0, "data": [rng.normal(0, 0.1, 128).tolist()]},
        {"time": 2, "label": "glasses", "id": 1, "data": [rng.normal(0, 0.1, 128).tolist()]},
    ]
    (models_dir / "alice.dat").write_text(json.dumps(models))
    return {
        "root": tmp_path,
        "paths_py": paths_py,
        "config_dir": config_dir,
        "dlib_dir": dlib_dir,
        "models_dir": models_dir,
        "models": models,
    }


@pytest.fixture(autouse=True)
def english(monkeypatch):
    """Assertions use the English source strings, whatever the host locale."""
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LANG", "C.UTF-8")
