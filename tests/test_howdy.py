import os

import pytest

from howdy360 import howdy


def test_parse_paths_py_reads_purepath_assignments():
    text = "config_dir = PurePath(\"/etc/howdy\")\nuser_models_dir = PurePath('/usr/share/dlib')\nother = 1\n"
    assert howdy.parse_paths_py(text) == {"config_dir": "/etc/howdy", "user_models_dir": "/usr/share/dlib"}


def test_discover_paths_uses_first_paths_py(howdy_tree):
    paths = howdy.discover_paths(candidates=("/nonexistent/paths.py", str(howdy_tree["paths_py"])))
    assert paths.config_dir == howdy_tree["config_dir"]
    assert paths.user_models_dir == howdy_tree["models_dir"]
    assert paths.model_file("alice") == howdy_tree["models_dir"] / "alice.dat"
    assert paths.source == str(howdy_tree["paths_py"])


def test_discover_paths_fallback_and_overrides(tmp_path):
    paths = howdy.discover_paths({"user_models_dir": str(tmp_path), "bogus": "x"}, candidates=())
    assert paths.user_models_dir == tmp_path
    assert str(paths.config_dir) == "/etc/howdy"
    assert paths.source == "fallback + overrides"


def test_read_config_values_and_fallbacks(howdy_tree, tmp_path):
    cfg = howdy.read_config(howdy_tree["config_dir"] / "config.ini")
    assert cfg.device_path == "/dev/null" and cfg.certainty == 3.5 and cfg.use_cnn is False
    minimal = tmp_path / "c.ini"
    minimal.write_text("[video]\ndevice_path = none\n")
    cfg = howdy.read_config(minimal)
    assert cfg.device_path is None and cfg.timeout == 4.0 and cfg.max_height == 320.0
    with pytest.raises(FileNotFoundError):
        howdy.read_config(tmp_path / "missing.ini")


def test_models_io(howdy_tree, tmp_path):
    models = howdy.load_models(howdy_tree["models_dir"] / "alice.dat")
    assert len(howdy.encodings(models)) == 2
    assert howdy.model_labels(models) == ["my-face", "glasses"]
    assert howdy.next_id(models) == 2 and howdy.next_id([]) == 0
    assert howdy.load_models(tmp_path / "none.dat") == []
    bad = tmp_path / "bad.dat"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        howdy.load_models(bad)


def test_sanitize_label_follows_howdy_rules():
    assert howdy.sanitize_label(" a,b ") == "ab"
    assert len(howdy.sanitize_label("x" * 40)) == 24


def test_pam_files_ignore_comments(tmp_path):
    (tmp_path / "gdm-password").write_text("auth sufficient /usr/lib/security/howdy/pam_howdy.so\n")
    (tmp_path / "sudo").write_text("# auth sufficient pam_howdy.so\n@include common-auth\n")
    (tmp_path / "sub").mkdir()
    assert howdy.pam_howdy_files(tmp_path) == ["gdm-password"]
    assert howdy.pam_howdy_files(tmp_path / "missing") == []


def test_file_mode(tmp_path):
    f = tmp_path / "m.dat"
    f.write_text("[]")
    os.chmod(f, 0o644)
    assert howdy.file_mode(f)["world_readable"] is True
    os.chmod(f, 0o600)
    assert howdy.file_mode(f) == {"mode": "0o600", "world_readable": False, "owner_uid": os.getuid()}


def test_current_user():
    assert howdy.current_user()
