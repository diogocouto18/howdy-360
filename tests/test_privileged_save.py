import json
import os
from pathlib import Path

import pytest

from howdy360 import privileged_save as ps

ME = {"PKEXEC_UID": str(os.getuid())}


def payload(n=2, mode="append", value=0.01):
    return {"mode": mode, "models": [{"label": f"360-left,{i}", "vector": [value] * 128} for i in range(n)]}


def write(tmp_path, data, name="p.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_invoking_user_requires_pkexec():
    with pytest.raises(PermissionError):
        ps.invoking_user({})
    with pytest.raises(PermissionError):
        ps.invoking_user({"PKEXEC_UID": "root"})
    assert ps.invoking_user(ME)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"mode": "delete", "models": [{"vector": [0.0] * 128}]},
        {"models": []},
        {"models": [{"vector": [0.0] * 127}]},
        {"models": [{"vector": ["x"] * 128}]},
        {"models": [{"vector": [True] * 128}]},
        {"models": [{"vector": [float("nan")] * 128}]},
        {"models": [{"vector": [99.0] * 128}]},
        {"models": ["nope"]},
        {"models": [{"vector": [0.0] * 128}] * 33},
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(ps.PayloadError):
        ps.validate(bad)


def test_validate_sanitizes_labels():
    mode, items = ps.validate({"models": [{"label": "a,b" + "x" * 30, "vector": [0] * 128}, {"vector": [0] * 128}]})
    assert mode == "append"
    assert items[0][0] == ("ab" + "x" * 30)[:24]
    assert items[1][0] == "360"


def test_merge_append_and_replace():
    existing = [{"time": 1, "label": "a", "id": 4, "data": [[0.0] * 128]}]
    items = [("x", [0.1] * 128)]
    appended = ps.merge(existing, items, "append", 99)
    assert [m["id"] for m in appended] == [4, 5] and appended[1]["time"] == 99
    assert ps.merge(existing, items, "replace", 99)[0]["id"] == 0


def test_main_appends_with_backup_and_keeps_mode(tmp_path):
    target_dir = tmp_path / "models"
    target_dir.mkdir()
    user = ps.invoking_user(ME)
    target = target_dir / f"{user}.dat"
    target.write_text(json.dumps([{"time": 1, "label": "old", "id": 0, "data": [[0.0] * 128]}]))
    os.chmod(target, 0o640)
    result = ps.main([write(tmp_path, payload())], ME, target_dir=target_dir)
    assert result["ok"] and result["added"] == 2 and result["total"] == 3
    saved = json.loads(target.read_text())
    assert [m["label"] for m in saved] == ["old", "360-left0", "360-left1"]
    assert oct(target.stat().st_mode & 0o777) == "0o640"
    backup = json.loads(Path(result["backup"]).read_text())
    assert len(backup) == 1
    assert not [p for p in target_dir.iterdir() if p.name.startswith(".howdy360-")]


def test_main_creates_new_file_without_backup(tmp_path):
    result = ps.main([write(tmp_path, payload(1, "replace"))], ME, target_dir=tmp_path / "new")
    assert result["backup"] is None and result["total"] == 1


def test_main_rejects_bad_existing_file_and_args(tmp_path):
    user = ps.invoking_user(ME)
    (tmp_path / f"{user}.dat").write_text("{}")
    with pytest.raises(ps.PayloadError):
        ps.main([write(tmp_path, payload())], ME, target_dir=tmp_path)
    with pytest.raises(ps.PayloadError):
        ps.main([], ME, target_dir=tmp_path)


def test_read_payload_refuses_symlink_and_large(tmp_path):
    real = write(tmp_path, payload())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(OSError):
        ps.read_payload(str(link))
    big = tmp_path / "big.json"
    big.write_text(" " * (ps.MAX_PAYLOAD_BYTES + 10))
    with pytest.raises(ps.PayloadError):
        ps.read_payload(str(big))


def test_models_dir_from_paths_py(tmp_path):
    p = tmp_path / "paths.py"
    p.write_text('user_models_dir = PurePath("/srv/models")\n')
    assert str(ps.models_dir((str(tmp_path / "x.py"), str(p)))) == "/srv/models"
    assert str(ps.models_dir(())) == ps.FALLBACK_MODELS_DIR
