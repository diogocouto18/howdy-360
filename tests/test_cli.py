import pytest

from howdy360 import cli, doctor


def test_parser_requires_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])
    args = cli.build_parser().parse_args(["enroll", "--dry-run"])
    assert args.dry_run and not args.replace


def test_doctor_no_camera_exit_code(howdy_tree, monkeypatch, capsys):
    monkeypatch.setattr(doctor.journal, "read", lambda days=7: None)
    monkeypatch.setattr(doctor, "pam_howdy_files", lambda pam_dir=None: ["gdm-password"])
    code = cli.main(
        [
            "--config-dir",
            str(howdy_tree["config_dir"]),
            "--models-dir",
            str(howdy_tree["models_dir"]),
            "--dlib-data-dir",
            str(howdy_tree["dlib_dir"]),
            "--user",
            "alice",
            "doctor",
            "--no-camera",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and "Recommendations" in out
