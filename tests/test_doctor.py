from collections import Counter

from howdy360 import doctor, howdy


def paths_for(tree):
    return howdy.discover_paths(candidates=(str(tree["paths_py"]),))


def test_run_without_camera(howdy_tree, monkeypatch):
    monkeypatch.setattr(doctor.journal, "read", lambda days=7: Counter(approved=3, timeout=7))
    monkeypatch.setattr(doctor, "pam_howdy_files", lambda pam_dir=None: ["gdm-password"])
    report = doctor.run(paths_for(howdy_tree), "alice", camera=False)
    text = doctor.render(report)
    assert "Howdy found" in text and "3 approved, 7 timeouts" in text
    assert report.failed  # more failures than approvals
    assert "world-readable" in text or "Model file permissions" in text


def test_missing_install(tmp_path):
    paths = howdy.discover_paths({"config_dir": str(tmp_path)}, candidates=())
    report = doctor.run(paths, "alice", camera=False)
    assert report.failed and "Howdy not found" in doctor.render(report)


def test_config_and_pam_recommendations(howdy_tree, tmp_path):
    report = doctor.Report()
    cfg = howdy.HowdyConfig(device_path=None, use_cnn=True, disabled=True)
    doctor.check_config(report, paths_for(howdy_tree), cfg)
    pam = tmp_path / "pam"
    pam.mkdir()
    (pam / "sudo").write_text("auth sufficient pam_howdy.so\n")
    doctor.check_pam(report, pam)
    text = doctor.render(report)
    assert "device_path is not set" in text and "disabled" in text
    assert len(report.recommendations) == 3
    empty = doctor.Report()
    doctor.check_pam(empty, tmp_path / "nothing")
    assert empty.sections[0].checks[0].status == "warn"


def test_missing_dlib_file(howdy_tree):
    (howdy_tree["dlib_dir"] / "shape_predictor_5_face_landmarks.dat").unlink()
    report = doctor.Report()
    doctor.check_install(report, paths_for(howdy_tree))
    assert report.failed


def test_history_unreadable_and_empty():
    report = doctor.Report()
    doctor.check_history(report, None)
    doctor.check_history(report, Counter())
    assert [s.checks[0].status for s in report.sections] == ["info", "ok"]
    assert "No problems found." in doctor.render(report)
