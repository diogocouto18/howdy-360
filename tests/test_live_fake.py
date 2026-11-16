"""Camera-bound logic of doctor/measure, driven by a fake Howdy pipeline."""

import numpy as np
import pytest

from howdy360 import doctor, howdy, measure, vision


class FakePipeline:
    script: list = []  # ("lit"|"dark"|"black"|None, embedding or None)
    fail_open = False
    width, cx = 80, 0.5

    def __init__(self, paths, config):
        self.timings = {"import_s": 0.3, "models_s": 0.4}

    def open(self):
        if self.fail_open:
            raise vision.CameraError("busy")
        self.timings["camera_s"] = 0.04
        self.queue = list(self.script)

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def read(self):
        if not self.queue:
            return None
        status, emb = self.queue.pop(0)
        if status is None:
            return None
        frame = np.zeros((4, 4, 3), np.uint8)
        fr = vision.Frame(status, 10.0, frame[..., 0], frame, frame[..., 0])
        fr.emb = emb
        return fr

    def analyze(self, fr, encode=True):
        if fr.emb is None:
            return None
        return vision.Face(self.width, 0.0, 0.5, 100.0, fr.emb if encode else None, self.cx)


@pytest.fixture
def setup(howdy_tree, monkeypatch):
    monkeypatch.setattr(vision, "HowdyPipeline", FakePipeline)
    monkeypatch.setattr(doctor.journal, "read", lambda days=7: None)
    monkeypatch.setattr(doctor, "pam_howdy_files", lambda pam_dir=None: ["gdm-password"])
    FakePipeline.fail_open = False
    known = np.array(howdy_tree["models"][0]["data"][0])
    return howdy.discover_paths(candidates=(str(howdy_tree["paths_py"]),)), known


def run_doctor(paths, script, user="alice"):
    FakePipeline.script = script * 20
    report = doctor.run(paths, user, seconds=0.05)
    return report, doctor.render(report)


def test_doctor_recognises_known_face(setup):
    paths, known = setup
    report, text = run_doctor(paths, [("lit", known + 0.001), ("dark", None), ("lit", None)])
    assert "IR emitter strobing detected" in text and "Recognised frames" in text
    assert "Only 2 stored model(s)" in text  # fewer than 3 models -> enroll
    assert "First match after" in text


def test_doctor_unknown_face_fails(setup):
    paths, known = setup
    report, text = run_doctor(paths, [("lit", known + 0.2)])
    assert report.failed and "No match during the test" in text


def test_doctor_dark_black_and_colour(setup, monkeypatch):
    paths, _ = setup
    _, text = run_doctor(paths, [("dark", None)])
    assert "All frames are too dark" in text and "linux-enable-ir-emitter" in text
    _, text = run_doctor(paths, [("black", None)])
    assert "only black frames" in text

    original = FakePipeline.read

    def colour_read(self):
        fr = original(self)
        if fr is not None:
            fr.frame = np.zeros((4, 4, 3), np.uint8)
            fr.frame[..., 2] = 9
        return fr

    monkeypatch.setattr(FakePipeline, "read", colour_read)
    _, text = run_doctor(paths, [("lit", None)])
    assert "colour" in text


def test_doctor_no_models_and_no_face(setup):
    paths, known = setup
    _, text = run_doctor(paths, [("lit", known)], user="bob")
    assert "No stored models for bob" in text
    _, text = run_doctor(paths, [("lit", None)])
    assert "No face seen" in text


def test_doctor_badly_placed_face(setup):
    paths, known = setup
    FakePipeline.width, FakePipeline.cx = 40, 0.1
    try:
        _, text = run_doctor(paths, [("lit", known + 0.2)])
    finally:
        FakePipeline.width, FakePipeline.cx = 80, 0.5
    assert "off-centre" in text and "run doctor again" in text and "howdy360 enroll" not in text


def test_doctor_camera_error(setup):
    paths, _ = setup
    FakePipeline.fail_open = True
    report, text = run_doctor(paths, [])
    assert report.failed and "Cannot open the camera" in text


def test_measure_capture(setup):
    paths, known = setup
    FakePipeline.script = [("lit", known), ("dark", None), (None, None), ("lit", None)] * 10
    dists, times, matches = measure.capture(paths, np.array([known]), 0.05)
    assert dists and all(d < 0.01 for d in dists) and set(matches) == {0}
