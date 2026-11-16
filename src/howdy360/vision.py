"""Camera + dlib pipeline that mirrors Howdy 3 (compare.py / cli/add.py).

Detection runs on the CLAHE-equalised greyscale frame; landmarks and the
128-d descriptor are computed on the original (scaled) frame, exactly as Howdy
does at login. Heavy imports (cv2, dlib) happen here only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .howdy import HowdyConfig, HowdyPaths
from .pose import head_pose


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    status: str  # "lit" | "dark" | "black"
    darkness: float
    display: np.ndarray  # CLAHE greyscale, full size (for previews)
    frame: np.ndarray | None = None  # original frame, scaled (BGR)
    gs: np.ndarray | None = None  # CLAHE greyscale, scaled


@dataclass
class Face:
    width: int
    yaw: float
    pitch: float
    sharp: float
    emb: np.ndarray | None
    center_x: float = 0.5  # face centre as a fraction of the frame width


class HowdyPipeline:
    def __init__(self, paths: HowdyPaths, config: HowdyConfig) -> None:
        if not config.device_path:
            raise CameraError("video.device_path is not set in Howdy's config.ini")
        self.config = config
        self.timings: dict[str, float] = {}
        t = time.perf_counter()
        import cv2
        import dlib

        self.cv2, self.dlib = cv2, dlib
        self.timings["import_s"] = time.perf_counter() - t

        t = time.perf_counter()
        if config.use_cnn:
            self.detector = dlib.cnn_face_detection_model_v1(str(paths.dlib_file("mmod_human_face_detector.dat")))
        else:
            self.detector = dlib.get_frontal_face_detector()
        self.pose = dlib.shape_predictor(str(paths.dlib_file("shape_predictor_5_face_landmarks.dat")))
        self.encoder = dlib.face_recognition_model_v1(str(paths.dlib_file("dlib_face_recognition_resnet_model_v1.dat")))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.timings["models_s"] = time.perf_counter() - t
        self.cap = None
        self.scale: float | None = None

    def open(self) -> None:
        t = time.perf_counter()
        self.cap = self.cv2.VideoCapture(self.config.device_path, self.cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise CameraError(f"cannot open camera {self.config.device_path}")
        self.timings["camera_s"] = time.perf_counter() - t

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> HowdyPipeline:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read(self) -> Frame | None:
        """One frame classified like Howdy: black, dark (skipped) or lit."""
        cv2 = self.cv2
        ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        gs = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        hist = cv2.calcHist([gs], [0], None, [8], [0, 256])
        total = float(hist.sum())
        darkness = float(hist[0][0] / total * 100) if total else 100.0
        if total == 0 or darkness == 100:
            return Frame("black", darkness, gs)
        if darkness > self.config.dark_threshold:
            return Frame("dark", darkness, gs)
        if self.scale is None:
            self.scale = (self.config.max_height / frame.shape[0]) or 1.0
        if self.scale != 1:
            small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            gs_small = cv2.resize(gs, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        else:
            small, gs_small = frame, gs
        return Frame("lit", darkness, gs, small, gs_small)

    def analyze(self, fr: Frame, encode: bool = True) -> Face | None:
        """Largest face in a lit frame, with pose, sharpness and descriptor."""
        faces = self.detector(fr.gs, 1)
        if not faces:
            return None
        rects = [f.rect if self.config.use_cnn else f for f in faces]
        r = max(rects, key=lambda x: x.width())
        shape = self.pose(fr.frame, r)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(5)], dtype=float)
        yaw, pitch = head_pose(pts)
        crop = fr.gs[max(r.top(), 0) : r.bottom(), max(r.left(), 0) : r.right()]
        sharp = float(self.cv2.Laplacian(crop, self.cv2.CV_64F).var()) if crop.size else 0.0
        emb = np.array(self.encoder.compute_face_descriptor(fr.frame, shape, 1)) if encode else None
        center_x = (r.left() + r.right()) / 2 / max(fr.gs.shape[1], 1)
        return Face(r.width(), yaw, pitch, sharp, emb, center_x)
