"""`howdy360 doctor`: explain why Howdy is slow or failing. Read-only."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import journal
from .howdy import (
    DLIB_FILES,
    HowdyConfig,
    HowdyPaths,
    encodings,
    file_mode,
    load_models,
    pam_howdy_files,
    read_config,
)
from .i18n import _
from .metrics import nearest, summarize

ICONS = {"ok": "✓", "warn": "!", "fail": "✗", "info": "•"}
MIN_FACE_WIDTH = 60  # pixels at Howdy's analysis scale
MAX_CENTRE_OFFSET = 0.25  # fraction of the frame width
WIDE_PAM = {"sudo", "common-auth", "system-auth", "password-auth", "su", "polkit-1"}


@dataclass
class Check:
    status: str  # ok | warn | fail | info
    text: str


@dataclass
class Section:
    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, status: str, text: str) -> None:
        self.checks.append(Check(status, text))


@dataclass
class Report:
    sections: list[Section] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def section(self, title: str) -> Section:
        sec = Section(title)
        self.sections.append(sec)
        return sec

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for s in self.sections for c in s.checks)


def render(report: Report, title: str | None = None) -> str:
    line = "─" * 60
    out = [title or _("Howdy 360 - diagnosis"), line]
    for sec in report.sections:
        out.append(sec.title)
        out += [f"  {ICONS[c.status]} {c.text}" for c in sec.checks]
        out.append("")
    out.append(_("Recommendations"))
    if report.recommendations:
        out += [f"  {i}. {r}" for i, r in enumerate(report.recommendations, start=1)]
    else:
        out.append(f"  {_('No problems found.')}")
    out.append(line)
    return "\n".join(out)


def check_install(report: Report, paths: HowdyPaths) -> HowdyConfig | None:
    sec = report.section(_("Install"))
    try:
        config = read_config(paths.config_file)
    except (FileNotFoundError, OSError, ValueError) as exc:
        sec.add("fail", _("Howdy not found: {error}", error=exc))
        return None
    sec.add("ok", _("Howdy found ({source})", source=paths.source))
    missing = [n for n in DLIB_FILES[:2] if not paths.dlib_file(n).exists()]
    if missing:
        for name in missing:
            sec.add("fail", _("dlib data file missing: {name}", name=name))
    else:
        sec.add("ok", _("dlib data files present"))
    return config


def check_config(report: Report, paths: HowdyPaths, config: HowdyConfig) -> None:
    sec = report.section(_("Configuration"))
    if config.device_path:
        exists = Path(config.device_path).exists()
        sec.add("ok" if exists else "fail", _("camera {device}", device=config.device_path))
    else:
        sec.add("fail", _("video.device_path is not set"))
        report.recommendations.append(_("Set video.device_path in {path} (sudo howdy config)", path=paths.config_file))
    if config.disabled:
        sec.add("warn", _("Howdy is disabled in config.ini (core.disabled = true)"))
    sec.add(
        "info",
        _(
            "certainty = {c}   timeout = {t} s   dark_threshold = {d}   use_cnn = {cnn}",
            c=config.certainty,
            t=config.timeout,
            d=config.dark_threshold,
            cnn=config.use_cnn,
        ),
    )
    if config.use_cnn:
        report.recommendations.append(_("use_cnn is slow on CPU (seconds per frame); consider use_cnn = false"))


def check_pam(report: Report, pam_dir: Path = Path("/etc/pam.d")) -> None:
    sec = report.section(_("PAM scope"))
    services = pam_howdy_files(pam_dir)
    wide = [s for s in services if s in WIDE_PAM]
    if not services:
        sec.add("warn", _("Howdy is not referenced by any PAM service"))
    elif wide:
        sec.add("warn", _("Enabled for sudo/system-wide auth: {services}", services=", ".join(wide)))
        report.recommendations.append(
            _("Face login is weaker than a password: consider removing Howdy from sudo/common-auth")
        )
    else:
        sec.add("ok", _("Enabled for: {services}", services=", ".join(services)))


def check_privacy(report: Report, model_file: Path) -> None:
    if not model_file.exists():
        return
    sec = report.section(_("Privacy"))
    info = file_mode(model_file)
    if info["world_readable"]:
        sec.add("info", _("Model file {path} is world-readable ({mode})", path=model_file, mode=info["mode"]))
    else:
        sec.add("ok", _("Model file permissions: {mode}", mode=info["mode"]))


def check_history(report: Report, counts: object, days: int = 7) -> None:
    sec = report.section(_("History (last {days} days)", days=days))
    if counts is None:
        sec.add("info", _("Journal not readable (needs systemd-journal or adm group)"))
        return
    approved = counts.get("approved", 0)
    timeouts = counts.get("timeout", 0)
    failed = counts.get("failed", 0)
    total = approved + timeouts + failed
    status = "ok" if total == 0 or timeouts + failed <= approved else "fail"
    sec.add(
        status,
        _(
            "{approved} approved, {timeout} timeouts, {failed} other failures, {skipped} skipped",
            approved=approved,
            timeout=timeouts,
            failed=failed,
            skipped=counts.get("skipped", 0),
        ),
    )


def check_live(report: Report, paths: HowdyPaths, config: HowdyConfig, user: str, seconds: float) -> None:
    """Camera, speed and recognition, measured the way Howdy runs at login."""
    from .vision import CameraError, HowdyPipeline

    cam = report.section(_("IR camera"))
    try:
        pipe = HowdyPipeline(paths, config)
        pipe.open()
    except (CameraError, OSError, RuntimeError) as exc:
        cam.add("fail", _("Cannot open the camera: {error}", error=exc))
        return
    models = encodings(load_models(paths.model_file(user)))
    known = np.array(models) if models else None
    counts = {"lit": 0, "dark": 0, "black": 0}
    colour_frames, analysis_times, dists, times, widths, offsets = 0, [], [], [], [], []
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            fr = pipe.read()
            if fr is None:
                continue
            counts[fr.status] += 1
            if fr.status != "lit":
                continue
            b, g, r = (fr.frame[..., i] for i in range(3))
            if not (np.array_equal(b, g) and np.array_equal(g, r)):
                colour_frames += 1
            ta = time.perf_counter()
            face = pipe.analyze(fr, encode=known is not None)
            analysis_times.append(time.perf_counter() - ta)
            if face is not None:
                widths.append(face.width)
                offsets.append(abs(face.center_x - 0.5))
            if face is not None and known is not None:
                dists.append(nearest(known, face.emb)[0])
                times.append(time.perf_counter() - t0)
    finally:
        pipe.close()

    cam.add(
        "ok" if counts["lit"] else "fail",
        _("Camera opened, {lit} lit / {dark} dark / {black} black frames", **counts),
    )
    if counts["lit"] == 0 and counts["dark"]:
        cam.add("fail", _("All frames are too dark"))
        report.recommendations.append(
            _("Frames are too dark: check the IR emitter (linux-enable-ir-emitter) or dark_threshold")
        )
    elif counts["lit"] == 0:
        cam.add("fail", _("Camera returns only black frames"))
    elif counts["dark"]:
        cam.add("ok", _("IR emitter strobing detected (normal)"))
    if colour_frames:
        cam.add("warn", _("Frames look colour, not IR greyscale"))

    speed = report.section(_("Speed"))
    speed.add(
        "ok",
        _(
            "Startup: libraries {imp:.2f} s + models {mod:.2f} s",
            imp=pipe.timings.get("import_s", 0),
            mod=pipe.timings.get("models_s", 0),
        ),
    )
    speed.add("ok", _("Open camera: {cam:.2f} s", cam=pipe.timings.get("camera_s", 0)))
    if analysis_times:
        ms = float(np.median(analysis_times) * 1000)
        speed.add("ok" if ms < 300 else "warn", _("Analysis per frame: {ms:.0f} ms", ms=ms))

    rec = report.section(_("Recognition"))
    if known is None:
        rec.add("fail", _("No stored models for {user}", user=user))
        report.recommendations.append(_("Only {n} stored model(s). Run:  howdy360 enroll", n=0))
        return
    rec.add("ok" if len(models) >= 3 else "warn", _("Stored models: {n}", n=len(models)))
    summary = summarize(dists, times, config.certainty, len(models))
    if not summary["frames"]:
        rec.add("warn", _("No face seen during the test - sit in front of the camera"))
        return
    rate = summary["recognized"] / summary["frames"]
    badly_placed = np.median(widths) < MIN_FACE_WIDTH or np.median(offsets) > MAX_CENTRE_OFFSET
    if badly_placed:
        rec.add("warn", _("Your face was small or off-centre - sit facing the screen and run again"))
    rec.add(
        "ok" if summary["median"] < config.certainty else "fail",
        _(
            "Distance to your face: median {median}, best {best} (limit {limit})",
            median=summary["median"],
            best=summary["best"],
            limit=config.certainty,
        ),
    )
    rec.add(
        "ok" if rate >= 0.8 else "warn" if rate >= 0.5 else "fail",
        _(
            "Recognised frames: {ok}/{total} ({pct}%)",
            ok=summary["recognized"],
            total=summary["frames"],
            pct=round(rate * 100),
        ),
    )
    if summary["first_match_s"] is None:
        rec.add("fail", _("No match during the test"))
    else:
        rec.add("ok", _("First match after {s} s", s=summary["first_match_s"]))
    if badly_placed:
        report.recommendations.append(_("Sit in front of the laptop, facing the screen, and run doctor again"))
    elif len(models) < 3:
        report.recommendations.append(_("Only {n} stored model(s). Run:  howdy360 enroll", n=len(models)))
        report.recommendations.append(_("Then measure again:  howdy360 measure"))
    elif rate < 0.8:
        report.recommendations.append(_("Most frames are not recognised. Add angles with:  howdy360 enroll"))


def run(paths: HowdyPaths, user: str, seconds: float = 10.0, camera: bool = True) -> Report:
    report = Report()
    config = check_install(report, paths)
    if config is None:
        return report
    check_config(report, paths, config)
    check_pam(report)
    if camera and config.device_path:
        print(_("Look at the screen normally for {s} seconds...", s=int(seconds)), flush=True)
        check_live(report, paths, config, user, seconds)
    check_history(report, journal.read())
    check_privacy(report, paths.model_file(user))
    return report
