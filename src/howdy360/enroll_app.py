"""`howdy360 enroll`: guided multi-angle enrollment (GTK4 + libadwaita)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .howdy import HowdyPaths, encodings, load_models, read_config
from .i18n import _
from .metrics import nearest, state_dir, summarize
from .pose import SECTOR_KEYS, Candidate, SectorTracker, keep_best, sector_of, selection

CAL_FRAMES, ROTATE_S, MEASURE_S, PAUSE_S = 8, 75.0, 10.0, 3
MIN_FACE_WIDTH = 60
BLUR_RATIO = 0.25
STEPS = ("Calibrate", "Rotate", "Measure", "Results")

CSS = """
.step { padding: 4px 14px; border-radius: 999px; color: alpha(currentColor, 0.5); font-weight: 600; }
.step.active { background: @accent_bg_color; color: @accent_fg_color; }
.step.done { color: @success_color; }
.instruction { font-size: 22px; font-weight: 800; }
.metric { font-size: 26px; font-weight: 800; }
.better { color: @success_color; }
.pad { padding: 18px; }
"""


class Cancelled(Exception):
    pass


class Engine(threading.Thread):
    """Camera worker: calibrate → rotate → measure. Publishes UI snapshots."""

    def __init__(self, paths: HowdyPaths, user: str, publish) -> None:
        super().__init__(daemon=True)
        self.paths, self.user, self.publish = paths, user, publish
        self.cancel = threading.Event()
        self.state: dict = {"step": 0, "title": _("Opening the camera..."), "hint": "", "progress": 0.0}
        self.chosen: list[tuple[str, np.ndarray]] = []
        self.report: dict = {}

    def emit(self, **kw) -> None:
        self.state.update(kw)
        self.publish(dict(self.state))

    def frames(self, pipe, deadline: float | None = None):
        """Lit frames until cancelled or the deadline passes (preview updated)."""
        while deadline is None or time.monotonic() < deadline:
            if self.cancel.is_set():
                raise Cancelled()
            fr = pipe.read()
            if fr is None or fr.status == "black":
                continue
            self.state["frame"] = np.ascontiguousarray(fr.display[:, ::-1])
            if fr.status == "lit":
                yield fr

    def run(self) -> None:
        from .vision import HowdyPipeline

        try:
            config = read_config(self.paths.config_file)
            known = np.array(encodings(load_models(self.paths.model_file(self.user))))
            with HowdyPipeline(self.paths, config) as pipe:
                self.report = self.phases(pipe, config, known)
            self.emit(step=3, done=True, report=self.report)
        except Cancelled:
            pass
        except Exception as exc:  # shown in the UI instead of dying silently
            self.emit(error=f"{type(exc).__name__}: {exc}")

    def phases(self, pipe, config, known: np.ndarray) -> dict:
        # 1. Calibrate the neutral pose.
        cal: list = []
        for fr in self.frames(pipe):
            face = pipe.analyze(fr)
            if face is None:
                self.emit(step=0, title=_("Put your face inside the circle"), hint=_("No face detected"), face_ok=False)
            elif face.width < MIN_FACE_WIDTH:
                self.emit(step=0, title=_("Move a little closer"), hint=_("Your face is too small"), face_ok=False)
            else:
                cal.append(face)
                self.emit(
                    step=0,
                    title=_("Look straight ahead"),
                    hint=_("Calibrating your normal position"),
                    progress=len(cal) / CAL_FRAMES,
                    face_ok=True,
                )
            if len(cal) >= CAL_FRAMES:
                break
        y0 = float(np.median([f.yaw for f in cal]))
        p0 = float(np.median([f.pitch for f in cal]))
        sharp_ref = float(np.median([f.sharp for f in cal]))
        center: list[Candidate] = []
        for f in cal:
            keep_best(center, Candidate(f.emb, f.sharp), 3)

        # 2. Rotate until every sector is covered.
        tracker = SectorTracker()
        sectors: list[list[Candidate]] = [[] for _ in range(8)]
        lost_since, last_mag, lit, no_face, blurry = None, 0.0, 0, 0, 0
        dot = None
        t0 = time.monotonic()
        for fr in self.frames(pipe, deadline=t0 + ROTATE_S):
            lit += 1
            face = pipe.analyze(fr)
            hint = None
            if face is None:
                no_face += 1
                lost_since = lost_since or time.monotonic()
                if time.monotonic() - lost_since > 0.6 and last_mag > 0.6:
                    hint = _("Lost your face - turn a little less that way")
            else:
                lost_since = None
                dot = tracker.update(face.yaw - y0, face.pitch - p0, time.monotonic() - t0)
                last_mag = math.hypot(*dot)
                cand = Candidate(face.emb, face.sharp)
                if face.sharp < BLUR_RATIO * sharp_ref:
                    blurry += 1
                    hint = _("Blurry - turn more slowly")
                elif (idx := sector_of(*dot)) is not None:
                    keep_best(sectors[idx], cand, 2)
                else:
                    keep_best(center, cand, 3)
            covered = [bool(s) for s in sectors]
            missing = [_(SECTOR_KEYS[i]) for i in range(8) if not covered[i]]
            self.emit(
                step=1,
                title=_("Slowly turn your head in a circle"),
                hint=hint or (_("Missing: {sectors}", sectors=", ".join(missing)) if missing else _("Complete!")),
                progress=sum(covered) / 8,
                covered=covered,
                dot=dot,
                face_ok=face is not None,
            )
            if all(covered):
                break
        self.chosen = selection(center, sectors)
        new = np.vstack([known.reshape(-1, 128), *[e for _, e in self.chosen]]) if self.chosen else known

        # 3. Pause, then measure sitting normally (current vs current + new).
        end = time.monotonic() + PAUSE_S
        for _fr in self.frames(pipe, deadline=end):
            left = max(1, math.ceil(end - time.monotonic()))
            self.emit(
                step=2,
                title=_("Great! Now sit normally"),
                hint=_("Measurement starts in {s} s", s=left),
                progress=0.0,
                dot=None,
                covered=None,
            )
        res: dict[str, list] = {"cur": [], "new": [], "t": []}
        t2 = time.monotonic()
        for fr in self.frames(pipe, deadline=t2 + MEASURE_S):
            face = pipe.analyze(fr)
            if face is not None:
                res["t"].append(time.monotonic() - t2)
                res["new"].append(nearest(new, face.emb)[0])
                if len(known):
                    res["cur"].append(nearest(known, face.emb)[0])
            self.emit(
                step=2,
                title=_("Measuring - look at the screen normally"),
                hint=f"{max(0, int(MEASURE_S - (time.monotonic() - t2)))} s",
                progress=(time.monotonic() - t2) / MEASURE_S,
                face_ok=face is not None,
            )
        cur_t = res["t"] if len(known) else []
        return {
            "sectors_covered": sum(bool(s) for s in sectors),
            "new_models": len(self.chosen),
            "reach": {k: round(v, 3) for k, v in tracker.reach.items()},
            "adapted_sides": tracker.adapted,
            "rotation": {"lit_frames": lit, "without_face": no_face, "blurry": blurry},
            "current": summarize(res["cur"], cur_t, config.certainty, len(known)),
            "with_360": summarize(res["new"], res["t"], config.certainty, len(new)),
        }


def save_with_pkexec(items: list[tuple[str, np.ndarray]], replace: bool) -> dict:
    """Hand the new encodings to the root helper through polkit."""
    payload = {
        "mode": "replace" if replace else "append",
        "models": [{"label": label, "vector": [float(x) for x in emb]} for label, emb in items],
    }
    runtime = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix="howdy360-", suffix=".json", dir=runtime)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        helper = Path(__file__).with_name("privileged_save.py")
        python = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
        proc = subprocess.run(
            ["pkexec", python, "-I", str(helper), path], capture_output=True, text=True, timeout=300, check=False
        )
    finally:
        os.unlink(path)
    if proc.returncode in (126, 127) and not proc.stdout.strip():
        return {"ok": False, "error": "authentication cancelled"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"}


def save_candidates(items: list[tuple[str, np.ndarray]]) -> Path:
    """Dry run: keep candidates (mode 600) for `howdy360 measure --candidates`."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f"candidates-{time.strftime('%Y%m%d-%H%M%S')}.npy"
    np.save(path, np.array([e for _, e in items]))
    os.chmod(path, 0o600)
    return path


def run(
    paths: HowdyPaths,
    user: str,
    dry_run: bool,
    replace: bool,
    demo: bool = False,
    screenshots: Path | None = None,
) -> int:
    import gi

    if screenshots is not None:
        # XWayland keeps drawing while the window is covered (Wayland stops sending
        # frames to occluded windows), so captures are reliable.
        os.environ.setdefault("GDK_BACKEND", "x11")
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    gi.require_version("Graphene", "1.0")
    from gi.repository import Adw, Gdk, GLib, Graphene, Gtk

    from .render import SIZE, compose

    class Capture(Gtk.Widget):
        """Bin container that can record its child's render node during a real frame."""

        def __init__(self, child) -> None:
            super().__init__()
            self.set_layout_manager(Gtk.BinLayout())
            child.set_parent(self)
            self.child = child
            self.request = None

        def do_snapshot(self, snapshot) -> None:
            if self.request is not None:
                done, self.request = self.request, None
                record = Gtk.Snapshot()
                self.snapshot_child(self.child, record)
                GLib.idle_add(done, record.to_node(), self.get_width(), self.get_height())
            self.snapshot_child(self.child, snapshot)

    class App(Adw.Application):
        def __init__(self) -> None:
            super().__init__(application_id="io.github.diogocouto18.Howdy360")
            self.snap: dict = {}
            self.pending = False
            self.engine: Engine | None = None
            self.exit_code = 0
            self.shots: set[str] = set()

        def do_activate(self) -> None:
            css = Gtk.CssProvider()
            css.load_from_string(CSS)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            self.win = Adw.ApplicationWindow(application=self, default_width=640, default_height=840, title="Howdy 360")
            self.win.connect("close-request", self.on_close)
            view = Adw.ToolbarView(css_classes=["background"])
            view.add_top_bar(
                Adw.HeaderBar(title_widget=Adw.WindowTitle(title="Howdy 360", subtitle=_("Guided face enrollment")))
            )
            self.toasts = Adw.ToastOverlay()
            self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
            self.stack.add_named(self.capture_page(), "capture")
            self.toasts.set_child(self.stack)
            view.set_content(self.toasts)
            self.capture = Capture(view)
            self.win.set_content(self.capture)
            self.win.present()
            if demo:
                from .demo import DemoEngine

                self.engine = DemoEngine(self.publish)
            else:
                self.engine = Engine(paths, user, self.publish)
            self.engine.start()

        def capture_page(self):
            box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                spacing=14,
                margin_top=12,
                margin_bottom=24,
                margin_start=24,
                margin_end=24,
            )
            steps = Gtk.Box(spacing=6, halign=Gtk.Align.CENTER)
            self.step_labels = [Gtk.Label(label=_(s), css_classes=["step"]) for s in STEPS]
            for label in self.step_labels:
                steps.append(label)
            box.append(steps)
            self.picture = Gtk.Picture(
                vexpand=True, can_shrink=True, content_fit=Gtk.ContentFit.CONTAIN, height_request=420
            )
            box.append(self.picture)
            self.title = Gtk.Label(
                label=_("Opening the camera..."),
                css_classes=["instruction"],
                wrap=True,
                justify=Gtk.Justification.CENTER,
            )
            self.hint = Gtk.Label(css_classes=["dim-label"], wrap=True, justify=Gtk.Justification.CENTER)
            self.bar = Gtk.ProgressBar(margin_start=40, margin_end=40)
            for widget in (self.title, self.hint, self.bar):
                box.append(widget)
            cancel = Gtk.Button(label=_("Cancel"), halign=Gtk.Align.CENTER, css_classes=["pill", "destructive-action"])
            cancel.connect("clicked", lambda *_a: self.win.close())
            box.append(cancel)
            return box

        def results_page(self, report: dict):
            cur, new = report["current"], report["with_360"]
            page = Adw.StatusPage(
                icon_name="emblem-ok-symbolic",
                title=_("Enrollment finished"),
                description=_(
                    "{covered}/8 angles covered · {n} new models",
                    covered=report["sectors_covered"],
                    n=report["new_models"],
                ),
            )
            card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, css_classes=["card", "pad"], halign=Gtk.Align.CENTER)
            grid = Gtk.Grid(column_spacing=36, row_spacing=10)
            grid.attach(Gtk.Label(label=_("Current ({n})", n=cur["n_models"]), css_classes=["dim-label"]), 1, 0, 1, 1)
            grid.attach(Gtk.Label(label=_("With 360 ({n})", n=new["n_models"]), css_classes=["dim-label"]), 2, 0, 1, 1)

            def fmt(summary, key):
                if not summary.get("frames"):
                    return "—"
                if key == "recognized":
                    return f"{summary['recognized']}/{summary['frames']}"
                value = summary.get(key)
                return "—" if value is None else (f"{value} s" if key == "first_match_s" else str(value))

            def better(key):
                a, b = cur.get(key), new.get(key)
                if a is None or b is None or not cur.get("frames"):
                    return False
                return b > a if key == "recognized" else b < a

            rows = (
                ("median", _("Median distance")),
                ("recognized", _("Recognised frames")),
                ("first_match_s", _("First match")),
            )
            for row, (key, name) in enumerate(rows, start=1):
                grid.attach(Gtk.Label(label=name, xalign=0), 0, row, 1, 1)
                grid.attach(Gtk.Label(label=fmt(cur, key), css_classes=["metric"]), 1, row, 1, 1)
                classes = ["metric", "better"] if better(key) else ["metric"]
                grid.attach(Gtk.Label(label=fmt(new, key), css_classes=classes), 2, row, 1, 1)
            card.append(grid)
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
            box.append(card)
            self.status = Gtk.Label(css_classes=["dim-label"], wrap=True, justify=Gtk.Justification.CENTER)
            box.append(self.status)
            buttons = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER)
            if demo:
                self.status.set_label(_("Demo mode: simulated run, nothing was saved."))
                close = Gtk.Button(label=_("Close"), css_classes=["pill", "suggested-action"])
                close.connect("clicked", lambda *_a: self.win.close())
                buttons.append(close)
            elif dry_run:
                path = save_candidates(self.engine.chosen) if self.engine.chosen else None
                self.status.set_label(_("Dry run: nothing was saved. Candidates: {path}", path=path))
                close = Gtk.Button(label=_("Close"), css_classes=["pill", "suggested-action"])
                close.connect("clicked", lambda *_a: self.win.close())
                buttons.append(close)
            else:
                self.discard = Gtk.Button(label=_("Discard"), css_classes=["pill"])
                self.discard.connect("clicked", lambda *_a: self.win.close())
                self.save = Gtk.Button(label=_("Save to Howdy"), css_classes=["pill", "suggested-action"])
                self.save.set_sensitive(bool(self.engine.chosen))
                self.save.connect("clicked", self.on_save)
                buttons.append(self.discard)
                buttons.append(self.save)
                self.status.set_label(_("Howdy's model file was not changed."))
            box.append(buttons)
            page.set_child(box)
            return page

        def on_save(self, *_a) -> None:
            if replace:
                dialog = Adw.AlertDialog(
                    heading=_("Replace all existing models?"), body=_("Your current models will be backed up first.")
                )
                dialog.add_response("cancel", _("Cancel"))
                dialog.add_response("replace", _("Replace"))
                dialog.set_response_appearance("replace", Adw.ResponseAppearance.DESTRUCTIVE)
                dialog.connect("response", lambda _d, r: r == "replace" and self.start_save())
                dialog.present(self.win)
            else:
                self.start_save()

        def start_save(self) -> bool:
            self.save.set_sensitive(False)
            self.discard.set_sensitive(False)
            self.status.set_label(_("Saving... (authentication required)"))
            items = list(self.engine.chosen)
            threading.Thread(
                target=lambda: GLib.idle_add(self.saved, save_with_pkexec(items, replace)), daemon=True
            ).start()
            return True

        def saved(self, result: dict) -> bool:
            if result.get("ok"):
                msg = (
                    _("Saved {n} models. Backup: {backup}", n=result["added"], backup=result["backup"])
                    if result.get("backup")
                    else _("Saved {n} models.", n=result["added"])
                )
                self.discard.set_label(_("Close"))
                self.discard.add_css_class("suggested-action")
            else:
                msg = _("Save failed: {error}", error=result.get("error"))
                self.save.set_sensitive(True)
                self.exit_code = 1
            self.status.set_label(msg)
            self.discard.set_sensitive(True)
            self.toasts.add_toast(Adw.Toast(title=msg, timeout=8))
            print(json.dumps(result), flush=True)
            return False

        def publish(self, snap: dict) -> None:
            self.snap = snap
            if not self.pending:
                self.pending = True
                GLib.idle_add(self.apply)

        def apply(self) -> bool:
            self.pending = False
            s = self.snap
            if s.get("error"):
                self.title.set_label(_("Error"))
                self.hint.set_label(s["error"])
                self.exit_code = 1
                return False
            self.title.set_label(s.get("title", ""))
            self.hint.set_label(s.get("hint", ""))
            self.bar.set_fraction(min(max(float(s.get("progress", 0.0)), 0.0), 1.0))
            step = s.get("step", 0)
            for i, label in enumerate(self.step_labels):
                label.remove_css_class("active")
                label.remove_css_class("done")
                if i == step:
                    label.add_css_class("active")
                elif i < step:
                    label.add_css_class("done")
            rgba = compose(
                s.get("frame"),
                bool(s.get("face_ok")),
                s.get("covered") if step == 1 else None,
                s.get("dot") if step == 1 else None,
            )
            self.picture.set_paintable(
                Gdk.MemoryTexture.new(SIZE, SIZE, Gdk.MemoryFormat.R8G8B8A8, GLib.Bytes.new(rgba.tobytes()), SIZE * 4)
            )
            if s.get("done") and self.stack.get_child_by_name("results") is None:
                print(json.dumps(s["report"], indent=2), flush=True)
                self.stack.add_named(self.results_page(s["report"]), "results")
                self.stack.set_visible_child_name("results")
                if screenshots is not None:
                    GLib.timeout_add(900, self.screenshot, "results", True)
            if screenshots is not None and step == 1 and sum(s.get("covered") or []) == 5:
                GLib.timeout_add(50, self.screenshot, "rotate", False)
            return False

        def screenshot(self, name: str, quit_after: bool) -> bool:
            """Render the window content to PNG (nothing else on screen is captured)."""
            if name in self.shots:
                return False
            self.shots.add(name)
            self.capture.request = lambda node, w, h: self.save_shot(node, w, h, name, quit_after)
            self.capture.queue_draw()
            return False

        def save_shot(self, node, width: int, height: int, name: str, quit_after: bool) -> bool:
            if node is not None:
                texture = self.win.get_renderer().render_texture(node, Graphene.Rect().init(0, 0, width, height))
                screenshots.mkdir(parents=True, exist_ok=True)
                texture.save_to_png(str(screenshots / f"enroll-{name}.png"))
            if quit_after:
                self.win.close()
            return False

        def on_close(self, *_a) -> bool:
            if self.engine is not None:
                self.engine.cancel.set()
            return False

    app = App()
    app.run([sys.argv[0]])
    return app.exit_code
