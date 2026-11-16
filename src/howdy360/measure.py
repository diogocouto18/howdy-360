"""`howdy360 measure`: recognition quality now vs the previous measurement."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .howdy import HowdyPaths, encodings, load_models, model_labels, read_config
from .i18n import _
from .metrics import append_history, compare, last_history, model_usage, nearest, summarize


def capture(paths: HowdyPaths, known: np.ndarray, seconds: float, certainty_cfg=None):
    """Distances, timestamps and closest-model indices for `seconds` of video."""
    from .vision import HowdyPipeline

    config = certainty_cfg or read_config(paths.config_file)
    dists, times, matches = [], [], []
    with HowdyPipeline(paths, config) as pipe:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fr = pipe.read()
            if fr is None or fr.status != "lit":
                continue
            face = pipe.analyze(fr)
            if face is None:
                continue
            d, i = nearest(known, face.emb)
            dists.append(d)
            times.append(time.perf_counter() - t0)
            matches.append(i)
    return dists, times, matches


def render(summary: dict, unused: list[str], previous: dict | None, extra: str) -> str:
    line = "─" * 60
    out = [_("Howdy 360 - measurement"), line, _("Stored models: {n}{extra}", n=summary["n_models"], extra=extra), ""]
    if not summary.get("frames"):
        out.append(_("No face seen during the test - sit in front of the camera"))
        return "\n".join([*out, line])
    rate = round(100 * summary["recognized"] / summary["frames"])
    first = "—" if summary["first_match_s"] is None else f"{summary['first_match_s']} s"
    out += [
        f"  {_('Median distance'):<24}{summary['median']}",
        f"  {_('Best distance'):<24}{summary['best']}",
        f"  {_('Recognised frames'):<24}{summary['recognized']}/{summary['frames']} ({rate}%)",
        f"  {_('First match'):<24}{first}",
    ]
    if unused:
        out.append(f"  {_('Never the closest model'):<24}{', '.join(unused)}")
    if previous:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(previous.get("time", 0)))
        out += ["", _("Compared with previous measurement ({when})", when=when)]
        names = {
            "median": _("Median distance"),
            "first_match_s": _("First match"),
            "recognized_pct": _("Recognised frames"),
        }
        for key, before, after, improved in compare(previous, summary):
            verdict = _("same") if improved is None or before == after else _("better") if improved else _("worse")
            out.append(f"  {names[key]:<24}{before} → {after}   {verdict}")
    return "\n".join([*out, line])


def run(paths: HowdyPaths, user: str, seconds: float, candidates: Path | None, history: bool) -> str:
    config = read_config(paths.config_file)
    models = load_models(paths.model_file(user))
    vectors, labels = encodings(models), model_labels(models)
    extra = ""
    if candidates is not None:
        cand = np.load(candidates)
        vectors += cand.tolist()
        labels += [f"candidate-{i + 1}" for i in range(len(cand))]
        extra = _(" (+ {n} candidates)", n=len(cand))
    if not vectors:
        return _("No stored models for {user}", user=user)
    print(_("Look at the screen normally for {s} seconds...", s=int(seconds)), flush=True)
    dists, times, matches = capture(paths, np.array(vectors), seconds, config)
    summary = summarize(dists, times, config.certainty, len(vectors))
    usage = model_usage(matches, labels)
    unused = [key.split(":", 1)[1] for key, n in usage.items() if n == 0] if summary.get("frames") else []
    previous = last_history() if history else None
    text = render(summary, unused, previous, extra)
    if history and summary.get("frames"):
        append_history({"kind": "measure", "candidates": candidates is not None, **summary})
    return text
