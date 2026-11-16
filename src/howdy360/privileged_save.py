#!/usr/bin/env python3
"""Root helper that adds face models to Howdy, run through pkexec.

Standard library only, so it can run as `pkexec /usr/bin/python3 -I <this file>`.
Security rules:
  * the target user comes from PKEXEC_UID, never from arguments;
  * the models directory is resolved from Howdy's own paths.py, never from input;
  * the payload is validated strictly (1-32 vectors of 128 finite floats);
  * a timestamped backup is written first, then the new file atomically, keeping
    the original owner and permissions.

Payload (JSON file given as the only argument):
  {"mode": "append" | "replace", "models": [{"label": str, "vector": [128 floats]}]}
Prints a JSON result on stdout.
"""

from __future__ import annotations

import json
import math
import os
import pwd
import re
import sys
import tempfile
import time
from pathlib import Path

PATHS_PY_CANDIDATES = (
    "/usr/lib/howdy/paths.py",
    "/usr/lib64/howdy/paths.py",
    "/usr/local/lib/howdy/paths.py",
    "/lib/security/howdy/paths.py",
    "/usr/lib/security/howdy/paths.py",
)
FALLBACK_MODELS_DIR = "/etc/howdy/models"
MAX_VECTORS = 32
MAX_PAYLOAD_BYTES = 1_000_000
LABEL_MAX = 24
_ASSIGN = re.compile(r"^user_models_dir\s*=\s*PurePath\(\s*[\"']([^\"']+)[\"']\s*\)", re.MULTILINE)


class PayloadError(ValueError):
    pass


def models_dir(candidates: tuple[str, ...] = PATHS_PY_CANDIDATES) -> Path:
    for candidate in candidates:
        try:
            text = Path(candidate).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = _ASSIGN.search(text)
        if match:
            return Path(match.group(1))
    return Path(FALLBACK_MODELS_DIR)


def invoking_user(env: dict[str, str]) -> str:
    """The user who ran pkexec; refuses to guess."""
    uid = env.get("PKEXEC_UID")
    if uid is None or not uid.isdigit():
        raise PermissionError("must be run through pkexec (PKEXEC_UID missing)")
    return pwd.getpwuid(int(uid)).pw_name


def sanitize_label(label: object) -> str:
    text = str(label).replace(",", "").strip()[:LABEL_MAX]
    return text or "360"


def validate(payload: object) -> tuple[str, list[tuple[str, list[float]]]]:
    if not isinstance(payload, dict):
        raise PayloadError("payload must be an object")
    mode = payload.get("mode", "append")
    if mode not in ("append", "replace"):
        raise PayloadError("mode must be 'append' or 'replace'")
    models = payload.get("models")
    if not isinstance(models, list) or not 1 <= len(models) <= MAX_VECTORS:
        raise PayloadError(f"models must be a list of 1-{MAX_VECTORS} items")
    items = []
    for item in models:
        if not isinstance(item, dict):
            raise PayloadError("each model must be an object")
        vec = item.get("vector")
        if not isinstance(vec, list) or len(vec) != 128:
            raise PayloadError("each vector must have 128 numbers")
        if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in vec):
            raise PayloadError("vectors must contain numbers only")
        if not all(math.isfinite(x) and abs(x) < 10 for x in vec):
            raise PayloadError("vector values out of range")
        items.append((sanitize_label(item.get("label", "360")), [float(x) for x in vec]))
    return mode, items


def merge(existing: list, items: list[tuple[str, list[float]]], mode: str, now: int) -> list:
    base = [] if mode == "replace" else list(existing)
    next_id = int(base[-1]["id"]) + 1 if base else 0
    for offset, (label, vec) in enumerate(items):
        base.append({"time": now, "label": label, "id": next_id + offset, "data": [vec]})
    return base


def read_payload(path: str) -> object:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(MAX_PAYLOAD_BYTES + 1)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise PayloadError("payload too large")
    return json.loads(data)


def write_models(target: Path, models: list, now: int) -> str | None:
    """Back up the current file, then replace it atomically."""
    backup = None
    uid, gid, mode = 0, 0, 0o644  # Howdy's own default for new model files
    if target.exists():
        st = target.stat()
        uid, gid, mode = st.st_uid, st.st_gid, st.st_mode & 0o777
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        backup = f"{target}.bak-howdy360-{stamp}"
        with open(target, "rb") as src, open(backup, "xb") as dst:
            dst.write(src.read())
        os.chmod(backup, mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".howdy360-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(models, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        if os.geteuid() == 0:
            os.chown(tmp, uid, gid)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return backup


def main(argv: list[str], env: dict[str, str], target_dir: Path | None = None) -> dict:
    if len(argv) != 1:
        raise PayloadError("usage: privileged_save.py PAYLOAD.json")
    user = invoking_user(env)
    mode, items = validate(read_payload(argv[0]))
    directory = target_dir if target_dir is not None else models_dir()
    target = directory / f"{user}.dat"
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        existing = []
    if not isinstance(existing, list):
        raise PayloadError(f"{target} is not a Howdy model file")
    now = int(time.time())
    merged = merge(existing, items, mode, now)
    backup = write_models(target, merged, now)
    return {"ok": True, "file": str(target), "backup": backup, "added": len(items), "total": len(merged)}


if __name__ == "__main__":
    try:
        result = main(sys.argv[1:], dict(os.environ))
    except (PayloadError, PermissionError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)
    print(json.dumps(result))
