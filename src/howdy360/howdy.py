"""Locate a Howdy installation, read its configuration and its face models."""

from __future__ import annotations

import configparser
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

PATHS_PY_CANDIDATES = (
    "/usr/lib/howdy/paths.py",
    "/usr/lib64/howdy/paths.py",
    "/usr/local/lib/howdy/paths.py",
    "/lib/security/howdy/paths.py",
    "/usr/lib/security/howdy/paths.py",
)
FALLBACK_PATHS = {
    "config_dir": "/etc/howdy",
    "dlib_data_dir": "/usr/share/dlib",
    "user_models_dir": "/etc/howdy/models",
}
DLIB_FILES = (
    "shape_predictor_5_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
    "mmod_human_face_detector.dat",
)
LABEL_MAX = 24  # Howdy truncates labels to 24 characters
_ASSIGN = re.compile(r"^(\w+)\s*=\s*PurePath\(\s*[\"']([^\"']+)[\"']\s*\)", re.MULTILINE)


@dataclass(frozen=True)
class HowdyPaths:
    config_dir: Path
    dlib_data_dir: Path
    user_models_dir: Path
    source: str  # where the paths came from (a paths.py file or "fallback")

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.ini"

    def model_file(self, user: str) -> Path:
        return self.user_models_dir / f"{user}.dat"

    def dlib_file(self, name: str) -> Path:
        return self.dlib_data_dir / name


def parse_paths_py(text: str) -> dict[str, str]:
    """Extract `name = PurePath("...")` assignments from Howdy's paths.py."""
    return {m.group(1): m.group(2) for m in _ASSIGN.finditer(text)}


def discover_paths(
    overrides: dict[str, str | None] | None = None,
    candidates: tuple[str, ...] = PATHS_PY_CANDIDATES,
) -> HowdyPaths:
    """Find Howdy's directories the same way Howdy does (its own paths.py)."""
    values = dict(FALLBACK_PATHS)
    source = "fallback"
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            parsed = parse_paths_py(path.read_text(encoding="utf-8", errors="replace"))
            if any(k in parsed for k in FALLBACK_PATHS):
                values.update({k: v for k, v in parsed.items() if k in FALLBACK_PATHS})
                source = str(path)
                break
    applied = [key for key, value in (overrides or {}).items() if value and key in FALLBACK_PATHS]
    for key in applied:
        values[key] = overrides[key]  # type: ignore[index]
    if applied:
        source = f"{source} + overrides"
    return HowdyPaths(
        config_dir=Path(values["config_dir"]),
        dlib_data_dir=Path(values["dlib_data_dir"]),
        user_models_dir=Path(values["user_models_dir"]),
        source=source,
    )


@dataclass(frozen=True)
class HowdyConfig:
    device_path: str | None
    certainty: float = 3.5
    timeout: float = 4.0
    dark_threshold: float = 60.0
    max_height: float = 320.0
    use_cnn: bool = False
    disabled: bool = False
    extra: dict[str, str] = field(default_factory=dict)


def read_config(path: Path) -> HowdyConfig:
    """Read config.ini with the same fallbacks Howdy uses in compare.py."""
    cp = configparser.ConfigParser(interpolation=None)
    if not cp.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)
    device = cp.get("video", "device_path", fallback="").strip() or None
    if device and device.lower() == "none":
        device = None
    return HowdyConfig(
        device_path=device,
        certainty=cp.getfloat("video", "certainty", fallback=3.5),
        timeout=cp.getfloat("video", "timeout", fallback=4.0),
        dark_threshold=cp.getfloat("video", "dark_threshold", fallback=60.0),
        max_height=cp.getfloat("video", "max_height", fallback=320.0),
        use_cnn=cp.getboolean("core", "use_cnn", fallback=False),
        disabled=cp.getboolean("core", "disabled", fallback=False),
        extra={
            "abort_if_ssh": cp.get("core", "abort_if_ssh", fallback=""),
            "abort_if_lid_closed": cp.get("core", "abort_if_lid_closed", fallback=""),
        },
    )


def load_models(path: Path) -> list[dict]:
    """Load a Howdy model file; a missing file means no models."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def encodings(models: list[dict]) -> list[list[float]]:
    """Flatten every stored 128-d vector, in file order."""
    return [vec for model in models for vec in model.get("data", [])]


def model_labels(models: list[dict]) -> list[str]:
    """One label per vector (a model may hold more than one vector)."""
    return [str(model.get("label", "?")) for model in models for _ in model.get("data", [])]


def sanitize_label(label: str) -> str:
    """Apply Howdy's label rules: no commas, at most 24 characters."""
    return label.replace(",", "").strip()[:LABEL_MAX]


def next_id(models: list[dict]) -> int:
    return int(models[-1]["id"]) + 1 if models else 0


def file_mode(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "mode": oct(stat.S_IMODE(st.st_mode)),
        "world_readable": bool(st.st_mode & stat.S_IROTH),
        "owner_uid": st.st_uid,
    }


def pam_howdy_files(pam_dir: Path = Path("/etc/pam.d")) -> list[str]:
    """PAM service files that load pam_howdy on a non-comment line."""
    hits = []
    try:
        entries = sorted(pam_dir.iterdir())
    except OSError:
        return hits
    for entry in entries:
        if not entry.is_file():
            continue
        try:
            lines = entry.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if any("pam_howdy" in ln and not ln.lstrip().startswith("#") for ln in lines):
            hits.append(entry.name)
    return hits


def current_user() -> str:
    import pwd

    return pwd.getpwuid(os.getuid()).pw_name
