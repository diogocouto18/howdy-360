"""Command line entry point: howdy360 doctor | enroll | measure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .howdy import current_user, discover_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="howdy360",
        description="Diagnose Howdy, enroll your face from many angles, and measure the result.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config-dir", help="Howdy config directory (default: from Howdy's paths.py)")
    parser.add_argument("--models-dir", help="Howdy face models directory (default: from Howdy's paths.py)")
    parser.add_argument("--dlib-data-dir", help="dlib data directory (default: from Howdy's paths.py)")
    parser.add_argument("--user", help="user whose models to read (default: you)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="explain why Howdy is slow or failing (read-only)")
    doctor.add_argument("--seconds", type=float, default=10.0, help="camera test length (default: 10)")
    doctor.add_argument("--no-camera", action="store_true", help="skip the camera and recognition tests")

    enroll = sub.add_parser("enroll", help="guided multi-angle enrollment window")
    enroll.add_argument("--dry-run", action="store_true", help="never save; keep candidates for `measure`")
    enroll.add_argument("--replace", action="store_true", help="replace existing models (asks first, backs up)")

    measure = sub.add_parser("measure", help="measure recognition and compare with last time (read-only)")
    measure.add_argument("--seconds", type=float, default=10.0, help="measurement length (default: 10)")
    measure.add_argument("--candidates", type=Path, help="also compare against a candidates .npy file")
    measure.add_argument("--no-history", action="store_true", help="do not read or write the history file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = discover_paths(
        {"config_dir": args.config_dir, "user_models_dir": args.models_dir, "dlib_data_dir": args.dlib_data_dir}
    )
    user = args.user or current_user()

    if args.command == "doctor":
        from . import doctor

        report = doctor.run(paths, user, seconds=args.seconds, camera=not args.no_camera)
        print(doctor.render(report))
        return 1 if report.failed else 0
    if args.command == "measure":
        from . import measure

        print(measure.run(paths, user, args.seconds, args.candidates, history=not args.no_history))
        return 0
    if args.command == "enroll":
        from . import enroll_app

        return enroll_app.run(paths, user, dry_run=args.dry_run, replace=args.replace)
    return 2


if __name__ == "__main__":
    sys.exit(main())
