# Security Policy

Howdy 360 reads biometric data (face encodings) and can write Howdy's model
file as root through `pkexec`. The guarantees below are part of its core
functionality, and a way around any of them is treated as a vulnerability:

- `doctor` and `measure` never write outside `~/.local/state/howdy-360/`.
- The root helper (`privileged_save.py`) only writes the model file of the
  user identified by polkit (`PKEXEC_UID`), resolves the models directory from
  Howdy's own `paths.py`, validates every value, refuses symlinked or oversized
  payloads, and backs up before replacing the file atomically.
- No camera images are stored.

Out of scope: weaknesses of face recognition itself (for example spoofing with
a photo), which belong to Howdy and to the hardware; see the README.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's
[private vulnerability reporting](https://github.com/diogocouto18/howdy-360/security/advisories/new)
with steps to reproduce and the version or commit you tested.

You can expect an acknowledgement within a week. Fixes are released as soon as
practical, and reporters are credited unless they prefer otherwise.

## Supported versions

Only the latest release on `master` receives security fixes.
