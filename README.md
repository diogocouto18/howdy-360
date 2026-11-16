# Howdy 360

Diagnose [Howdy](https://github.com/boltgolt/howdy) face unlock, enroll your face
from many angles in a native GNOME window, and measure whether it actually got
better.

> **Unofficial.** Howdy 360 is a companion tool for Howdy. It is not affiliated
> with Howdy or its authors, and it does not replace Howdy's PAM module or its
> recognition engine.

## The problem

Howdy is often described as "slow" or "doesn't always recognise me". On the
machine this tool was built on, the engine itself was fast: about 0.75 s to load
its libraries and models, 0.04 s to open the camera and 75 ms per analysed
frame. The real cause was enrollment. One stored face model gave distances of
4.0–5.5 against a `certainty` of 3.5, so 7 of 10 unlock attempts ran into the
4-second timeout and fell back to the password.

Howdy 360 finds this kind of problem (`doctor`), fixes it by storing your face
from many head angles (`enroll`), and proves the result with numbers
(`measure`).

## How it works

Howdy does not keep pictures of your face. For each stored model it keeps a list
of **128 numbers** that a neural network (dlib) computes from an IR image. At
login, Howdy computes the same 128 numbers from the camera and measures the
distance to every stored model; if the closest one is below `certainty`, you are
in.

A single model only matches you well from roughly the angle and lighting it was
taken in. Howdy 360 guides you through a slow head rotation and keeps the
sharpest, most distinct frames from 8 directions plus the front, so there is
always a stored model close to how you are actually sitting.

It is **not** a 3D scan like Face ID: typical laptop IR cameras have no depth
sensor. "360" means many 2D angles. See [docs/how-it-works.md](docs/how-it-works.md)
for the details (head pose, sector coverage, frame selection).

Every number Howdy 360 shows is computed exactly the way Howdy computes it at
login (same CLAHE-based detection, same descriptor on the original frame, same
dark-frame filtering, same distance scale), read from your Howdy config.

## Requirements

- Linux with Howdy 3 installed and working (IR camera configured)
- The Python packages Howdy already uses: `python3-opencv`, `python3-dlib`
  (or `dlib` via Howdy's installer), `python3-numpy`
- For `enroll`: GTK 4 and libadwaita with PyGObject (`python3-gi`,
  `gir1.2-gtk-4.0`, `gir1.2-adw-1`) and `pkexec` (polkit) to save
- Tested on: ASUS ZenBook UX425UA (Quanta IR camera), Ubuntu 26.04, Howdy 3.0

## Installation

Howdy 360 reuses your system's OpenCV, dlib and GTK instead of downloading its
own copies, so install it with access to system packages:

```bash
pipx install --system-site-packages git+https://github.com/diogocouto18/howdy-360
```

Or run it from a clone:

```bash
git clone https://github.com/diogocouto18/howdy-360
cd howdy-360
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/howdy360 doctor
```

The user interface is in English, and in Portuguese when your `LANG` starts
with `pt`.

## Usage

The usual flow is `doctor` → `enroll` → `measure`.

### `howdy360 doctor`: what is wrong?

Read-only, no root. Sit in front of the laptop and look at the screen for about
10 seconds. It checks your installation, configuration, PAM scope, IR camera,
speed, recognition, the last 7 days of unlock attempts and the privacy of your
model file, then lists what to do, in order. Example from the machine this tool
was built on, while it still had a single stored model:

```
Howdy 360 - diagnosis
────────────────────────────────────────────────────────────
Configuration
  ✓ camera /dev/v4l/by-path/pci-0000:04:00.3-usb-0:3:1.2-video-index0
  • certainty = 3.5   timeout = 4.0 s   dark_threshold = 60.0   use_cnn = False

PAM scope
  ✓ Enabled for: gdm-password

IR camera
  ✓ Camera opened, 61 lit / 12 dark / 0 black frames
  ✓ IR emitter strobing detected (normal)

Speed
  ✓ Startup: libraries 0.23 s + models 0.38 s
  ✓ Open camera: 0.14 s
  ✓ Analysis per frame: 37 ms

Recognition
  ! Stored models: 1
  ✗ Distance to your face: median 4.5, best 4.0 (limit 3.5)
  ✗ Recognised frames: 0/35 (0%)
  ✗ No match during the test

History (last 7 days)
  ✗ 3 approved, 7 timeouts, 0 other failures, 0 skipped

Recommendations
  1. Only 1 stored model(s). Run:  howdy360 enroll
  2. Then measure again:  howdy360 measure
────────────────────────────────────────────────────────────
```

`✓` fine · `!` worth a look · `✗` a problem · `•` information.
The exit code is 1 when there is at least one `✗`.

| Flag | Meaning |
|---|---|
| `--seconds N` | length of the camera test (default 10) |
| `--no-camera` | skip camera, speed and recognition checks |

### `howdy360 enroll`: register your face from many angles

Opens a GTK4/libadwaita window that follows your light/dark theme:

1. **Calibrate**: look straight ahead; your normal position becomes the
   reference.
2. **Rotate**: slowly turn your head in a circle. The camera preview sits in a
   circle with a ring of 8 sectors around it; each sector turns green when that
   direction is covered, and a blue dot shows where you are facing. Hints tell
   you to turn less when your face is lost, or more slowly when frames are
   blurry. If one side cannot reach the ring (for example because the camera
   sits off-centre), its threshold adapts after 15 seconds, down to a minimum.
3. **Measure**: sit normally for 10 seconds; your current models are compared
   with current + new.
4. **Results**: a before/after table, then **Save to Howdy** or **Discard**.

Saving asks for your password through the standard GNOME dialog (polkit) and
writes a timestamped backup of your model file before changing it. New models
are labelled `360-front-1`, `360-left-1`, `360-up-right-2`… so `howdy list`
shows where each came from.

| Flag | Meaning |
|---|---|
| `--dry-run` | never save; keep the candidates (mode 600) for `measure --candidates` |
| `--replace` | replace your existing models instead of adding to them (asks first, backs up) |

### `howdy360 measure`: did it get better?

Read-only. Sit normally for 10 seconds. Example output (illustrative numbers):

```
Howdy 360 - measurement
────────────────────────────────────────────────────────────
Stored models: 20

  Median distance         2.32
  Best distance           1.81
  Recognised frames       67/67 (100%)
  First match             0.24 s
  Never the closest model my-face

Compared with previous measurement (2026-11-16 11:20)
  Median distance         3.24 → 2.32   better
  First match             0.54 → 0.24   better
  Recognised frames       82.0 → 100.0  better
────────────────────────────────────────────────────────────
```

Each run appends the numbers (never encodings or images) to
`~/.local/state/howdy-360/history.jsonl`, so you can compare across days,
lighting and positions. "Never the closest model" points at stored models that
do not help.

| Flag | Meaning |
|---|---|
| `--seconds N` | measurement length (default 10) |
| `--candidates FILE` | also compare against candidates from `enroll --dry-run` |
| `--no-history` | do not read or write the history file |

Global options for unusual installs: `--config-dir`, `--models-dir`,
`--dlib-data-dir`, `--user`. By default all paths come from your Howdy
installation's own `paths.py`.

## Security and privacy

- **No images are stored.** Only the 128-number encodings, and only in Howdy's
  model file after you press Save (or in a mode-600 candidates file with
  `--dry-run`).
- `doctor` and `measure` never change anything outside
  `~/.local/state/howdy-360/`.
- Root is used only to save, through `pkexec`, by a small standard-library
  helper that:
  - only writes the model file of the user who ran it (taken from polkit, not
    from arguments);
  - finds Howdy's model directory itself;
  - validates every value;
  - backs up first and replaces the file atomically, keeping its owner and
    permissions.
- **Face unlock is weaker than a password.** Keep Howdy on the lock/login
  screen; `doctor` warns when it is enabled for `sudo` or system-wide
  authentication.
- Howdy's model file is usually readable by every local user (mode 644).
  `doctor` reports it; the encodings cannot be turned back into a photo, but
  they are biometric data.

Please report vulnerabilities privately (see [SECURITY.md](SECURITY.md)).

## FAQ

**Why not use dlib's CNN face detector? It handles turned faces better.**
On this laptop's CPU it took 1.2–4.6 s per frame against 37 ms for the default
detector (7 samples each, first discarded). That is slower than Howdy's 4 s
timeout per frame, and far too slow for a live preview.

**Does it work without an IR camera?**
Howdy can be configured with a normal webcam, and Howdy 360 will follow your
config, but recognition is much less reliable without IR and `doctor` will warn
that the frames look like colour frames.

**How do I undo an enrollment?**
Restore the backup printed after saving
(`<models dir>/<user>.dat.bak-howdy360-<date>`), or remove individual models
with `sudo howdy remove <id>`.

**Does it change my Howdy configuration?**
No. It only recommends changes.

## Development

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . pytest pytest-cov ruff
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest --cov
```

The camera-dependent code is tested with a fake pipeline; the window and the
real camera path are checked by hand on real hardware.

## License

[MIT](LICENSE) © 2026 Diogo Couto
