# howdy-360 — Specification

Status: v0.1 (first public release)

## 1. Problem

[Howdy](https://github.com/boltgolt/howdy) unlocks Linux with an IR camera, but
many users report it as slow or unreliable. Measured on a real machine
(ASUS ZenBook, Ubuntu 26.04, Howdy 3.0):

- The engine itself is fast: ~0.75 s to load libraries and models, ~0.04 s to
  open the camera, ~75 ms per analysed frame.
- The real cause was enrollment: a single stored face model gave distances of
  4.0–5.5 against a `certainty` of 3.5, so 7 of 10 unlock attempts ran into the
  4 s timeout and fell back to the password.
- Adding models from more head angles fixed it.

`howdy-360` turns that investigation into a tool. It does **not** replace
Howdy, its PAM module or its recognition engine.

## 2. Goals and non-goals

Goals

1. `doctor`: explain why Howdy is slow or failing, with concrete, ordered
   recommendations. Read-only, no root.
2. `enroll`: guided multi-angle enrollment in a native GTK4/libadwaita window,
   saving into Howdy's own model file only after explicit confirmation, with a
   backup.
3. `measure`: measure recognition quality against the stored models and compare
   with the previous measurement. Read-only (except its own history file).

Non-goals

- No PAM module, no daemon, nothing running in the background.
- No true 3D face model: common laptop IR cameras have no depth sensor, so
  "360" means many 2D angles, not depth.
- No anti-spoofing claims beyond what Howdy already provides.
- No change to Howdy's configuration (the tool may *recommend* changes).

## 3. Fidelity to Howdy

All measurements and all stored encodings must match what Howdy computes at
login, otherwise numbers are meaningless. The pipeline mirrors Howdy 3
`compare.py` / `cli/add.py`:

1. Read the frame from `video.device_path` with OpenCV (V4L2).
2. `gsframe = CLAHE(clipLimit=2.0, tile=8x8)(gray(frame))`.
3. Skip frames whose darkness (first of 8 histogram bins, %) exceeds
   `video.dark_threshold`, or is exactly 100.
4. Scale both `frame` and `gsframe` by `video.max_height / frame_height`
   (`INTER_AREA`).
5. Detect faces on `gsframe` with dlib's HOG frontal detector (upsample 1), or
   the CNN detector when `core.use_cnn = true`.
6. Landmarks (5-point) and the 128-d descriptor are computed on the **original
   scaled `frame`**, not on the CLAHE image.
7. Distance = Euclidean norm × 10, compared with `video.certainty`.

Paths are discovered the way Howdy does: by reading `paths.py` of the installed
Howdy (`config_dir`, `dlib_data_dir`, `user_models_dir`), with documented
fallbacks and CLI overrides.

## 4. Commands

### 4.1 `howdy360 doctor`

Runs for ~10 s and prints a report of checks, each `ok`, `warn`, `fail` or
`info`, then numbered recommendations.

| Area | Checks |
|---|---|
| Install | Howdy found; version; config readable; dlib data files present |
| Config | `device_path` exists and is an IR (greyscale) device; `certainty`, `timeout`, `dark_threshold`, `use_cnn`; Howdy disabled flag |
| PAM scope | Which `/etc/pam.d` files reference `pam_howdy` (warn when `sudo`/`common-auth` include it) |
| Camera | Opens; lit vs dark vs black frames; IR emitter strobe detected |
| Speed | Library import, model load, camera open, per-frame analysis time |
| Recognition | Number of stored models; median / best distance; % frames under `certainty`; time to first match |
| History | Last 7 days of `pam_howdy` journal outcomes (approved / timeout / skipped) |
| Privacy | Model file permissions (warn when world-readable) |

Exit code: 0 when no `fail`, 1 otherwise.

### 4.2 `howdy360 enroll`

GTK4 + libadwaita window (follows the system light/dark preference):

1. **Calibrate**: look straight ahead; the neutral pose becomes the reference.
2. **Rotate**: a ring of 8 sectors around a circular live preview fills as each
   head direction is covered; a dot shows the current direction.
   Per-side thresholds adapt (never below a floor) when one side cannot reach
   the ring; hints for "lost face — turn less" and "blurry — slower".
3. **Measure**: 10 s sitting normally; current models vs current + new.
4. **Results**: comparison table; **Save to Howdy** or **Discard**.

Selection: per sector the sharpest distinct frames (distance between kept
encodings > 1.5), plus up to 3 neutral frames; at most 19 new models.

Saving goes through `pkexec` to a small, stdlib-only helper that:

- writes only the invoking user's model file (user derived from `PKEXEC_UID`,
  never from arguments);
- resolves the models directory itself (never from user input);
- validates the input strictly (list of 1–32 vectors of 128 finite floats);
- writes a timestamped backup first, then the new file atomically, keeping the
  original owner and mode;
- labels models `360-<sector>-<n>` (≤ 24 chars, no commas, as Howdy requires).

Flags: `--dry-run` (never saves), `--replace` (replace existing models; asks
for confirmation in the UI).

### 4.3 `howdy360 measure`

10 s measurement while sitting normally. Prints model count, median and best
distance, recognised frames, time to first match, and the model that matched
most / never matched. Appends numbers only (no encodings, no images) to
`$XDG_STATE_HOME/howdy-360/history.jsonl` and compares with the previous entry.
`--candidates FILE` compares against current + an `.npy` of candidate encodings
(used to validate an enrollment in a later session).

## 5. Security and privacy

- No images are stored. Encodings exist in memory, in Howdy's model file (only
  after Save), and in an optional candidates file written with mode 600.
- `doctor` and `measure` never write outside the user's state directory.
- Root is used only for Save, through polkit (`pkexec`), with the checks above.
- Face recognition is weaker than a password; the README says so and `doctor`
  warns when Howdy is enabled for `sudo`.

## 6. Architecture

```
src/howdy360/
  cli.py            argparse entry point (doctor | enroll | measure)
  i18n.py           English UI strings with Portuguese translation (LANG=pt*)
  howdy.py          locate install, read config, load models, PAM scope
  vision.py         Howdy-faithful camera + dlib pipeline (lazy imports)
  pose.py           head pose, sectors, adaptive thresholds, selection (pure)
  metrics.py        distances, summaries, history file (pure + small I/O)
  journal.py        parse pam_howdy journal lines (pure) + runner
  doctor.py         checks and recommendations
  measure.py        measurement command
  enroll_app.py     GTK4/libadwaita UI + capture engine thread
  render.py         circular preview + ring compositing (OpenCV → GdkTexture)
  privileged_save.py  stdlib-only helper run through pkexec
```

Heavy dependencies (`cv2`, `dlib`, `gi`) are the system packages Howdy already
needs; they are imported lazily so the pure modules are testable anywhere.

## 7. Quality

- `ruff check` + `ruff format --check`.
- `pytest` with coverage gate (≥ 85 %) on the pure modules: `pose`, `metrics`,
  `journal`, `howdy`, `privileged_save`, `i18n`, `doctor` rendering.
- CI on GitHub Actions: lint + tests on Python 3.12 and 3.13, secret scanning
  (gitleaks).
- Hardware paths (`vision`, `enroll_app`, `render`) are verified manually on a
  real IR camera; the README states which hardware was tested.

## 8. Known limitations

- The HOG detector loses faces turned beyond ~30–40°; the CNN detector is too
  slow on CPU (measured 1.2–4.6 s per frame vs 37 ms), so it is not used for
  live enrollment.
- Results depend on lighting; validate with `measure --candidates` in a later
  session.
- Tested on one machine (ASUS ZenBook UX425UA, Quanta IR camera, Howdy 3.0,
  Ubuntu 26.04).
