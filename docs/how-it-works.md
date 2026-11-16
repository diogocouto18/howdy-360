# How it works

## Fidelity to Howdy

Every frame goes through the same steps as Howdy 3's `compare.py`:

1. The frame is read from `video.device_path` with OpenCV (V4L2).
2. A greyscale copy is equalised with CLAHE (clip limit 2.0, 8×8 tiles).
3. Darkness is the share of pixels in the first of 8 histogram bins. Frames
   above `video.dark_threshold` are skipped (on IR cameras the emitter often
   strobes, so every other frame is dark; that is normal).
4. Both images are scaled to `video.max_height`.
5. Faces are **detected on the CLAHE image**; the 5 landmarks and the 128-number
   descriptor are **computed on the original frame**.
6. Distance = Euclidean distance × 10, compared with `video.certainty`.

Getting step 5 wrong (describing the CLAHE image) produces numbers that do not
match what Howdy sees at login, so tests and measurements always use this
exact pipeline.

## Head pose

dlib's 5-point model gives the two corners of each eye and the base of the
nose. With the eye centres `L` and `R` (left/right on the image) and nose `N`:

- yaw = `(N.x − L.x) / (R.x − L.x) − 0.5`, mirrored so positive means the
  user's own right (the preview is a mirror);
- pitch = `(N.y − mean(L.y, R.y)) / (R.x − L.x)`.

Both are taken relative to the median of 8 calibration frames looking straight
ahead, which removes the effect of an off-centre camera.

## Sectors and adaptive thresholds

The relative pose is divided by a per-side threshold (0.10 for left/right,
0.07 for up/down), giving a point on the ring where 1.0 is the ring itself.
Beyond 1.0 the point belongs to one of 8 sectors of 45°.

dlib's default face detector stops finding faces turned more than about
30–40°, and an off-centre camera makes one side reach less than the other. If
after 15 s a side's maximum reach is still short of its threshold, the
threshold becomes 90 % of that reach, but never less than 0.05. The final
thresholds and reaches are printed in the JSON report.

## Choosing frames

- Frames whose sharpness (variance of the Laplacian on the face) is below 25 %
  of the calibration median are skipped as blurry.
- Each sector keeps its 2 sharpest frames, the front keeps 3, and a frame is
  only kept if its encoding differs from the ones already kept (distance > 1.5
  in Howdy units), so near-duplicates are not stored.
- At most 19 new models are added. Howdy notes that each model slows
  recognition "slightly": comparing against 20 vectors is a tiny cost next to
  the ~37 ms detection per frame.

## Saving

The window writes the selected encodings to a mode-600 file in
`$XDG_RUNTIME_DIR` and runs `pkexec /usr/bin/python3 -I privileged_save.py
<file>`. The helper follows the rules in [SECURITY.md](../SECURITY.md) and
prints a JSON result (file, backup, number added). The temporary file is
removed afterwards.
