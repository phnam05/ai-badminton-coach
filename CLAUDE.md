# Badminton AI Coach

## What this is
A computer-vision tool that analyzes a player's badminton technique from video and
gives coach-like feedback. The end goal of this first stage: tell whether a player's
**forehand smash** and **forehand slice** are executed well, and explain *why*.

The user is not a strong player — the whole point is to replace a human coach when
practicing alone. This has a critical design consequence (see "Ground truth" below):
we must NOT invent "correct" technique numbers ourselves.

## Core approach
Video → per-frame body pose (skeleton) → **view-invariant, body-scale-normalized
features** (joint angles, ratios, relative positions) → evaluation → coach feedback.

Two evaluation engines, treated as EQUAL PARTNERS, both consuming the same feature layer:
1. **Reference comparison** — run the same pipeline on a strong player's clip, extract
   their feature trajectories, align to the user's with **Dynamic Time Warping (DTW)**,
   and report meaningful deviations. This supplies *calibrated targets* measured off
   someone actually good, instead of numbers we made up.
2. **Coaching principles** — a curated set of documented, widely-agreed technique
   principles (BWF / established coaching material), each written as measurable
   geometry. This supplies the *meaning and direction* of a deviation ("contact too
   low → get it higher and in front"), turning raw "you differ" into real coaching.

Reference gives targets; principles interpret them. Neither alone is coaching.

## Ground truth (the key design constraint)
The user cannot be the source of "good." So evaluation standards come from (a) measured
reference footage and (b) documented coaching principles — never from our own guesses at
threshold numbers. Any hardcoded threshold must trace to one of those two sources.

## View invariance (the hard problem, not DTW)
We never compare raw pixel coordinates — camera distance, zoom, body size, and framing
would dominate. Everything is expressed as **joint angles, ratios (e.g. contact height /
torso length), and relative positions**, which are invariant to those. This is why the
feature layer is shared across both engines and both camera views.

Limitation: monocular 2D pose flattens out-of-plane motion. We manage this with a
consistent filming protocol and by choosing metrics robust in each view's plane. True
3D (multi-camera or 3D pose) is a later phase, only if needed.

## Camera views
Two INDEPENDENT single-view analyses — we do NOT fuse them into 3D.
- **Side-on (primary):** contact height, contact in-front/behind, elbow extension,
  trunk arch/lean, swing arc, follow-through. Built first; fully self-sufficient.
- **Back view (added later):** shoulder/hip rotation, swing-path straightness
  (over-the-top vs slung around the side), lateral tilt, base/footwork. Leans more on
  the coaching-principles engine early, since clean behind-the-player references are rare.

A session = one or more clips, each tagged with its view. A single side clip is the
normal, fully-supported case. Back view is a bonus layer when a clip is provided.

## Tech stack
- Python 3.14, `.venv` in repo root (`.venv\Scripts\python.exe`)
- MediaPipe Pose — per-frame skeleton (CPU, single player). Uses the **Tasks API**
  (`mediapipe.tasks.python.vision.PoseLandmarker`, model at
  `models/pose_landmarker_full.task`, auto-downloaded by `tests/smoke_test.py`) —
  NOT the deprecated legacy `mp.solutions.pose`.
- OpenCV — video I/O and overlay rendering
- NumPy / SciPy — angle math, smoothing, peak/event detection, DTW
- Matplotlib (optional) — metric plots in the report

## Architecture / modules
```
src/
  pose/       # video -> keypoints (+ overlay video)
  segment/    # detect backswing / CONTACT / follow-through (wrist-speed peak)
  metrics/    # keypoints -> normalized view-invariant features + at-contact metrics
  evaluate/   # smash & slice engines: (1) coaching-principle rules, (2) reference+DTW
  report/     # annotated video + written verdict and top fixes
  config/     # tunable thresholds and metric target ranges (sourced, not invented)
cli.py        # analyze a clip (or clips) end-to-end
data/raw/     # input clips (gitignored)
data/output/  # annotated videos + reports (gitignored)
docs/         # design artifacts (feature spec, filming protocol, coaching principles)
```

## Build order (locked)
Principles-first, so we validate end-to-end on the user's own clips before we're blocked
on sourcing a clean reference clip.
1. Setup (venv, deps, folders, MediaPipe smoke test)
2. Stage 1 — Pose extraction + overlay video. **Checkpoint: user eyeballs tracking.**
3. Stage 2 — Contact-frame detection. **Checkpoint: verify contact frame is right.**
4. Stage 3 — Normalized feature + at-contact metric extraction (side view)
5. Stage 4a — Coaching-principle rule engine (works with zero reference footage)
6. Stage 5 — Annotated video + text report
7. Stage 4b — Reference comparison + DTW (once a good pro clip is sourced)
8. Tune thresholds against real footage; then add back-view module.

## Decisions locked
- Player is right-handed (racket = right arm); keep configurable.
- Reference footage source: curated YouTube/social clips (vetted per docs/filming-protocol.md).
- Smash vs slice is NOT auto-classified in v1 — the racket-face cut isn't reliably visible
  in body pose. The user tells the tool which shot to evaluate; it scores against that profile.
- Output: annotated video + text feedback.
- Clips are pre-trimmed to roughly one shot.
- **Contact is anchored to a WINDOW, not a single frame**, with a confidence measure and a
  fallback: on ambiguous detection the tool reports low confidence and asks for a re-film,
  never emitting a silently-wrong answer.
- **v1 feedback is directional/relative, not absolute pass/fail** — thresholds are untuned
  starting points, so absolute grades on provisional numbers are forbidden (that's the
  invented-threshold trap this project exists to avoid).
- **Handedness resolved once at the feature layer:** user AND reference clips are normalized
  to a canonical right-handed frame (left-handed inputs mirrored).
- **Frontness ("in front of body") is body-relative**, from detected facing direction — not
  net/court-relative.
- **Slice v1 judges setup + reduced power only; the slicing action itself is unverified**,
  and the report must say so to the user.
- **Kinetic-chain timing (P14) is deferred** — unresolvable at 30fps; revisit at ≥60fps as a
  coarse binary only.

## Status
Setup (build step 1) complete: venv (Python 3.14), deps installed, folder structure,
`tests/smoke_test.py` PASSING (33 landmarks detected on a real-person sample image).
Next: Stage 1 — pose extraction + overlay video, checkpointed on the user's own clip.

## Working notes for Claude
- Match surrounding code style once code exists. Keep modules decoupled around the
  shared feature layer — that layer is the contract; don't leak raw coordinates past it.
- Any threshold committed to `src/config/` must cite its source (reference-derived or a
  specific coaching principle), never a guess.
