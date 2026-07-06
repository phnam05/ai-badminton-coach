# Badminton AI Coach

Analyzes a video of your badminton technique and gives coach-like feedback — starting
with the forehand smash and forehand slice. Built to stand in for a human coach when
practicing alone.

## How it works

**video → per-frame skeleton → view-invariant features → evaluation → coach feedback**

1. **Pose** — MediaPipe Pose turns every frame into a skeleton of body keypoints.
2. **Features** — raw pixel coordinates are never compared. Everything becomes joint
   angles, ratios (e.g. contact height / torso length), and body-relative positions,
   which don't care about camera distance, zoom, or player size.
3. **Evaluation** — two engines, equal partners, reading the same features:
   - **Reference comparison** — the same pipeline runs on a strong player's clip; the
     feature trajectories are aligned to yours with Dynamic Time Warping, and
     meaningful deviations are reported. This supplies *measured* targets.
   - **Coaching principles** — documented, widely-agreed technique principles (BWF /
     established coaching material), each written as measurable geometry. This supplies
     the *meaning*: "contact too low → get it higher and in front."
4. **Report** — annotated video plus a written verdict with the top fixes.

## The rule everything traces to

The user of this tool is not a strong player — so "correct" cannot come from the user,
and it must not come from us. Every evaluation standard traces to measured reference
footage or documented coaching material, never to numbers we made up. The same value
governs uncertainty: when detection is ambiguous, the tool reports low confidence and
asks for a re-film — it never delivers a confident wrong answer.

## Scope (v1)

Side-on camera view first (back view later, as an independent second analysis — no 3D
fusion). Right-handed player by default (configurable). Clips are pre-trimmed to one
shot, and you tell the tool which shot to judge — smash vs slice is not auto-classified,
since the racket-face cut isn't visible in body pose (slice v1 judges setup and reduced
power only). Feedback is directional ("higher, more in front"), not pass/fail grades.

## Status

- **Stage 1 — pose extraction + skeleton overlay: built.** Validated on a synthetic
  pan/zoom clip (100% detection); checkpoint on real footage pending.
- **Stage 2 — swing segmentation: built.** Wrist speed locates the swing; peak arm
  extension pins the suspected hit, anchored to a contact *window* with a confidence
  verdict. Checkpoint (verifying the marked contact) pending.
- **Next:** Stage 3 feature extraction, Stage 4 principle-rule engine, Stage 5 report.
  Reference comparison + DTW lands once a good reference clip is sourced.

## Usage

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python tests\smoke_test.py          # first run: downloads the pose model
.venv\Scripts\python cli.py extract data\raw\clip.mp4   # keypoints + overlay video
.venv\Scripts\python cli.py segment data\raw\clip.mp4   # swing events + contact window
```

Outputs land in `data/output/` (pose JSON, overlay videos, checkpoint chart).

## Layout

```
src/pose/      video -> keypoints + skeleton overlay
src/segment/   swing events: backswing, CONTACT window, follow-through
src/metrics/   (next) normalized features + at-contact metrics
src/evaluate/  (next) principle rules + reference/DTW comparison
src/report/    (next) annotated video + written feedback
docs/          feature spec, filming protocol, coaching principles
data/          input clips and outputs (gitignored)
```
