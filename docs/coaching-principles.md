# Coaching Principles

Curated, widely-agreed technique principles for the forehand overhead, drawn from
established coaching material (BWF-style fundamentals), each written as **measurable
geometry** tied to the features in `feature-spec.md`. These are the *interpretation* layer:
they give a deviation meaning and direction. Thresholds shown as ranges are starting points
to be tuned against real/reference footage, not invented gospel.

Each principle notes whether it's reliably **pose-observable**. Anything gated by the
racket face is flagged as a limitation.

## Forehand smash

### Preparation / backswing
- **P1 — Non-racket arm points up at the shuttle.** Raised for tracking, balance, and to
  pre-load rotation. Signal: `left_arm_elevation` high at backswing apex. *Pose-observable.*
- **P2 — Racket-side elbow leads and stays high.** Signal: `right_arm_elevation` high;
  elbow above shoulder in prep. *Pose-observable.*
- **P3 — Trunk arches back ("bow").** Chest up, slight hyperextension before the forward
  swing. Signal: `trunk_incline` leaning back at backswing. *Pose-observable.*
- **P4 — Loaded base.** Knees bent / weight loaded to drive up into the shot. Signal:
  `right_knee_angle` flexed pre-swing. *Pose-observable (side).*

### Contact
- **P5 — Contact is HIGH.** Hit at close to full reach. Signal: `right_wrist_height` at
  contact near its max for the swing. *Pose-observable — a top-priority smash marker.*
  **Non-monotonic — must be gated by P6/P7:** height gained by reaching *behind* the head
  is a fault, not a virtue. The engine only credits height when contact is also in front
  (P6) and the arm is extended (P7). Never reward raw wrist-height alone.
- **P6 — Contact is IN FRONT of the body.** Not behind the head. Signal:
  `right_wrist_frontness` positive (toward net) at contact. *Pose-observable.*
- **P7 — Arm near full extension at contact.** Reach up through the shuttle. Signal:
  `right_elbow_angle` near straight at contact. *Pose-observable.*
- **P8 — Steep downward racket face.** The defining smash quality. Signal: NOT reliably
  recoverable from body pose. *Limitation — inferred weakly from forearm orientation; do
  not score confidently.*

### Rotation & swing path (best seen from BACK view)
- **P9 — Shoulder-over-shoulder rotation.** Racket shoulder drives from back to front,
  finishing lower. Signal: `shoulder_tilt` reverses through the swing. *Pose-observable
  (back view).*
- **P10 — Hips lead shoulders (separation).** Signal: `shoulder_hip_separation` peaks then
  unwinds. *Pose-observable (back view).*
- **P11 — Swing comes OVER THE TOP, not around the side.** Common power-killing fault.
  Signal: `wrist_overhead_deviation` stays near the body midline overhead. *Pose-observable
  (back view) — invisible side-on.*

### Follow-through
- **P12 — Non-racket arm pulls down and in.** Drives the rotation. Signal:
  `left_arm_elevation` drops sharply backswing→contact. *Pose-observable.*
- **P13 — Racket follows through down and across** to the non-racket side; trunk flexes
  forward. Signal: `trunk_incline` swings from back-arch to forward-flex; wrist path ends
  low across body. *Pose-observable.*

### Whole-motion
- **P14 — Kinetic chain fires proximal-to-distal:** legs → hips → trunk → shoulder → elbow
  → wrist, in sequence (not all at once). **DEFERRED from v1 scoring** — at 30fps the peaks
  fall within 2–3 frames and can't be resolved, so reporting them adds noise, not coaching.
  Revisit only at measured ≥60fps, and even then as a coarse binary (did the lower body
  initiate before the arm?), not a full 6-link ordering.

## Forehand slice

> **v1 SCOPE — say this in the report, not just here.** v1 evaluates the slice's *setup*
> (does it convincingly mimic your smash) and its *reduced power / follow-through*. It does
> **NOT** judge the slicing action itself — the racket-face cut across the shuttle is not
> recoverable from body pose. A user filming a slice must be told in the output that the
> cut itself is unverified, so they don't mistake "setup looks good" for "your slice is good."

Setup should look identical to the smash (deception is the point), so P1–P7, P9–P14 apply
as preparation principles. The differences:

- **S1 — Preparation mimics the smash.** Deception depends on the backswing looking the
  same. Signal: prep features (P1–P4) match the user's own smash prep. *Pose-observable —
  and a genuinely useful thing to coach.*
- **S2 — Contact is a brushing/cutting action across the shuttle**, not a flat drive
  through it (forearm pronation, glancing racket face). Signal: the racket-face cut is NOT
  reliably visible in pose. *Hard limitation — reported to the user as unverified.* Weak
  side-view proxy only: `wrist_cut_path` (wrist trajectory curving *across* at contact vs
  driving straight down). Distinct feature from the back-view `wrist_overhead_deviation`.
- **S3 — Reduced power / shorter, often abbreviated follow-through** vs the smash. Signal:
  lower `right_wrist_speed` at contact and shorter follow-through arc than the user's smash.
  *Pose-observable, relative to the user's own smash.*

## How these are used
- **Directional verdicts, not pass/fail.** v1 thresholds are untuned starting points, so
  the engine reports *direction and relative comparison* ("contact looks low relative to
  your reach") — never an absolute score/grade on provisional numbers. Absolute scoring on
  invented thresholds is the exact failure this project is built to avoid.
- The rule engine checks each pose-observable principle and reports violations with
  direction ("contact too low — reach higher").
- Principles flagged as limitations are reported as unverified/low-confidence or omitted,
  never presented as certain.
- Where a matching reference clip exists, DTW-aligned reference trajectories refine or
  replace the starting-point thresholds here. **Both user and reference are normalized to a
  canonical right-handed frame**, so a left-handed reference clip is mirrored to match a
  right-handed user (and vice versa). Principles still choose *which* channels matter and
  what a deviation *means*.
