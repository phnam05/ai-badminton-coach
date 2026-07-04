# Feature Spec

The contract both evaluation engines depend on. Everything here is **view-invariant and
body-scale-normalized** — no raw pixel coordinates escape this layer.

Assumptions: single player in frame, right-handed (racket arm = right). All mirrored for
left-handed via a config flag.

## Normalization
- **Scale unit = torso length** = distance(mid-shoulder, mid-hip). Any distance metric is
  divided by this, so it's invariant to camera distance / player size.
- **Angles** are naturally scale- and translation-invariant; preferred wherever possible.
- Coordinates are only used internally to derive angles/ratios, never compared directly.
- **Canonical right-handed frame.** Handedness is resolved ONCE here: every clip — user
  AND reference — is normalized into a canonical right-handed representation (left-handed
  inputs are horizontally mirrored). Downstream code never handles handedness again.
- **Body-local facing.** "In front of / behind" is defined along the direction the player
  faces (from hip/foot orientation), detected per clip — NOT relative to the net/court.
  This makes frontness invariant to which way the player faces in the image.

## Per-frame features — SIDE VIEW (primary)
Computed every frame; produces a time-series per feature.

| Feature | Definition | What it captures |
|---|---|---|
| `right_elbow_angle` | angle(shoulder, elbow, wrist) | arm extension (smash wants near-straight at contact) |
| `right_arm_elevation` | angle of upper arm vs trunk axis | how high the arm is raised |
| `trunk_incline` | angle(shoulder-hip line, vertical) | arch back / lean / forward flex |
| `right_wrist_height` | (wrist.y - shoulder.y) / torso_len | contact height relative to body |
| `right_wrist_frontness` | wrist offset from shoulder along the body-facing axis / torso_len | contact in front vs behind (body-relative; see Normalization) |
| `left_arm_elevation` | angle of non-racket upper arm vs trunk | non-racket arm raise (tracking/balance) |
| `right_knee_angle` | angle(hip, knee, ankle) | leg loading |
| `right_wrist_speed` | frame-to-frame wrist displacement (smoothed) | used for event detection |
| `wrist_cut_path` | shape of the wrist trajectory across the contact window (lateral curvature vs straight-down drive) | slice brushing proxy (S2) — weak, pose can't see the racket face |

## Events (side view)
Detected from the time-series, anchor the at-contact metrics.
- **Backswing apex** — racket wrist highest / most retracted before the forward swing.
- **Contact WINDOW (not a single frame)** — a small window around the `right_wrist_speed`
  peak near the top of the arc. 2D speed peak does not reliably coincide with true contact
  (projection hides toward/away motion), and at 30fps a single frame is fragile — one wrong
  frame corrupts every scored metric. Each at-contact metric defines how it aggregates over
  the window (e.g. contact height uses the max-reach frame within the window).
- **Follow-through end** — wrist speed decays after contact.
- **Detection confidence + fallback.** If the speed peak is flat/ambiguous or the window is
  ill-defined, the tool reports LOW CONFIDENCE and asks the user to re-film/re-trim rather
  than emitting a wrong answer. Never silently score an unreliable contact.

## At-contact metrics (side view) — the scored quantities
Derived at (or across a window around) the contact frame.

| Metric | From | Good-form intuition (target comes from reference/principles, not invented here) |
|---|---|---|
| Contact height (gated) | `right_wrist_height` at contact, **gated by frontness (P6) + extension (P7)** | height is NON-monotonic: height earned by reaching *behind* the head is a fault. Only reward height when contact is also in front and the arm is extended. |
| Elbow extension at contact | `right_elbow_angle` at contact | near-full extension for smash |
| Contact frontness | `right_wrist_frontness` at contact | contact ahead of the body |
| Trunk arch→flex swing | `trunk_incline` backswing→follow-through | bow-and-arrow whip |
| Non-racket arm pull | `left_arm_elevation` backswing→contact delta | raised then pulled down |
| ~~Kinetic-chain timing~~ | — | **DEFERRED from v1 scoring.** At 30fps the segmental peaks fall within 2–3 frames and can't be resolved; reporting them is noise. Revisit only at measured ≥60fps, and even then start with a coarse binary (lower body initiates before arm?), not a 6-link ordering. |

## Per-frame features — BACK VIEW (added later)
Independent set; compared only against back-view references.

| Feature | Definition | What it captures |
|---|---|---|
| `shoulder_tilt` | left/right shoulder height difference / torso_len | shoulder rotation proxy |
| `hip_tilt` | left/right hip height difference / torso_len | hip rotation proxy |
| `shoulder_hip_separation` | shoulder-line angle − hip-line angle | hip-lead separation |
| `wrist_overhead_deviation` | overhead wrist-path deviation from the body midline | over-the-top vs slung around the side (P11). Distinct from the side-view `wrist_cut_path` — different plane, different meaning. |
| `stance_width` | ankle-to-ankle distance / torso_len | base / footwork |

## Notes
- The **racket face angle** (what most distinguishes slice from smash) is NOT reliably
  recoverable from body pose. It is intentionally absent. Slice evaluation leans on
  preparation similarity (deception), contact height, and wrist-path brushing shape.
- All "good-form intuition" columns are directional hints only. Actual thresholds come
  from reference footage (DTW-aligned trajectories) or a cited coaching principle, per
  CLAUDE.md's ground-truth rule.
