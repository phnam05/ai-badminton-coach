# Filming Protocol

Comparison quality is capped by footage quality. This covers both filming yourself and
vetting reference clips. The two must be comparable: **a metric can only be compared to a
reference if both clips are filmed from the plane that reveals it.**

## Filming yourself
- **Side-on (primary):** camera perpendicular to your swing, at roughly shoulder/net-post
  height, ~4–6 m away, filming your racket-arm side. The whole swing (raised arm to
  follow-through) must stay in frame — don't crop the racket at the top.
- **Back view (optional, when using a 2nd phone):** camera directly behind you, facing the
  net, on the swing's center line. Feet to raised racket in frame.
- Steady camera (tripod / propped). No panning or zooming during the shot.
- Good, even lighting; uncluttered background; you are the only person in frame.
- **Trim to roughly one shot** per clip (backswing → follow-through, plus a little margin).
- Highest frame rate you can (60 fps ideal) — the swing is fast; more frames = cleaner
  contact detection.
- Two-camera capture doesn't need to be synced — views are analyzed independently.

## Vetting reference clips (YouTube / social)
A clip is usable as reference only if:
- **Single clear shot** of the target stroke (forehand smash or forehand slice), trimmable.
- **Minimal camera movement** — no pan/zoom/cut across the stroke. This rules out most
  broadcast match footage.
- **Angle matches a view bucket** — either clean side-on or clean behind-the-player. Tag it.
- Player fully in frame through the whole swing; single subject; decent resolution.
- Any handedness is fine — a left-handed reference is mirrored into the canonical
  right-handed frame at the feature layer (see feature-spec.md). Just tag it correctly.
- MediaPipe tracks it cleanly (we verify with the overlay before trusting it).

Store references bucketed by `shot × view` (e.g. `smash/side/`, `slice/side/`). The
back-view buckets will likely be sparse at first — expected; back view leans on the
coaching-principles engine until good behind-view references turn up.

## Reality check
Side-on demo tutorials are common and usually clean. Behind-the-player reference clips are
rarer, and no raw footage will be perfect. We start with side view precisely because good
references and robust 2D metrics are most available there.
