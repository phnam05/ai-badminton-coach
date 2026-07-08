"""Badminton AI Coach — command-line entry point.

Stage 1 exposes `extract`: run pose estimation on a clip and produce
  - <name>.pose.json   (per-frame keypoints, consumed by later stages)
  - <name>.overlay.mp4 (skeleton overlay — eyeball this to verify tracking)

Usage:
   extract data\\raw\\my_smash.mp4
"""

import argparse
import sys
from pathlib import Path

from src.metrics import compute_features, measure_contact, render_feature_sheet
from src.pose import PoseExtractor, render_overlay
from src.pose.extractor import PoseSequence
from src.segment import detect_events, render_speed_plot, review_clip

OUTPUT_DIR = Path("data/output")


def _load_or_extract(path: Path, out_dir: Path) -> PoseSequence:
    """Accept a video or a .pose.json; reuse cached keypoints when present."""
    if path.suffix == ".json":
        return PoseSequence.load(path)
    cached = out_dir / f"{path.stem}.pose.json"
    if cached.exists():
        print(f"reusing cached keypoints: {cached}")
        return PoseSequence.load(cached)
    print(f"extracting pose from {path} ...")
    seq = PoseExtractor().extract(path)
    seq.save(cached)
    return seq


def cmd_extract(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.exists():
        print(f"error: video not found: {video}")
        return 1

    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    stem = video.stem

    print(f"extracting pose from {video} ...")
    seq = PoseExtractor().extract(video)

    n = len(seq.frames)
    rate = seq.detection_rate
    print(f"  {n} frames @ {seq.fps:.1f} fps, {seq.width}x{seq.height}")
    print(f"  detection rate: {rate:.0%}")
    if n == 0:
        print("error: no frames read from video")
        return 1
    if rate < 0.9:
        print("  WARNING: pose lost in >10% of frames — tracking may be unreliable.")
        print("  Check lighting, framing, and that you are the only person in frame.")

    pose_path = out_dir / f"{stem}.pose.json"
    seq.save(pose_path)
    print(f"  keypoints -> {pose_path}")

    overlay_path = out_dir / f"{stem}.overlay.mp4"
    render_overlay(video, seq, overlay_path)
    print(f"  overlay   -> {overlay_path}")
    print("Open the overlay video and check the skeleton sticks to the player through the whole swing.")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"error: not found: {path}")
        return 1
    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    stem = path.stem.removesuffix(".pose")

    seq = _load_or_extract(path, out_dir)
    events = detect_events(seq, side=args.hand)
    print(events.summary())

    plot_path = render_speed_plot(events, out_dir / f"{stem}.speed.png", title=stem)
    print(f"  speed plot -> {plot_path}")

    # Annotated overlay needs the source video; skip gracefully without it.
    video = path if path.suffix != ".json" else Path(seq.source)
    if video.exists() and video.suffix != ".json":
        events_overlay = render_overlay(video, seq, out_dir / f"{stem}.events.mp4", events=events)
        print(f"  events overlay -> {events_overlay}")
    else:
        print(f"  (source video not found at '{seq.source}' — skipping events overlay)")

    if events.confidence == "low":
        print("\nLOW CONFIDENCE — do not trust metrics from this clip; see reasons above.")
    else:
        print("\nCheck the plot and events overlay: is the marked contact window the actual hit?")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"error: not found: {path}")
        return 1
    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    stem = path.stem.removesuffix(".pose")

    seq = _load_or_extract(path, out_dir)
    # The viewer shows real frames, so it needs the source video, not just keypoints.
    video = path if path.suffix != ".json" else Path(seq.source)
    if not video.exists() or video.suffix == ".json":
        print(f"error: source video not found at '{seq.source}' — review needs the actual clip")
        return 1

    events = detect_events(seq, side=args.hand)
    print(events.summary())
    print("\nOpening viewer — step with arrow keys, press g on the true hit frame, q to quit.")

    gt_path = out_dir / f"{stem}.groundtruth.json"
    truth = review_clip(video, seq, events, gt_path)
    if truth is not None:
        off = events.contact_frame - truth
        print(f"ground truth: frame {truth} (detector said {events.contact_frame}, "
              f"off by {off:+d}) -> {gt_path}")
    else:
        print("no ground truth marked.")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"error: not found: {path}")
        return 1
    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    stem = path.stem.removesuffix(".pose")

    seq = _load_or_extract(path, out_dir)
    events = detect_events(seq, side=args.hand)
    print(events.summary())
    print()

    features = compute_features(seq, handedness=args.hand)
    contact = measure_contact(features, events)
    print(contact.summary())

    features_path = out_dir / f"{stem}.features.json"
    features.save(features_path)
    print(f"\n  features   -> {features_path}")

    contact_path = out_dir / f"{stem}.contact.json"
    contact.save(contact_path)
    print(f"  at-contact -> {contact_path}")

    sheet_path = render_feature_sheet(features, events, contact, out_dir / f"{stem}.features.png", title=stem)
    print(f"  plot sheet -> {sheet_path}")

    if contact.confidence == "low":
        print("\nLOW CONFIDENCE — do not trust these numbers; see reasons above.")
    else:
        print("\nCheck the plot sheet against the overlay video: smooth curves, contact window on the real hit?")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="badminton-ai-coach")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="video -> keypoints + skeleton overlay")
    p_extract.add_argument("video", help="path to a clip (pre-trimmed to one shot)")
    p_extract.add_argument("--out", help=f"output directory (default: {OUTPUT_DIR})")
    p_extract.set_defaults(func=cmd_extract)

    p_segment = sub.add_parser("segment", help="find backswing / contact window / follow-through")
    p_segment.add_argument("input", help="video or .pose.json from extract")
    p_segment.add_argument("--hand", choices=["right", "left"], default="right", help="racket hand")
    p_segment.add_argument("--out", help=f"output directory (default: {OUTPUT_DIR})")
    p_segment.set_defaults(func=cmd_segment)

    p_features = sub.add_parser("features", help="normalized feature time series + at-contact metrics")
    p_features.add_argument("input", help="video or .pose.json from extract")
    p_features.add_argument("--hand", choices=["right", "left"], default="right", help="racket hand")
    p_features.add_argument("--out", help=f"output directory (default: {OUTPUT_DIR})")
    p_features.set_defaults(func=cmd_features)

    p_review = sub.add_parser("review", help="step through detected events frame by frame; mark ground truth")
    p_review.add_argument("input", help="video or .pose.json from extract")
    p_review.add_argument("--hand", choices=["right", "left"], default="right", help="racket hand")
    p_review.add_argument("--out", help=f"output directory (default: {OUTPUT_DIR})")
    p_review.set_defaults(func=cmd_review)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
