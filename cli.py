"""Badminton AI Coach — command-line entry point.

Stage 1 exposes `extract`: run pose estimation on a clip and produce
  - <name>.pose.json   (per-frame keypoints, consumed by later stages)
  - <name>.overlay.mp4 (skeleton overlay — eyeball this to verify tracking)

Usage:
  .venv\\Scripts\\python.exe cli.py extract data\\raw\\my_smash.mp4
"""

import argparse
import sys
from pathlib import Path

from src.pose import PoseExtractor, render_overlay

OUTPUT_DIR = Path("data/output")


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


def main() -> int:
    parser = argparse.ArgumentParser(prog="badminton-ai-coach")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="video -> keypoints + skeleton overlay")
    p_extract.add_argument("video", help="path to a clip (pre-trimmed to one shot)")
    p_extract.add_argument("--out", help=f"output directory (default: {OUTPUT_DIR})")
    p_extract.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
