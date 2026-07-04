"""Smoke test: prove MediaPipe pose estimation runs on this machine.

This exists because the venv runs Python 3.14 and mediapipe ships a
version-agnostic wheel — installation succeeding does not prove the native
runtime works. This script is the judge. It:

  1. imports the full stack (cv2, mediapipe, numpy, scipy)
  2. downloads the pose landmarker model if missing (-> models/)
  3. runs pose detection on a test image of a person and reports landmarks

Run:  .venv\\Scripts\\python.exe tests\\smoke_test.py
Pass: exits 0 and prints "SMOKE TEST PASSED" with landmarks detected.
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
# MediaPipe's own sample image of a person, used in their pose examples.
TEST_IMAGE_PATH = ROOT / "data" / "raw" / "smoke_pose_sample.jpg"
TEST_IMAGE_URL = "https://storage.googleapis.com/mediapipe-assets/pose.jpg"


def step(msg: str) -> None:
    print(f"[smoke] {msg}")


def download(url: str, dest: Path, what: str) -> bool:
    if dest.exists():
        step(f"{what} already present: {dest.name}")
        return True
    step(f"downloading {what} ...")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        step(f"{what} saved to {dest} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:  # noqa: BLE001 - report anything, this is a smoke test
        step(f"WARNING: could not download {what}: {e}")
        return False


def main() -> int:
    step(f"python {sys.version}")

    # 1. Imports — the first thing that breaks if the 3.14 wheel is bad.
    import numpy as np
    import scipy  # noqa: F401
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    step(f"imports OK (mediapipe {mp.__version__}, opencv {cv2.__version__}, numpy {np.__version__})")

    # 2. Model.
    if not download(MODEL_URL, MODEL_PATH, "pose landmarker model"):
        step("FAILED: cannot proceed without the model")
        return 1

    # 3. Build the landmarker — native runtime initialization happens here.
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    step("pose landmarker initialized")

    # 4. Inference. Prefer a real image of a person; fall back to a blank
    #    frame (still proves the runtime executes, just detects nobody).
    have_person = download(TEST_IMAGE_URL, TEST_IMAGE_PATH, "test image (person)")
    if have_person:
        bgr = cv2.imread(str(TEST_IMAGE_PATH))
        have_person = bgr is not None
    if not have_person:
        step("falling back to blank frame (runtime check only)")
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    landmarker.close()

    if have_person:
        if not result.pose_landmarks:
            step("FAILED: runtime ran but detected no person in a known-person image")
            return 1
        n = len(result.pose_landmarks[0])
        nose = result.pose_landmarks[0][0]
        step(f"detected {n} landmarks; nose at ({nose.x:.3f}, {nose.y:.3f}), visibility {nose.visibility:.2f}")
    else:
        step(f"inference ran on blank frame; poses found: {len(result.pose_landmarks)} (expected 0)")
        step("NOTE: person-detection not verified (test image unavailable) — verify at Stage 1 checkpoint")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
