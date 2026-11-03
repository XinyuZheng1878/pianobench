"""Extract a MediaPipe 21-point hand skeleton from a video, in PianoVAM's Handskeleton JSON format.

  python3 -B scripts/extract_skeleton.py --video CAM_5.avi --model hand_landmarker.task --out rec.json \
      [--every 1] [--upscale 2] [--max-frames 0]

Output: {"<frame>": {"Left": [[x, y, z] * 21], "Right": [...]}, ...} with x, y normalised to the
original frame size, exactly what scripts/export_pianovam.py reads from PianoVAM's Handskeleton/.
"Left"/"Right" are MediaPipe's handedness labels, unchanged; which one is the player's left depends on
the camera, so export decides the swap (PianoVAM's top view needs it swapped).
The model file ships with PianoVAM-Code: FingeringDetection/hand_landmarker.task.
"""
import argparse
import json
import sys

import cv2
import mediapipe as mp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1, help="process every N-th frame")
    ap.add_argument("--upscale", type=float, default=1.0, help="enlarge frames before detection (small hands)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.3)
    a = ap.parse_args()

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=a.model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=a.min_conf, min_hand_presence_confidence=a.min_conf,
        min_tracking_confidence=a.min_conf)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i, n_seen, n_both, n_any = {}, 0, 0, 0, 0
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as det:
        while True:
            ok, frame = cap.read()
            if not ok or (a.max_frames and i >= a.max_frames):
                break
            if i % a.every == 0:
                img = frame if a.upscale == 1 else cv2.resize(frame, None, fx=a.upscale, fy=a.upscale,
                                                              interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
                fr = {}
                for lm, hd in zip(res.hand_landmarks, res.handedness):
                    fr[hd[0].category_name] = [[p.x, p.y, p.z] for p in lm]
                out[str(i)] = fr
                n_seen += 1
                n_any += bool(fr)
                n_both += len(fr) == 2
            i += 1
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{a.video}: {n_seen} frames processed, any hand {n_any / max(n_seen, 1):.3f}, "
          f"both hands {n_both / max(n_seen, 1):.3f} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
