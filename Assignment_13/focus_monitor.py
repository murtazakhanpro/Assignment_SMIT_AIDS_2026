"""Focus & Slouch Webcam Monitor (OpenCV + MediaPipe).

Install requirements.txt, then run: python focus_monitor.py
Press q/Esc to quit, c to recalibrate, or b to toggle the blur demo overlay.
"""
from __future__ import annotations

import csv
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# Live-demo tuning: calibration records upright nose-to-shoulder position.
# Increase SLOUCH_MARGIN for fewer false alarms; lower it to trigger sooner.
CAMERA_INDEX = 0
CALIBRATION_SECONDS = 3.0
SLOUCH_MARGIN = 0.055          # normalized vertical position below baseline
SLOUCH_BUFFER_SECONDS = 5.0    # set to 0 for instant slouch alerts
EYE_CLOSED_EAR = 0.20          # raise to detect more readily; typical range .18-.23
EYES_CLOSED_BUFFER_SECONDS = 0.5  # set to 0 for instant demo response
LOOK_AWAY_MARGIN = 0.22        # iris displacement from eye center, normalized by eye width
LOOK_AWAY_BUFFER_SECONDS = 0.6
LOG_FILE = Path(__file__).with_name("focus_log.csv")

# Face Mesh indices (MediaPipe canonical topology): eye contours and irises.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)


def ear(points: np.ndarray, ids: tuple[int, ...]) -> float:
    """Eye Aspect Ratio: vertical lid distances divided by eye width."""
    p = points[list(ids)]
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-6:
        return 0.0
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width)


def gaze_is_away(points: np.ndarray) -> bool:
    """Approximate horizontal gaze from iris position inside each eye."""
    offsets = []
    for corners, iris_ids in ((LEFT_EYE, LEFT_IRIS), (RIGHT_EYE, RIGHT_IRIS)):
        start, end = points[corners[0]], points[corners[3]]
        width = np.linalg.norm(end - start)
        if width > 1e-6:
            center = points[list(iris_ids)].mean(axis=0)
            offsets.append(abs(float(np.dot(center - (start + end) / 2,
                                             (end - start) / width))) / width)
    return bool(offsets and float(np.mean(offsets)) > LOOK_AWAY_MARGIN)


def play_alert_sound() -> None:
    """Play two short tones with Windows' built-in sound API (no pygame needed)."""
    try:
        import winsound
        winsound.Beep(660, 160)
        winsound.Beep(880, 160)
    except (ImportError, RuntimeError):
        # Portable fallback: terminal bell. Some terminals may not play a sound.
        print("\a", end="", flush=True)


@dataclass
class Totals:
    focus: float = 0.0
    distracted: float = 0.0


def append_log(totals: Totals) -> None:
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(("ended_at", "focus_seconds", "distracted_seconds"))
        writer.writerow((datetime.now().isoformat(timespec="seconds"),
                         round(totals.focus, 1), round(totals.distracted, 1)))


def main() -> None:
    if not hasattr(mp, "solutions"):
        version = getattr(mp, "__version__", "unknown")
        raise SystemExit(
            f"This script needs MediaPipe 0.10.21 (installed: {version}) and Python 3.12. "
            "Create a Python 3.12 environment, then run: python -m pip install -r requirements.txt"
        )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Change CAMERA_INDEX in the script.")

    pose_api = mp.solutions.pose
    face_api = mp.solutions.face_mesh
    totals = Totals()
    baseline: Optional[float] = None
    calibration_values: list[float] = []
    calibrating_until = time.monotonic() + CALIBRATION_SECONDS
    slouch_since = eyes_since = gaze_since = None
    alert_playing = False
    blur_demo = False
    last_time = time.monotonic()

    try:
        with pose_api.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                           model_complexity=1) as pose, \
             face_api.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face:
            while True:
                ok, frame = cap.read()
                now = time.monotonic()
                dt = min(now - last_time, 0.25)
                last_time = now
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb)
                face_result = face.process(rgb)

                posture_value = None
                slouch = False
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    nose = lm[pose_api.PoseLandmark.NOSE]
                    ls = lm[pose_api.PoseLandmark.LEFT_SHOULDER]
                    rs = lm[pose_api.PoseLandmark.RIGHT_SHOULDER]
                    if min(nose.visibility, ls.visibility, rs.visibility) > 0.45:
                        shoulder_mid_y = (ls.y + rs.y) / 2
                        shoulder_width = max(abs(ls.x - rs.x), 0.08)
                        posture_value = (nose.y - shoulder_mid_y) / shoulder_width
                        if baseline is None or now < calibrating_until:
                            calibration_values.append(posture_value)
                        elif posture_value > baseline + SLOUCH_MARGIN:
                            slouch_since = slouch_since or now
                            slouch = now - slouch_since >= SLOUCH_BUFFER_SECONDS
                        else:
                            slouch_since = None

                eyes_closed = looking_away = False
                if face_result.multi_face_landmarks:
                    face_lm = face_result.multi_face_landmarks[0].landmark
                    pts = np.array([(p.x, p.y) for p in face_lm], dtype=np.float32)
                    mean_ear = (ear(pts, LEFT_EYE) + ear(pts, RIGHT_EYE)) / 2
                    eyes_closed = mean_ear < EYE_CLOSED_EAR
                    looking_away = not eyes_closed and gaze_is_away(pts)

                if eyes_closed:
                    eyes_since = eyes_since or now
                else:
                    eyes_since = None
                if looking_away:
                    gaze_since = gaze_since or now
                else:
                    gaze_since = None
                sleepy = bool(eyes_since and now - eyes_since >= EYES_CLOSED_BUFFER_SECONDS)
                away = bool(gaze_since and now - gaze_since >= LOOK_AWAY_BUFFER_SECONDS)
                distracted = slouch or sleepy or away
                totals.distracted += dt if distracted else 0
                totals.focus += dt if not distracted else 0

                # Sound once per continuous alert; reset when posture/attention recovers.
                if distracted and not alert_playing:
                    # Keep the camera loop responsive while the tone plays.
                    threading.Thread(target=play_alert_sound, daemon=True).start()
                    alert_playing = True
                elif not distracted:
                    alert_playing = False

                if baseline is None or now < calibrating_until:
                    remaining = max(0, calibrating_until - now)
                    cv2.putText(frame, f"Sit upright - calibrating {remaining:.1f}s",
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                    if now >= calibrating_until and calibration_values:
                        baseline = float(np.median(calibration_values))
                        calibration_values.clear()
                else:
                    status = "DISTRACTED" if distracted else "FOCUSED"
                    color = (0, 0, 255) if distracted else (50, 220, 50)
                    cv2.putText(frame, f"{status} | focus {totals.focus:.0f}s | away {totals.distracted:.0f}s",
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
                    if slouch_since and not slouch:
                        cv2.putText(frame, "Sit up...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    if sleepy:
                        cv2.putText(frame, "Wake up!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "q: quit  c: calibrate  b: blur demo", (20, frame.shape[0] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

                # Placeholder effect: overlay blur to demonstrate where an effect can go.
                if blur_demo and distracted:
                    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
                    frame = cv2.addWeighted(frame, 0.15, blurred, 0.85, 0)
                cv2.imshow("Focus & Slouch Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("b"):
                    blur_demo = not blur_demo
                if key == ord("c"):
                    baseline = None
                    calibration_values.clear()
                    calibrating_until = time.monotonic() + CALIBRATION_SECONDS
                    slouch_since = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        append_log(totals)
        print(f"Session saved to {LOG_FILE}\nFocus: {totals.focus:.1f}s | Distracted: {totals.distracted:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Install with: pip install -r requirements.txt") from exc
