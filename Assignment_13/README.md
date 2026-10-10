# Focus & Slouch Webcam Monitor

A real-time webcam monitor that estimates posture and attention with MediaPipe, OpenCV, and NumPy. It tracks focus and distracted time and writes each session summary to `focus_log.csv`.

## Run locally

Use Python 3.12, install the dependencies, and start the monitor:

```bash
python -m pip install -r requirements.txt
python focus_monitor.py
```

The webcam opens with a short upright calibration. Press `c` to calibrate again, `b` to toggle the blur demonstration, or `q`/`Esc` to quit. Adjust thresholds and alert delays near the top of `focus_monitor.py` for a live demonstration.
