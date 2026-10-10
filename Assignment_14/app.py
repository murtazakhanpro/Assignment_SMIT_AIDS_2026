"""Flask web app for four local Ultralytics YOLO models (CPU inference)."""
from __future__ import annotations

import base64
import os
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "runs" / "web"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

MODEL_FILES = {
    "detection": "object_detection.pt",
    # Keep the spelling of the supplied segmentation weight file.
    "segmentation": "segmentaion_model.pt",
    "pose": "pose_model.pt",
    "obb": "obb.pt",
}
MODEL_LABELS = {
    "detection": "Object Detection",
    "segmentation": "Segmentation",
    "pose": "Pose Estimation",
    "obb": "OBB (Oriented Bounding Boxes)",
}


def load_models() -> dict[str, YOLO]:
    missing = [filename for filename in MODEL_FILES.values() if not (BASE_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError("Missing model weight files beside app.py: " + ", ".join(missing))
    return {key: YOLO(str(BASE_DIR / filename)) for key, filename in MODEL_FILES.items()}


MODELS = load_models()
MODEL_LOCKS = {key: threading.Lock() for key in MODELS}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


PAGE = r"""<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Vision Studio</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; padding: 36px 18px; color: #e9efff;
      background: radial-gradient(ellipse at 15% 0%, #24366b 0, transparent 42%), #0b1020; }
    main { width: min(940px, 100%); margin: auto; }
    .card { padding: clamp(22px, 5vw, 44px); border: 1px solid #2b3655; border-radius: 24px;
      background: #121a2d; box-shadow: 0 24px 80px #0006; }
    header { text-align: center; margin-bottom: 28px; }
    .eyebrow { color: #8ea9ff; font-size: .78rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 10px 0; font-size: clamp(2rem, 6vw, 3.2rem); letter-spacing: -.04em; }
    header p, .hint { color: #aeb9d3; line-height: 1.55; }
    label { display: block; margin: 0 0 9px; font-weight: 700; }
    select, .button { border: 1px solid #3b4b72; border-radius: 12px; color: #f5f7ff; background: #19243d; }
    select { width: 100%; padding: 14px 15px; font: inherit; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 24px; }
    .button { min-height: 48px; padding: 12px 15px; cursor: pointer; font: inherit; font-weight: 750; transition: .15s ease; }
    .button:hover { transform: translateY(-1px); border-color: #93a9ff; background: #26365b; }
    .button.primary { border-color: #728cff; background: #536cf1; }
    .button:disabled { opacity: .55; cursor: wait; transform: none; }
    input[type=file] { display: none; }
    .panel { border: 1px solid #2b3655; border-radius: 16px; padding: 18px; background: #0c1324; }
    .panel h2 { margin: 0 0 10px; font-size: 1.1rem; }
    .media { display: block; width: 100%; max-width: 100%; max-height: 65vh; object-fit: contain;
      margin: 12px auto 0; border-radius: 10px; background: #050812; }
    .camera-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .credit { margin: 22px 0 0; color: #aeb9d3; text-align: center; font-size: .9rem; }
    .credit a { color: #aebaff; text-decoration: none; }
    .credit a:hover { text-decoration: underline; }
    [hidden] { display: none !important; }
    @media (max-width: 650px) { .actions, .camera-grid { grid-template-columns: 1fr; } body { padding: 14px 9px; } }
  </style>
</head><body><main><div class="card">
  <header><div class="eyebrow">Computer Vision · YOLO</div><h1>YOLO Vision Studio</h1>
    <p>Choose a trained model and try it on an image, video, or live webcam.</p></header>
  <label for="modelSelect">Active model</label>
  <select id="modelSelect">
    <option value="detection">Object Detection</option>
    <option value="segmentation">Segmentation</option>
    <option value="pose">Pose Estimation</option>
    <option value="obb">OBB (Oriented Bounding Boxes)</option>
  </select>
  <div class="actions">
    <button class="button primary" id="webcamButton" type="button">1) Use Web Cam</button>
    <button class="button" id="videoButton" type="button">2) Upload Video</button>
    <button class="button" id="imageButton" type="button">3) Upload Image</button>
  </div>
  <input id="videoInput" type="file" accept="video/*"><input id="imageInput" type="file" accept="image/*">

  <section class="panel" id="webcamPanel" hidden>
    <h2>Live webcam</h2><p class="hint" id="webcamStatus">Allow camera access in your browser.</p>
    <div class="camera-grid">
      <video class="media" id="webcamVideo" autoplay muted playsinline aria-label="Webcam preview"></video>
      <img class="media" id="webcamResult" alt="Live model predictions">
    </div>
  </section>
  <section class="panel" id="videoPanel" hidden>
    <h2>Video processing</h2><p class="hint" id="videoStatus" aria-live="polite"></p>
    <a class="button primary" id="videoDownload" hidden>Download processed video</a>
  </section>
  <section class="panel" id="imagePanel" hidden>
    <h2>Image result</h2><p class="hint" id="imageStatus" aria-live="polite"></p>
    <img class="media" id="imageResult" alt="Annotated image result">
  </section>
</div>
<footer class="credit">Designed &amp; created by <strong>Murtaza Ali</strong> ·
  <a href="https://www.murtazaalikhan.com" target="_blank" rel="noopener noreferrer">Website</a> ·
  <a href="https://www.kaggle.com/murtazakhanpro" target="_blank" rel="noopener noreferrer">Kaggle</a>
</footer></main>
<script>
  const modelSelect = document.querySelector('#modelSelect');
  const webcamPanel = document.querySelector('#webcamPanel');
  const webcamVideo = document.querySelector('#webcamVideo');
  const webcamResult = document.querySelector('#webcamResult');
  const webcamStatus = document.querySelector('#webcamStatus');
  const videoInput = document.querySelector('#videoInput');
  const imageInput = document.querySelector('#imageInput');
  const canvas = document.createElement('canvas');
  let cameraStream = null, webcamOn = false;
  const modelKey = () => modelSelect.value;
  function stopWebcam() {
    webcamOn = false;
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null; webcamVideo.srcObject = null; webcamResult.removeAttribute('src'); webcamPanel.hidden = true;
  }
  function showOnly(which) {
    stopWebcam();
    document.querySelector('#videoPanel').hidden = which !== 'video';
    document.querySelector('#imagePanel').hidden = which !== 'image';
  }
  async function streamFrames() {
    while (webcamOn && cameraStream) {
      if (webcamVideo.readyState >= 2 && webcamVideo.videoWidth > 0) {
        const scale = Math.min(1, 800 / webcamVideo.videoWidth);
        canvas.width = Math.round(webcamVideo.videoWidth * scale);
        canvas.height = Math.round(webcamVideo.videoHeight * scale);
        canvas.getContext('2d').drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.82));
        if (!blob) break;
        const form = new FormData(); form.append('model', modelKey()); form.append('file', blob, 'webcam.jpg');
        try {
          const response = await fetch('/api/frame', {method: 'POST', body: form});
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || 'Webcam inference failed.');
          webcamResult.src = `data:image/jpeg;base64,${data.image}`;
        } catch (error) { webcamStatus.textContent = error.message; break; }
      }
      await new Promise(resolve => setTimeout(resolve, 80));
    }
  }
  document.querySelector('#webcamButton').addEventListener('click', async () => {
    if (webcamOn) { stopWebcam(); return; }
    showOnly('webcam'); webcamPanel.hidden = false;
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
      webcamVideo.srcObject = cameraStream; webcamOn = true;
      webcamStatus.textContent = 'Camera active. Predictions appear beside the preview.';
      streamFrames();
    } catch (error) { webcamStatus.textContent = `Could not access the camera: ${error.message}`; }
  });
  document.querySelector('#videoButton').addEventListener('click', () => videoInput.click());
  document.querySelector('#imageButton').addEventListener('click', () => imageInput.click());
  videoInput.addEventListener('change', async () => {
    const file = videoInput.files[0]; if (!file) return;
    showOnly('video');
    const status = document.querySelector('#videoStatus'), link = document.querySelector('#videoDownload');
    status.textContent = `Uploading and processing ${file.name} with ${modelSelect.selectedOptions[0].text}...`;
    link.hidden = true; link.removeAttribute('href');
    const form = new FormData(); form.append('model', modelKey()); form.append('file', file);
    try {
      const response = await fetch('/api/video', {method: 'POST', body: form});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Video processing failed.');
      status.textContent = data.message; link.href = data.download_url; link.download = data.filename; link.hidden = false;
    } catch (error) { status.textContent = error.message; }
    finally { videoInput.value = ''; }
  });
  imageInput.addEventListener('change', async () => {
    const file = imageInput.files[0]; if (!file) return;
    showOnly('image');
    const status = document.querySelector('#imageStatus'), image = document.querySelector('#imageResult');
    status.textContent = `Processing ${file.name} with ${modelSelect.selectedOptions[0].text}...`;
    image.removeAttribute('src');
    const form = new FormData(); form.append('model', modelKey()); form.append('file', file);
    try {
      const response = await fetch('/api/image', {method: 'POST', body: form});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Image processing failed.');
      image.src = `data:${data.mime_type};base64,${data.image}`; status.textContent = 'Inference complete.';
    } catch (error) { status.textContent = error.message; }
    finally { imageInput.value = ''; }
  });
</script>
</body></html>"""


def request_model() -> str:
    key = request.form.get("model", "detection")
    if key not in MODELS:
        raise ValueError("Unknown model. Choose detection, segmentation, pose, or obb.")
    return key


def save_upload(file, extensions: set[str]):
    if file is None or not file.filename:
        return None, (jsonify(error="Choose a file to upload."), 400)
    name = secure_filename(file.filename)
    if not name or Path(name).suffix.lower() not in extensions:
        return None, (jsonify(error="Unsupported file type."), 400)
    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}_{name}"
    file.save(destination)
    return destination, None


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/frame")
def process_frame():
    try:
        key = request_model()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    frame_file = request.files.get("file")
    if frame_file is None:
        return jsonify(error="No webcam frame was received."), 400
    try:
        raw = frame_file.read()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="The webcam frame could not be decoded."), 400
        with MODEL_LOCKS[key]:
            result = MODELS[key](source=frame, device=DEVICE, verbose=False)[0]
            annotated = result.plot()
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("Could not encode the webcam frame.")
        return jsonify(image=base64.b64encode(buffer.tobytes()).decode("ascii"))
    except Exception as exc:
        app.logger.exception("Webcam inference failed")
        return jsonify(error=f"Webcam inference failed: {exc}"), 500


@app.post("/api/image")
def process_image():
    try:
        key = request_model()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    path, error = save_upload(request.files.get("file"), {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
    if error:
        return error
    try:
        with MODEL_LOCKS[key]:
            result = MODELS[key](source=str(path), device=DEVICE, verbose=False)[0]
            annotated = result.plot()
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("Could not encode the annotated image.")
        return jsonify(image=base64.b64encode(buffer.tobytes()).decode("ascii"), mime_type="image/jpeg")
    except Exception as exc:
        app.logger.exception("Image inference failed")
        return jsonify(error=f"Image inference failed: {exc}"), 500
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/video")
def process_video():
    try:
        key = request_model()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    path, error = save_upload(request.files.get("file"), {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"})
    if error:
        return error
    try:
        with MODEL_LOCKS[key]:
            results = MODELS[key](source=str(path), device=DEVICE, save=True,
                                  project=str(OUTPUT_DIR), name=uuid.uuid4().hex, stream=True)
            for _result in results:
                pass
            predictor = MODELS[key].predictor
            output_dir = Path(predictor.save_dir).resolve() if predictor and predictor.save_dir else None
        if output_dir is None or not output_dir.is_dir():
            raise RuntimeError("YOLO completed without creating a video output folder.")
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
        files = [item for item in output_dir.iterdir() if item.is_file() and item.suffix.lower() in video_exts]
        if not files:
            raise RuntimeError("YOLO completed, but no processed video was created.")
        output_file = max(files, key=lambda item: item.stat().st_mtime)
        download_url = f"/outputs/{output_dir.name}/{output_file.name}"
        return jsonify(message=f"Video processed with {MODEL_LABELS[key]} on CPU.",
                       output_path=str(output_file), download_url=download_url, filename=output_file.name)
    except Exception as exc:
        app.logger.exception("Video inference failed")
        return jsonify(error=f"Video inference failed: {exc}"), 500
    finally:
        path.unlink(missing_ok=True)


@app.get("/outputs/<run_id>/<path:filename>")
def download_output(run_id: str, filename: str):
    if not run_id.isalnum():
        return jsonify(error="Invalid output identifier."), 400
    output_folder = (OUTPUT_DIR / run_id).resolve()
    if OUTPUT_DIR.resolve() not in output_folder.parents:
        return jsonify(error="Invalid output path."), 400
    return send_from_directory(output_folder, filename, as_attachment=True)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error="Upload exceeds the 500 MB limit."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), debug=False, threaded=True)
