"""Gradio app for four local Ultralytics YOLO models, running on CPU."""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "runs" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"

MODEL_FILES = {
    "detection": "object_detection.pt",
    # Keep the spelling of the supplied model file.
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
    """Load all weights once when the Space starts."""
    missing = [filename for filename in MODEL_FILES.values() if not (BASE_DIR / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model weight files beside app.py: " + ", ".join(missing)
        )
    return {key: YOLO(str(BASE_DIR / filename)) for key, filename in MODEL_FILES.items()}


MODELS = load_models()
MODEL_LOCKS = {key: threading.Lock() for key in MODELS}


def predict_frame(image_rgb: np.ndarray, model_key: str) -> np.ndarray:
    """Run the selected task and return an RGB annotated image for Gradio."""
    if model_key not in MODELS:
        raise gr.Error("Select one of the four available models.")
    if image_rgb is None or image_rgb.size == 0:
        raise gr.Error("Provide an image or webcam frame first.")
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with MODEL_LOCKS[model_key]:
        results = MODELS[model_key](source=frame_bgr, device=DEVICE, verbose=False)
        annotated_bgr = results[0].plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def process_image(model_key: str, image_rgb: np.ndarray | None):
    if image_rgb is None:
        raise gr.Error("Upload an image to run inference.")
    return predict_frame(image_rgb, model_key), f"Image processed with **{MODEL_LABELS[model_key]}** on CPU."


def process_video(model_key: str, video_path: str | None):
    if model_key not in MODELS:
        raise gr.Error("Select one of the four available models.")
    if not video_path:
        raise gr.Error("Upload a video to run inference.")

    run_id = uuid.uuid4().hex
    try:
        with MODEL_LOCKS[model_key]:
            results = MODELS[model_key](
                source=video_path,
                device=DEVICE,
                save=True,
                project=str(OUTPUT_DIR),
                name=run_id,
                stream=True,
            )
            # Stream frames so long videos do not keep every result in memory.
            for _result in results:
                pass
            predictor = MODELS[model_key].predictor
            output_dir = Path(predictor.save_dir).resolve() if predictor and predictor.save_dir else None

        if output_dir is None or not output_dir.is_dir():
            raise RuntimeError("YOLO completed without creating an output folder.")
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
        output_files = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in video_extensions]
        if not output_files:
            raise RuntimeError("YOLO completed, but no processed video was created.")
        output_path = max(output_files, key=lambda p: p.stat().st_mtime)
        return str(output_path), str(output_path), f"Video processed with **{MODEL_LABELS[model_key]}** on CPU."
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Video inference failed: {exc}") from exc


CSS = """
.gradio-container {max-width: 1120px !important; margin: auto !important;}
.hero {text-align: center; padding: 1.15rem 0 .6rem;}
.hero h1 {margin-bottom: .25rem;}
.credit {text-align: center; opacity: .82; font-size: .95rem;}
"""


with gr.Blocks(title="YOLO Vision Studio") as demo:
    gr.Markdown(
        """
        <div class="hero">
        <h1>🔎 YOLO Vision Studio</h1>
        <p>Run four computer-vision models on images, live webcam frames, and videos.</p>
        </div>
        """
    )

    model_selector = gr.Dropdown(
        choices=[(label, key) for key, label in MODEL_LABELS.items()],
        value="detection",
        label="Active model",
        info="Choose the task model to use for this input.",
    )

    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                image_input = gr.Image(sources=["upload"], type="numpy", label="Upload an image")
                image_output = gr.Image(type="numpy", label="Annotated result")
            image_button = gr.Button("Run image inference", variant="primary")
            image_status = gr.Markdown()
            image_button.click(
                fn=process_image,
                inputs=[model_selector, image_input],
                outputs=[image_output, image_status],
            )

        with gr.Tab("Live Webcam"):
            gr.Markdown("Allow camera access, then start the webcam. Each frame is processed on CPU.")
            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam"], type="numpy", streaming=True, label="Camera"
                )
                webcam_output = gr.Image(type="numpy", label="Live predictions")
            webcam_input.stream(
                fn=predict_frame,
                inputs=[webcam_input, model_selector],
                outputs=webcam_output,
                stream_every=0.35,
                time_limit=300,
                concurrency_limit=1,
            )

        with gr.Tab("Video"):
            video_input = gr.Video(sources=["upload"], label="Upload a video")
            video_button = gr.Button("Process video", variant="primary")
            video_status = gr.Markdown()
            with gr.Row():
                video_output = gr.Video(label="Processed video", format="mp4")
                video_download = gr.File(label="Download processed video")
            video_button.click(
                fn=process_video,
                inputs=[model_selector, video_input],
                outputs=[video_output, video_download, video_status],
            )

    gr.Markdown(
        """<div class="credit">Designed &amp; created by <strong>Murtaza Ali</strong> ·
        <a href="https://www.murtazaalikhan.com" target="_blank" rel="noopener noreferrer">Website</a> ·
        <a href="https://www.kaggle.com/murtazakhanpro" target="_blank" rel="noopener noreferrer">Kaggle</a>
        </div>"""
    )


if __name__ == "__main__":
    demo.queue(max_size=12, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=CSS,
        allowed_paths=[str(OUTPUT_DIR)],
    )
