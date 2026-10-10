---
title: YOLO Vision Studio
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: 3.11
app_file: app.py
short_description: CPU-powered detection, segmentation, pose, and OBB demos
---

# YOLO Vision Studio

A Gradio app for trying four trained Ultralytics YOLO models on images, live webcam frames, and videos. All inference runs on CPU.

**Designed & created by Murtaza Ali**

- Website: [murtazaalikhan.com](https://www.murtazaalikhan.com)
- Kaggle: [murtazakhanpro](https://www.kaggle.com/murtazakhanpro)

## How to use

Choose a model from the dropdown at the top, then open the matching input tab. The app loads all four local model files at startup.

![Select a model and workflow](docs/images/01-choose-model.svg)

| Model | Weight file |
| --- | --- |
| Object Detection | `object_detection.pt` |
| Segmentation | `segmentaion_model.pt` |
| Pose Estimation | `pose_model.pt` |
| OBB (Oriented Bounding Boxes) | `obb.pt` |

The segmentation weight filename keeps the spelling of the supplied file. If you rename it, also change `MODEL_FILES` in `app.py`.

### Image

Upload a still image and select **Run image inference**. The annotated image appears beside the input.

### Live Webcam

Open the **Live Webcam** tab and allow your browser to use the camera. Gradio streams webcam frames to the selected model and shows annotated frames. Camera access requires a secure browser context, such as the HTTPS Hugging Face Space page.

![Allow the browser camera and view predictions](docs/images/02-webcam.svg)

### Video

Upload a video and select **Process video**. Processing runs frame by frame; the result appears in the video player and as a downloadable file.

![Upload and view image or video results](docs/images/03-upload-results.svg)

## Deploy to the Hugging Face Space

The target Space is [Murtazakhanpro/Vision_Studio](https://huggingface.co/spaces/Murtazakhanpro/Vision_Studio). Upload or push the contents of this `Assignment_14` folder to the root of that Space repository. Keep `README.md`, `app.py`, `requirements.txt`, all four `.pt` files, and `docs/images/` in the repository. The README YAML config selects the Gradio SDK and `app.py` as the app entry point.

The weights total about 167 MiB. Use Git LFS (the included `.gitattributes` tracks `.pt` files) or the Hugging Face web uploader for the model files. Allow the Space build to finish installing dependencies and loading all four weights before opening the app.

## Run locally

Use Python 3.11, then from this directory:

```bash
python -m pip install -r requirements-cpu.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860). The CPU build processes video more slowly than a GPU, and webcam frame rate depends on the CPU available to the Space. Uploaded files and generated outputs use temporary local storage.

## Files

- `app.py` — Gradio interface and CPU inference functions.
- `requirements.txt` — Gradio, Ultralytics, OpenCV, and CPU PyTorch dependencies.
- `requirements-cpu.txt` — requirements entry point for local setup.
- `object_detection.pt`, `segmentaion_model.pt`, `pose_model.pt`, `obb.pt` — model weights.
- `docs/images/` — quick-start visuals.

## License and model provenance

Set a license for the Space only after confirming the license and redistribution terms for the application and all four model weight files. This project does not assert that the supplied weights are licensed for redistribution.
