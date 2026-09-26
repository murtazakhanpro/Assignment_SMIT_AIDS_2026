Act as an expert Python Developer and Computer Vision Engineer. I have trained a custom YOLO object detection model for Pakistani traffic. Based on my project directory, I need you to write a complete, production-ready, and error-free `app.py` script using Gradio.

Here is my environment and project details:
- Project Root Directory: D:\Assignment_SMIT_AIDS_2026-main\Assignment_12_ObjectDetection_CV\
- Trained Model Path: D:\Assignment_SMIT_AIDS_2026-main\Assignment_12_ObjectDetection_CV\Pakistani_Trafic_V2.pt
- Frameworks to use: Ultralytics (YOLO), Gradio, OpenCV, Pandas, and Time/Logging.

Please implement the Gradio application with the following strict requirements and features:

1. Core Functionality (Detection, Tracking & Prediction):
   - Load the model using Ultralytics YOLO from the exact absolute path provided.
   - The app must accept both Video and Image inputs.
   - Implement real-time object detection AND tracking (using model.track() or model.predict(track=True)).

2. Object Counting & Live UI Categories:
   - Calculate the real-time count of detected objects grouped by their specific categories/classes.
   - Display this category-wise breakdown dynamically on the Gradio UI screen alongside the processed output.

3. Video Download & Resolution Options:
   - Provide a download option for processed videos.
   - Include a dropdown or radio button in the UI to select output resolutions: 720p (1280x720) or 1080p (1920x1080).
   - Use OpenCV (cv2.VideoWriter) with 'mp4v' or suitable codec to scale and export the final output as an MP4 file.

4. Log Reports & CSV Export:
   - Generate a detailed log/report of the detection session (timestamp, class name, object count).
   - Automatically save this report as a CSV or text log file in the project folder and provide a file download component in the Gradio UI.

5. Webcam Support (Real-time Processing):
   - Add a separate tab or component that connects to the user's Webcam.
   - Process the webcam feed frame-by-frame in real-time, overlaying bounding boxes, tracks, and showing the live count.

6. UI Design:
   - Organize the interface cleanly using Gradio Blocks, separating Image/Video processing, Webcam streaming, and Logs/Downloads into logical tabs or columns.
   - Ensure the code handles file-saving paths correctly without throwing permission or missing-directory errors.

Write the clean, fully commented, and complete `app.py` code ready to run.
