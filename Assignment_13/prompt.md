Act as an expert Python developer. Write a complete, ready-to-run script for a "Focus & Slouch Webcam Monitor" using OpenCV, MediaPipe, and pygame.

The application needs to run in the background and perform the following:
1. Slouch Detection: Use MediaPipe Pose Landmarker or Face Mesh to track the distance between the nose/ears and shoulders. If the user's posture drops below a certain calibrated baseline for more than 5 seconds, play a funny alert sound using pygame (and provide a placeholder for a screen-blur effect).
2. Distraction/Sleep Detection: Calculate the Eye Aspect Ratio (EAR) to detect if the user's eyes are closed or if they look away. Track and log their total "focus time" vs "distracted time".
3. Demo Mode: Ensure there is an easy calibration step at startup so I can demonstrate it live. If I deliberately slouch or close my eyes, the response must trigger instantly (or after a customizable 5-second buffer).

Provide clean, modular code with comments explaining how to adjust the thresholds for the live presentation.
