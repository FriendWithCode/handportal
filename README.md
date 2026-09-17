# AI-Powered Interactive Hand Portal

## Overview
 This project leverages state-of-the-art machine learning models to detect and track up to 4 hands simultaneously, creating a seamless and smooth "portal" visual effect that follows the user's movements.

## Features
- **Advanced Dual-Hand & Multi-Hand Tracking**: Stable and reliable tracking for up to 4 hands with a 15-frame dual-hand persistence tracker, preventing "ghost" movements and jittering.
- **Gesture-Based Filter Control**: Intuitively switch between different visual filters using natural hand gestures. 
- **Smooth Visual Transitions**: Features graceful fade-in and fade-out effects (`portal_alpha`) for both the portal mesh and the hand skeletons, ensuring visual elements transition synchronously without popping when hands enter or exit the camera frame.
- **Robust Real-Time Compositing**: Custom `Compositor` and `HandSkeletonVisualizer` to render HUD interfaces, skeletons, and particle meshes smoothly over a live video feed.

## Technologies Used
- **Python**: Core programming language.
- **OpenCV (cv2)**: Used for real-time video capture (utilizing `CAP_DSHOW` on Windows for stability), frame manipulation, and image compositing.
- **MediaPipe Tasks (HandLandmarker)**: Google's powerful machine learning framework for high-performance, real-time hand tracking and landmark detection.
- **NumPy**: Employed for efficient matrix operations, alpha blending, and image array manipulations.

## How to Run
1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. Run the application:
   ```bash
   python hand_portal.py
   ```
