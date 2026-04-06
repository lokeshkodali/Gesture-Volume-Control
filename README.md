# Gesture Volume Control

## Description
This project controls system volume using hand gestures through a webcam. It uses computer vision techniques to detect hand landmarks and adjust volume based on finger distance.

## 🛠 Technologies Used
- Python
- OpenCV
- MediaPipe
- Streamlit

## How It Works
- Captures live video using webcam
- Detects hand using MediaPipe
- Tracks finger positions (landmarks)
- Calculates distance between fingers
- Maps distance to system volume

## How to Run
1. Install required libraries:
   pip install opencv-python mediapipe streamlit

2. Run the main file:
   python main.py

## Features
- Real-time gesture detection
- Smooth volume control
- User-friendly interface

## Future Improvements
- Add brightness control
- Improve gesture accuracy
- Add more gesture commands
