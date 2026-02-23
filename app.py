import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import comtypes
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

st.set_page_config(layout="wide")
st.title("Volume Hand Gesture")

# Session state
if "run" not in st.session_state:
    st.session_state.run = False

if "capture" not in st.session_state:
    st.session_state.capture = False

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    start_btn = st.button("Start Camera")

with col2:
    stop_btn = st.button("Stop Camera")
    capture_btn = st.button("Capture")

if start_btn:
    st.session_state.run = True

if stop_btn:
    st.session_state.run = False

if capture_btn:
    st.session_state.capture = True

# Display placeholders
frame_window = col1.image([])
volume_text = col1.empty()

status_box = col2.empty()
params_box = col2.empty()
info_box = col2.empty()

# Sliders for parameters (like your screenshot)
col2.markdown("### Detection Parameters")

detection_conf = col2.slider("Detection Confidence", 0.1, 1.0, 0.7, 0.05)
tracking_conf = col2.slider("Tracking Confidence", 0.1, 1.0, 0.7, 0.05)
max_hands = col2.slider("Max Number of Hands", 1, 2, 1)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=detection_conf,
    min_tracking_confidence=tracking_conf,
    max_num_hands=max_hands
)
mp_draw = mp.solutions.drawing_utils

if st.session_state.run:

    comtypes.CoInitialize()

    # Audio setup
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol, max_vol = volume_ctrl.GetVolumeRange()[:2]

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.run:

        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        vol_percent = 0
        hands_detected = 0

        if results.multi_hand_landmarks:
            hands_detected = len(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = []

                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                # Pinch gesture
                x1, y1 = lm_list[4]
                x2, y2 = lm_list[8]

                distance = np.hypot(x2 - x1, y2 - y1)
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                volume_ctrl.SetMasterVolumeLevel(vol, None)

                vol_percent = int(np.interp(distance, [30, 200], [0, 100]))

                cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                mp_draw.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # FPS
        current_time = time.time()
        fps = int(1 / (current_time - prev_time)) if current_time != prev_time else 0
        prev_time = current_time

        # Update UI
        frame_window.image(img, channels="BGR")
        volume_text.write(f"Volume: {vol_percent} %")

        # Detection Status Panel
        status_box.markdown(f"""
        ### Detection Status
        Camera Status: **🟢 Active**  
        Hands Detected: **{hands_detected}**  
        Detection FPS: **{fps}**  
        Model Status: **Loaded**
        """)

        # Detection Info Panel
        info_box.markdown(f"""
        ### Detection Info
        Landmarks: **21**  
        Connections: **20**  
        Resolution: **{w} x {h}**
        """)

        # Capture image
        if st.session_state.capture:
            cv2.imwrite("capture.jpg", img)
            col2.success("Image Saved")
            st.session_state.capture = False

        time.sleep(0.03)

    cap.release()

else:
    status_box.markdown("""
    ### Detection Status
    Camera Status: **🔴 Stopped**  
    Hands Detected: **0**  
    Detection FPS: **0**
    """)
