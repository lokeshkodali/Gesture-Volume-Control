import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import comtypes
import time
import plotly.graph_objects as go
import pandas as pd
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 0.3rem;
    padding-bottom: 0rem;
}

html, body, [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

/* Main title smaller */
h1 {
    font-size: 34px !important;
    margin-bottom: 0.5rem;
}

/* Volume display styling */
.volume-box {
    text-align: center;
    margin-top: 20px;
}

.volume-percent {
    font-size: 72px;
    font-weight: 700;
}

/* Reduce column spacing */
[data-testid="column"] {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Hand Gesture Volume Control")

# ---------------- SESSION ----------------
if "run" not in st.session_state:
    st.session_state.run = False
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "current_volume" not in st.session_state:
    st.session_state.current_volume = 50

# ---------------- AUDIO ----------------
if "volume_ctrl" not in st.session_state:
    comtypes.CoInitialize()
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_,
        CLSCTX_ALL,
        None
    )
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol, max_vol = volume_ctrl.GetVolumeRange()[:2]

    st.session_state.volume_ctrl = volume_ctrl
    st.session_state.min_vol = min_vol
    st.session_state.max_vol = max_vol

volume_ctrl = st.session_state.volume_ctrl
min_vol = st.session_state.min_vol
max_vol = st.session_state.max_vol

# ---------------- LAYOUT ----------------
left, center, right = st.columns([1, 2.4, 1.2])

# Center Buttons
with center:
    col1, col2 = st.columns(2)
    if col1.button("▶ Start"):
        st.session_state.run = True
    if col2.button("⏹ Stop"):
        st.session_state.run = False

frame_placeholder = center.empty()

# Left Panel
left.markdown("### 🔊 Current Volume")
volume_display = left.empty()

# Right Panel
right.markdown("### 📊 Distance → Volume Mapping")
mapping_chart = right.empty()

right.markdown("### 📈 Volume History")
history_chart = right.empty()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw = mp.solutions.drawing_utils

min_dist = 30
max_dist = 200

# ---------------- SMOOTH FUNCTION ----------------
def smooth_volume_change(current, target, step=2):
    if abs(target - current) < step:
        return target
    if target > current:
        return current + step
    else:
        return current - step

# ---------------- CAMERA ----------------
if st.session_state.run:

    cap = cv2.VideoCapture(0)

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        distance_mm = 0
        vol_percent = st.session_state.current_volume

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:

                lm = [(int(p.x*w), int(p.y*h)) for p in hand.landmark]

                x1, y1 = lm[4]
                x2, y2 = lm[8]

                distance = np.hypot(x2-x1, y2-y1)
                distance_mm = int(distance)

                # Target %
                target_percent = int(np.interp(
                    distance,
                    [min_dist, max_dist],
                    [0, 100]
                ))

                # Smooth transition
                smoothed_percent = smooth_volume_change(
                    st.session_state.current_volume,
                    target_percent
                )

                st.session_state.current_volume = smoothed_percent
                vol_percent = smoothed_percent

                # Convert % → system volume
                device_volume = float(np.interp(
                    smoothed_percent,
                    [0, 100],
                    [min_vol, max_vol]
                ))

                volume_ctrl.SetMasterVolumeLevel(device_volume, None)

                # Store history
                st.session_state.volume_history.append(vol_percent)
                if len(st.session_state.volume_history) > 30:
                    st.session_state.volume_history.pop(0)

                # Draw
                cv2.circle(frame, (x1, y1), 8, (255, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (255, 0, 255), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # -------- LEFT PANEL UPDATE --------
        volume_display.markdown(
            f"""
            <div class="volume-box">
                <div class="volume-percent">{vol_percent}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------- GRAPH 1 --------
        distances = np.linspace(min_dist, max_dist, 50)
        volumes = np.interp(distances, [min_dist, max_dist], [0, 100])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=distances, y=volumes, mode='lines'))
        fig.add_trace(go.Scatter(
            x=[distance_mm],
            y=[vol_percent],
            mode='markers',
            marker=dict(size=10)
        ))

        fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Distance",
            yaxis_title="Volume %"
        )

        mapping_chart.plotly_chart(
            fig,
            use_container_width=True,
            key=str(time.time())
        )

        # -------- GRAPH 2 --------
        hist_df = pd.DataFrame({
            "Time": list(range(len(st.session_state.volume_history))),
            "Volume": st.session_state.volume_history
        })

        history_chart.line_chart(
            hist_df.set_index("Time"),
            height=200,
            use_container_width=True
        )

        # -------- CAMERA --------
        frame_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True
        )

    cap.release()