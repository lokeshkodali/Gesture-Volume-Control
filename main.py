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

# ---------- COMPACT RIGHT PANEL CSS ----------
st.markdown("""
<style>
.small-title {font-size:18px; font-weight:600; margin-bottom:0px;}
.small-value {font-size:26px; font-weight:700; margin-top:-5px;}
.compact hr {margin:8px 0;}
</style>
""", unsafe_allow_html=True)

st.title("Hand Gesture Volume + Gesture Recognition")

# ---------- SESSION ----------
if "run" not in st.session_state:
    st.session_state.run = False
if "capture" not in st.session_state:
    st.session_state.capture = False

# ---------- LAYOUT ----------
left, center, right = st.columns([1.2, 2.6, 1.2])

with center:
    if st.button("▶ Start"):
        st.session_state.run = True

with right:
    if st.button("⏹ Stop"):
        st.session_state.run = False
    if st.button("📸 Capture"):
        st.session_state.capture = True

frame_placeholder = center.image([])

# ---------- LEFT PANEL ----------
status_box = left.empty()
info_box = left.empty()
gesture_status_box = left.empty()

# ---------- RIGHT PANEL ----------
distance_box = right.empty()
distance_bar = right.empty()
gesture_box = right.empty()
volume_box = right.empty()
volume_bar = right.empty()

right.markdown("### Detection Parameters")
det_conf = right.slider("Detection Confidence",0.1,1.0,0.7,0.05)
track_conf = right.slider("Tracking Confidence",0.1,1.0,0.7,0.05)
max_hands = right.slider("Max Hands",1,2,1)

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=det_conf,
    min_tracking_confidence=track_conf
)
draw = mp.solutions.drawing_utils

# ---------- AUDIO ----------
comtypes.CoInitialize()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol = volume_ctrl.GetVolumeRange()[:2]

# ---------- CAMERA ----------
if st.session_state.run:

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        distance_mm = 0
        vol_percent = 0
        gesture = "No Hand"
        hands_detected = 0

        if result.multi_hand_landmarks:
            hands_detected = len(result.multi_hand_landmarks)

            for hand in result.multi_hand_landmarks:

                lm = [(int(p.x*w), int(p.y*h)) for p in hand.landmark]

                x1,y1 = lm[4]
                x2,y2 = lm[8]

                distance = np.hypot(x2-x1, y2-y1)
                distance_mm = int(distance)

                vol = np.interp(distance,[30,200],[min_vol,max_vol])
                volume_ctrl.SetMasterVolumeLevel(vol,None)
                vol_percent = int(np.interp(distance,[30,200],[0,100]))

                if distance_mm > 120:
                    gesture = "Open Hand"
                    color = (0,255,0)
                elif 40 < distance_mm <= 120:
                    gesture = "Pinch"
                    color = (0,165,255)
                else:
                    gesture = "Closed"
                    color = (0,0,255)

                cv2.circle(frame,(x1,y1),8,(255,0,255),-1)
                cv2.circle(frame,(x2,y2),8,(255,0,255),-1)
                cv2.line(frame,(x1,y1),(x2,y2),color,3)

                cv2.rectangle(frame,(0,0),(320,70),(128,0,128),-1)
                cv2.putText(frame,gesture,(20,45),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        now = time.time()
        fps = int(1/(now-prev_time)) if now!=prev_time else 0
        prev_time = now

        # ---------- LEFT ----------
        status_box.markdown(f"""
        ### Detection Status
        🟢 Camera Active  
        Hands: **{hands_detected}**  
        FPS: **{fps}**
        """)

        info_box.markdown(f"""
        ### Detection Info
        Landmarks: **21**  
        Connections: **20**  
        Resolution: **{w} x {h}**
        """)

        gesture_status_box.markdown(f"""
        ### ✋ Gesture Status
        🟢 Open (>120)  
        🟠 Pinch (40-120)  
        🔴 Closed (<40)

        ---
        **Current:** {gesture}
        """)

        # ---------- RIGHT COMPACT ----------
        distance_box.markdown(
            f"<div class='small-title'>Distance</div><div class='small-value'>{distance_mm}</div>",
            unsafe_allow_html=True
        )
        distance_bar.progress(min(distance_mm/200,1.0))

        gesture_box.markdown(f"<div class='small-title'>Gesture</div>{gesture}",
                             unsafe_allow_html=True)

        volume_box.markdown(
            f"<div class='small-title'>🔊 Volume</div><div class='small-value'>{vol_percent}%</div>",
            unsafe_allow_html=True
        )
        volume_bar.progress(vol_percent/100)

        if st.session_state.capture:
            cv2.imwrite("capture.jpg", frame)
            right.success("Saved")
            st.session_state.capture = False

        frame_placeholder.image(frame, channels="BGR")

    cap.release()