import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(layout="wide")

st.title("Gesture Recognition Interface")

# ---------- SESSION STATE ----------
if "run" not in st.session_state:
    st.session_state.run = False

col_video, col_panel = st.columns([4,1])

with col_video:
    if st.button("▶ Start"):
        st.session_state.run = True

with col_panel:
    if st.button("⏸ Pause"):
        st.session_state.run = False

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

frame_placeholder = col_video.image([])

# ---------- RIGHT PANEL PLACEHOLDERS ----------
distance_value = col_panel.empty()
distance_bar = col_panel.empty()
gesture_box = col_panel.empty()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

while st.session_state.run:

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not detected")
        break

    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    distance_mm = 0
    gesture = "No Hand"
    color = (255,255,255)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            lm = [(int(p.x*w), int(p.y*h)) for p in hand.landmark]

            x1,y1 = lm[4]
            x2,y2 = lm[8]

            distance = np.hypot(x2-x1, y2-y1)
            distance_mm = int(distance)

            # ---------- GESTURE CLASSIFICATION ----------
            if distance_mm > 120:
                gesture = "Open Hand"
                color = (0,255,0)

            elif 40 < distance_mm <= 120:
                gesture = "Pinch"
                color = (0,165,255)

            else:
                gesture = "Closed"
                color = (0,0,255)

            # ---------- DRAW ----------
            cv2.circle(frame,(x1,y1),8,(255,0,255),-1)
            cv2.circle(frame,(x2,y2),8,(255,0,255),-1)
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),3)

            cv2.putText(frame,f"{distance_mm} mm",(x2+10,y2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

            # ---------- TOP GESTURE BADGE ----------
            cv2.rectangle(frame,(0,0),(320,70),(128,0,128),-1)
            cv2.putText(frame,gesture,(20,45),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ---------- RIGHT PANEL UI ----------
    distance_value.markdown(
        f"<h2 style='text-align:center;margin-bottom:0'>{distance_mm}</h2>"
        f"<p style='text-align:center;margin-top:0'>millimeters</p>",
        unsafe_allow_html=True
    )

    distance_bar.progress(min(distance_mm/200,1.0))

    gesture_box.markdown(f"""
    ### ✋ Gesture States
    🟢 **Open Hand** (>120mm)  
    🟠 **Pinch** (40-120mm)  
    🔴 **Closed** (<40mm)

    ---
    ## Current: **{gesture}**
    """)

    frame_placeholder.image(frame, channels="BGR")

    time.sleep(0.03)

cap.release()