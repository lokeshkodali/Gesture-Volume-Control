import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ================= MediaPipe Setup =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= Audio Setup =================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol = volume_ctrl.GetVolumeRange()[:2]

# ================= Webcam =================
cap = cv2.VideoCapture(0)

# ================= Mute Debounce =================
open_palm_counter = 0
debounce_threshold = 15
is_muted = False


def is_open_palm(lm_list):
    """
    Returns True if all fingers are extended.
    """
    finger_tips = [4, 8, 12, 16, 20]
    finger_knuckles = [2, 6, 10, 14, 18]

    for tip, knuckle in zip(finger_tips, finger_knuckles):
        if lm_list[tip][1] > lm_list[knuckle][1]:
            return False
    return True


# ================= Main Loop =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []

            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # ========== PINCH GESTURE (VOLUME CONTROL) ==========
            x1, y1 = lm_list[4]   # Thumb tip
            x2, y2 = lm_list[8]   # Index tip

            distance = np.hypot(x2 - x1, y2 - y1)

            vol = np.interp(distance, [30, 200], [min_vol, max_vol])
            volume_ctrl.SetMasterVolumeLevel(vol, None)

            vol_percent = np.interp(distance, [30, 200], [0, 100])

            # Visuals
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            bar_height = int(np.interp(vol_percent, [0, 100], [400, 150]))
            cv2.rectangle(img, (50, bar_height), (85, 400), (0, 255, 0), -1)

            cv2.putText(
                img, f'{int(vol_percent)} %',
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 3
            )

            # ========== OPEN PALM (MUTE / UNMUTE) ==========
            if is_open_palm(lm_list):
                open_palm_counter += 1
            else:
                open_palm_counter = 0

            if open_palm_counter >= debounce_threshold:
                is_muted = not is_muted
                volume_ctrl.SetMute(is_muted, None)
                print("Mute toggled:", is_muted)
                open_palm_counter = 0

            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()