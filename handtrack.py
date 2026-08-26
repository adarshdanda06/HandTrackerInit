import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download once:
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="hand_landmarker.task",
        delegate=python.BaseOptions.Delegate.CPU,  # Metal/GPU delegate crashes on macOS
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

cap = cv2.VideoCapture(0)  # 0 is the built-in FaceTime camera on Mac
start = time.monotonic()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror so movement feels natural
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int((time.monotonic() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    for lm in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        xs = [px for px, _ in pts]
        ys = [py for _, py in pts]
        x, y, bw, bh = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        cx, cy = x + bw // 2, y + bh // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # palm center: mean of wrist + 4 knuckles (rigid, ignores finger motion)
        PALM = [0, 5, 9, 13, 17]
        tx = sum(pts[i][0] for i in PALM) / len(PALM)
        ty = sum(pts[i][1] for i in PALM) / len(PALM)
        cv2.circle(frame, (int(tx), int(ty)), 6, (255, 0, 255), -1)

        # error of the hand from frame center -> drives the pan/tilt motors
        frame_cx = w / 2
        frame_cy = h / 2
        err_x = tx - frame_cx  # + : hand is right of center  -> pan right
        err_y = ty - frame_cy  # + : hand is below center     -> tilt down
        print(f'err_x: {err_x}')
        print(f'err_y: {err_y}')
    cv2.imshow("Hand tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
