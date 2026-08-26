import cv2

# Download once: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (320, 320),
    score_threshold=0.6
)

cap = cv2.VideoCapture(0)  # 0 is the built-in FaceTime camera on Mac

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, bw, bh = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cx, cy = x + bw // 2, y + bh // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imshow("Face tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
