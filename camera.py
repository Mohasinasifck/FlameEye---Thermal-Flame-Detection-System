import cv2
from ultralytics import YOLO
import torch

# 🔥 Load model
model = YOLO("FireAndSmoke.pt")
model.to("cuda")

print("Classes:", model.names)

# 🎥 Camera sources
camera_sources = [0, 1, 3]

camera_names = ["Hall", "Kitchen", "Warehouse"]

caps = []
for src in camera_sources:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Camera {src} not opening")
    caps.append(cap)

while True:
    frames = []
    valid_indices = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640))
            frames.append(frame)
            valid_indices.append(i)

    if len(frames) == 0:
        break

    # 🔥 Batch inference
    results = model(frames, conf=0.5)

    for idx, result in zip(valid_indices, results):

        original_frame = frames[valid_indices.index(idx)].copy()
        fire_detected = False

        # 🔥 Draw ONLY fire boxes
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # 🔥 Only Fire (ignore smoke completely)
            if cls_id == 0 and conf > 0.5:
                fire_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(original_frame, "FIRE",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

        # 🔔 Alert
        if fire_detected:
            location = camera_names[idx]
            print(f"🚨 ALERT: Fire in {location}")

            cv2.putText(original_frame,
                        f"ALERT: FIRE IN {location.upper()}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        cv2.imshow(camera_names[idx], original_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

for cap in caps:
    cap.release()

cv2.destroyAllWindows()