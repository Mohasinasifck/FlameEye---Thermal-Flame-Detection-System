import cv2
import time
from ultralytics import YOLO

# ==========================
# LOAD TRAINED MODEL
# ==========================
model = YOLO("FireAndSmoke.pt")   # your trained model

# ==========================
# OPEN WEBCAM
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found")
    exit()

# Camera resolution (optional)
cap.set(3, 1280)
cap.set(4, 720)

print("Press Q to Quit")

# ==========================
# FPS CALCULATION
# ==========================
prev_time = 0

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model(frame, conf=0.5)

    # Draw detections
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 60/(current_time - prev_time)
    prev_time = current_time

    # Show FPS on screen
    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    # Show output
    cv2.imshow("YOLO Realtime Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# RELEASE
# ==========================
cap.release()
cv2.destroyAllWindows()
