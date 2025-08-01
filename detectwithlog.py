import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("runs/detect/yolov8_custom/weights/best.pt")

# Open video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0

# Output video writer
out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
)

# Open log file
log_file = open("detections_log.txt", "w")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, verbose=False)[0]
    annotated_frame = frame.copy()

    # Store detections in a list for this frame
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detections.append(f"{label} ({conf:.2f})")


    timestamp = frame_number / fps
    if detections:
        line = f"Frame {frame_number}, Time {timestamp:.2f}s: " + ", ".join(detections)
    else:
        line = f"Frame {frame_number}, Time {timestamp:.2f}s: No objects detected"


    log_file.write(line + "\n")
    print(line)


    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(annotated_frame)
    frame_number += 1


cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()

print("✅ Detection and logging complete.")

