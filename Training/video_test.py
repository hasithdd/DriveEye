# import cv2

# from ultralytics import YOLO

# model = YOLO("/home/hasith/Personal/DriveEye/Training/runs/detect/train/weights/best.pt")

# video_path = "/home/hasith/Personal/DriveEye/Training/onedirection_h264.mp4"
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         results = model(frame)

#         annotated_frame = results[0].plot()

#         cv2.imshow("YOLO Inference", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO
# import time

# model = YOLO("/home/hasith/Personal/DriveEye/Training/runs/detect/train/weights/best.pt")

# video_path = "/home/hasith/Personal/DriveEye/Training/onedirection_h264.mp4"
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# frames_threshold = int(fps * 3)  # 3 seconds worth of frames

# # Track how many consecutive frames each class is detected
# class_frame_counts = {}

# # File to log detections
# log_file = "detection_log.txt"

# def log_message(message):
#     print(message)
#     with open(log_file, "a") as f:
#         f.write(message + "\n")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     results = model(frame)
#     detections = results[0]

#     detected_classes_this_frame = set()

#     for det in results[0].boxes.data.tolist():
#         x1, y1, x2, y2, conf, cls_id = det
#         conf = float(conf)
#         cls_id = int(cls_id)
#         cls_name = results[0].names[cls_id]

#         # Filter out "Safe Driving" and confidence threshold
#         if conf > 0.50 and cls_name != "Safe Driving":
#             detected_classes_this_frame.add(cls_name)
#             class_frame_counts[cls_name] = class_frame_counts.get(cls_name, 0) + 1

#             # Log only when detection reaches 3 seconds continuously
#             if class_frame_counts[cls_name] == frames_threshold:
#                 timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#                 message = f"[{timestamp}] ALERT: '{cls_name}' detected continuously for 3 seconds (conf > 0.50)"
#                 log_message(message)

#     # Reset counts for classes not detected in this frame
#     for cls in list(class_frame_counts.keys()):
#         if cls not in detected_classes_this_frame:
#             class_frame_counts[cls] = 0

#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLO Inference", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
import time
import os

model = YOLO("/home/hasith/Personal/DriveEye/Training/runs/detect/train/weights/best.pt")

video_path = "/home/hasith/Personal/DriveEye/Training/testing.mp4"
# web_cam = 0  # Use 0 for the default webcam
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frames_threshold = int(fps * 3)  # 3 seconds worth of frames

class_frame_counts = {}
log_file = "detection_log.txt"

# Create directory for misbehaviours if not exists
misbehaviour_dir = "misbehaviours"
os.makedirs(misbehaviour_dir, exist_ok=True)

def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    detections = results[0]

    detected_classes_this_frame = set()

    for det in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = det
        conf = float(conf)
        cls_id = int(cls_id)
        cls_name = results[0].names[cls_id]

        if conf > 0.50 and cls_name != "Safe Driving":
            detected_classes_this_frame.add(cls_name)
            class_frame_counts[cls_name] = class_frame_counts.get(cls_name, 0) + 1

            # When threshold reached, log and save image once
            if class_frame_counts[cls_name] == frames_threshold:
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                message = f"[{timestamp}] ALERT: '{cls_name}' detected continuously for 3 seconds (conf > 0.50)"
                log_message(message)

                # Save image with timestamp and class name
                filename = f"{cls_name.replace(' ', '_')}_{timestamp}.jpg"
                filepath = os.path.join(misbehaviour_dir, filename)
                cv2.imwrite(filepath, frame)
                
    # Reset counts for classes not detected in this frame
    for cls in list(class_frame_counts.keys()):
        if cls not in detected_classes_this_frame:
            class_frame_counts[cls] = 0

    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
