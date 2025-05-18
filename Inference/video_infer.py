import cv2
import time
from collections import deque
from Inference.inference import InferenceEngine
from Logger.incident_logger import log_incident
from DataCollector.save_samples import save_sample

def run_inference():
    CONF_THRESHOLD = 0.5
    FPS = 15  # Adjust based on camera
    SECS_THRESHOLD = 3
    FRAME_THRESHOLD = int(FPS * SECS_THRESHOLD)

    buffer = deque(maxlen=FRAME_THRESHOLD)
    engine = InferenceEngine("./Models/best.engine")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = engine.infer(frame)
        boxes = results.boxes
        drowsy_this_frame = False

        for det in boxes.data.tolist():
            _, _, _, _, conf, cls_id = det
            if conf < CONF_THRESHOLD:
                continue
            cls_name = results.names[int(cls_id)]

            if cls_name == "drowsy":
                drowsy_this_frame = True

            if conf > 0.95:
                save_sample(frame, cls_name)

        buffer.append(drowsy_this_frame)

        if buffer.count(True) == FRAME_THRESHOLD:
            log_incident(frame, "drowsy")
            buffer.clear()

        cv2.imshow("DriveEye - Inference", results.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
