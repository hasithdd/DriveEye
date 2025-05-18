import os
import time
import cv2
import json

def log_incident(frame, cls_name="drowsy", save_dir="../logs/incidents"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    img_path = os.path.join(save_dir, f"{cls_name}_{timestamp}.jpg")
    cv2.imwrite(img_path, frame)

    meta = {"event": cls_name, "timestamp": timestamp}
    with open(img_path.replace(".jpg", ".json"), "w") as f:
        json.dump(meta, f)
