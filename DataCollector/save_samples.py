import os
import time
import cv2

def save_sample(frame, cls_name, save_dir="../logs/high_confidence"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{cls_name}_{timestamp}.jpg"
    cv2.imwrite(os.path.join(save_dir, filename), frame)
