from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, engine_path="../Models/best.engine"):
        self.model = YOLO(engine_path)

    def infer(self, frame):
        return self.model(frame)[0]
