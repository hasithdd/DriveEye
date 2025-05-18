from ultralytics import YOLO

model = YOLO("../Training/runs/detect/train/weights/best.pt") 

model.export(format="engine")  

tensorrt_model = YOLO("best.engine")


