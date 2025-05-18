from ultralytics import YOLO

# Load PyTorch-trained model
model = YOLO("../Training/runs/detect/train/weights/best.pt")

# Export to TensorRT
model.export(format="engine")  # produces best.engine in root directory

# Load and test engine
tensorrt_model = YOLO("../Models/best.engine")
results = tensorrt_model("../Training/testing.mp4", conf=0.5, show=True)

results.save("output_results.mp4")
results.show()
