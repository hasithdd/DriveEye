from ultralytics import YOLO
import os
import shutil

# Paths
pt_path = "./Training/runs/detect/weights/best.pt"
engine_name = "best.engine"
engine_copy_path = "./Models/"

# Check if model exists
if not os.path.exists(pt_path):
    raise FileNotFoundError(f"Model not found at: {pt_path}")

# Export model to TensorRT (creates best.engine in current dir)
print(f"Exporting {pt_path} to TensorRT...")
model = YOLO(pt_path)
model.export(format="engine")

# Copy engine to Models/ folder
if os.path.exists(engine_name):
    os.makedirs("Models", exist_ok=True)
    shutil.copy(engine_name, engine_copy_path)
    print(f"Copied {engine_name} to {engine_copy_path}")
else:
    raise FileNotFoundError(f"Export failed: {engine_name} not found.")

# Optional: Test the engine on a video
print("🧪 Running test inference...")
tensorrt_model = YOLO(engine_copy_path)
results = tensorrt_model("Training/testing.mp4", conf=0.5, show=True)

results.save("output_results.mp4")
print("Inference complete. Saved: output_results.mp4")
