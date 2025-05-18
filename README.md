<div align="center">

# 🚗 DriveEye  
### AI-Powered Driver Monitoring System (Jetson Optimized)

[![Jetson Tested](https://img.shields.io/badge/Jetson-Orin%20NX-green?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)
[![JetPack](https://img.shields.io/badge/JetPack-6.2-blue)](https://developer.nvidia.com/embedded/jetpack)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**Real-time drowsiness detection using YOLOv5 + TensorRT, built for public transport safety.**

</div>

---

## ✨ Features

- 🧠 **Binary YOLOv5 model** (`drowsy` / `not_drowsy`)
- 🔁 **3-second buffer detection logic**
- ⚡ **TensorRT optimized** for Jetson (Orin NX)
- 📷 Live USB/CSI camera feed
- 📝 Incident logging with timestamped frame & metadata
- 🧪 High-confidence data collection for retraining
- 🐳 Jetson-compatible **Docker deployment** using `l4t-ml:r36.4.0`

---

## 🚀 Jetson Setup

### 1. ✅ Flash Jetson Orin NX with JetPack 6.2

Use [NVIDIA SDK Manager](https://developer.nvidia.com/embedded/jetpack) or `flash.sh`.

### 2. 🐳 Pull Docker Image

```bash
sudo docker pull dustynv/l4t-ml:r36.4.0
```

---

## 📦 Project Setup

### Clone Repository

```bash
git clone https://github.com/your-username/DriveEye.git
cd DriveEye
```

### Dataset (Roboflow)

Use [Driver Drowsiness v3](https://universe.roboflow.com/driver-no-yawn/driver-drowsiness1/dataset/3) in **YOLOv5 format**:

```
Dataset1/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

---

## 🏋️ Model Training

```bash
python3 Training/train.py
```

- Output: `Training/runs/detect/weights/best.pt`

---

## 🧪 Testing the Model

```bash
python3 Training/test.py
```

---

## 🧠 Export to TensorRT

```bash
python3 Models/tensorrt_export.py
```

- Outputs: `Models/best.engine`

---

## 🎥 Run Inference

Update camera index if needed in `Inference/video_infer.py`:

```python
cap = cv2.VideoCapture(0)  # Replace 0 if needed
```

Then run:

```bash
python3 main.py --mode inference
```

---

## 📂 Directory Structure

```
DriveEye/
├── Dataset1/
├── Inference/
├── Models/
├── Training/
├── Logger/
├── DataCollector/
├── logs/
│   ├── incidents/
│   └── high_confidence/
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

---

## 🐳 Docker (Jetson Deployment)

```bash
sudo docker build -t driveeye .
sudo docker run -it --runtime nvidia --network host \
  -v $PWD:/app --workdir /app driveeye \
  python3 main.py --mode inference
```

---

## 🧩 Troubleshooting

- 📷 Camera not detected? Try another index:
```bash
v4l2-ctl --list-devices
```

---

## 📌 Roadmap

- ✅ GPS integration
- 📈 MLflow experiment tracking
- ♻️ Retraining pipeline w/ real-world data
- 🚦 Multi-behavior detection (texting, yawning, etc.)

---

## 🤝 Contributing

PRs and issue reports are welcome!

---

## 📜 License

Licensed under the [MIT License](LICENSE).
