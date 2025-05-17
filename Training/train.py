import logging
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  
        logging.StreamHandler()  
    ]
)

def train_model(model_path: str, data_path: str, epochs: int, img_size: int):
    """
    Train the YOLO model with the given parameters.

    Args:
        model_path (str): Path to the YOLO model file.
        data_path (str): Path to the dataset configuration file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
    """
    logging.info("Initializing YOLO model...")
    model = YOLO(model_path)

    logging.info("Starting training...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        save=True,
        verbose=True 
    )

    logging.info("Training completed.")
    logging.info(f"Results: {results}")

if __name__ == "__main__":
    MODEL_PATH = "yolo11s.pt"
    DATA_PATH = "/home/hasith/Personal/DriveEye/Dataset/data.yaml"
    EPOCHS = 200
    IMG_SIZE = 640

    train_model(MODEL_PATH, DATA_PATH, EPOCHS, IMG_SIZE)