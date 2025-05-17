import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_predictions.log"),  # Logs to a file
        logging.StreamHandler()  # Logs to the console
    ]
)

def run_predictions(model_path: str, test_images_path: str, output_folder: str, output_name: str):
    """
    Run predictions on test images and save outputs.

    Args:
        model_path (str): Path to the YOLO model file.
        test_images_path (str): Path to the folder containing test images.
        output_folder (str): Folder to save the output results.
        output_name (str): Name of the output folder for results.
    """
    logging.info("Loading YOLO model...")
    model = YOLO(model_path)

    logging.info(f"Running predictions on test images in: {test_images_path}")
    results = model.predict(
        source=test_images_path,
        save=True,
        project=output_folder,
        name=output_name
    )

    logging.info("Predictions completed.")
    logging.info(f"Results saved in: {output_folder}/{output_name}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/home/hasith/Personal/DriveEye/Training/runs/detect/train/weights/best.pt"  # Path to the best model
    TEST_IMAGES_PATH = "/home/hasith/Personal/DriveEye/Dataset/test/images"
    OUTPUT_FOLDER = "/home/hasith/Personal/DriveEye/Training/runs/detect/test_predictions"
    OUTPUT_NAME = "test_results"

    run_predictions(MODEL_PATH, TEST_IMAGES_PATH, OUTPUT_FOLDER, OUTPUT_NAME)