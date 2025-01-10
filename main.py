import requests
import numpy as np
import cv2
import logging
import time
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Measure model loading time
start_time = time.time()
model_path = "coral_models/efficientdet_lite0/efficientdet_lite0_320_ptq_edgetpu.tflite"
logging.info(f"Loading model: {model_path}")
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
model_load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
logging.info(f"Model loaded and interpreter initialized in {model_load_time:.2f} ms.")
size = input_size(interpreter)


def load_labels(path):
    """
    Load labels from a file into a dictionary.

    Args:
        path (str): Path to the labels file.

    Returns:
        dict: Dictionary mapping label indices to names.
    """
    start_time = time.time()
    logging.info(f"Loading labels from: {path}")
    with open(path, "r") as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}
    label_load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    logging.info(f"Loaded {len(labels)} labels in {label_load_time:.2f} ms.")
    return labels


def fetch_and_preprocess_frame(url, size):
    """
    Fetch an image from a URL and preprocess it for model input.

    Args:
        url (str): URL of the image.
        size (tuple): Desired width and height for resizing.

    Returns:
        tuple: Original frame and resized frame.
    """
    logging.info(f"Fetching frame from URL: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        resized_frame = cv2.resize(frame, size)
        logging.info(f"Frame fetched and resized to: {size}")
        return frame, resized_frame
    else:
        logging.error(f"Failed to fetch image. Status code: {response.status_code}")
        raise ValueError(f"Failed to fetch image. Status code: {response.status_code}")


def perform_detection(interpreter, frame, threshold=0.25):
    """
    Run object detection on a preprocessed frame.

    Args:
        interpreter (Interpreter): TensorFlow Lite interpreter.
        frame (ndarray): Preprocessed image.
        threshold (float): Confidence threshold for detected objects.

    Returns:
        tuple: List of detected objects and inference time in milliseconds.
    """
    logging.info(f"Running inference with threshold: {threshold}")
    start_time = time.time()
    input_data = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    objects = get_objects(interpreter, threshold)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    logging.info(f"Inference completed in {inference_time:.2f} ms. Detected {len(objects)} objects.")
    return objects, inference_time


def draw_bounding_boxes(frame, results, labels):
    """
    Draw bounding boxes and labels on an image.

    Args:
        frame (ndarray): Original image.
        results (list): List of detected objects.
        labels (dict): Dictionary of label indices to names.

    Returns:
        tuple: Annotated frame and time taken to draw bounding boxes in milliseconds.
    """
    logging.info("Drawing bounding boxes on the frame.")
    start_time = time.time()
    for obj in results:
        bbox = obj.bbox
        label = labels.get(obj.id, "Unknown")
        score = obj.score
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        label_text = f"{label} ({score:.2f})"
        cv2.putText(frame, label_text, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    draw_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    logging.info(f"Bounding boxes drawn in {draw_time:.2f} ms.")
    return frame, draw_time


if __name__ == "__main__":
    """
    Main function to perform object detection on a camera feed.
    """

    logging.info(f"Expected input size for the model: {size}")

    try:
        original_frame, resized_frame = fetch_and_preprocess_frame(url, size)
        results, inference_time = perform_detection(interpreter, resized_frame)

        if results:
            annotated_frame, draw_time = draw_bounding_boxes(original_frame, results, labels)
            output_path = "output_with_bboxes.jpg"
            cv2.imwrite(output_path, annotated_frame)
            logging.info(f"Annotated image saved as {output_path}")
            logging.info(f"Total processing time (inference + drawing): {inference_time + draw_time:.2f} ms.")
        else:
            logging.info("No objects detected.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
