import requests
import numpy as np
import cv2
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import input_size
from pycoral.adapters.classify import get_classes
from pycoral.utils.edgetpu import make_interpreter

# Load the model
model_path = "coral_models/efficientdet_lite0/efficientdet_lite0_320_ptq_edgetpu.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Load labels from a file
def load_labels(path):
    with open(path, "r") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

labels = load_labels("coral_models/efficientdet_lite0/coco_labels.txt")

# Fetch and preprocess the frame
def fetch_and_preprocess_frame(url, size):
    # Fetch the image
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Convert to a NumPy array
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize the frame to the required size
        resized_frame = cv2.resize(frame, size)

        # Convert to a 3D array and return
        return resized_frame
    else:
        raise ValueError(f"Failed to fetch image. Status code: {response.status_code}")

# Perform inference
def perform_inference(interpreter, frame):
    # Preprocess input
    input_data = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # Get the classification results
    return get_classes(interpreter, top_k=25)

def perform_detection(interpreter, frame, threshold=0.5):
    # Preprocess input
    input_data = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # Extract detection results
    objects = get_objects(interpreter, threshold)
    return objects


# Main loop
if __name__ == "__main__":
    # Set the camera feed URL
    url = "http://localhost:1984/api/frame.jpeg?src=webrtc_camera_voor"
    size = input_size(interpreter)

    try:
        # Fetch and preprocess the frame
        frame = fetch_and_preprocess_frame(url, size)

        # Perform inference
        results = perform_detection(interpreter, frame)
        print(results)

        # Display results
        print("Detected classes:")
        #for result in results:
        #    label = labels.get(result.id, "Unknown")
        #    print(f"Class: {label}, Score: {result.score:.2f}")

        for obj in results:
            bbox = obj.bbox  # Bounding box coordinates
            label = labels.get(obj.id, "Unknown")
            print(f"Detected {label} with confidence {obj.score:.2f} at {bbox}")

    except Exception as e:
        print(f"Error: {e}")

