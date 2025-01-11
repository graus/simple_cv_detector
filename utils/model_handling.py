import time
import numpy as np
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter


def load_model(model_path):
    """
    Load and initialize the Edge TPU model.

    Args:
        model_path (str): Path to the TensorFlow Lite model file.

    Returns:
        tuple: Initialized interpreter, input size (width, height), and load time in ms.
    """
    start_time = time.time()
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    load_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return interpreter, input_size(interpreter), load_time


def run_inference(interpreter, frame, threshold=0.3):
    """
    Perform object detection inference on the given frame.

    Args:
        interpreter (Interpreter): Initialized Edge TPU interpreter.
        frame (ndarray): Preprocessed image for inference.
        threshold (float): Confidence threshold for detected objects.

    Returns:
        tuple: List of detected objects and inference time in ms.
    """
    start_time = time.time()
    input_data = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    objects = get_objects(interpreter, threshold)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return objects, inference_time
