import time
from pycoral.adapters.detect import get_objects
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter

def load_model(model_path):
    """
    Load and initialize the Edge TPU model.
    """
    start_time = time.time()
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    load_time = (time.time() - start_time) * 1000
    return interpreter, input_size(interpreter), load_time

def run_inference(interpreter, frame, threshold=0.3):
    """
    Perform object detection inference on the given frame.
    """
    import numpy as np
    start_time = time.time()
    input_data = np.expand_dims(frame, axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    objects = get_objects(interpreter, threshold)
    inference_time = (time.time() - start_time) * 1000
    return objects, inference_time
