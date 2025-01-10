import cv2
import requests
import numpy as np
import time

def fetch_and_resize_image(url, size):
    """
    Fetch an image from a URL and resize it for model input.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        resized_frame = cv2.resize(frame, size)
        return frame, resized_frame
    else:
        raise ValueError(f"Failed to fetch image. Status code: {response.status_code}")

def draw_bounding_boxes(frame, objects, labels):
    """
    Draw bounding boxes and labels on the frame.
    """
    start_time = time.time()
    for obj in objects:
        bbox = obj.bbox
        label = labels.get(obj.id, "Unknown")
        score = obj.score
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (bbox.xmin, bbox.ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    draw_time = (time.time() - start_time) * 1000
    return frame, draw_time
