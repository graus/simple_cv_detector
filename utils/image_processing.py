import cv2
import time


def draw_bounding_boxes(frame, objects, labels):
    """
    Draw bounding boxes and labels on the given frame.

    Args:
        frame (ndarray): Original image frame.
        objects (list): List of detected objects with bounding box details.
        labels (dict): Mapping of label indices to names.

    Returns:
        tuple: Annotated frame and time taken to draw bounding boxes in ms.
    """
    start_time = time.time()
    for obj in objects:
        bbox = obj.bbox
        label = labels.get(obj.id, "Unknown")
        score = obj.score

        # Draw bounding box and label
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        label_text = f"{label} ({score:.2f})"
        cv2.putText(frame, label_text, (bbox.xmin, bbox.ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    draw_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return frame, draw_time
