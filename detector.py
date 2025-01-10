import json
import cv2
from config import (MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CAMERA_URL, MODEL_PATH,
                    LABELS_PATH, CONFIDENCE_THRESHOLD, OUTPUT_IMAGE_PATH)
from utils.logging_setup import setup_logging
from utils.model_handling import load_model, run_inference
from utils.image_processing import fetch_and_resize_image, draw_bounding_boxes
from mqtt_handler import setup_mqtt

# Set up logging
setup_logging()

# Load model and labels
interpreter, size, model_load_time = load_model(MODEL_PATH)
print(f"Model loaded in {model_load_time:.2f} ms.")
with open(LABELS_PATH, "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Set up MQTT
mqtt_client = setup_mqtt(MQTT_BROKER, MQTT_PORT)

if __name__ == "__main__":
    try:
        # Fetch and preprocess frame
        original_frame, resized_frame = fetch_and_resize_image(CAMERA_URL, size)

        # Run inference
        objects, inference_time = run_inference(interpreter, resized_frame, CONFIDENCE_THRESHOLD)

        # Draw bounding boxes if objects detected
        if objects:
            annotated_frame, draw_time = draw_bounding_boxes(original_frame, objects, labels)
            cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)
            print(f"Annotated image saved as {OUTPUT_IMAGE_PATH}")

        # Publish results to MQTT
        results = [{"label": labels.get(obj.id, "Unknown"),
                    "score": obj.score,
                    "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]} for obj in objects]
        payload = {"objects": results, "total_objects": len(results)}
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), retain=False)
        print(f"Published: {payload}")

    except Exception as e:
        print(f"Error: {e}")