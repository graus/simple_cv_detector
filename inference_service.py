import json
import cv2
from config import (MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CAMERA_URL,
                    CONFIDENCE_THRESHOLD, OUTPUT_IMAGE_PATH)
from utils.logging_setup import setup_logging
from utils.model_handling import load_model, run_inference
from utils.image_processing import fetch_and_resize_image, draw_bounding_boxes
from mqtt_handler import setup_mqtt

# Set up logging
setup_logging()

# Load the model once
interpreter, size, model_load_time = load_model("coral_models/efficientdet_lite0/efficientdet_lite0_320_ptq_edgetpu.tflite")
print(f"Model loaded in {model_load_time:.2f} ms.")

# Load labels
with open("coral_models/efficientdet_lite0/coco_labels.txt", "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

# Initialize MQTT client
mqtt_client = setup_mqtt(MQTT_BROKER, MQTT_PORT)

def handle_inference():
    """
    Perform inference when triggered via MQTT.
    """
    try:
        try:
            original_frame, resized_frame = fetch_and_resize_image(CAMERA_URL, size)
            print("Frame fetched successfully!")
        except Exception as e:
            print(f"Error during fetch: {e}")

        # Run inference
        objects, inference_time = run_inference(interpreter, resized_frame, CONFIDENCE_THRESHOLD)

        # Draw bounding boxes if objects are detected
        if objects:
            annotated_frame, draw_time = draw_bounding_boxes(original_frame, objects, labels)
            output_path = OUTPUT_IMAGE_PATH
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved as {output_path}")

        # Publish results to MQTT
        results = [{"label": labels.get(obj.id, "Unknown"),
                    "score": obj.score,
                    "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]} for obj in objects]
        payload = {"objects": results, "total_objects": len(results)}
        mqtt_client.publish(f"{MQTT_TOPIC}/results", json.dumps(payload), retain=False)
        print(f"Published: {payload}")

    except Exception as e:
        print(f"Error: {e}")

def on_message(client, userdata, message):
    """
    Handle incoming MQTT messages to trigger inference.
    """
    print(f"Received message on topic {message.topic}: {message.payload.decode()}")
    if message.topic == f"{MQTT_TOPIC}/trigger":
        handle_inference()

if __name__ == "__main__":
    # Set up MQTT subscription
    mqtt_client.subscribe(f"{MQTT_TOPIC}/trigger")
    mqtt_client.on_message = on_message

    print("Inference service is running. Waiting for triggers...")
    mqtt_client.loop_forever()
