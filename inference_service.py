import json
import cv2
from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC,
    CONFIDENCE_THRESHOLD, MODEL_PATH, LABELS_PATH
)
from utils.logging_setup import setup_logging
from utils.model_handling import load_model, run_inference
from utils.image_processing import draw_bounding_boxes
from mqtt_handler import setup_mqtt

# Initialize logging, model, and MQTT client
setup_logging()
interpreter, size, model_load_time = load_model(MODEL_PATH)
print(f"Model loaded in {model_load_time:.2f} ms.")

with open(LABELS_PATH, "r") as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

mqtt_client = setup_mqtt(MQTT_BROKER, MQTT_PORT)


def handle_inference(camera_id):
    """
    Run object detection on a snapshot for the specified camera.
    """
    snapshot_path = f"/tmp/snapshot_{camera_id}.jpg"
    original_frame = cv2.imread(snapshot_path)

    if original_frame is None:
        print(f"Failed to load snapshot from {snapshot_path}")
        return

    resized_frame = cv2.resize(original_frame, size)
    objects, inference_time = run_inference(interpreter, resized_frame, CONFIDENCE_THRESHOLD)

    if objects:
        annotated_frame, draw_time = draw_bounding_boxes(original_frame, objects, labels)
        output_path = f"/tmp/output_with_bboxes_{camera_id}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated image saved: {output_path}")

    results = [
        {
            "label": labels.get(obj.id, "Unknown"),
            "score": obj.score,
            "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax],
        }
        for obj in objects
    ]
    payload = {
        "camera_id": camera_id,
        "objects": results,
        "total_objects": len(results),
        "image_path": f"/tmp/output_with_bboxes_{camera_id}.jpg",
    }
    mqtt_client.publish(f"{MQTT_TOPIC}/results", json.dumps(payload), retain=False)
    print(f"Published: {payload}")


def on_message(client, userdata, message):
    """
    Process incoming MQTT messages and trigger inference.
    """
    try:
        payload = json.loads(message.payload.decode())
        camera_id = payload.get("camera_id")

        if not camera_id:
            print("No camera_id provided. Skipping inference.")
            return

        print(f"Received message on topic {message.topic}: {payload}")
        handle_inference(camera_id)

    except json.JSONDecodeError:
        print("Invalid MQTT payload: Unable to decode JSON.")
    except Exception as e:
        print(f"Error in on_message: {e}")


if __name__ == "__main__":
    mqtt_client.subscribe(f"{MQTT_TOPIC}/trigger")
    mqtt_client.on_message = on_message

    print(f"Inference service running. Listening on {MQTT_TOPIC}/trigger...")
    mqtt_client.loop_forever()
