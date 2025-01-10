import json
import paho.mqtt.client as mqtt
import cv2

from main import draw_bounding_boxes, perform_detection, fetch_and_preprocess_frame, load_labels, interpreter, size

# Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "object_detection/state"
CAMERA_URL = "http://localhost:1984/api/frame.jpeg?src=webrtc_camera_achter"
MODEL_PATH = "coral_models/efficientdet_lite0/efficientdet_lite0_320_ptq_edgetpu.tflite"
LABELS_PATH = "coral_models/efficientdet_lite0/coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.3

# Load model and labels
labels = load_labels(LABELS_PATH)

# Initialize MQTT client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
print(mqtt_client)


if __name__ == "__main__":

    try:
        original_frame, resized_frame = fetch_and_preprocess_frame(CAMERA_URL, size)
        objects, inference_time = perform_detection(interpreter, resized_frame)
        
        if objects:
            annotated_frame, draw_time = draw_bounding_boxes(original_frame, objects, labels)
            output_path = "output_with_bboxes_detected.jpg"
            cv2.imwrite(output_path, annotated_frame)
        
        results = [
            {
                "label": labels.get(obj.id, "Unknown"),
                "score": obj.score,
                "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax],
            }
            for obj in objects
        ]
        payload = {
            "objects": results,
            "total_objects": len(results),
        }

        # Publish results to MQTT
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), retain=False)
        print(f"Published: {payload}")

    except Exception as e:
        print(f"Error: {e}")
