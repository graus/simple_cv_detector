MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "object_detection/state"
CAMERA_URL = "http://localhost:1984/api/frame.jpeg?src=webrtc_camera_achter"
MODEL_PATH = "coral_models/efficientdet_lite0/efficientdet_lite0_320_ptq_edgetpu.tflite"
LABELS_PATH = "coral_models/efficientdet_lite0/coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.3
OUTPUT_IMAGE_PATH = "output_with_bboxes.jpg"