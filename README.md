# Coral-based Object Detection Inference Service

This project provides a simple and lightweight object detection inference service designed to work in conjunction with Home Assistant. It serves as the AI-powered second-tier in a two-tier approach for camera monitoring:

1. **Simple Motion Detection:** Handled by a custom Home Assistant component ([simple_motion_detector](https://github.com/graus/simple_motion_detector/)) that detects motion.
2. **Object Detection:** Once motion is detected, this service uses a Coral USB Accelerator to run efficient object detection on the camera snapshots.

> **Note:** This service runs as a standalone Python program (using Python 3.9 due to the `pycoral` package constraints) and is configured to run as a systemd service. It cannot be deployed as a custom component inside Home Assistant directly.

## Features

- **Efficient Inference:** Utilizes the Coral USB Accelerator with TensorFlow Lite models for high-speed object detection.
- **MQTT Integration:** Listens for trigger messages and publishes detailed detection results.
- **Two-Tier Approach:** Complements a simple motion detection system in Home Assistant, handling intensive object detection only when motion is detected.
- **Customizable and Lightweight:** Designed for resource efficiency and ease of integration with existing Home Assistant setups.

## Project Structure

```
.
├── config.py
├── inference_service.py
├── README.md
├── coral_models/
│   └── efficientdet_lite3/
│       ├── efficientdet_lite3_512_ptq_edgetpu.tflite
│       └── coco_labels.txt
└── utils
    ├── image_processing.py
    ├── logging_setup.py
    └── model_handling.py
└── mqtt_handler.py
```

### Key Files

- **inference_service.py**  
  The main entry point for the inference service. It initializes logging, loads the Edge TPU model, listens for MQTT messages, processes images for inference, and publishes detection results.

- **config.py**  
  Contains configuration parameters such as the MQTT broker settings, paths to the model and labels, and the confidence threshold.

- **utils/model_handling.py**  
  Provides functions for loading the TensorFlow Lite model and running inference on image data.

- **utils/image_processing.py**  
  Contains functions for drawing bounding boxes and overlaying labels on the images based on detection results.

- **mqtt_handler.py**  
  Handles MQTT client setup and subscription management.

## Prerequisites

- **Python 3.9**  
  *Required for compatibility with the `pycoral` package.*

- **Hardware:**  
  - Coral USB Accelerator ([https://coral.ai/products/accelerator](https://coral.ai/products/accelerator)) is required to run the AI-based object detection inference efficiently.
  - A supported camera that provides snapshots (saved temporarily on disk).

- **Software Dependencies:**
  - **OpenCV:** For image reading, processing, and annotation.
  - **NumPy:** For array manipulations.
  - **pycoral:** For Edge TPU integration and object detection.  
  - **MQTT Broker:** Ensure that an MQTT broker (such as Mosquitto) is running and that its connection details are set in `config.py`.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/graus/simple_cv_detector.git
   cd simple_cv_detector
   ```

2. **Install Dependencies:**
   It is recommended to use a virtual environment:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare Model and Labels:**
   - Ensure the model file (`efficientdet_lite3_512_ptq_edgetpu.tflite`) and the label file (`coco_labels.txt`) are in the correct paths as specified in `config.py`.
   - You can use any compatible model from Coral's [object detection models repository](https://coral.ai/models/object-detection/) by updating the model and label paths in `config.py`.

## Usage

1. **Configure the Service:**
   - Open `config.py` and adjust the following settings as needed:
     - **MQTT_BROKER**, **MQTT_PORT**, **MQTT_TOPIC**
     - **MODEL_PATH** and **LABELS_PATH**
     - **CONFIDENCE_THRESHOLD**

2. **Start the Service as a Systemd Service or Standalone Script:**
   ```bash
   python inference_service.py
   ```
   You should see output indicating that the model has been loaded and that the service is listening on the configured MQTT trigger topic.

3. **Trigger Object Detection:**
   - Publish a JSON message to the MQTT topic (e.g., `object_detection/state/trigger`) to trigger inference.  
     Example payload:
     ```json
     {
       "camera_id": "camera1"
     }
     ```
   - The service will:
     - Load a snapshot from `/tmp/snapshot_camera1.jpg`.
     - Run object detection using the Coral USB Accelerator.
     - Annotate the image with bounding boxes and labels (saved as `/tmp/output_with_bboxes_camera1.jpg`).
     - Publish the detection results to the MQTT results topic (e.g., `object_detection/state/results`).

## MQTT Message Flow

- **Trigger Message:**  
  - **Topic:** `object_detection/state/trigger` (modifiable via `config.py`).
  - **Payload:** JSON containing a `camera_id` key.

- **Results Message:**  
  - **Topic:** `object_detection/state/results`
  - **Payload Structure:**
    ```json
    {
      "camera_id": "camera1",
      "objects": [
        {
          "label": "person",
          "score": 0.97,
          "bbox": [xmin, ymin, xmax, ymax]
        }
      ],
      "total_objects": 1,
      "image_path": "/tmp/output_with_bboxes_camera1.jpg"
    }
    ```

## Integration with Home Assistant

- **Motion Detection Trigger:**  
  The service is designed to work with the [simple_motion_detector](https://github.com/graus/simple_motion_detector/) component for Home Assistant. The custom component detects motion in camera feeds, and can be used in an automation to publish a trigger to the MQTT topic.

- **Systemd Service:**  
  Since the service runs on Python 3.9 and relies on system-level libraries and hardware (Coral USB Accelerator), it is deployed as a standalone systemd service rather than as a Home Assistant custom component.  

  *Example systemd service file:*
  ```ini
  [Unit]
  Description=Coral Object Detection Inference Service
  After=network.target

  [Service]
  User=your_username
  WorkingDirectory=/path/to/your/project
  ExecStart=/path/to/venv/bin/python inference_service.py
  Restart=always

  [Install]
  WantedBy=multi-user.target
  ```
  Adjust the paths and user as necessary.

## Acknowledgements

- [PyCoral](https://coral.ai/software/) – For providing the libraries and support for Edge TPU integration.
- [Home Assistant Community](https://www.home-assistant.io/) – For continuous inspiration in smart home automation.