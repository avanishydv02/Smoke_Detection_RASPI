# Smoke Detection on Raspberry Pi

This project implements real-time smoke detection using a TensorFlow Lite model on a Raspberry Pi. It is designed to process a live video feed from a camera module or webcam and run efficient edge inference.

## Project Structure

* `best_nano.pt`: Trained PyTorch YOLOv8 model checkpoint.
* `best_float16.tflite`: Converted TensorFlow Lite model with float16 quantization for edge device deployment.
* `smoke_detect_live.py`: Main script to capture camera frames, pre-process them, and run real-time inference using the TFLite model.
* `tflite_extract.py`: A helper script to test webcam connectivity and functionality.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install opencv-python numpy tensorflow
```

Note: For lightweight deployment on a Raspberry Pi, you can install `tflite-runtime` instead of the full `tensorflow` package.

## Setup and Usage

### 1. Test Camera Connection
Verify that your camera is working and accessible by running:
```bash
python tflite_extract.py
```

### 2. Configure the Model Path
Open `smoke_detect_live.py` and update the `model_path` (line 6) to point to the local `best_float16.tflite` file:
```python
interpreter = tf.lite.Interpreter(model_path="best_float16.tflite")
```

### 3. Run Live Detection
Start the live detection system:
```bash
python smoke_detect_live.py
```
Press `q` on the keyboard to exit the camera window.
