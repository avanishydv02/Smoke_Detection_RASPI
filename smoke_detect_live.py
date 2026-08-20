import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\shard\OneDrive\Desktop\Smoke Detect\best_saved_model\best_float16.tflite")  # Path to your converted TFLite model
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame (resize and normalize as needed for your model)
    input_shape = input_details[0]['shape']
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
    frame_normalized = np.expand_dims(frame_resized / 255.0, axis=0).astype(np.float32)

    # Set the tensor to the input frame
    interpreter.set_tensor(input_details[0]['index'], frame_normalized)

    # Run the inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process and display the results (add your own logic here)
    # For now, we'll just show the raw frame for simplicity
    cv2.imshow("YOLOv8 Live Detection (TFLite)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
