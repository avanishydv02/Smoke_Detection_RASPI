import cv2

cap = cv2.VideoCapture(0)  # Try changing to 1 if it doesn't work

if cap.isOpened():
    print("Camera is working!")
    cap.release()  # Release the camera
else:
    print("Camera not found.")