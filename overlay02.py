import cv2
import numpy as np
from deepface import DeepFace

# Load the Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the glasses image (with a white or black background to be removed)
glasses_img = cv2.imread('glasses2.png', -1)  # Ensure the correct path is used

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()


def overlay_glasses(frame, glasses_img, x, y, w, h):
    """ Overlay the glasses on the detected eyes, making them 30 units larger. """
    glasses_width = w + 30  # Increase the width by 30 units
    glasses_height = int(glasses_img.shape[0] * (
                glasses_width / glasses_img.shape[1])) + 30  # Increase height proportionally by 30 units

    # Resize the glasses image to the larger size
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # Adjust x
    x = x - 10
    # Adjust the position slightly upwards to keep glasses centered (optional)
    y = max(0, y - 27)

    # Ensure the region of interest (ROI) doesn't exceed frame boundaries
    if y + glasses_height > frame.shape[0]:
        glasses_height = frame.shape[0] - y
    if x + glasses_width > frame.shape[1]:
        glasses_width = frame.shape[1] - x

    # Get the region of interest (ROI) on the face where the glasses will be placed
    roi = frame[y:y + glasses_height, x:x + glasses_width]

    # Resize the glasses image to exactly match the ROI size (if needed)
    resized_glasses = cv2.resize(resized_glasses, (glasses_width, glasses_height))

    # Split the resized glasses image into its color and alpha channels
    glasses_color = resized_glasses[:, :, :3]  # RGB channels
    glasses_alpha = resized_glasses[:, :, 3]  # Alpha channel

    # Create a mask where both white ([255, 255, 255]) and black ([0, 0, 0]) pixels are considered transparent
    white_mask = np.all(glasses_color == [255, 255, 255], axis=-1)
    black_mask = np.all(glasses_color == [0, 0, 0], axis=-1)

    # Combine the white and black masks
    transparent_mask = np.logical_or(white_mask, black_mask)

    # Create the mask and inverse mask using the alpha channel
    mask = glasses_alpha / 255.0
    mask[transparent_mask] = 0  # Set both white and black pixels to be fully transparent
    inv_mask = 1.0 - mask

    # Overlay the glasses on the ROI using the mask
    for c in range(0, 3):
        roi[:, :, c] = (mask * glasses_color[:, :, c] + inv_mask * roi[:, :, c])

    # Put the modified ROI back on the frame
    frame[y:y + glasses_height, x:x + glasses_width] = roi


while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    # If frame not captured correctly, break the loop
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Detect eyes within each face and overlay glasses
    for (x, y, w, h) in faces:
        # Get the region of interest (ROI) for the face in grayscale
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

        if len(eyes) >= 2:  # Overlay glasses only if two eyes are detected
            # Get the coordinates of both eyes
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            # Calculate the midpoint between the eyes and the width to overlay glasses
            eye_center_x = min(ex1, ex2)
            eye_center_y = min(ey1, ey2)
            eye_width = max(ex1 + ew1, ex2 + ew2) - eye_center_x

            # Overlay the glasses on the face
            overlay_glasses(face_color, glasses_img, eye_center_x, ey1, eye_width, eh1)

        # Analyze the face with DeepFace for age detection
        try:
            face_roi = frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['age'], enforce_detection=False)
            age = result['age']

            # Display the age on the frame
            cv2.putText(frame, f"Age: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"DeepFace error: {e}")

    # Display the resulting frame with face, eye detection, and age
    cv2.imshow('Face Detection with Glasses Overlay and Age Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
