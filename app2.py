# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image
# import os
#
# # Define the path where the glasses images are stored
# GLASSES_FOLDER = "glasses"
#
# # Function to load all available glasses from the directory
# def load_glasses_images():
#     glasses_files = [f for f in os.listdir(GLASSES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     glasses_images = {file: cv2.imread(os.path.join(GLASSES_FOLDER, file), cv2.IMREAD_UNCHANGED) for file in glasses_files}
#     return glasses_images
#
# # Function to overlay glasses on detected eyes
# def overlay_glasses(frame, glasses_img, x, y, w, h):
#     glasses_width = w + 30  # Increase the width by 30 units
#     glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1])) + 30  # Adjust height
#
#     # Resize glasses image to fit
#     resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))
#
#     # Adjust the position slightly upwards to keep glasses centered
#     x= x-10
#     y = max(0, y - 27)
#
#     # Ensure the region of interest (ROI) doesn't exceed frame boundaries
#     if y + glasses_height > frame.shape[0]:
#         glasses_height = frame.shape[0] - y
#     if x + glasses_width > frame.shape[1]:
#         glasses_width = frame.shape[1] - x
#
#     # Extract the region of interest (ROI) from the frame
#     roi = frame[y:y+glasses_height, x:x+glasses_width]
#
#     # Check if the resized glasses image has an alpha channel
#     if resized_glasses.shape[2] == 4:  # Image has an alpha channel
#         # Split the resized glasses image into its color and alpha channels
#         glasses_color = resized_glasses[:, :, :3]
#         glasses_alpha = resized_glasses[:, :, 3]
#
#         # Create the mask and inverse mask
#         mask = glasses_alpha / 255.0
#         inv_mask = 1.0 - mask
#
#         # Resize the mask and colors to match the ROI
#         mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
#         inv_mask = 1.0 - mask
#         glasses_color = cv2.resize(glasses_color, (roi.shape[1], roi.shape[0]))
#
#         # Overlay the glasses on the ROI using the mask
#         for c in range(0, 3):
#             roi[:, :, c] = (mask * glasses_color[:, :, c] + inv_mask * roi[:, :, c])
#
#     else:  # If no alpha channel, just overlay the glasses directly
#         roi_resized = cv2.resize(resized_glasses, (glasses_width, glasses_height))
#         frame[y:y+glasses_height, x:x+glasses_width] = roi_resized
#
# # Load Haar Cascades for face and eye detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#
# # Streamlit app UI
# st.title("Virtual Glasses Try-On")
# st.write("Use the buttons below to control the webcam.")
#
# # Load the available glasses images
# glasses_images = load_glasses_images()
# glasses_names = list(glasses_images.keys())
#
# # Display available glasses options
# selected_glasses = st.selectbox("Choose Glasses", glasses_names)
#
# # Display the selected glasses image as a preview
# selected_glasses_image = Image.open(os.path.join(GLASSES_FOLDER, selected_glasses))
# st.image(selected_glasses_image, caption="Selected Glasses Preview", use_column_width=True)
#
# # Start webcam session state
# if 'webcam_active' not in st.session_state:
#     st.session_state.webcam_active = False
#
# # Function to start the webcam
# def start_webcam():
#     st.session_state.webcam_active = True
#
# # Function to stop the webcam
# def stop_webcam():
#     st.session_state.webcam_active = False
#
# # Buttons to control the webcam
# start_button = st.button("Start Webcam", on_click=start_webcam)
# stop_button = st.button("Stop Webcam", on_click=stop_webcam)
#
# # Create a placeholder for the video feed
# video_placeholder = st.empty()
#
# if st.session_state.webcam_active:
#     # Start webcam feed
#     cap = cv2.VideoCapture(0)
#
#     while st.session_state.webcam_active:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Unable to access the webcam.")
#             break
#
#         # Convert the frame to RGB
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Convert image to grayscale for face and eye detection
#         gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#
#         # Detect faces
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
#
#         # Get the selected glasses image
#         glasses_img = glasses_images[selected_glasses]
#
#         # Process each detected face
#         for (x, y, w, h) in faces:
#             face_gray = gray[y:y+h, x:x+w]
#             face_color = img_rgb[y:y+h, x:x+w]
#
#             # Detect eyes within the face
#             eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
#
#             if len(eyes) >= 2:
#                 # Get the coordinates of the eyes
#                 (ex1, ey1, ew1, eh1) = eyes[0]
#                 (ex2, ey2, ew2, eh2) = eyes[1]
#
#                 # Calculate the midpoint between the eyes and the width for the glasses
#                 eye_center_x = min(ex1, ex2)
#                 eye_width = max(ex1 + ew1, ex2 + ew2) - eye_center_x
#
#                 # Overlay the glasses on the face
#                 overlay_glasses(face_color, glasses_img, eye_center_x, ey1, eye_width, eh1)
#
#         # Display the image with glasses overlay in Streamlit
#         video_placeholder.image(img_rgb, caption="Glasses Overlay", use_column_width=True)
#
#     # Release the webcam when stopped
#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os

# Define the path where the glasses images are stored
GLASSES_FOLDER = "glasses"


# Function to load all available glasses from the directory
def load_glasses_images():
    glasses_files = [f for f in os.listdir(GLASSES_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    glasses_images = {file: cv2.imread(os.path.join(GLASSES_FOLDER, file), cv2.IMREAD_UNCHANGED) for file in
                      glasses_files}
    return glasses_images


# Function to overlay glasses on detected eyes
# def overlay_glasses(frame, glasses_img, x, y, w, h):
#     glasses_width = w + 30  # Increase the width by 30 units
#     glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1])) + 30  # Adjust height
#
#     # Resize glasses image to fit
#     resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))
#
#     # Adjust the position slightly upwards to keep glasses centered
#     x = x - 10
#     y = max(0, y - 27)
#
#     # Ensure the region of interest (ROI) doesn't exceed frame boundaries
#     if y + glasses_height > frame.shape[0]:
#         glasses_height = frame.shape[0] - y
#     if x + glasses_width > frame.shape[1]:
#         glasses_width = frame.shape[1] - x
#
#     # Extract the region of interest (ROI) from the frame
#     roi = frame[y:y + glasses_height, x:x + glasses_width]
#
#     # Check if the resized glasses image has an alpha channel
#     if resized_glasses.shape[2] == 4:  # Image has an alpha channel
#         # Split the resized glasses image into its color and alpha channels
#         glasses_color = resized_glasses[:, :, :3]
#         glasses_alpha = resized_glasses[:, :, 3]
#
#         # Create the mask and inverse mask
#         mask = glasses_alpha / 255.0
#         inv_mask = 1.0 - mask
#
#         # Resize the mask and colors to match the ROI
#         mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
#         inv_mask = 1.0 - mask
#         glasses_color = cv2.resize(glasses_color, (roi.shape[1], roi.shape[0]))
#
#         # Overlay the glasses on the ROI using the mask
#         for c in range(0, 3):
#             roi[:, :, c] = (mask * glasses_color[:, :, c] + inv_mask * roi[:, :, c])
#
#     else:  # If no alpha channel, just overlay the glasses directly
#         roi_resized = cv2.resize(resized_glasses, (glasses_width, glasses_height))
#         frame[y:y + glasses_height, x:x + glasses_width] = roi_resized
def overlay_glasses(frame, glasses_img, x, y, w, h):
    glasses_width = w + 30  # Increase the width by 30 units
    glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1])) + 30  # Adjust height

    # Resize glasses image to fit
    resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # Adjust the position slightly upwards to keep glasses centered
    x=x-10
    y = max(0, y - 27)

    # Ensure the region of interest (ROI) doesn't exceed frame boundaries
    if y + glasses_height > frame.shape[0]:
        glasses_height = frame.shape[0] - y
    if x + glasses_width > frame.shape[1]:
        glasses_width = frame.shape[1] - x

    # Extract the region of interest (ROI) from the frame
    roi = frame[y:y+glasses_height, x:x+glasses_width]

    # Check if the resized glasses image has an alpha channel (4 channels)
    if resized_glasses.shape[2] == 4:
        # Split the resized glasses image into its color and alpha channels
        glasses_color = resized_glasses[:, :, :3]
        glasses_alpha = resized_glasses[:, :, 3]

        # Resize the glasses and alpha mask to match the ROI
        glasses_color = cv2.resize(glasses_color, (roi.shape[1], roi.shape[0]))
        glasses_alpha = cv2.resize(glasses_alpha, (roi.shape[1], roi.shape[0]))

        # Create the mask and inverse mask
        mask = glasses_alpha / 255.0
        inv_mask = 1.0 - mask

        # Overlay the glasses on the ROI using the mask
        for c in range(0, 3):
            roi[:, :, c] = (mask * glasses_color[:, :, c] + inv_mask * roi[:, :, c])

    else:  # If the image doesn't have an alpha channel, just overlay it directly
        if resized_glasses.shape[2] == 3:
            resized_glasses = cv2.resize(resized_glasses, (roi.shape[1], roi.shape[0]))
            roi[:, :, :] = resized_glasses

    # Place the modified ROI back on the frame
    frame[y:y+glasses_height, x:x+glasses_width] = roi


# Load Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Streamlit app UI
st.title("Virtual Glasses Try-On")
st.write("Use the buttons below to control the webcam.")

# Load the available glasses images
glasses_images = load_glasses_images()
glasses_names = list(glasses_images.keys())

# Display available glasses options with image
selected_glasses = st.selectbox("Choose Glasses", glasses_names)

# Directly display the selected glasses image as a preview
if selected_glasses:
    st.image(os.path.join(GLASSES_FOLDER, selected_glasses), caption="Selected Glasses", use_column_width=True)

# Start webcam session state
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False


# Function to start the webcam
def start_webcam():
    st.session_state.webcam_active = True


# Function to stop the webcam
def stop_webcam():
    st.session_state.webcam_active = False


# Buttons to control the webcam
start_button = st.button("Start Webcam", on_click=start_webcam)
stop_button = st.button("Stop Webcam", on_click=stop_webcam)

# Create a placeholder for the video feed
video_placeholder = st.empty()

if st.session_state.webcam_active:
    # Start webcam feed
    cap = cv2.VideoCapture(0)

    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the webcam.")
            break

        # Convert the frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image to grayscale for face and eye detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Get the selected glasses image
        glasses_img = glasses_images[selected_glasses]

        # Process each detected face
        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            face_color = img_rgb[y:y + h, x:x + w]

            # Detect eyes within the face
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

            if len(eyes) >= 2:
                # Get the coordinates of the eyes
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]

                # Calculate the midpoint between the eyes and the width for the glasses
                eye_center_x = min(ex1, ex2)
                eye_width = max(ex1 + ew1, ex2 + ew2) - eye_center_x

                # Overlay the glasses on the face
                overlay_glasses(face_color, glasses_img, eye_center_x, ey1, eye_width, eh1)

        # Display the image with glasses overlay in Streamlit
        video_placeholder.image(img_rgb, caption="Glasses Overlay", use_column_width=True)

    # Release the webcam when stopped
    cap.release()
  #  cv2.destroyAllWindows()
