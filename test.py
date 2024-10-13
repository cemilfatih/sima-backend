import cv2
import numpy as np
from deepface import DeepFace

# Load the pre-trained model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or could not be read")

    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
        raise ValueError("No faces found in the image")

    # Get the first detected face
    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]  # Keep the color image

    # Resize the face ROI to 48x48
    resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    return resized_face  # Return the resized face image

# Function to predict emotion
def predict_emotion(image_path):
    # Preprocess the image
    input_image = preprocess_image(image_path)

    print("Preprocessing done")
    print(input_image.shape)

    try:
        # Analyze the face
        result = DeepFace.analyze(input_image, actions=['emotion'], enforce_detection=False)
        emotion = max(result[0]['emotion'], key=result[0]['emotion'].get)
    except ValueError as e:
        print(f"Error during analysis: {e}")
        emotion = "No Emotion Detected"

    return emotion

# Example: Test the model on an image
image_path = 'test.jpg'  # Path to your test image
predicted_emotion = predict_emotion(image_path)
print(f'Predicted Emotion: {predicted_emotion}')
