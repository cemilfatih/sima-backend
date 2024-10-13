import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import sqlite3
from deepface import DeepFace

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (only once when the server starts)

# Emotion labels (based on your model's output classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database setup
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            emotion1 TEXT NOT NULL,
            weight1 REAL NOT NULL,
            emotion2 TEXT NOT NULL,
            weight2 REAL NOT NULL,
            filename TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

# Function to check allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or could not be read")

    # Detect faces in the original image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) == 0:
        raise ValueError("No faces found in the image")

    # Get the first detected face
    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]  # Use the original color image

    # Resize the face ROI to 48x48
    resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    normalized_face = resized_face / 255.0  # Normalize the pixel values

    # Reshape for model input

    return resized_face

    

# Function to predict emotion
def predict_emotion(image_path):
    # Preprocess the image
    print("Preprocessing image")
    input_image = preprocess_image(image_path)
    print("Preprocessing done")
    print(input_image.shape)
    result = DeepFace.analyze(input_image, actions=['emotion'], enforce_detection=False)
    print(result)
    emotion = max(result[0]['emotion'], key=result[0]['emotion'].get)
    emotion_weight = result[0]['emotion'][emotion]
    print(emotion)
    # Get the second highest emotion
    print("Removing the first emotion")
    del result[0]['emotion'][emotion]  # Remove the first emotion
    emotion2 = max(result[0]['emotion'], key=result[0]['emotion'].get)
    emotion_weight2 = result[0]['emotion'][emotion2]
    print(emotion2)


    print(type(emotion_weight))

    return {
        "result": {
            "emotion_result1": emotion,
            "weight_result1": float(emotion_weight),  # Convert to float for JSON serializability
            "emotion_result2": emotion2,
            "weight_result2": float(emotion_weight2),
        }
    }

# Function to save predictions to the database
def save_to_db(emotion1, weight1, emotion2, weight2, filename):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    # Insert the prediction into the database
    c.execute('''
        INSERT INTO predictions (emotion1, weight1, emotion2, weight2, filename)
        VALUES (?, ?, ?, ?, ?)
    ''', (emotion1, weight1, emotion2, weight2, filename))

    conn.commit()
    conn.close()

# Route for handling file uploads and emotion prediction
@app.route('/predict_emotion', methods=['POST'])
def upload_image():
    print("Request received")
    
    # Check if the request contains the image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    print("File received")
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(os.getcwd(), filename)  # Save it in the current directory
        
        # Save the uploaded image to the current directory
        file.save(filepath)
        
        # Predict emotion
        try:
            predicted_emotions = predict_emotion(filepath)
            print("here")
            print(predicted_emotions)  # Print the predicted emotion
            
            # Save predictions to the database
            save_to_db(
                predicted_emotions['result']['emotion_result1'],
                predicted_emotions['result']['weight_result1'],
                predicted_emotions['result']['emotion_result2'],
                predicted_emotions['result']['weight_result2'],
                filename
            )
            
            return jsonify(predicted_emotions), 200  # Return the result directly
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

# Start the Flask app
if __name__ == '__main__':
    print("Server started")
    app.run(host='0.0.0.0', port=8000)
