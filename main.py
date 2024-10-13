from datetime import datetime
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
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Emotion labels (based on your model's output classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database setup
def init_db():
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    # Create a table to store predictions if it doesn't already exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT NOT NULL PRIMARY KEY,
            emotion1 TEXT NOT NULL,
            weight1 REAL NOT NULL,
            emotion2 TEXT NOT NULL,
            weight2 REAL NOT NULL
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
    input_image = preprocess_image(image_path)

    # Make a prediction
    result = DeepFace.analyze(input_image, actions=['emotion'], enforce_detection=False)

    emotion = max(result[0]['emotion'], key=result[0]['emotion'].get)
    emotion_weight = result[0]['emotion'][emotion]

    del result[0]['emotion'][emotion]  # Remove the first emotion
    emotion2 = max(result[0]['emotion'], key=result[0]['emotion'].get)
    emotion_weight2 = result[0]['emotion'][emotion2]


    id = generate_id()

    return {
        "result": {
            "id": id,
            "emotion_result1": emotion,
            "weight_result1": float(emotion_weight),  # Convert to float for JSON serializability
            "emotion_result2": emotion2,
            "weight_result2": float(emotion_weight2),
        }
    }

# Function to save predictions to the database
def save_to_db(id, emotion1, weight1, emotion2, weight2):


    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    # Insert the prediction into the database
    c.execute('''
        INSERT INTO predictions (id, emotion1, weight1, emotion2, weight2)
        VALUES (?, ?, ?, ?, ?)
    ''', (id,emotion1, weight1, emotion2, weight2))

    conn.commit()
    conn.close()


def generate_id():
    return datetime.now().strftime('%Y%m%d%H%M%S')


@app.route('/predictions/count', methods=['GET'])
def get_predictions_count():
    try:
        conn = sqlite3.connect('predictions.db')  # Update with your database file name
        cursor = conn.cursor()

        # Execute the query to count the number of rows in the 'predictions' table
        cursor.execute("SELECT COUNT(*) FROM predictions")

        # Fetch the count result
        count = cursor.fetchone()[0]

        return jsonify({'count': count})
    except Exception as e:
        print(f"Error fetching predictions count: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()




@app.route('/predictions/delete/<id>', methods=['DELETE'])
def delete_prediction_by_id(id):
    try:
        print(f"Delete request received for ID: {id}")

        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        print("Database connected.")

        # Check if the record exists based on the ID
        cursor.execute('SELECT id FROM predictions WHERE id = ?', (id,))
        result = cursor.fetchone()
        print(f"Query result: {result}")

        if result:
            # Delete the record from the database
            cursor.execute('DELETE FROM predictions WHERE id = ?', (id,))
            conn.commit()
            print(f"Record with ID {id} deleted from database.")

            # Get the new count of predictions
            count_response = get_predictions_count()
            count = count_response.get_json().get('count')  # Accessing the count from the JSON response
            print("Item count after delete: ", count)

            return jsonify({'message': 'Record deleted successfully'}), 200
        else:
            return jsonify({'error': 'No record found for the given ID'}), 404
    except Exception as e:
        print(f"Error deleting prediction by ID {id}: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/predictions/all', methods=['GET'])
def get_all_predictions():

    # Connect to the SQLite database
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    # Execute the query to get all results from the predictions table
    c.execute('SELECT * FROM predictions')

    # Fetch all the rows from the table
    rows = c.fetchall()

    # Close the database connection
    conn.close()


    #20241009133122

    # Format the result into a list of dictionaries
    results = []
    for row in rows:
        result = {
            "id": row[0],  # Assuming the first column is 'id'
            "emotion_result1": row[1],  # Assuming second column is 'emotion1'
            "weight_result1": row[2],   # Assuming third column is 'weight1'
            "emotion_result2": row[3],  # Assuming fourth column is 'emotion2'
            "weight_result2": row[4],   # Assuming fifth column is 'weight2'
        }
        results.append(result)

    # Return the results as a JSON response
    return jsonify({"results": results})

# Route for handling file uploads and emotion prediction
@app.route('/predict_emotion', methods=['POST'])
def upload_image():
    
    # Check if the request contains the image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
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
            print(predicted_emotions)  # Print the predicted emotion
            
            # Save predictions to the database
            save_to_db(
                predicted_emotions['result']['id'],
                predicted_emotions['result']['emotion_result1'],
                predicted_emotions['result']['weight_result1'],
                predicted_emotions['result']['emotion_result2'],
                predicted_emotions['result']['weight_result2'],
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
