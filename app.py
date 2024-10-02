from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
import base64

app = Flask(__name__)

# Load the labels
try:
    with open('CLASSES.json', 'r') as json_file:
        labels_dict = json.load(json_file)
    labels_dict = {int(k): v for k, v in labels_dict.items()}
except FileNotFoundError:
    print("CLASSES.json file not found. Please ensure it's in the correct directory.")
    exit(1)
except json.JSONDecodeError:
    print("Error decoding CLASSES.json. Please ensure it's a valid JSON file.")
    exit(1)

# Load the model
try:
    with open('model.p', 'rb') as model_file:
        model_dict = pickle.load(model_file)
    model = model_dict['model']
except FileNotFoundError:
    print("model.p file not found. Please ensure it's in the correct directory.")
    exit(1)
except (pickle.UnpicklingError, KeyError):
    print("Error loading the model from model.p. The file may be corrupted or in an unexpected format.")
    exit(1)

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the frame data from the request
    frame_data = request.json['frame']
    # Decode the base64 image
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = "No hand detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

    return jsonify({'gesture': predicted_character})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)