from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Hugging Face API details
API_URL = os.getenv('API_URL')
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            break
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make API call to Hugging Face
            response = requests.post(API_URL, headers=headers, json={"inputs": data_aux})
            result = response.json()
            
            if isinstance(result, list):
                result = result[0]  # Hugging Face sometimes returns a list
            
            predicted_index = result.get('label')
            predicted_character = labels_dict.get(int(predicted_index), "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)