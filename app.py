from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import requests
import base64
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Hugging Face API details
# API_URL = os.getenv('API_URL')
API_URL = "https://api-inference.huggingface.co/models/CloudStone/ASL_MODEL"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        frame_data = request.json['frame']
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

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

            # Make API call to Hugging Face
            response = requests.post(API_URL, headers=headers, json={"inputs": data_aux})
            result = response.json()
            
            if isinstance(result, list):
                result = result[0]  # Hugging Face sometimes returns a list
            
            predicted_gesture = result.get('gesture', 'Unknown')
        else:
            predicted_gesture = "No hand detected"

        return jsonify({'gesture': predicted_gesture})
    except Exception as e:
        app.logger.error(f"Error in process_frame: {str(e)}")
        return jsonify({'gesture': 'Error occurred', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)