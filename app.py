from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
import base64
from threading import Thread
from queue import Queue

app = Flask(__name__)

# Load the labels
with open('CLASSES.json', 'r') as json_file:
    labels_dict = json.load(json_file)
labels_dict = {int(k): v for k, v in labels_dict.items()}

# Load the model
with open('model.p', 'rb') as model_file:
    model_dict = pickle.load(model_file)
model = model_dict['model']

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a queue for frame processing
frame_queue = Queue(maxsize=1)  # Only keep the latest frame

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            continue

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

        app.predicted_gesture = predicted_character

# Start the frame processing thread
processing_thread = Thread(target=process_frames)
processing_thread.daemon = True
processing_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.json['frame']
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Put the frame in the queue for processing
    if frame_queue.full():
        frame_queue.get()  # Remove the old frame
    frame_queue.put(frame)

    # Return the last predicted gesture
    return jsonify({'gesture': getattr(app, 'predicted_gesture', 'Processing...')})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000, threaded=True)