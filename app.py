from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
import base64
from threading import Thread
from queue import Queue
import logging
import traceback

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load the labels
try:
    with open('CLASSES.json', 'r') as json_file:
        labels_dict = json.load(json_file)
    labels_dict = {int(k): v for k, v in labels_dict.items()}
    app.logger.info(f"Labels loaded: {labels_dict}")
except Exception as e:
    app.logger.error(f"Error loading labels: {str(e)}")
    labels_dict = {}

# Load the model
try:
    with open('model.p', 'rb') as model_file:
        model_dict = pickle.load(model_file)
    model = model_dict['model']
    app.logger.info("Model loaded successfully")
    
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    model = None

# Set up MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

frame_queue = Queue(maxsize=1)

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            continue

        try:
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
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                app.logger.info(f"Predicted character: {predicted_character}")
            else:
                app.logger.info("No hand detected in the frame")

            app.predicted_gesture = predicted_character
        except Exception as e:
            app.logger.error(f"Error processing frame: {str(e)}")
            app.logger.error(traceback.format_exc())
            app.predicted_gesture = "Error in processing"

processing_thread = Thread(target=process_frames)
processing_thread.daemon = True
processing_thread.start()

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

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

        gesture = getattr(app, 'predicted_gesture', 'Processing...')
        app.logger.info(f"Returning gesture: {gesture}")
        return jsonify({'gesture': gesture})
    except Exception as e:
        app.logger.error(f"Error in process_frame: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'gesture': 'Error occurred', 'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())
    return jsonify({'gesture': 'Server error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000, threaded=True)