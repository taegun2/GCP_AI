from flask import Flask, jsonify
import cv2
import mediapipe as mp
import numpy as np
from google.cloud import storage, pubsub_v1
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import requests
import json
import threading
import os

app = Flask(__name__)

# GOOGLE_APPLICATION_CREDENTIALS 환경 변수 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# GCS 클라이언트 초기화
storage_client = storage.Client()

# 모델 및 라벨 인코더 로드
model = load_model('sign_language_model.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)

def download_video_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """GCS에서 파일을 다운로드하여 로컬에 저장"""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def detect_hand_movement_threshold(hand_landmarks, prev_hand_landmarks, threshold=0.02):
    """현재 손 랜드마크와 이전 프레임의 랜드마크 사이의 이동 거리를 계산하여 움직임을 감지."""
    if prev_hand_landmarks is None:
        return False
    
    movement = np.linalg.norm(hand_landmarks - prev_hand_landmarks, axis=1)
    return np.any(movement > threshold)

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7)

    keypoints = []
    prev_hand_landmarks = None
    recording = False
    movement_detected_at_least_once = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)

        hand_landmarks = np.zeros((42, 3))
        face_landmarks = np.zeros((468, 3))

        if results_hands.multi_hand_landmarks:
            for hand_id, hand_landmark in enumerate(results_hands.multi_hand_landmarks):
                if hand_id < 2:  # 두 손까지만 처리
                    for i, landmark in enumerate(hand_landmark.landmark):
                        hand_landmarks[i + hand_id * 21] = [landmark.x, landmark.y, landmark.z]
            
            # 손의 움직임 감지
            if detect_hand_movement_threshold(hand_landmarks, prev_hand_landmarks):
                recording = True  # 첫 움직임이 감지되면 기록 시작
                movement_detected_at_least_once = True  # 첫 번째 움직임이 감지되었음을 표시
            prev_hand_landmarks = hand_landmarks.copy()

        if recording:
            # 얼굴 랜드마크 처리
            if results_face.multi_face_landmarks:
                for i, landmark in enumerate(results_face.multi_face_landmarks[0].landmark):
                    face_landmarks[i] = [landmark.x, landmark.y, landmark.z]

            keypoints.append(np.concatenate([hand_landmarks.flatten(), face_landmarks.flatten()]))

        # 중간에 움직임이 멈추더라도, 첫 움직임 이후에는 계속 기록 유지
        if movement_detected_at_least_once and not detect_hand_movement_threshold(hand_landmarks, prev_hand_landmarks):
            recording = True

    cap.release()
    return np.array(keypoints)

def predict_video_class(model, video_path, max_sequence_length=99):
    keypoints = extract_keypoints(video_path)
    keypoints_padded = pad_sequences([keypoints], maxlen=max_sequence_length, dtype='float32', padding='post')
    predictions = model.predict(keypoints_padded)
    return predictions

def decode_predictions(predictions, label_encoder):
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_class_names = label_encoder.inverse_transform(predicted_classes)
    return predicted_class_names[0]

def process_message(message):
    data = json.loads(message.data)
    bucket_name = data['bucket_name']
    video_path_in_gcs = data['video_path']
    callback_url = data['callback_url']
    local_video_path = 'downloaded_video.mp4'
    
    # GCS에서 비디오 다운로드
    download_video_from_gcs(bucket_name, video_path_in_gcs, local_video_path)

    # 비디오 예측 수행
    predictions = predict_video_class(model, local_video_path)
    
    # 예측 결과 해석
    predicted_class_name = decode_predictions(predictions, label_encoder)
    
    # 예측된 클래스명을 콜백 URL로 전송
    response = requests.post(callback_url, json={'predicted_class': predicted_class_name})

    print(f"Predicted class sent to {callback_url}: {predicted_class_name}")
    message.ack()

def start_pubsub_listener():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path('andong-24-team-103', 'url-sub')
    
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=process_message)
    print(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except Exception as e:
        streaming_pull_future.cancel()
        print(f"Listening for messages was interrupted: {e}")

@app.route('/')
def home():
    return "Flask server is running on a fixed port!"

if __name__ == '__main__':
    # Pub/Sub listener를 별도의 스레드에서 실행
    listener_thread = threading.Thread(target=start_pubsub_listener)
    listener_thread.start()

    # Flask 서버를 항상 포트 5000에서 실행
    app.run(host='0.0.0.0', port=5000)
