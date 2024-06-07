
import os
import re
import time
import queue
import threading
import multiprocessing
import numpy as np
import cv2
import wave
import torch
import streamlit as st
import sounddevice as sd
import librosa
import pyaudio
import pandas as pd
from PIL import Image
from transformers import pipeline, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import speech
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av

st.set_page_config(page_title="Emotion Analysis App", layout="wide")

page = st.sidebar.selectbox("Choose a page", ["Emotion Detection", "Dashboard"])

# Initialize models and processors
emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-latest")
text_emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-latest")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = torch.load('ser_modelv2_oversampled.pth', map_location=device)
emotion_model.to(device)
text_emotion_model.to(device)

emotion_model.eval()
text_emotion_model.eval()

model_path = 'facial_emotions_image_detection/checkpoint-3136'
pipe = pipeline('image-classification', model=model_path, device=-1)
face_model = YOLO('yolov8l-face.pt')



if 'emotions_data' not in st.session_state:
    st.session_state.emotions_data = pd.DataFrame()

def analyze_facial_emotion(frame, pipe, face_model):
    face_result = face_model.predict(frame, conf=0.40)
    all_face_emotions = []

    for info in face_result:
        for box in info.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face_img = frame[y1:y2, x1:x2]
            face_img_pil = Image.fromarray(face_img)
            emotion_result = pipe(face_img_pil)
            emotion_text = emotion_result[0]['label']
            all_face_emotions.append(emotion_text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return all_face_emotions, frame

def is_silent(audio_chunk, threshold=0.01):
    return np.mean(np.abs(audio_chunk)) < threshold

def process_audio_chunk(audio, text):
    sample_rate = 16000
    chunk_length = 10 * sample_rate
    ignore_length = 2 * sample_rate

    all_audio_emotions = []
    all_text_emotions = []

    for start in range(0, len(audio) - ignore_length, chunk_length):
        end = start + chunk_length
        audio_chunk = audio[start:end]
        if len(audio_chunk) == 0 or is_silent(audio_chunk):
            continue
        audio_chunk = torch.from_numpy(audio_chunk).to(torch.float32).to(device)
        inputs = emotion_feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).squeeze()
            score_dict = {emotion_model.config.id2label[i]: score.item() for i, score in enumerate(scores)}
            initial_prediction = max(score_dict, key=score_dict.get)
            predicted_emotion = initial_prediction
            all_audio_emotions.append(predicted_emotion)

    text_chunks = split_text_into_segments(text, 4)
    for chunk in text_chunks:
        chunk = chunk.replace('exit', '')
        if len(chunk) == 0 or 'exit' in chunk.lower():
            continue
        text_inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            text_outputs = text_emotion_model(**text_inputs)
        text_prediction = torch.argmax(text_outputs.logits, dim=-1)
        text_emotion = text_emotion_model.config.id2label[text_prediction.item()]
        all_text_emotions.append(text_emotion)

    return all_audio_emotions, all_text_emotions

def split_text_into_segments(text, num_segments):
    segment_length = len(text) // num_segments
    segments = []
    last_index = 0
    for _ in range(num_segments - 1):
        split_index = text.rfind(' ', last_index, last_index + segment_length + 1)
        if split_index == -1:
            split_index = last_index + segment_length
        segments.append(text[last_index:split_index])
        last_index = split_index + 1
    segments.append(text[last_index:])
    return segments

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    face_emotions, annotated_frame = analyze_facial_emotion(img, pipe, face_model)
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def audio_frame_callback(audio_frame):
    audio = np.array(audio_frame.to_ndarray())
    return audio

def start_emotion_detection():
    st.session_state.emotions_data = pd.DataFrame()
    st.session_state.audio_data = []
    st.session_state.transcription = ""

    webrtc_ctx = webrtc_streamer(
        key="emotion_detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True
    )

    while webrtc_ctx.state.playing:
        audio_data = np.array(st.session_state.audio_data)
        transcription = st.session_state.transcription
        if audio_data is not None and transcription:
            predicted_audio_emotions, predicted_text_emotions = process_audio_chunk(audio_data, transcription)
            st.write(f"Predicted Audio Emotions: {predicted_audio_emotions}")
            st.write(f"Predicted Text Emotions: {predicted_text_emotions}")

if page == 'Emotion Detection':
    st.title("Emotion Detection App")
    st.markdown("Start by pressing 'Start Camera and Microphone' button. Then navigate to the next page using the sidebar to access the dashboard for comprehensive insights.")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Face & Audio Emotion Detection")
        # if st.button("Start Camera and Microphone"):
        start_emotion_detection()
