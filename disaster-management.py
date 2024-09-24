import streamlit as st
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import json
import os

# Set page config at the very beginning
st.set_page_config(layout="wide", page_title="Disaster Management Detection")

# Load pre-trained model
@st.cache_resource
def load_model():
    return fasterrcnn_resnet50_fpn(pretrained=True)

model = load_model()

# File handling functions
def save_detection(frame, label, confidence, box):
    data = {
        "frame": frame,
        "label": int(label),
        "confidence": float(confidence),
        "box": box.tolist()
    }
    with open("detections.json", "a") as f:
        f.write(json.dumps(data) + "\n")

def get_detections():
    if not os.path.exists("detections.json"):
        return pd.DataFrame()
    
    with open("detections.json", "r") as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    df['x1'] = df['box'].apply(lambda x: x[0])
    df['y1'] = df['box'].apply(lambda x: x[1])
    df['x2'] = df['box'].apply(lambda x: x[2])
    df['y2'] = df['box'].apply(lambda x: x[3])
    return df

# Detection function
def detect_objects(frame, model):
    img_tensor = F.to_tensor(frame).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    confidence_threshold = 0.5
    mask = scores > confidence_threshold
    return boxes[mask], scores[mask], labels[mask]

# Streamlit app
st.title("Disaster Management Video Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Analysis")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the video
        video = cv2.VideoCapture("temp_video.mp4")
        
        # Clear previous detections
        if os.path.exists("detections.json"):
            os.remove("detections.json")
        
        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 10 == 0:  # Process every 10th frame
                boxes, scores, labels = detect_objects(frame, model)
                
                for box, score, label in zip(boxes, scores, labels):
                    save_detection(frame_count, label, score, box)
                
                # Update progress
                progress = frame_count / video.get(cv2.CAP_PROP_FRAME_COUNT)
                progress_bar.progress(progress)
                status_text.text(f"Processed frame: {frame_count}")
        
        video.release()
        st.success("Video processing complete!")

with col2:
    st.subheader("Detection Data Dashboard")
    
    df = get_detections()
    
    if not df.empty:
        st.write(df.head())
        
        # Example visualizations
        st.subheader("Data Visualizations")
        
        # Bar chart of object counts
        object_counts = df['label'].value_counts()
        bar_chart = px.bar(object_counts, x=object_counts.index, y=object_counts.values, 
                           labels={'x': 'Object Class', 'y': 'Count'}, title="Object Counts")
        st.plotly_chart(bar_chart, use_container_width=True)
        
        # Line chart of confidence over frames
        line_chart = px.line(df, x='frame', y='confidence', color='label',
                             title="Confidence Scores Over Frames")
        st.plotly_chart(line_chart, use_container_width=True)
        
        # Scatter plot of object positions
        scatter_plot = px.scatter(df, x='x1', y='y1', color='label', 
                                  title="Object Positions (Top-Left Corner)")
        st.plotly_chart(scatter_plot, use_container_width=True)
    else:
        st.write("No data available. Please process a video first.")
