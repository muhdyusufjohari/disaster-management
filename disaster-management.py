import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
import plotly.express as px
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained models
@st.cache_resource
def load_models():
    flood_model = fasterrcnn_resnet50_fpn(pretrained=True)
    landslide_model = fasterrcnn_resnet50_fpn(pretrained=True)
    damage_model = fasterrcnn_resnet50_fpn(pretrained=True)
    erosion_model = fasterrcnn_resnet50_fpn(pretrained=True)
    return flood_model, landslide_model, damage_model, erosion_model

flood_model, landslide_model, damage_model, erosion_model = load_models()

# Detection functions
def detect_objects(image, model):
    # Convert PIL Image to tensor
    img_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        prediction = model(img_tensor)
    
    # Process predictions
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Filter predictions based on confidence threshold
    confidence_threshold = 0.5
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels

def visualize_detection(image, boxes, scores, labels):
    img = np.array(image)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"Class: {label}, Score: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    return img

# Streamlit app
st.set_page_config(layout="wide", page_title="Disaster Management Detection")

st.sidebar.title("Disaster Management Detection")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Object Detection", "Data Dashboard"])

if app_mode == "Object Detection":
    st.title("Disaster Management Object Detection")
    
    detection_type = st.sidebar.selectbox(
        "Choose Detection Type",
        ("Flood Detection", "Landslide Risk Assessment", "Infrastructure Damage Detection", "Coastal Erosion Monitoring")
    )
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Detecting...")
        
        if detection_type == "Flood Detection":
            boxes, scores, labels = detect_objects(image, flood_model)
        elif detection_type == "Landslide Risk Assessment":
            boxes, scores, labels = detect_objects(image, landslide_model)
        elif detection_type == "Infrastructure Damage Detection":
            boxes, scores, labels = detect_objects(image, damage_model)
        elif detection_type == "Coastal Erosion Monitoring":
            boxes, scores, labels = detect_objects(image, erosion_model)
        
        result_image = visualize_detection(image, boxes, scores, labels)
        st.image(result_image, caption="Detection Result", use_column_width=True)
        
        # Display detection results
        st.write(f"Number of objects detected: {len(boxes)}")
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            st.write(f"Object {i+1}: Class {label}, Confidence: {score:.2f}, Bounding Box: {box}")

elif app_mode == "Data Dashboard":
    st.title("Disaster Management Data Dashboard")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        
        # Example visualizations
        st.subheader("Data Visualizations")
        
        # Bar chart
        st.write("Bar Chart")
        bar_chart = px.bar(data, x=data.columns[0], y=data.columns[1])
        st.plotly_chart(bar_chart)
        
        # Line chart
        st.write("Line Chart")
        line_chart = px.line(data, x=data.columns[0], y=data.columns[1])
        st.plotly_chart(line_chart)
        
        # Scatter plot
        st.write("Scatter Plot")
        scatter_plot = px.scatter(data, x=data.columns[0], y=data.columns[1])
        st.plotly_chart(scatter_plot)

