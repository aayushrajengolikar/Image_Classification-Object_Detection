# app.py
import streamlit as st
import numpy as np
from PIL import Image
import cv2

# -----------------------------
# TensorFlow / Keras
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import load_model

# -----------------------------
# YOLO (Ultralytics)
# -----------------------------
from ultralytics import YOLO

# -----------------------------
# Load Models
# -----------------------------
st.sidebar.title("Model Selection")
task = st.sidebar.radio("Choose Task:", ["Image Classification", "Object Detection"])

# Load classifier
classifier_model = load_model("best_image_classifier.keras")  # <-- updated path

# Load YOLO object detector
yolo_model = YOLO("best.pt")  # <-- updated path

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img: Image.Image, target_size=(256, 256)):
    img = img.resize(target_size)
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# YOLO Prediction
# -----------------------------
def predict_objects_yolo(image: Image.Image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = yolo_model(img)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_model.names[int(cls)]} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Image Analysis App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    if task == "Image Classification":
        st.write("Processing classification...")
        img_array = preprocess_image(image)
        prediction = classifier_model.predict(img_array)[0][0]

        if prediction >= 0.5:
            predicted_class = "Drone"
        else:
            predicted_class = "Bird"

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {prediction*100:.2f}%")

    elif task == "Object Detection":
        st.write("Processing object detection...")
        detected_image = predict_objects_yolo(image)
        st.image(detected_image, caption="Detected Objects", width=500)

