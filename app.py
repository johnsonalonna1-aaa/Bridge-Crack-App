import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Concrete Crack Detection (YOLOv8)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Running detection...")

    model = YOLO("yolov8n.pt")
    results = model(image)

    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
