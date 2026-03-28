import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="RC Defect Detector", layout="centered")
st.title("RC Defect Detector")
st.write("Upload a concrete image to detect cracks and spalling.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run detection"):
        try:
            from ultralytics import YOLO

            model = YOLO("best.pt")
            results = model.predict(image, conf=0.25)

            plotted = results[0].plot()
            plotted = Image.fromarray(plotted[:, :, ::-1])  # BGR to RGB

            st.image(plotted, caption="Detection result", use_container_width=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.subheader("Detections")
                names = results[0].names
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = names.get(cls_id, str(cls_id))
                    st.write(f"{i+1}. {label} — confidence: {conf:.2f}")
            else:
                st.info("No defects detected.")
        except Exception as e:
            st.error(f"Detection failed: {e}")
