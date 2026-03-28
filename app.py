import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

NUM_CLASSES = 3 # Background + Crack + Spalling

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load('rc_defect_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title("Civil Engineering AI Application: RC Defect Detector")
st.write("Upload an image of a concrete pier or column to inspect it for defects.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Original Upload', use_column_width=True)

    if st.button('Analyze Structure'):
        st.write("Running Structure AI inference...")

        # Transform image for the model
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Prepare image for drawing bounding boxes
        img_cv2 = np.array(image)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

        boxes = prediction['boxes'].numpy()
        scores = prediction['scores'].numpy()
        labels = prediction['labels'].numpy()

        # TODO: Students, define your class names here to match Step 3, complete the part in ' '.
        class_names = {1: 'crack', 2: 'spalling'}

        # Draw boxes
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.4:  # Adjust this threshold if needed
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 3)
                text = f"{class_names.get(label, 'Unknown')}: {score:.2f}"
                cv2.putText(img_cv2, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        st.image(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), caption='Structure AI Inspection Results', use_column_width=True)
        st.success("RC Structure Inspection Complete!")
