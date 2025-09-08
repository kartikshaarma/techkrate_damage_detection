import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Path to your trained weights
MODEL_PATH = "runs/vehicle_damage_yolov8m/weights/best.pt"

# Load model once at startup
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Vehicle Damage Detector", layout="centered")
st.title("🚗 Vehicle Damage Detection")

st.write("Upload a vehicle image, and the model will detect and mark damages.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    pil_img = Image.open(uploaded_file).convert("RGB")

    # Display original
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    st.write("Running detection...")
    results = model.predict(
        source=np.array(pil_img),
        imgsz=640,
        conf=0.30,
        augment=True,
        workers=0,
        verbose=False
    )

    # Show image with bounding boxes
    res = results[0]
    plotted = res.plot()  # numpy array with annotations
    st.image(plotted, caption="Detections", use_column_width=True)

    # Show table of detections
    if res.boxes is not None and len(res.boxes) > 0:
        st.subheader("Detections")
        data = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            data.append({"class": "damage", "confidence": f"{conf:.2f}"})
        st.table(data)
    else:
        st.info("No damages detected in this image.")
