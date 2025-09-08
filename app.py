# app.py
import os
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import time

# ---------------- CONFIG ----------------
MODEL_DIR = Path("runs/vehicle_damage_yolov8m/weights")
MODEL_PATH = MODEL_DIR / "best.pt"
# Google Drive file id expected in env var / Streamlit secret MODEL_GDRIVE_ID
GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "")
MIN_UPSCALE_SIZE = 640
# ----------------------------------------

st.set_page_config(page_title="Vehicle Damage Detector", layout="centered")
st.title("🚗 Vehicle Damage Detection")

# ---------- downloader ----------
def download_model_from_gdrive(gdrive_id, out_path):
    try:
        import gdown
    except Exception:
        st.info("Installing gdown...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st.info("Downloading model weights (this may take a few minutes)...")
    gdown.download(url, str(out_path), quiet=False)
    return out_path.exists()

# Ensure model exists, otherwise download
if not MODEL_PATH.exists():
    if not GDRIVE_ID:
        st.error("Model not found locally and MODEL_GDRIVE_ID not set. Set MODEL_GDRIVE_ID in Streamlit secrets or environment.")
        st.stop()
    ok = download_model_from_gdrive(GDRIVE_ID, MODEL_PATH)
    if not ok:
        st.error("Failed to download model from Google Drive.")
        st.stop()

# ---------- load model (cached) ----------
@st.cache_resource
def load_model(path):
    return YOLO(str(path))

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# ---------- UI sidebar ----------
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence threshold", 0.01, 0.9, 0.30, 0.01)
imgsz = st.sidebar.selectbox("Inference image size (imgsz)", [320, 416, 512, 640, 768], index=3)
use_tta = st.sidebar.checkbox("Use TTA (augment=True)", value=True)
upscale_toggle = st.sidebar.checkbox(f"Upscale images smaller than {MIN_UPSCALE_SIZE}px", value=True)
tile_toggle = st.sidebar.checkbox("Use tile inference (advanced, slower)", value=False)
tile_size = st.sidebar.slider("Tile size (when tile inference enabled)", 512, 1024, 640, step=64)
tile_overlap = st.sidebar.slider("Tile overlap fraction", 0.0, 0.5, 0.2, step=0.05)
st.sidebar.markdown("---")
st.sidebar.write("Model path:")
st.sidebar.code(str(MODEL_PATH))

# ---------- helpers ----------
def upscale_if_small(pil_img: Image.Image, min_size=MIN_UPSCALE_SIZE):
    w, h = pil_img.size
    long_side = max(w, h)
    if long_side >= min_size:
        return pil_img, False
    scale = float(min_size) / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    arr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
    up = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    up_pil = Image.fromarray(up[:, :, ::-1])
    return up_pil, True

def non_max_suppression(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return [], []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    xywh = []
    for b in boxes_np:
        x1, y1, x2, y2 = b
        xywh.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    indices = cv2.dnn.NMSBoxes(xywh, scores_np.tolist(), 0.0, iou_thresh)
    keep = []
    if len(indices) > 0:
        try:
            flat = [int(i) for i in indices.flatten()]
        except Exception:
            flat = [int(indices)]
        keep = flat
    kept_boxes = [boxes[i] for i in keep]
    kept_scores = [float(scores[i]) for i in keep]
    return kept_boxes, kept_scores

def tile_and_predict(img_pil, model, tile_size=640, overlap=0.2, imgsz=640, conf=0.25, augment=False):
    w, h = img_pil.size
    step = int(tile_size * (1 - overlap))
    all_boxes, all_scores, all_labels = [], [], []
    xs = list(range(0, max(1, w - tile_size + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile_size + 1), step)) or [0]
    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            crop = img_pil.crop((x1, y1, x2, y2)).convert("RGB")
            arr = np.array(crop)
            res = model.predict(source=arr, imgsz=imgsz, conf=conf, augment=augment, workers=0, verbose=False)
            if len(res) == 0 or not hasattr(res[0], "boxes") or len(res[0].boxes) == 0:
                continue
            r = res[0]
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            labels = r.boxes.cls.cpu().numpy().astype(int).tolist()
            for b, s, l in zip(boxes, scores, labels):
                x1b, y1b, x2b, y2b = b
                all_boxes.append([x1b + x1, y1b + y1, x2b + x1, y2b + y1])
                all_scores.append(s)
                all_labels.append(l)
    kept_boxes, kept_scores = non_max_suppression(all_boxes, all_scores, 0.5)
    kept_labels = [0] * len(kept_boxes)
    return kept_boxes, kept_scores, kept_labels

# ---------- prediction & display ----------
def predict_and_display(pil_img):
    if model is None:
        st.error("Model not loaded")
        return
    original = pil_img.copy()
    upscaled_flag = False
    if upscale_toggle:
        pil_img, upscaled_flag = upscale_if_small(pil_img, min_size=MIN_UPSCALE_SIZE)
    if upscaled_flag:
        st.info(f"Image upscaled to {pil_img.size[0]}x{pil_img.size[1]} to improve small-object detection.")
    start = time.time()
    if tile_toggle:
        boxes, scores, labels = tile_and_predict(pil_img, model, tile_size=tile_size, overlap=tile_overlap,
                                                 imgsz=imgsz, conf=conf, augment=use_tta)
    else:
        arr = np.array(pil_img.convert("RGB"))
        res = model.predict(source=arr, imgsz=imgsz, conf=conf, augment=use_tta, workers=0, verbose=False)
        if len(res) == 0 or not hasattr(res[0], "boxes") or len(res[0].boxes) == 0:
            boxes, scores, labels = [], [], []
        else:
            r = res[0]
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            labels = r.boxes.cls.cpu().numpy().astype(int).tolist()
    elapsed = time.time() - start
    disp_img = original.convert("RGB")
    draw = ImageDraw.Draw(disp_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    if len(boxes) == 0:
        st.image(disp_img, caption=f"No detections (in {elapsed:.2f}s)", use_column_width=True)
        return
    if upscaled_flag:
        ow, oh = original.size
        nw, nh = pil_img.size
        sx = ow / float(nw)
        sy = oh / float(nh)
    else:
        sx = sy = 1.0
    rows = []
    for i, (b, s, l) in enumerate(zip(boxes, scores, labels), start=1):
        x1, y1, x2, y2 = b
        x1o = int(x1 * sx); y1o = int(y1 * sy); x2o = int(x2 * sx); y2o = int(y2 * sy)
        draw.rectangle([x1o, y1o, x2o, y2o], outline="red", width=3)
        label_txt = f"damage {s:.2f}"
        try:
            bbox = font.getbbox(label_txt)
            text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(label_txt, font=font)
        draw.rectangle([x1o, max(0, y1o - text_h - 4), x1o + text_w + 4, y1o], fill="red")
        draw.text((x1o + 2, max(0, y1o - text_h - 3)), label_txt, fill="white", font=font)
        rows.append({"#": i, "class": "damage", "confidence": f"{s:.3f}"})
    st.image(disp_img, caption=f"Detections (in {elapsed:.2f}s)", use_column_width=True)
    st.table(rows)

# ---------- main ----------
st.header("Upload an image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(pil_img, use_column_width=True)
    if st.button("Run detection"):
        predict_and_display(pil_img)
else:
    st.info("Upload an image to run detection.")
