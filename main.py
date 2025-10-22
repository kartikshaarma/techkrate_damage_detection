# main.py
import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from pydantic import BaseModel
from google.cloud import storage
from ultralytics import YOLO
from PIL import Image
import numpy as np

MODEL_BUCKET = os.environ.get("MODEL_BUCKET")  # e.g. "gs://cobalt-column-472507-k4-models"
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best.pt")
API_KEY = os.environ.get("API_KEY")  # mobile app must send this header value
LOCAL_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"

app = FastAPI(title="YOLOv8 Cloud Run API")

_model = None

def download_model_if_needed():
    global _model
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.getsize(LOCAL_MODEL_PATH) > 1000:
        return LOCAL_MODEL_PATH
    if not MODEL_BUCKET:
        raise RuntimeError("MODEL_BUCKET env var not set.")
    client = storage.Client()
    bucket_name = MODEL_BUCKET.replace("gs://", "")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(MODEL_FILENAME)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    return LOCAL_MODEL_PATH

def get_model():
    global _model
    if _model is None:
        model_path = download_model_if_needed()
        _model = YOLO(model_path)
    return _model

class PredictResponse(BaseModel):
    boxes: list
    scores: list
    classes: list
    names: dict

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Optional[UploadFile] = None,
    image_base64: Optional[str] = None,
    x_api_key: Optional[str] = Header(None)
):
    # Simple API key gate (mobile app supplies header "x-api-key")
    if API_KEY:
        if x_api_key is None or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if file is None and image_base64 is None:
        raise HTTPException(status_code=400, detail="Provide file or image_base64")

    if file:
        image_bytes = await file.read()
    else:
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",")[1]
        image_bytes = base64.b64decode(image_base64)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)

    model = get_model()
    results = model.predict(source=arr, imgsz=640, conf=0.25, device="cpu")  # cpu inference on Cloud Run
    r = results[0]

    boxes = []
    scores = []
    classes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            boxes.append([float(x) for x in box])
            scores.append(float(conf))
            classes.append(int(cls))

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "names": model.names
    }

@app.get("/health")
def health():
    return {"status": "ok"}
