#!/usr/bin/env python3
"""
ensemble_and_eval.py

- Runs predictions on validation images with two checkpoints (model_a, model_b).
- Applies Weighted Box Fusion (WBF) to fuse detections per image.
- Converts GT (YOLO .txt) and fused detections to COCO json format.
- Evaluates mAP50 and mAP50-95 using pycocotools.

Edit CONFIG below to point to your files if needed.
"""

import json
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------- CONFIG ----------------
MODEL_A = "runs/vehicle_damage_yolov8m/weights/best.pt"
MODEL_B = "runs/vehicle_damage_yolov8m/weights/last.pt"
VAL_IMAGES_DIR = Path(r"C:\Users\ksh16\Downloads\techkrate vehicle damage detection\vehicle_damage_yolo\valid\images")
VAL_LABELS_DIR = Path(r"C:\Users\ksh16\Downloads\techkrate vehicle damage detection\vehicle_damage_yolo\valid\labels")
IMG_SIZE = 640
CONF_THRESH = 0.30   # use 0.30 since it gave you best results
IOU_FOR_FUSION = 0.55
WBF_WEIGHT = [0.6, 0.4]
OUTPUT_GT_JSON = Path("tmp_gt_coco.json")
OUTPUT_PRED_JSON = Path("tmp_pred_coco.json")
# ----------------------------------------

def read_yolo_labels(txt_path, img_w, img_h):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        x_center = xc * img_w
        y_center = yc * img_h
        bw = w * img_w
        bh = h * img_h
        x1 = x_center - bw/2
        y1 = y_center - bh/2
        boxes.append({
            "category_id": cls + 1,
            "bbox_xywh": [x1, y1, bw, bh]
        })
    return boxes

def run_model_preds(model_path, images, imgsz=640, conf=0.25):
    model = YOLO(model_path)
    preds_per_image = {}
    for img_path in tqdm(images, desc=f"Predicting {Path(model_path).name}"):
        res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, workers=0, verbose=False)
        if len(res) == 0 or len(res[0].boxes) == 0:
            preds_per_image[img_path.name] = {"boxes": [], "scores": [], "labels": []}
            continue
        boxes = res[0].boxes.xyxy.cpu().numpy().tolist()
        scores = res[0].boxes.conf.cpu().numpy().tolist()
        labels = res[0].boxes.cls.cpu().numpy().astype(int).tolist()
        preds_per_image[img_path.name] = {"boxes": boxes, "scores": scores, "labels": labels}
    return preds_per_image

def main():
    images = sorted([p for p in VAL_IMAGES_DIR.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]])
    if len(images) == 0:
        raise SystemExit("No images found in VAL_IMAGES_DIR")

    # Run both models
    preds_a = run_model_preds(MODEL_A, images, imgsz=IMG_SIZE, conf=CONF_THRESH)
    preds_b = run_model_preds(MODEL_B, images, imgsz=IMG_SIZE, conf=CONF_THRESH)

    # Build COCO GT JSON (with required fields)
    coco_gt = {
        "info": {"description": "vehicle_damage_yolo ground truth"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "damage"}]
    }
    ann_id = 1
    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        coco_gt["images"].append({"id": idx, "file_name": img_path.name, "height": h, "width": w})
        gt_boxes = read_yolo_labels(VAL_LABELS_DIR / (img_path.stem + ".txt"), w, h)
        for g in gt_boxes:
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": idx,
                "category_id": g["category_id"],
                "bbox": [float(v) for v in g["bbox_xywh"]],
                "area": float(g["bbox_xywh"][2] * g["bbox_xywh"][3]),
                "iscrowd": 0
            })
            ann_id += 1
    OUTPUT_GT_JSON.write_text(json.dumps(coco_gt))
    print(f"Wrote GT COCO JSON to {OUTPUT_GT_JSON}")

    # Build fused predictions JSON
    coco_dets = []
    for idx, img_path in enumerate(tqdm(images, desc="Fusing detections"), start=1):
        pa = preds_a.get(img_path.name, {"boxes": [], "scores": [], "labels": []})
        pb = preds_b.get(img_path.name, {"boxes": [], "scores": [], "labels": []})

        img_cv = cv2.imread(str(img_path)); h, w = img_cv.shape[:2]
        boxes_a = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in pa["boxes"]]
        boxes_b = [[b[0]/w, b[1]/h, b[2]/w, b[3]/h] for b in pb["boxes"]]
        scores_a = pa["scores"]; scores_b = pb["scores"]
        labels_a = pa["labels"]; labels_b = pb["labels"]

        if (len(boxes_a) + len(boxes_b)) == 0:
            continue

        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            [boxes_a, boxes_b], [scores_a, scores_b], [labels_a, labels_b],
            weights=WBF_WEIGHT, iou_thr=IOU_FOR_FUSION, skip_box_thr=0.0
        )

        for b, sc, lb in zip(boxes_fused, scores_fused, labels_fused):
            x1 = b[0] * w; y1 = b[1] * h; x2 = b[2] * w; y2 = b[3] * h
            xywh = [float(x1), float(y1), float(max(0, x2-x1)), float(max(0, y2-y1))]
            coco_dets.append({
                "image_id": idx,
                "category_id": int(lb) + 1,
                "bbox": xywh,
                "score": float(sc)
            })

    OUTPUT_PRED_JSON.write_text(json.dumps(coco_dets))
    print(f"Wrote fused detections to {OUTPUT_PRED_JSON}")

    # Evaluate
    cocoGt = COCO(str(OUTPUT_GT_JSON))
    cocoDt = cocoGt.loadRes(str(OUTPUT_PRED_JSON))
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    main()
