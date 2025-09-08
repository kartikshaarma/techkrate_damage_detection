#!/usr/bin/env python3
"""
conf_sweep_val.py

Run YOLOv8 validation at multiple confidence thresholds to see
how mAP50 changes. Useful for fine-tuning eval settings.
"""

from ultralytics import YOLO

MODEL_PATH = "runs/vehicle_damage_yolov8m/weights/best.pt"
DATA_PATH = r"C:\Users\ksh16\Downloads\techkrate vehicle damage detection\vehicle_damage_yolo\data.yaml"
IMG_SIZE = 640

conf_thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

def main():
    model = YOLO(MODEL_PATH)
    print(f"Validating {MODEL_PATH} on {DATA_PATH} with conf sweep...\n")
    results = {}

    for conf in conf_thresholds:
        print(f"--- Conf={conf:.2f} ---")
        metrics = model.val(
            data=DATA_PATH,
            imgsz=IMG_SIZE,
            conf=conf,
            save_json=False,
            verbose=False,
            workers=0  # force single-worker to avoid spawn issues
        )
        results[conf] = metrics.results_dict["metrics/mAP50(B)"]
        print(f"mAP50: {results[conf]:.3f}\n")

    print("=== Summary ===")
    for conf, score in results.items():
        print(f"Conf={conf:.2f} → mAP50={score:.3f}")

if __name__ == "__main__":
    main()
