#!/usr/bin/env python3
"""
train_yolov8m.py

Usage:
    python train_yolov8m.py --data ./vehicle_damage_yolo/data.yaml --model yolov8m.pt --epochs 150 --batch 6 --imgsz 640

This script uses the ultralytics package (YOLOv8) to fine-tune a pre-trained yolov8m model.
Install ultralytics via: pip install ultralytics
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: str, model_checkpoint: str = "yolov8m.pt",
          epochs: int = 150, batch: int = 6, imgsz: int = 640,
          project: str = "runs", name: str = "vehicle_damage_yolov8m",
          device: str = "0", workers: int = 0, augment: bool = True):
    """
    data_yaml: path to data.yaml produced in Step 2
    model_checkpoint: pre-trained checkpoint (e.g., yolov8m.pt)
    """
    data_yaml = str(Path(data_yaml).resolve())
    print(f"Training with data: {data_yaml}")
    model = YOLO(model_checkpoint)  # loads pretrained weights

    # start training
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        augment=augment,
        project=project,
        name=name,
        exist_ok=True  # allows overwriting experiment folder if needed
    )
    print("Training finished. Best weights are in the ultralytics runs/ folder.")

    # optional: evaluate on validation set and export best model
    # model.val(data=data_yaml)
    # model.export(format="onnx")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8m.pt", help="Pre-trained checkpoint to start from")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="Device id or 'cpu'")
    p.add_argument("--project", default="runs", help="Ultralytics project folder")
    p.add_argument("--name", default="vehicle_damage_yolov8m", help="Experiment name")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--augment", action="store_true", help="Use data augmentation")
    args = p.parse_args()

    train(args.data, args.model, args.epochs, args.batch, args.imgsz,
          args.project, args.name, args.device, args.workers, args.augment)
