#!/usr/bin/env python3
"""
coco_to_yolo_split.py

Usage:
    python coco_to_yolo_split.py --master master_dataset --output vehicle_damage_yolo --train-ratio 0.8 --seed 42

Produces:
vehicle_damage_yolo/
  ├── data.yaml
  ├── train/
  │   ├── images/
  │   └── labels/
  └── valid/
      ├── images/
      └── labels/
"""
import argparse
import json
from pathlib import Path
import random
import shutil
from collections import defaultdict
from PIL import Image
import os

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def convert(master_dir: Path, out_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    master_ann = master_dir / "_annotations.coco.json"
    images_dir = master_dir / "images"
    if not master_ann.exists():
        raise SystemExit(f"Master annotation file not found: {master_ann}")
    with open(master_ann, "r", encoding="utf-8") as fh:
        coco = json.load(fh)

    imgs = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    # Build category id -> index (0..nc-1) mapping
    # Sort categories by id to make mapping deterministic
    cats_sorted = sorted(cats, key=lambda x: x["id"])
    catid_to_idx = {c["id"]: idx for idx, c in enumerate(cats_sorted)}
    names = [c["name"] for c in cats_sorted]
    nc = len(names)
    print(f"Detected {len(imgs)} images, {len(anns)} annotations, {nc} classes.")

    # Build annotations_by_image
    annotations_by_image = defaultdict(list)
    for ann in anns:
        annotations_by_image[ann["image_id"]].append(ann)

    # deterministic shuffle & split
    random.seed(seed)
    imgs_shuffled = imgs.copy()
    random.shuffle(imgs_shuffled)
    n_train = int(len(imgs_shuffled) * train_ratio)
    train_imgs = imgs_shuffled[:n_train]
    val_imgs = imgs_shuffled[n_train:]

    # prepare directories
    train_images_dir = out_dir / "train" / "images"
    train_labels_dir = out_dir / "train" / "labels"
    val_images_dir = out_dir / "valid" / "images"
    val_labels_dir = out_dir / "valid" / "labels"

    for d in (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
        ensure_dir(d)

    def process_split(img_list, images_dst_dir: Path, labels_dst_dir: Path):
        for img in img_list:
            fname = img["file_name"]
            src_path = images_dir / fname
            if not src_path.exists():
                # try to find by basename (robustness)
                candidates = list(images_dir.rglob(Path(fname).name))
                if candidates:
                    src_path = candidates[0]
                else:
                    print(f"WARNING: image file {fname} not found in {images_dir}. Skipping.")
                    continue
            # copy image
            dst_img = images_dst_dir / fname
            shutil.copy2(src_path, dst_img)

            # get image size (use metadata if present, else read image)
            img_w = img.get("width")
            img_h = img.get("height")
            if img_w is None or img_h is None:
                with Image.open(src_path) as im:
                    img_w, img_h = im.size

            # create label file
            label_path = labels_dst_dir / (Path(fname).stem + ".txt")
            anns_for_img = annotations_by_image.get(img["id"], [])
            lines = []
            for a in anns_for_img:
                catid = a["category_id"]
                cls = catid_to_idx[catid]
                bbox = a["bbox"]  # [x, y, w, h] in COCO
                x, y, w, h = bbox
                # convert to x_center, y_center
                xc = x + w / 2.0
                yc = y + h / 2.0
                # normalize
                xc_norm = xc / img_w
                yc_norm = yc / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                # clamp to [0,1]
                xc_norm = max(0.0, min(1.0, xc_norm))
                yc_norm = max(0.0, min(1.0, yc_norm))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                lines.append(f"{cls} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

            # write label file (may be empty)
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines))

    print(f"Writing {len(train_imgs)} train and {len(val_imgs)} val images.")
    process_split(train_imgs, train_images_dir, train_labels_dir)
    process_split(val_imgs, val_images_dir, val_labels_dir)

    # write data.yaml
    data_yaml = out_dir / "data.yaml"
    yaml_content = f"""train: train/images
val: valid/images
nc: {nc}
names: {names}
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"Created dataset at {out_dir} with data.yaml (nc={nc}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True, help="Path to master_dataset (must contain _annotations.coco.json and images/)")
    parser.add_argument("--output", required=True, help="Output dir for YOLO dataset (e.g., vehicle_damage_yolo)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of images to use for training (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()
    master_dir = Path(args.master)
    out_dir = Path(args.output)
    convert(master_dir, out_dir, train_ratio=args.train_ratio, seed=args.seed)
