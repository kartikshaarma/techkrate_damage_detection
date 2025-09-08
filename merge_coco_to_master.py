#!/usr/bin/env python3
"""
merge_coco_to_master.py

Usage:
    python merge_coco_to_master.py --parent Trained_dataset --output master_dataset

This script:
 - Looks for all *_annotations.coco.json files under parent_dir (recursively).
 - Re-indexes image ids and annotation ids to unique integers.
 - Merges categories by (name, supercategory) and remaps category ids.
 - Copies images into output_dir/images with unique filenames (img_00000001.jpg, ...).
 - Writes merged annotations to output_dir/_annotations.coco.json
"""
import argparse
import json
from pathlib import Path
import shutil
import re
from collections import defaultdict
from PIL import Image

def sanitize(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s)

def find_image_file(base_dir: Path, file_name: str) -> Path:
    """Try multiple heuristics to find the image file inside base_dir."""
    cand = base_dir / file_name
    if cand.exists():
        return cand
    # try basename (in case file_name contains path segments)
    cand = base_dir / Path(file_name).name
    if cand.exists():
        return cand
    # try images/ subfolder
    cand = base_dir / "images" / Path(file_name).name
    if cand.exists():
        return cand
    # recursive search for file with same basename
    for p in base_dir.rglob(Path(file_name).name):
        if p.is_file():
            return p
    return None

def merge(parent_dir: Path, out_dir: Path):
    out_images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    master = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # mapping structures
    category_key_to_newid = {}  # key=(name,supercat) -> new_id
    dataset_catid_to_newid = {} # key=(dataset_marker, old_cat_id) -> new_id
    next_cat_id = 1

    dataset_image_map = {}  # key=(dataset_marker, old_img_id) -> new_img_id
    next_image_id = 1
    next_ann_id = 1

    # optional: collect info/licenses from first file if available
    aggregated_info = None
    aggregated_licenses = []

    # find all _annotations.coco.json under parent_dir
    ann_files = list(parent_dir.rglob("_annotations.coco.json"))
    if not ann_files:
        raise SystemExit(f"No files named '_annotations.coco.json' found under {parent_dir}")

    print(f"Found {len(ann_files)} annotation files. Merging...")

    for ann_file in ann_files:
        train_dir = ann_file.parent  # the train/ folder
        dataset_marker = sanitize(str(train_dir.parent.name))  # e.g., '1st.v1i.coco'
        print(f"Processing dataset '{dataset_marker}' at {train_dir}")

        with open(ann_file, "r", encoding="utf-8") as fh:
            coco = json.load(fh)

        # collect info/licenses only from first
        if aggregated_info is None and "info" in coco:
            aggregated_info = coco["info"]
        if "licenses" in coco:
            for lic in coco.get("licenses", []):
                if lic not in aggregated_licenses:
                    aggregated_licenses.append(lic)

        # map categories for this dataset
        for cat in coco.get("categories", []):
            key = (cat.get("name"), cat.get("supercategory", ""))
            if key not in category_key_to_newid:
                new_id = next_cat_id
                category_key_to_newid[key] = new_id
                next_cat_id += 1
            else:
                new_id = category_key_to_newid[key]
            dataset_catid_to_newid[(dataset_marker, cat["id"])] = new_id

        # images: copy & remap
        for img in coco.get("images", []):
            old_img_id = img["id"]
            src_path = find_image_file(train_dir, img["file_name"])
            if src_path is None:
                print(f"WARNING: image file {img['file_name']} not found under {train_dir}. Skipping image id {old_img_id}.")
                continue

            # ensure dimensions are present; if not, read via PIL
            width = img.get("width")
            height = img.get("height")
            if width is None or height is None:
                try:
                    with Image.open(src_path) as im:
                        width, height = im.size
                except Exception as e:
                    raise RuntimeError(f"Cannot open image to get size: {src_path}") from e

            new_img_id = next_image_id
            next_image_id += 1
            ext = Path(src_path).suffix.lower()
            new_fname = f"img_{new_img_id:08d}{ext}"
            dst_path = out_images_dir / new_fname
            shutil.copy2(src_path, dst_path)

            new_img_entry = {
                "id": new_img_id,
                "file_name": new_fname,
                "width": width,
                "height": height
            }
            # keep other optional fields if present (license, coco_url, etc.)
            for k in ("license", "flickr_url", "coco_url", "date_captured"):
                if k in img:
                    new_img_entry[k] = img[k]

            master["images"].append(new_img_entry)
            dataset_image_map[(dataset_marker, old_img_id)] = new_img_id

        # annotations: remap ids and image references
        for ann in coco.get("annotations", []):
            old_ann_id = ann.get("id")
            old_image_id = ann.get("image_id")
            new_image_id = dataset_image_map.get((dataset_marker, old_image_id))
            if new_image_id is None:
                # image was not copied (missing) so skip the annotation.
                continue
            new_ann_id = next_ann_id
            next_ann_id += 1

            # map category
            new_cat_id = dataset_catid_to_newid.get((dataset_marker, ann["category_id"]))
            if new_cat_id is None:
                raise RuntimeError(f"Category id {ann['category_id']} in dataset {dataset_marker} not mapped")

            new_ann = dict(ann)  # shallow copy
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = new_cat_id
            # keep bbox, segmentation, area, iscrowd
            master["annotations"].append(new_ann)

    # build master categories list
    for (name, supercat), new_id in sorted(category_key_to_newid.items(), key=lambda x: x[1]):
        master["categories"].append({
            "id": new_id,
            "name": name,
            "supercategory": supercat
        })

    # optional attach info/licenses if we collected them
    if aggregated_info is not None:
        master["info"] = aggregated_info
    if aggregated_licenses:
        master["licenses"] = aggregated_licenses

    # write out
    out_ann_path = out_dir / "_annotations.coco.json"
    with open(out_ann_path, "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)
    print(f"Written merged annotations to {out_ann_path}")
    print(f"Copied {len(master['images'])} images and merged {len(master['annotations'])} annotations.")
    print(f"{len(master['categories'])} categories in master dataset.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True, help="Parent directory containing your datasets (e.g., Trained_dataset)")
    p.add_argument("--output", required=True, help="Output master dataset directory (e.g., master_dataset)")
    args = p.parse_args()
    parent = Path(args.parent)
    out = Path(args.output)
    if not parent.exists():
        raise SystemExit(f"Parent dir {parent} does not exist.")
    merge(parent, out)
