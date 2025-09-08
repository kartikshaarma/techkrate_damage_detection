#!/usr/bin/env python3
"""
print_summary.py (robust)

Runs YOLOv8 validation (safe on Windows) and prints a clean summary.
Also writes results_summary.txt in the project root.

Usage:
    python print_summary.py
"""
from ultralytics import YOLO
from pathlib import Path
import time
import json

MODEL_PATH = "runs/vehicle_damage_yolov8m/weights/best.pt"
DATA_PATH = r"C:\Users\ksh16\Downloads\techkrate vehicle damage detection\vehicle_damage_yolo\data.yaml"
IMG_SIZE = 640
CONF = 0.30
AUGMENT = True

def find_metric_key(results_dict, keywords):
    """Find the first key in results_dict that contains any of the keywords (case-insensitive)."""
    for k in results_dict.keys():
        lk = k.lower()
        for kw in keywords:
            if kw.lower() in lk:
                return k
    return None

def fmt(val):
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)

def main():
    t0 = time.time()
    model = YOLO(MODEL_PATH)
    print(f"Running validation: model={MODEL_PATH}, data={DATA_PATH}, imgsz={IMG_SIZE}, conf={CONF}, augment={AUGMENT}")
    # run safe for windows: workers=0
    res = model.val(data=DATA_PATH, imgsz=IMG_SIZE, conf=CONF, augment=AUGMENT, workers=0, verbose=False)
    elapsed = time.time() - t0

    rd = {}
    # ultralytics returns results object with .results_dict or .metrics; be defensive
    if hasattr(res, "results_dict") and isinstance(res.results_dict, dict):
        rd = res.results_dict
    elif isinstance(res, dict):
        rd = res
    else:
        # try to coerce via __dict__
        rd = getattr(res, "__dict__", {})
    # keys for humans to search
    # typical possible substrings: precision, recall, mAP50, mAP50-95
    precision_key = find_metric_key(rd, ["precision", "prec"])
    recall_key = find_metric_key(rd, ["recall", "rec"])
    map50_key = find_metric_key(rd, ["map50", "mAP50", "map_50", "mAP@.5", "map@50"])
    map5095_key = find_metric_key(rd, ["map50-95", "map50_95", "map_0.5_0.95", "map@.5:.95", "map"])

    # Fallbacks: sometimes Ultralytics prefixes with "metrics/..."
    if map50_key is None:
        map50_key = find_metric_key(rd, ["metrics/mAP50", "metrics/map50", "metrics/mAP50(B)"])
    if precision_key is None:
        precision_key = find_metric_key(rd, ["metrics/precision", "metrics/precision(B)"])
    if recall_key is None:
        recall_key = find_metric_key(rd, ["metrics/recall"])
    if map5095_key is None:
        map5095_key = find_metric_key(rd, ["metrics/mAP50-95", "metrics/mAP50-95(B)"])

    summary_lines = []
    summary_lines.append("=== Model Summary ===")
    summary_lines.append(f"Model: {MODEL_PATH}")
    summary_lines.append(f"Dataset: {DATA_PATH}")
    summary_lines.append(f"Images evaluated (elapsed): {int(elapsed)}s")
    # print metrics if found
    if precision_key:
        summary_lines.append(f"Precision ({precision_key}): {fmt(rd.get(precision_key))}")
    else:
        summary_lines.append("Precision: (not found)")

    if recall_key:
        summary_lines.append(f"Recall ({recall_key}): {fmt(rd.get(recall_key))}")
    else:
        summary_lines.append("Recall: (not found)")

    if map50_key:
        summary_lines.append(f"mAP50 ({map50_key}): {fmt(rd.get(map50_key))}")
    else:
        summary_lines.append("mAP50: (not found)")

    if map5095_key:
        # If map5095_key is equal to map50_key (both map entries), try to find a different key
        if map5095_key == map50_key:
            # look for any other 'map' key
            other_map = None
            for k in rd.keys():
                if ("map" in k.lower() or "mAP" in k) and k != map50_key:
                    other_map = k; break
            if other_map:
                map5095_key = other_map
        summary_lines.append(f"mAP50-95 ({map5095_key}): {fmt(rd.get(map5095_key))}")
    else:
        summary_lines.append("mAP50-95: (not found)")

    summary_lines.append("======================")
    out_text = "\n".join(summary_lines)
    print("\n" + out_text + "\n")

    # save to results_summary.txt
    Path("results_summary.txt").write_text(out_text + "\n\nraw_results_keys:\n" + json.dumps(list(rd.keys()), indent=2))
    print("Wrote results_summary.txt")

if __name__ == "__main__":
    main()
