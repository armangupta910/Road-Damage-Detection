import os
import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Enhanced Road Damage Detection",
    page_icon="",
    layout="centered",
)

HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "xxx"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# ------------------ MODEL ------------------
cache_key = "yolo_model"
if cache_key not in st.session_state:
    st.session_state[cache_key] = YOLO(MODEL_LOCAL_PATH)

net = st.session_state[cache_key]

# ------------------ UTILS ------------------

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    union = bbox_area(box1) + bbox_area(box2) - inter

    return inter / union if union > 0 else 0


def apply_nms(detections, iou_thresh=0.5):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    keep = []

    for det in detections:
        if all(compute_iou(det["bbox"], k["bbox"]) < iou_thresh for k in keep):
            keep.append(det)

    return keep

def compute_union_area(detections, image_shape):
    H, W = image_shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])

        # clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        mask[y1:y2, x1:x2] = 1

    union_area = np.sum(mask)

    return union_area

# ------------------ ANALYTICS ------------------

def compute_severity(detections, image_shape):
    H, W = image_shape
    image_area = H * W

    class_weights = {
        "Longitudinal Crack": 0.5,
        "Transverse Crack": 0.6,
        "Alligator Crack": 0.8,
        "Potholes": 1.0
    }

    # ------------------ UNION MASK ------------------
    union_mask = np.zeros((H, W), dtype=np.uint8)

    # ------------------ WEIGHT MAP ------------------
    weight_map = np.zeros((H, W), dtype=np.float32)

    # ------------------ CLASS-WISE ------------------
    class_pixel_map = {}

    for d in detections:
        cls = d["class"]
        conf = d["confidence"]
        x1, y1, x2, y2 = map(int, d["bbox"])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # union mask (binary)
        union_mask[y1:y2, x1:x2] = 1

        # weight
        weight = conf * class_weights.get(cls, 1.0)

        # weighted map (NO double counting)
        weight_map[y1:y2, x1:x2] = np.maximum(
            weight_map[y1:y2, x1:x2],
            weight
        )

        # class-wise pixel tracking
        if cls not in class_pixel_map:
            class_pixel_map[cls] = np.zeros((H, W), dtype=np.uint8)

        class_pixel_map[cls][y1:y2, x1:x2] = 1

    # ------------------ FINAL METRICS ------------------

    union_area = np.sum(union_mask)
    weighted_area = np.sum(weight_map)

    severity = union_area / image_area
    weighted_severity = weighted_area / image_area

    # class-wise severity (pixel accurate)
    class_severity = {
        k: np.sum(v) / image_area for k, v in class_pixel_map.items()
    }

    return severity, weighted_severity, class_severity


def compute_density(detections, image_shape):
    if len(detections) <= 1:
        return 0

    centers = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        centers.append(((x1+x2)/2, (y1+y2)/2))

    visited = [False] * len(centers)
    clusters = 0

    threshold = 0.05 * max(image_shape)  # adaptive

    for i in range(len(centers)):
        if visited[i]:
            continue

        clusters += 1
        visited[i] = True

        for j in range(i+1, len(centers)):
            if visited[j]:
                continue

            dist = np.linalg.norm(
                np.array(centers[i]) - np.array(centers[j])
            )

            if dist < threshold:
                visited[j] = True

    return clusters


def compute_spread(detections, image_shape):
    if not detections:
        return 0

    x_min = min(d["bbox"][0] for d in detections)
    y_min = min(d["bbox"][1] for d in detections)
    x_max = max(d["bbox"][2] for d in detections)
    y_max = max(d["bbox"][3] for d in detections)

    spread_area = (x_max - x_min) * (y_max - y_min)
    return spread_area / (image_shape[0] * image_shape[1])


def compute_risk(severity, weighted_severity, count, density, spread):
    count_norm = min(count / 20, 1.0)
    density_norm = min(density / 10, 1.0)

    score = (
        0.35 * severity +
        0.25 * weighted_severity +
        0.15 * count_norm +
        0.15 * density_norm +
        0.10 * spread
    )

    return int(score * 100)


def classify_risk(score):
    if score < 30:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


def compute_rci(weighted_severity):
    return max(0, int((1 - weighted_severity) * 100))


# ------------------ UI ------------------

st.title("Enhanced Road Damage Detection")

image_file = st.file_uploader("Upload Image", type=["png", "jpg"])

conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

if image_file:
    image = Image.open(image_file)
    img = np.array(image)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (640, 640))

    results = net.predict(img, conf=conf_thresh)

    # ------------------ DETECTIONS ------------------
    detections = []
    for r in results:
        if r.boxes:
            for i in range(len(r.boxes)):
                detections.append({
                    "class": r.names[int(r.boxes.cls[i])],
                    "confidence": float(r.boxes.conf[i]),
                    "bbox": r.boxes.xyxy[i].tolist()
                })

    # Filtering + NMS
    detections = [d for d in detections if d["confidence"] > conf_thresh]
    detections = apply_nms(detections)

    # ------------------ METRICS ------------------
    severity, weighted_severity, class_severity = compute_severity(detections, (h, w))
    density = compute_density(detections, (h, w))
    spread = compute_spread(detections, (h, w))
    count = len(detections)

    risk_score = compute_risk(severity, weighted_severity, count, density, spread)
    risk_label = classify_risk(risk_score)
    RCI = compute_rci(weighted_severity)

    # ------------------ DISPLAY ------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original")

    with col2:
        pred_img = cv2.resize(results[0].plot(), (w, h))
        st.image(pred_img, caption="Prediction")

        buffer = BytesIO()
        Image.fromarray(pred_img).save(buffer, format="PNG")
        st.download_button("Download Image", buffer.getvalue())

    # ------------------ ANALYTICS ------------------
    st.write("## Damage Analytics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Severity", f"{severity:.3f}")
    c2.metric("Weighted Severity", f"{weighted_severity:.3f}")
    c3.metric("RCI", f"{RCI}")

    st.write(f"### Risk Score: {risk_score} ({risk_label})")

    if risk_label == "LOW":
        st.success(" Safe Road")
    elif risk_label == "MEDIUM":
        st.warning(" Moderate Damage")
    else:
        st.error(" Severe Damage")

    # ------------------ CLASS BREAKDOWN ------------------
    st.write("## Class-wise Severity")

    if class_severity:
        df = pd.DataFrame({
            "Class": list(class_severity.keys()),
            "Severity": list(class_severity.values())
        })

        st.bar_chart(df.set_index("Class"))

        worst = max(class_severity, key=class_severity.get)
        st.warning(f"⚠️ Most Severe: {worst}")

    # Extra insights
    st.write(f"**Density (clusters):** {density}")
    st.write(f"**Spread:** {spread:.3f}")
    st.write(f"**Detections Count:** {count}")