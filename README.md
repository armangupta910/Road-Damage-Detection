# 🚧 Enhanced Road Damage Detection with Severity Estimation and Intelligent Risk Scoring

## Overview

This project extends traditional road damage detection by introducing **quantitative severity estimation** and an **intelligent risk scoring mechanism**.

Instead of only detecting damages, the system analyzes:
- Spatial extent of damage
- Detection confidence
- Distribution patterns

to generate a **real-time road condition assessment**.

---

## Key Contributions

- YOLOv8-based road damage detection  
- Pixel-level **severity estimation (union of regions)**  
- **Confidence-weighted severity**  
- Spatial analysis:
  - Density (clustering)
  - Spread (coverage)
- Multi-factor **risk scoring**
- Road Condition Index (RCI)

---

## Pipeline

```
Image → YOLOv8 → Bounding Boxes → Pixel Union Mask
→ Severity → Density + Spread → Risk Score → RCI
```

---

## Dataset Preparation

- Dataset: **RDD2022 (India subset)**
- Annotation format: Pascal VOC → YOLOv8

### Steps:
1. Parse XML annotations  
2. Convert to YOLO format `(class, cx, cy, w, h)`  
3. Normalize coordinates  
4. Filter classes:
   - Longitudinal Crack
   - Transverse Crack
   - Alligator Crack
   - Potholes  

---

## Dataset Split

- Train / Validation: **90 / 10**
- Background images: ~10%
- Seed: `1337`

---

## Training

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data=_data,
    epochs=100,
    warmup_epochs=5,
    batch=32,
    imgsz=640,
    save_period=10,
    workers=1,
    project="runs/RDD_India",
    name="Baseline_YOLOv8Small_Filtered",
    seed=1337,
    cos_lr=True,
    mosaic=0.0
)
```

---

## Validation

```python
model = YOLO(weight_path)
metrics = model.val(data=_data)
```

---

## Inference Pipeline

* Input image
* YOLO detection
* Extract:
  * Bounding boxes
  * Confidence scores
  * Class labels
* Apply:
  * Confidence filtering
  * Non-Max Suppression (NMS)

---

## Metrics (Core Contribution)

---

### 1. Severity (Pixel-Based)

Measures actual damaged area using **union of bounding boxes**

```
Severity = Union Area of Damage / Total Image Area
```

✔ No double counting  
✔ Pixel-accurate

---

### 2. Weighted Severity

Accounts for detection confidence and class importance

```
Weighted Severity = Sum(max(pixel weights)) / Image Area
```

Where:

* Pixel weight = Confidence × Class Weight

### Class Weights:

| Class              | Weight |
| ------------------ | ------ |
| Longitudinal Crack | 0.5    |
| Transverse Crack   | 0.6    |
| Alligator Crack    | 0.8    |
| Potholes           | 1.0    |

---

### 3. Density

Measures clustering of damages based on spatial proximity.

---

### 4. Spread

Measures how widely damage is distributed:

```
Spread = Bounding Region Area / Image Area
```

---

### 5. Risk Score

Multi-factor weighted score:

```
Risk =
0.35 * Severity +
0.25 * Weighted Severity +
0.15 * Count +
0.15 * Density +
0.10 * Spread
```

---

### 6. Road Condition Index (RCI)

```
RCI = (1 - Weighted Severity) * 100
```

### Interpretation:

| RCI    | Condition |
| ------ | --------- |
| 80–100 | Good      |
| 50–80  | Moderate  |
| <50    | Severe    |

---

## 🚀 Installation

```bash
pip install ultralytics streamlit opencv-python numpy pandas
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## Output

* Bounding boxes on detected damage
* Severity score
* Risk score
* RCI value
* Class-wise breakdown

---

## Limitations

* Bounding-box based (not segmentation)
* No real-world scale calibration
* Single image analysis (no temporal tracking)

---

## Future Work

* Segmentation-based severity
* GPS-based road mapping
* Temporal damage tracking
* ML-based risk prediction
* Real-world unit estimation