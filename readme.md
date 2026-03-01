# 🛒 Retail OS — Object Detection & Share of Shelf Analytics

> **YOLOv8x-powered retail shelf analysis with optimized recall and SKU-level share of shelf insights.**

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Task Breakdown](#task-breakdown)
3. [Dataset Structure](#dataset-structure)
4. [Methodology](#methodology)
5. [Model Architecture & Optimization](#model-architecture--optimization)
6. [Share of Shelf Analytics](#share-of-shelf-analytics)
7. [Installation & Setup](#installation--setup)
8. [How to Run](#how-to-run)
9. [Results & Outputs](#results--outputs)
10. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This project trains a state-of-the-art YOLOv8 object detector on a retail shelf dataset to:

1. **Detect all products (SKUs)** on store shelves with high accuracy
2. **Optimize Recall** — the baseline model misses ~32% of items (recall = 67.6%). We push this significantly higher without sacrificing meaningful precision.
3. **Share of Shelf Analytics** — treat the entire test set as one representative store shelf and compute the percentage presence of each product class (SKU).

---

## Task Breakdown

| # | Task | Approach |
|---|------|----------|
| 1 | **Model Optimization** | YOLOv8x + lower confidence threshold + heavy augmentation + AdamW + cosine LR |
| 2 | **Share of Shelf Analytics** | Aggregate all test-set detections, compute count-based and area-weighted shelf share per SKU |
| 3 | **Visualization** | Dashboard with bar charts, pie chart, donut, heatmap, and sample inference overlays |

---

## Dataset Structure

```
DATASET/
├── train/
│   ├── images/        # Training images (.jpg / .png)
│   └── labels/        # YOLO-format .txt label files
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml          # Class names + split paths
└── requirements.txt
```

### Label Format (YOLO)
Each `.txt` label file contains one bounding box per line:
```
<class_id>  <cx>  <cy>  <width>  <height>
```
All values are **normalized** to [0, 1] relative to image dimensions.

---

## Methodology

### Problem: Low Recall (67.6%)
Low recall means the model is **missing 32% of products** on the shelf — leading to inaccurate stock counts in a real store environment.

### Solution Strategy

```
High Recall Strategy
─────────────────────────────────────
1. Larger model (YOLOv8x) → better feature extraction
2. Heavy augmentation (mosaic, mixup, copy-paste) → more robust representations
3. Lower confidence threshold at inference → fewer missed detections
4. Longer training with cosine LR schedule → better convergence
5. Multi-scale training → detect small/occluded items
6. Threshold sweep analysis → find optimal precision/recall tradeoff
```

---

## Model Architecture & Optimization

### Backbone: YOLOv8x
- **YOLOv8x** is the largest/most powerful variant in the YOLOv8 family
- 68.2M parameters, highest mAP on COCO benchmark
- CSPDarknet backbone with C2f modules
- Decoupled detection head for classification and regression

### Hyperparameter Choices

| Hyperparameter | Value | Reason |
| `epochs` | 100 | Full convergence |
| `imgsz` | 640 | Standard YOLO input, captures small items |
| `optimizer` | AdamW | Better generalization than SGD for this task |
| `cos_lr` | True | Smooth decay avoids local minima |
| `mosaic` | 1.0 | Context-rich augmentation |
| `mixup` | 0.2 | Class boundary robustness |
| `copy_paste` | 0.3 | Rare class augmentation |
| `conf` (inference) | 0.15 | Lower threshold → higher recall |
| `patience` | 30 | Early stopping |

### Confidence Threshold Analysis
We evaluate the model across a sweep of confidence thresholds `[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]` and select the optimal threshold that **maximizes recall while keeping precision ≥ 0.65**.

```
conf=0.10  →  High Recall, Lower Precision
conf=0.15  →  Sweet spot (recommended)
conf=0.50  →  Low Recall, High Precision (baseline behavior)
```

### Concept
The **entire test dataset** is treated as a single representative store shelf. All product detections across all test images are aggregated.

### Two Metrics Computed

**1. Count-Based Share (%)**
```
Share(SKU_i) = (Detections of SKU_i / Total Detections) × 100
```

**2. Area-Weighted Share (%)**
```
AreaShare(SKU_i) = (Sum of BBox Areas for SKU_i / Total BBox Area) × 100
```

Area-weighted share accounts for **pack size** — a large bottle occupies more shelf space than a small sachet even if both are detected once.

### Output Visualizations
| File | Description |
|---|---|
| `share_of_shelf_dashboard.png` | 4-panel dashboard (bars, pie, donut, comparison) |
| `shelf_heatmap.png` | Heatmap of count share, area share & confidence per SKU |
| `share_of_shelf_analytics.csv` | Full tabular data per SKU |


## Results & Outputs

### Model Performance (Expected after optimization)

| Metric | Baseline | After Optimization |
|---|---|---|
| **Recall** | 67.6% | **74.63%** ↑ |
| **Precision** | — | 55.87% |
| **mAP@50** | 73.97% |
| **F1 Score** | 63.90% | 

### Generated Files

```
retail_os/
├── yolov8x_optimized/
│   ├── weights/
│   │   ├── best.pt          ← Best model checkpoint
│   │   └── last.pt
│   ├── results.csv          ← Per-epoch training metrics
│   └── plots/               ← Training curves, confusion matrix

class_distribution.png       ← EDA: class counts per split
bbox_statistics.png          ← EDA: width/height/area distributions
threshold_analysis.png       ← Precision/Recall vs conf threshold
share_of_shelf_dashboard.png ← 4-panel shelf analytics dashboard
shelf_heatmap.png            ← SKU metrics heatmap
share_of_shelf_analytics.csv ← Tabular shelf data per SKU
sample_predictions.png       ← Inference overlay on 6 test images
```

---

## Key Design Decisions

### Why YOLOv8x over YOLOv8n/s/m?
Larger models learn richer representations of occluded, small, or partially-visible products on dense retail shelves. The compute cost is worth the significant mAP gain in this task.

### Why lower confidence threshold to improve recall?
The model assigns a confidence score to every detection. By lowering the threshold from 0.5 to 0.15, we accept more detections — including lower-confidence ones that are still correct. This recovers many true positives the baseline missed.

### Why area-weighted share?
Count-based share treats a large cereal box and a small candy bar equally. Area-weighted share reflects actual physical shelf space occupied, which is more meaningful for planogram compliance and retail analytics.

### Why mosaic + mixup + copy_paste augmentation?
Retail shelves have dense, cluttered scenes with overlapping products. These augmentations force the model to handle partial occlusion and multi-product contexts, directly improving recall on hard cases.

---

## Requirements File

```
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
albumentations>=1.3.0
PyYAML>=6.0
torch>=2.0.0
torchvision>=0.15.0
```



