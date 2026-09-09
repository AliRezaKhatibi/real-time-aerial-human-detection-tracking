# Real-Time Aerial Human Detection, Tracking & Monitoring

<p align="center">

**A Deep Learning and Computer Vision Framework for Real-Time Aerial Human Detection, Multi-Object Tracking, and Optimized Inference**

<br>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-red?logo=pytorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.21-purple?logo=onnx)](https://onnx.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-11-green)](https://developer.nvidia.com/tensorrt)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-blue?logo=opencv)](https://opencv.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia)](https://developer.nvidia.com/cuda)

</p>
## 🎥 Real-Time Tracking Demo

| YOLO26s + ByteTrack | RT-DETR-R18 + ByteTrack |
|:---:|:---:|
| Demo video | Demo video |
---

## Overview

This repository presents a complete experimental framework for **real-time aerial human detection, multi-object tracking, inference optimization, and performance evaluation**.

The project investigates the complete computer-vision pipeline from dataset preparation and controlled model training to optimized inference and multi-object tracking.

Three detector architectures were systematically evaluated:

* **YOLO26s**
* **RT-DETR-R18**
* **BPD-YOLOn/L-FPN**

Two multi-object tracking algorithms were evaluated:

* **ByteTrack**
* **BoT-SORT**

Three inference backends were benchmarked:

* **PyTorch**
* **ONNX Runtime**
* **NVIDIA TensorRT FP16**

The final experimental analysis demonstrates a clear trade-off between **accuracy, throughput, latency, memory consumption, and identity stability** rather than a single universally optimal architecture.

---

## Research Pipeline

```text
                     ┌─────────────────────────┐
                     │   Aerial Image / Video  │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   Dataset Preparation   │
                     │   & Class Normalization │
                     └────────────┬────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │          Object Detection            │
               │                                     │
               │ YOLO26s │ RT-DETR-R18 │ BPD-YOLO   │
               └────────────────┬────────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────────┐
               │       Inference Optimization         │
               │                                     │
               │ PyTorch → ONNX → TensorRT FP16      │
               └────────────────┬────────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────────┐
               │      Multi-Object Tracking           │
               │                                     │
               │       ByteTrack │ BoT-SORT          │
               └────────────────┬────────────────────┘
                                │
                                ▼
               ┌─────────────────────────────────────┐
               │       Multi-Criteria Evaluation      │
               │                                     │
               │ Accuracy │ FPS │ Latency │ VRAM     │
               │ HOTA │ MOTA │ IDF1 │ IDSW            │
               └────────────────┬────────────────────┘
                                │
                                ▼
                     ┌─────────────────────────┐
                     │   Real-Time Monitoring  │
                     └─────────────────────────┘
```

---

# Key Contributions

### 1. Controlled detector comparison

Three fundamentally different detector profiles were evaluated under a common experimental protocol:

* YOLO26s — speed-oriented detector
* RT-DETR-R18 — accuracy-oriented transformer detector
* BPD-YOLOn/L-FPN — specialized small-object baseline

### 2. Cross-backend inference evaluation

Each detector was evaluated through:

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT FP16
```

This enabled direct analysis of the effect of inference backend on accuracy, latency, throughput, memory and deployment artifacts.

### 3. Multi-object tracking evaluation

Six detector–tracker combinations were evaluated:

```text
3 Detectors × 2 Trackers = 6 Configurations
```

using:

* HOTA
* MOTA
* IDF1
* IDSW
* IDSW / 100 frames
* DetA
* AssA
* MOTP
* MOTP-related tracking statistics
* Fragmentation

### 4. Pareto analysis

The project does not select a model using accuracy alone.

Two complementary Pareto perspectives were investigated:

* **Accuracy vs. Speed**
* **Tracking Quality vs. Identity Switching**

### 5. Aerial Tiny-Person Curriculum (ATPC)

An additional training strategy was investigated to improve sensitivity to very small aerial persons.

ATPC augments the training distribution using **2,800 derived tiny-context tiles** without modifying the inference architecture.

---

# Dataset

## Detection Dataset

The controlled detector experiments were performed using the **person-only VisDrone2019-DET** configuration.

The original categories:

```text
pedestrian
people
```

were mapped into a unified target class:

```text
person
```

### Dataset Split

| Split                 |      Images |
| --------------------- | ----------: |
| Official training set |       6,471 |
| Validation set        |         548 |
| Target class          |      person |
| Input resolution      | 1280 × 1280 |

The independent final test set was intentionally excluded from intermediate training, conversion, calibration and benchmarking procedures.

---

# Training Protocol

The three detector architectures were trained under a controlled comparison protocol.

| Model           | Best Epoch | Best mAP50-95 |        AP50 |
| --------------- | ---------: | ------------: | ----------: |
| YOLO26s         |         30 |   **0.31060** | **0.67438** |
| RT-DETR-R18     |         45 |   **0.34854** | **0.70839** |
| BPD-YOLOn/L-FPN |         43 |   **0.29366** | **0.65557** |

### Interpretation

RT-DETR-R18 achieved the highest raw detection accuracy.

However, the final selection was not based on detection accuracy alone because real-time operation also requires:

* high throughput
* low latency
* manageable VRAM
* stable tracking
* low identity switching

---

# Inference Optimization

Each detector was exported and evaluated through multiple inference backends.

```text
PyTorch
   │
   ▼
ONNX
   │
   ▼
TensorRT FP16
```

### Experimental Environment

| Component    | Version / Hardware |
| ------------ | ------------------ |
| GPU          | NVIDIA L4          |
| Python       | 3.12.13            |
| PyTorch      | 2.13.0             |
| CUDA         | 13.0               |
| Ultralytics  | 8.4.116            |
| ONNX         | 1.21.0             |
| ONNX Runtime | 1.24.4             |
| TensorRT     | 11.2.1.2           |
| OpenCV       | 5.0.0              |

### Common Evaluation Configuration

| Parameter                 | Value |
| ------------------------- | ----: |
| Input size                |  1280 |
| Batch size                |     1 |
| Warm-up runs              |    10 |
| Operating confidence      |  0.25 |
| Operating IoU             |  0.50 |
| NMS IoU                   |  0.70 |
| Maximum detections        | 3,000 |
| Accuracy confidence floor | 0.001 |

---

# Detection Accuracy

The final TensorRT-FP16 comparison produced the following results:

| Model               |     mAP50-95 |         AP50 |  Precision |     Recall |      AP Tiny |
| ------------------- | -----------: | -----------: | ---------: | ---------: | -----------: |
| **YOLO26s**         |     0.312938 |     0.679592 | **0.7532** |     0.5872 |     0.194015 |
| **RT-DETR-R18**     | **0.353666** | **0.726303** |     0.3520 | **0.8286** | **0.221301** |
| **BPD-YOLOn/L-FPN** |     0.294211 |     0.654936 |     0.7390 |     0.5673 |     0.188526 |

### Main observation

**RT-DETR-R18** achieved the highest:

* mAP50-95
* AP50
* Recall
* AP Tiny

while **YOLO26s** achieved the highest Precision.

---

# Detection by Object Size

| Model           |      AP Tiny |     AP Small |    AP Medium |     AP Large |
| --------------- | -----------: | -----------: | -----------: | -----------: |
| YOLO26s         |     0.194015 |     0.388334 |     0.472091 |     0.458569 |
| BPD-YOLOn/L-FPN |     0.188526 |     0.366153 |     0.434958 |     0.330066 |
| RT-DETR-R18     | **0.221301** | **0.423062** | **0.533897** | **0.745512** |

RT-DETR-R18 maintained the strongest detection performance across the evaluated object-size categories.

---

# Backend Accuracy Comparison

## YOLO26s

| Backend       | mAP50-95 |     AP50 |     AP75 | Precision | Recall |
| ------------- | -------: | -------: | -------: | --------: | -----: |
| PyTorch       | 0.313769 | 0.678953 | 0.241723 |    0.7520 | 0.5865 |
| ONNX          | 0.314200 | 0.679744 | 0.243708 |    0.7536 | 0.5866 |
| TensorRT FP16 | 0.312938 | 0.679592 | 0.239396 |    0.7532 | 0.5872 |

## RT-DETR-R18

| Backend       | mAP50-95 |     AP50 |     AP75 | Precision | Recall |
| ------------- | -------: | -------: | -------: | --------: | -----: |
| PyTorch       | 0.357379 | 0.729532 | 0.297640 |    0.3538 | 0.8320 |
| ONNX          | 0.357441 | 0.729552 | 0.297755 |    0.3538 | 0.8321 |
| TensorRT FP16 | 0.353666 | 0.726303 | 0.290430 |    0.3520 | 0.8286 |

## BPD-YOLOn/L-FPN

| Backend       | mAP50-95 |     AP50 |     AP75 | Precision | Recall |
| ------------- | -------: | -------: | -------: | --------: | -----: |
| PyTorch       | 0.295872 | 0.658070 | 0.205344 |    0.7448 | 0.5685 |
| ONNX          | 0.295906 | 0.657923 | 0.207742 |    0.7463 | 0.5678 |
| TensorRT FP16 | 0.294211 | 0.654936 | 0.207369 |    0.7390 | 0.5673 |

The ONNX conversion preserved accuracy very closely to PyTorch, while TensorRT FP16 introduced only limited changes.

---

# Inference Performance

All performance measurements below were obtained on an **NVIDIA L4**.

## Final TensorRT-FP16 Profile

| Model           |        FPS | Mean Latency (ms) | P95 Latency (ms) | Peak VRAM (MiB) | Engine Size (MiB) |
| --------------- | ---------: | ----------------: | ---------------: | --------------: | ----------------: |
| **YOLO26s**     | **72.944** |        **13.709** |           14.509 |             668 |            256.53 |
| RT-DETR-R18     |     50.412 |            19.836 |           20.575 |             820 |             63.81 |
| BPD-YOLOn/L-FPN |     70.240 |            14.237 |           17.524 |         **594** |            150.55 |

### Speed ranking

```text
YOLO26s             72.944 FPS
BPD-YOLOn/L-FPN     70.240 FPS
RT-DETR-R18         50.412 FPS
```

### Latency ranking

```text
YOLO26s             13.709 ms
BPD-YOLOn/L-FPN     14.237 ms
RT-DETR-R18         19.836 ms
```

Therefore, YOLO26s provided the strongest throughput/latency profile.

---

# Full Backend Speed Comparison

## YOLO26s

| Backend       |        FPS | Mean Latency (ms) |   P95 (ms) |
| ------------- | ---------: | ----------------: | ---------: |
| PyTorch       |     53.316 |            18.756 |     19.821 |
| ONNX          |     47.357 |            21.116 |     21.646 |
| TensorRT FP16 | **72.944** |        **13.709** | **14.509** |

## RT-DETR-R18

| Backend       |        FPS | Mean Latency (ms) |   P95 (ms) |
| ------------- | ---------: | ----------------: | ---------: |
| PyTorch       |     23.981 |            41.700 |     42.623 |
| ONNX          |     20.429 |            48.950 |     49.983 |
| TensorRT FP16 | **50.412** |        **19.836** | **20.575** |

## BPD-YOLOn/L-FPN

| Backend       |        FPS | Mean Latency (ms) |   P95 (ms) |
| ------------- | ---------: | ----------------: | ---------: |
| PyTorch       |     53.874 |            18.562 |     19.715 |
| ONNX          |     45.672 |            21.895 |     22.554 |
| TensorRT FP16 | **70.240** |        **14.237** | **17.524** |

TensorRT FP16 produced the highest throughput for all three architectures.

---

# Latency Decomposition

TensorRT-FP16 mean latency was decomposed into preprocessing, inference and postprocessing:

| Model           | Pre (ms) | Inference (ms) | Post (ms) | Total Mean (ms) | P95 (ms) |
| --------------- | -------: | -------------: | --------: | --------------: | -------: |
| YOLO26s         |    7.709 |          4.832 |     0.769 |      **13.709** |   14.509 |
| BPD-YOLOn/L-FPN |    7.667 |          5.390 |     0.768 |      **14.237** |   17.524 |
| RT-DETR-R18     |   13.919 |          5.548 |     0.353 |      **19.836** |   20.575 |

The results show that preprocessing is a significant component of end-to-end latency, particularly for RT-DETR-R18.

---

# Memory and Artifact Analysis

## Peak VRAM

| Model           | PyTorch |      ONNX |    TensorRT |
| --------------- | ------: | --------: | ----------: |
| YOLO26s         | 512 MiB | 1,352 MiB |     668 MiB |
| RT-DETR-R18     | 782 MiB |   724 MiB |     820 MiB |
| BPD-YOLOn/L-FPN | 562 MiB |   922 MiB | **594 MiB** |

## Artifact Size

| Model           | Checkpoint |      ONNX |   TensorRT |
| --------------- | ---------: | --------: | ---------: |
| YOLO26s         |  19.43 MiB | 36.88 MiB | 256.53 MiB |
| RT-DETR-R18     | 307.19 MiB | 77.72 MiB |  63.81 MiB |
| BPD-YOLOn/L-FPN |   3.64 MiB | 11.11 MiB | 150.55 MiB |

Artifact size and runtime memory consumption are therefore not directly interchangeable measures.

---

# Pareto Analysis

The accuracy–speed analysis produced three distinct profiles.

### YOLO26s

**Speed-oriented profile**

* Highest FPS
* Lowest mean latency
* Competitive detection accuracy
* Strong tracking stability

### RT-DETR-R18

**Accuracy-oriented profile**

* Highest mAP50-95
* Highest AP50
* Highest Recall
* Highest AP Tiny
* Higher latency and VRAM consumption

### BPD-YOLOn/L-FPN

**Memory-oriented baseline**

* Lowest TensorRT VRAM
* High throughput
* Lower detection accuracy
* Dominated by YOLO26s in the accuracy–speed plane

YOLO26s and RT-DETR-R18 form the principal accuracy–speed Pareto trade-off, while BPD-YOLOn/L-FPN is dominated by YOLO26s in this comparison.

---

# Multi-Object Tracking

Tracking evaluation was conducted on:

* **5 video clips**
* **300 frames per clip**
* **1,500 total frames**

Six detector–tracker combinations were evaluated.

## Complete Tracking Results

| Detector        | Tracker   |        HOTA |        MOTA |        IDF1 |   IDSW | IDSW / 100 |
| --------------- | --------- | ----------: | ----------: | ----------: | -----: | ---------: |
| BPD-YOLOn/L-FPN | BoT-SORT  |     0.37480 |     0.30541 |     0.50824 |     18 |       1.20 |
| BPD-YOLOn/L-FPN | ByteTrack |     0.36092 |     0.29320 |     0.47098 |     30 |       2.00 |
| RT-DETR-R18     | BoT-SORT  | **0.46474** |     0.34265 | **0.61090** |     59 |       3.93 |
| RT-DETR-R18     | ByteTrack |     0.45780 |     0.34625 |     0.59828 |     82 |       5.47 |
| YOLO26s         | BoT-SORT  |     0.40225 | **0.34658** |     0.55767 | **12** |   **0.80** |
| YOLO26s         | ByteTrack |     0.39019 |     0.34457 |     0.53266 |     23 |       1.53 |

---

# Tracking Component Analysis

The detector component metrics obtained during tracking evaluation were:

| Model           |   Precision |      Recall |          F1 |
| --------------- | ----------: | ----------: | ----------: |
| YOLO26s         | **0.63394** |     0.52973 |     0.57352 |
| RT-DETR-R18     |     0.55981 | **0.65946** | **0.60405** |
| BPD-YOLOn/L-FPN |     0.55606 |     0.51899 |     0.53605 |

RT-DETR-R18 provides higher recall and F1, whereas YOLO26s provides higher precision.

---

# Effect of Tracker Selection

BoT-SORT consistently improved tracking quality relative to ByteTrack.

### YOLO26s

```text
HOTA:  0.39019 → 0.40225
IDF1:  0.53266 → 0.55767
IDSW:  23 → 12
```

### RT-DETR-R18

```text
HOTA:  0.45780 → 0.46474
IDF1:  0.59828 → 0.61090
IDSW:  82 → 59
```

### BPD-YOLOn/L-FPN

```text
HOTA:  0.36092 → 0.37480
IDF1:  0.47098 → 0.50824
IDSW:  30 → 18
```

The improvement of BoT-SORT over ByteTrack was therefore consistent across all three detector architectures.

---

# Tracking Pareto Perspective

Two complementary profiles emerged.

### RT-DETR-R18 + BoT-SORT

Best for:

* HOTA
* IDF1
* Overall association quality

```text
HOTA = 0.46474
IDF1 = 0.61090
```

### YOLO26s + BoT-SORT

Best for:

* MOTA
* Identity stability
* Lowest ID switching

```text
MOTA     = 0.34658
IDSW     = 12
IDSW/100 = 0.80
```

Thus, the tracking results reinforce the same fundamental trade-off observed during detection.

---

# Aerial Tiny-Person Curriculum (ATPC)

An additional training strategy was investigated for improving detection of very small aerial persons.

ATPC does **not modify the inference architecture**.

Instead, it changes the training distribution by adding derived tiny-context samples.

## Training Configuration

| Component                  |       Value |
| -------------------------- | ----------: |
| Base architecture          |     YOLO26s |
| Official training images   |       6,471 |
| Derived tiny-context tiles |       2,800 |
| Total training images      |       9,271 |
| Validation images          |         548 |
| Input resolution           | 1280 × 1280 |
| Batch size                 |           8 |
| Optimizer                  |       AdamW |
| Seed                       |          42 |
| Baseline training budget   |   35 epochs |
| Target class               |      person |

---

# ATPC Results

| Metric        | YOLO26s Baseline | YOLO26s + ATPC |         Change |
| ------------- | ---------------: | -------------: | -------------: |
| Best mAP50-95 |          0.31060 |   **≈0.34058** |   **+0.02998** |
| AP50 / mAP50  |          0.67438 |     **≈0.727** | **≈ +0.05262** |
| Precision     |                — |         ≈0.765 |              — |
| Recall        |                — |         ≈0.650 |              — |

The best observed mAP50-95 improvement was approximately:

```text
0.31060 → 0.34058
```

corresponding to approximately:

```text
+0.02998 absolute
≈ 9.65% relative improvement
```

### Important experimental qualification

ATPC should be interpreted as a **training-recipe improvement**, not as a perfectly budget-matched architectural ablation.

The baseline uses:

```text
6,471 official training images
```

whereas ATPC uses:

```text
6,471 official images
+
2,800 derived tiny-context tiles
=
9,271 training samples
```

Therefore, the result demonstrates that the ATPC training recipe improves YOLO26s performance within this project, but does not isolate a purely architectural contribution.

---

# Final Model Selection

The project deliberately avoids selecting a model using a single metric.

The final decision considered:

```text
Detection Accuracy
        +
Throughput
        +
Latency
        +
VRAM
        +
Tracking Quality
        +
Identity Stability
```

## Final Operational Configuration

```text
YOLO26s
   +
TensorRT FP16
   +
BoT-SORT
```

### Final Detection Profile

```text
FPS           = 72.944
Mean Latency  = 13.709 ms
mAP50-95      = 0.312938
Peak VRAM     = 668 MiB
```

### Final Tracking Profile

```text
HOTA         = 0.40225
IDF1         = 0.55767
MOTA         = 0.34658
IDSW         = 12
IDSW / 100   = 0.80
```

---

# Why YOLO26s?

YOLO26s was selected as the final operational detector because it provides the strongest overall real-time profile.

### Key advantages

* **72.944 FPS**
* **13.709 ms mean latency**
* Strong precision
* Competitive mAP50-95
* Highest throughput among the evaluated models
* Lowest mean latency
* Lowest identity-switch rate when combined with BoT-SORT
* Highest MOTA among the six evaluated detector–tracker configurations

This is not a claim that YOLO26s is the most accurate detector.

Instead:

> **RT-DETR-R18 is the accuracy leader, while YOLO26s is the real-time operational leader.**

---

# Final Model Profiles

| Criterion           | YOLO26s       | RT-DETR-R18       | BPD-YOLOn/L-FPN |
| ------------------- | ------------- | ----------------- | --------------- |
| Detection accuracy  | High          | **Highest**       | Lower           |
| FPS                 | **Highest**   | Lowest            | High            |
| Latency             | **Lowest**    | Highest           | Low             |
| AP Tiny             | Medium        | **Highest**       | Lowest          |
| Peak VRAM           | Medium        | Highest           | **Lowest**      |
| HOTA + BoT-SORT     | Medium        | **Highest**       | Lowest          |
| IDF1 + BoT-SORT     | Medium        | **Highest**       | Lowest          |
| MOTA + BoT-SORT     | **Highest**   | High              | Lower           |
| IDSW + BoT-SORT     | **Lowest**    | Higher            | Medium          |
| Operational profile | **Real-time** | Accuracy-oriented | Memory-oriented |

---

# Reproducibility

All performance figures reported in this repository should be interpreted within the corresponding experimental protocol.

In particular:

* Detection benchmarking used the VisDrone2019-DET validation set.
* The validation set contained **548 images**.
* Tracking evaluation used **1,500 frames** across five 300-frame clips.
* Inference benchmarks were performed on an **NVIDIA L4**.
* TensorRT engines are hardware-dependent.
* TensorRT engines should be rebuilt for a different target GPU.
* Accuracy, latency, FPS and VRAM should be re-evaluated after rebuilding the engine on different hardware.

The reported NVIDIA L4 performance should therefore **not be interpreted as universal hardware-independent performance**.

---

# Software Architecture

The complete experimental system follows a modular architecture:

```text
                 ┌────────────────────────┐
                 │       Input Video      │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │   Pre-processing       │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Object Detector        │
                 │                        │
                 │ YOLO26s                │
                 │ RT-DETR-R18            │
                 │ BPD-YOLOn/L-FPN        │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Inference Backend      │
                 │                        │
                 │ PyTorch                │
                 │ ONNX Runtime           │
                 │ TensorRT FP16          │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Multi-Object Tracker   │
                 │                        │
                 │ ByteTrack              │
                 │ BoT-SORT               │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ Evaluation & Monitoring │
                 │                        │
                 │ FPS / Latency          │
                 │ HOTA / MOTA / IDF1     │
                 │ IDSW / VRAM            │
                 └────────────────────────┘
```

---

# Project Structure

```text
real-time-aerial-human-detection-tracking/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── methodology.md
│   ├── dataset.md
│   ├── detection.md
│   ├── tracking.md
│   ├── inference-optimization.md
│   ├── experiments.md
│   └── reproducibility.md
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
│
├── configs/
│   ├── detection/
│   ├── tracking/
│   └── inference/
│
├── scripts/
│
├── notebooks/
│
├── experiments/
│
├── results/
│   ├── detection/
│   ├── tracking/
│   ├── benchmarks/
│   └── figures/
│
├── assets/
│   ├── architecture/
│   ├── qualitative-results/
│   └── figures/
│
└── tests/
```

---

# Limitations

Several limitations should be considered when interpreting the reported results.

### Hardware dependency

The reported FPS, latency and VRAM measurements were obtained on an NVIDIA L4 and should not be directly generalized to other GPUs.

### Dataset dependency

Detection performance depends strongly on the characteristics and object-size distribution of the VisDrone validation set.

### ATPC comparison

The ATPC experiment uses additional derived training samples and therefore is not a strictly equal-budget architectural ablation.

### Tracking evaluation size

Tracking results were obtained on five 300-frame clips, corresponding to 1,500 frames in total.

### TensorRT portability

TensorRT engines are hardware- and configuration-dependent and should be rebuilt on the target GPU.

---

# Future Work

Potential research directions include:

* Independent final-test evaluation after all model decisions are frozen
* Rebuilding TensorRT engines on additional GPU architectures
* More extensive tiny-person evaluation
* Larger-scale tracking benchmarks
* Additional detector architectures
* Quantization beyond FP16
* End-to-end pipeline optimization
* Preprocessing acceleration
* Temporal modeling for aerial video
* Improved small-object feature representation
* More systematic ATPC ablation studies
* Confidence and threshold sensitivity analysis
* Cross-dataset generalization

---

# Experimental Summary

The complete experimental study can be summarized as:

```text
                         ACCURACY
                            ▲
                            │
                   RT-DETR-R18
                            │
                            │
                            │
                            │
          BPD ──────────────┼──────────── YOLO26s
                            │
                            │
                            └──────────────────► SPEED
```

The experiments demonstrate that the best model depends on the optimization objective:

```text
Maximum Detection Accuracy
        ↓
RT-DETR-R18

Maximum Real-Time Throughput
        ↓
YOLO26s

Minimum TensorRT VRAM
        ↓
BPD-YOLOn/L-FPN

Best HOTA / IDF1
        ↓
RT-DETR-R18 + BoT-SORT

Best MOTA / Identity Stability
        ↓
YOLO26s + BoT-SORT
```

---

# Final Experimental Configuration

## Detector

```text
YOLO26s
```

## Inference Backend

```text
TensorRT FP16
```

## Tracker

```text
BoT-SORT
```

## Reference Hardware

```text
NVIDIA L4
```

## Detection Performance

```text
72.944 FPS
13.709 ms mean latency
0.312938 mAP50-95
```

## Tracking Performance

```text
HOTA       0.40225
IDF1       0.55767
MOTA       0.34658
IDSW       12
IDSW/100   0.80
```

---

# Citation

If you use this repository, methodology, experimental results, or derived implementations in academic work, please cite this project according to the citation information provided in `CITATION.cff`.

---

# License

See [`LICENSE`](LICENSE) for the applicable license and redistribution terms.

---

<p align="center">

**Real-Time Aerial Human Detection, Tracking & Monitoring**

<br>

*Deep Learning · Computer Vision · Object Detection · Multi-Object Tracking · TensorRT · Real-Time Inference*

</p>
