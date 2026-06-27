# ECG Arrhythmia Detection using CNN · BiLSTM · Attention Mechanism

> An end-to-end deep learning framework for automated heartbeat classification trained on the MIT-BIH Arrhythmia Database, deployed via FastAPI and Streamlit.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://ecg-arrhythmia-detection1.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Abstract

Electrocardiogram (ECG) analysis plays a critical role in detecting cardiac arrhythmias and supporting clinical decision-making. Manual interpretation of ECG recordings is labour-intensive and demands significant domain expertise. This project presents an end-to-end deep learning framework for automated heartbeat classification using a hybrid **Convolutional Neural Network (CNN)**, **Bidirectional Long Short-Term Memory (BiLSTM)**, and **Attention** architecture.

The system is trained on the **MIT-BIH Arrhythmia Database** and classifies individual heartbeats into AAMI-standard arrhythmia categories. The complete solution encompasses data preprocessing, heartbeat segmentation, class balancing via SMOTE, model training, REST API deployment using FastAPI, and a Streamlit-based user interface.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Mathematical Motivation](#2-mathematical-motivation)
3. [Dataset](#3-dataset)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Deep Learning Architecture](#5-deep-learning-architecture)
6. [Training Strategy](#6-training-strategy)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results](#8-results)
9. [Project Structure](#9-project-structure)
10. [REST API Documentation](#10-rest-api-documentation)
11. [Deployment Architecture](#11-deployment-architecture)
12. [Technology Stack](#12-technology-stack)
13. [Future Work](#13-future-work)
14. [Author](#14-author)

---

## 1. Problem Statement

The objective is to develop an automated arrhythmia detection system capable of classifying ECG heartbeats into clinically meaningful categories.

Given a discrete ECG signal:

```math
X = \{x_1,\, x_2,\, \ldots,\, x_n\} \quad x_i \in \mathbb{R}
```

the goal is to learn a mapping:

```math
f_\theta : X \rightarrow Y, \quad Y \in \{\,N,\; S,\; V,\; F,\; Q\,\}
```

where $\theta$ denotes the trainable parameters of the neural network and $Y$ represents the predicted heartbeat class.

**Core challenges addressed:**

| Challenge | Mitigation Strategy |
|---|---|
| Noisy physiological signals | Bandpass filtering + Z-score normalisation |
| Temporal dependencies | BiLSTM sequence modelling |
| Severe class imbalance | SMOTE oversampling on training split |
| Inter-patient variability | Patient-independent train/test split |
| Real-time deployment | FastAPI + async inference endpoint |

---

## 2. Mathematical Motivation

### 2.1 Convolutional Feature Extraction

A 1-D convolution over the ECG signal $x$ with learnable kernel $w$ of length $k$ is defined as:

```math
y(t) = \sum_{i=1}^{k} w_i \cdot x_{t-i} + b
```

This operation learns to detect localised morphological patterns such as the QRS complex, P-wave, and T-wave within each heartbeat segment.

### 2.2 Bidirectional LSTM

Standard LSTM gates at time step $t$:

```math
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
```
```math
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
```
```math
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
```
```math
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
```
```math
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
```
```math
h_t = o_t \odot \tanh(c_t)
```

BiLSTM concatenates forward and backward hidden states to capture bidirectional context:

```math
h_t = \left[\,\overrightarrow{h_t}\;;\;\overleftarrow{h_t}\,\right] \in \mathbb{R}^{2d}
```

### 2.3 Additive Attention Mechanism

An alignment score $e_t$ is computed via a learned scoring function, and the attention weights are normalised using a softmax:

```math
e_t = v^\top \tanh(W_h h_t + b_a)
```

```math
\alpha_t = \frac{\exp(e_t)}{\displaystyle\sum_{j=1}^{T} \exp(e_j)}
```

The resulting context vector is a weighted sum over all hidden states:

```math
c = \sum_{t=1}^{T} \alpha_t \, h_t
```

This enables the model to focus selectively on diagnostically significant segments of the ECG beat, improving both performance and interpretability.

---

## 3. Dataset

**Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — PhysioNet

### 3.1 Dataset Characteristics

| Attribute | Value |
|---|---|
| Total Records | 48 |
| Sampling Frequency | 360 Hz |
| Signal Channels | 2 (MLII + V1 or V5) |
| Duration per Record | ~30 minutes |
| Annotation Source | Expert Cardiologists |
| Total Heartbeats | > 100,000 |

### 3.2 File Format

Each record comprises three files:

```
100.dat    # Raw binary ECG waveform signal
100.hea    # Header: sampling frequency, signal length, gain, units
100.atr    # Annotation: R-peak sample indices + beat labels
```

### 3.3 AAMI Class Mapping

MIT-BIH uses over 20 beat symbols. Following AAMI EC57 recommendations, these are consolidated into five clinically relevant classes:

| AAMI Class | Description | MIT-BIH Symbols |
|---|---|---|
| **N** | Normal Beat | N, L, R, e, j |
| **S** | Supraventricular Ectopic | A, a, J, S |
| **V** | Ventricular Ectopic | V, E |
| **F** | Fusion Beat | F |
| **Q** | Unknown / Paced | /, f, Q, others |

### 3.4 Class Distribution

The dataset exhibits severe class imbalance, motivating the use of SMOTE:

| Class | Label | Count | Proportion |
|---|---|---|---|
| Normal | N | 90,608 | 82.8% |
| Unknown | Q | 8,042 | 7.3% |
| Ventricular | V | 7,235 | 6.6% |
| Supraventricular | S | 2,781 | 2.5% |
| Fusion | F | 802 | 0.7% |

```
Class Distribution (Pre-SMOTE)

N  |████████████████████████████████████████████| 90,608
Q  |████                                        |  8,042
V  |███                                         |  7,235
S  |█                                           |  2,781
F  |                                            |    802
```

---

## 4. Data Preprocessing Pipeline

```
Raw MIT-BIH Records (.dat / .hea / .atr)
              │
              ▼
    ┌─────────────────────┐
    │  Step 1: Signal      │   wfdb.rdrecord()  +  wfdb.rdann()
    │  Loading             │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 2: Heartbeat   │   R-peak ± 90 samples  →  180-sample window
    │  Segmentation        │   Shape: (180, 1)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 3: Normalisa-  │   Z-score per beat: x' = (x - μ) / σ
    │  tion                │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 4: Label       │   N→0  S→1  V→2  F→3  Q→4
    │  Encoding            │   One-hot for categorical crossentropy
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 5: Train /     │   Train 80%  │  Val 10%  │  Test 10%
    │  Val / Test Split    │   (patient-independent)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Step 6: SMOTE       │   Applied to training set ONLY
    │  Oversampling        │   Balances minority classes
    └─────────────────────┘
```

### Heartbeat Segmentation

R-peak annotations provide the centre of each beat. A fixed-width window is extracted:

```
Sample index:  [R - 90]  ···  [R]  ···  [R + 90]
                  └──────── 180 samples ────────┘
```

This window reliably captures the full P-QRS-T morphology at 360 Hz.

---

## 5. Deep Learning Architecture

### 5.1 Network Overview

```
Input ECG Beat  (180 × 1)
        │
        ▼
┌───────────────────┐
│  Conv1D (32, k=5) │   Local morphology detection
│  BatchNorm + ReLU │
│  MaxPooling (2)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Conv1D (64, k=3) │   Hierarchical feature extraction
│  BatchNorm + ReLU │
│  MaxPooling (2)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  BiLSTM (128)     │   Bidirectional temporal modelling
│  h_t = [fwd; bwd]│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Attention Layer  │   Adaptive focus on salient timesteps
│  Context vector c │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dense (64, ReLU) │
│  Dropout (0.4)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Dense (5)        │   One neuron per AAMI class
│  Softmax          │
└───────────────────┘

Output: P(Y = c | X),  c ∈ {N, S, V, F, Q}
```

### 5.2 Component Responsibilities

| Component | Role | ECG Relevance |
|---|---|---|
| Conv1D (1st) | Low-level feature maps | P-wave, QRS onset detection |
| Conv1D (2nd) | Higher-order patterns | QRS morphology, ST segment |
| BiLSTM | Sequential context | Beat rhythm and rate dynamics |
| Attention | Selective weighting | Focuses on QRS peak and T-wave |
| Dropout | Regularisation | Reduces overfitting to training patients |

---

## 6. Training Strategy

### 6.1 Configuration

| Hyperparameter | Value |
|---|---|
| Framework | TensorFlow 2.x / Keras |
| Optimiser | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 512 |
| Max Epochs | 100 (Early Stopping) |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping | patience=10, monitor=val\_loss |
| Input Shape | (180, 1) |

### 6.2 Learning Rate Schedule

```
LR
 │
1e-3 ──┐
       │ plateau detected
       └──── 5e-4 ──┐
                    └──── 2.5e-4 ──── ...
                              │
                         Early Stop
```

### 6.3 SMOTE Details

SMOTE synthesises new minority-class samples by interpolating between existing feature vectors in the training split only:

```math
x_{\text{new}} = x_i + \lambda \cdot (x_{\hat{k}} - x_i), \quad \lambda \sim \mathcal{U}(0,1)
```

where $x_{\hat{k}}$ is a randomly selected k-nearest neighbour of $x_i$ in the same class.

---

## 7. Evaluation Metrics

All metrics are computed per-class and macro-averaged over the five AAMI categories.

### Accuracy

```math
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
```

### Precision

```math
\text{Precision} = \frac{TP}{TP + FP}
```

### Recall (Sensitivity)

```math
\text{Recall} = \frac{TP}{TP + FN}
```

### F1 Score

```math
F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

### Matthews Correlation Coefficient

For imbalanced datasets, MCC provides a more robust single-figure summary:

```math
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
```

---

## 8. Results

### 8.1 Overall Performance

```
Test Accuracy:  99.18%
```

### 8.2 Per-Class Classification Report

The model achieved an overall test accuracy of **99.18%** on the MIT-BIH Arrhythmia Database.

Class-wise Precision, Recall, and F1-Scores were computed during evaluation using `sklearn.metrics.classification_report`. Due to ongoing experimentation and hyperparameter tuning, only the overall performance metric is reported here.

Detailed evaluation reports and confusion matrix visualizations are available in the `reports/` directory.

| Class | Description |
|---------|-------------|
| N | Normal Beat |
| S | Supraventricular Ectopic Beat |
| V | Ventricular Ectopic Beat |
| F | Fusion Beat |
| Q | Unknown Beat |

### 8.3 Example Prediction — MIT-BIH Record 100

```json
{
  "record": "100",
  "beat_distribution": {
    "N": 2237,
    "S": 33,
    "V": 1
  },
  "risk_assessment": {
    "risk_level": "Low",
    "dominant_class": "N",
    "ventricular_burden_pct": 0.04
  }
}
```

### 8.4 Confusion Matrix

A confusion matrix was generated during evaluation to analyze class-wise prediction performance.

The matrix exhibited strong diagonal dominance, indicating that the majority of heartbeats were correctly classified into their respective categories.

The complete confusion matrix visualization can be found in:

```text
reports/confusion_matrix.png
```

This analysis helps identify misclassification patterns between clinically similar arrhythmia categories and provides additional insight beyond overall accuracy.
```
## 9. Project Structure


ECG-ARRHYTHMIA-DETECTION/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── predict.py
│   │   │       ├── waveform.py
│   │   │       ├── rpeaks.py
│   │   │       └── heartbeats.py
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   └── signal_processing.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── utils/
│   │   │   ├── ecg_utils.py
│   │   │   └── risk_engine.py
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── components/
│   │   ├── waveform_plot.py
│   │   └── result_card.py
│   ├── pages/
│   │   ├── home.py
│   │   └── analysis.py
│   ├── services/
│   │   └── api_client.py
│   ├── utils/
│   │   └── formatting.py
│   └── app.py
│
├── training/
│   ├── preprocessing/
│   │   ├── segmentation.py
│   │   ├── normalisation.py
│   │   └── smote_balancing.py
│   ├── datasets/
│   │   └── mitbih_loader.py
│   ├── models/
│   │   ├── architecture.py
│   │   └── attention.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualise.py
│   └── notebooks/
│       ├── 01_eda.ipynb
│       ├── 02_preprocessing.ipynb
│       ├── 03_training.ipynb
│       └── 04_evaluation.ipynb
│
├── models/
│   ├── best_model.keras
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── architecture.png
│   ├── training_curves.png
│   └── metrics.csv
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 10. REST API Documentation

Base URL: `https://<your-render-service>.onrender.com`

### Endpoints

| Endpoint | Method | Description | Input |
|---|---|---|---|
| `/api/v1/predict` | POST | Full arrhythmia prediction | MIT-BIH record number |
| `/api/v1/waveform` | POST | ECG waveform data | Record number, channel |
| `/api/v1/rpeaks` | POST | R-peak detection | Record number |
| `/api/v1/heartbeats` | POST | Heartbeat segment extraction | Record number |

### Request / Response Example — `/api/v1/predict`

**Request**

```http
POST /api/v1/predict
Content-Type: application/json

{
  "record_id": "100",
  "channel": 0
}
```

**Response**

```json
{
  "status": "success",
  "record": "100",
  "total_beats": 2271,
  "distribution": {
    "N": 2237,
    "S": 33,
    "V": 1,
    "F": 0,
    "Q": 0
  },
  "risk": {
    "risk_level": "Low",
    "ventricular_burden_pct": 0.04,
    "supraventricular_burden_pct": 1.45
  },
  "inference_time_ms": 312
}
```

### Error Responses

| Code | Meaning |
|---|---|
| 400 | Invalid record ID or malformed request |
| 404 | Record not found in PhysioNet cache |
| 500 | Model inference failure |

---

## 11. Deployment Architecture

```
                    ┌──────────────────────────┐
                    │         User              │
                    │   (Browser / Mobile)      │
                    └────────────┬─────────────┘
                                 │  HTTPS
                                 ▼
                    ┌──────────────────────────┐
                    │   Streamlit Frontend      │
                    │   Streamlit Community     │
                    │   Cloud                   │
                    └────────────┬─────────────┘
                                 │  REST (JSON)
                                 ▼
                    ┌──────────────────────────┐
                    │   FastAPI Backend         │
                    │   Render (Docker)         │
                    │   Uvicorn ASGI Server     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  CNN · BiLSTM · Attention │
                    │  TensorFlow / Keras       │
                    │  best_model.keras         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   MIT-BIH PhysioNet       │
                    │   WFDB Signal Loader      │
                    └──────────────────────────┘
```

**Frontend:** [Streamlit Community Cloud](https://streamlit.io/cloud)
**Backend:** [Render](https://render.com) — containerised FastAPI service

---

## 12. Technology Stack

### Machine Learning

| Library | Purpose |
|---|---|
| TensorFlow 2.x | Model training and inference |
| Keras | High-level neural network API |
| Scikit-Learn | Preprocessing, metrics, splitting |
| Imbalanced-Learn | SMOTE oversampling |

### Signal Processing

| Library | Purpose |
|---|---|
| WFDB | MIT-BIH record loading and annotation parsing |
| NumPy | Numerical array operations |
| SciPy | Digital filtering (bandpass) |

### Data & Visualisation

| Library | Purpose |
|---|---|
| Pandas | Tabular data management |
| Matplotlib | Static training curves, confusion matrix |
| Plotly | Interactive ECG waveform plots |

### Backend

| Library | Purpose |
|---|---|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request / response validation |

### Frontend

| Library | Purpose |
|---|---|
| Streamlit | Interactive web application |

### Deployment & DevOps

| Tool | Purpose |
|---|---|
| Render | Backend cloud deployment |
| Streamlit Cloud | Frontend hosting |
| Docker | Containerisation |
| Git / GitHub | Version control |

---

## 13. Future Work

- **Transformer-based architectures** — replace BiLSTM with a time-series Transformer (e.g., PatchTST) for improved long-range dependency modelling
- **Multi-lead ECG classification** — extend from single-lead (MLII) to full 12-lead input for richer clinical representation
- **Explainability** — integrate Grad-CAM and SHAP visualisations to highlight diagnostically relevant waveform regions
- **Real-time ECG streaming** — WebSocket-based pipeline for live beat-by-beat classification
- **Federated learning** — patient-private model training across hospital sites without centralising raw ECG data
- **Mobile deployment** — TensorFlow Lite conversion for edge inference on wearable devices
- **Clinical report generation** — automated structured PDF reports with rhythm analysis and risk stratification

---

## 14. Author

**Harshith Devaraja**
M.Sc. Applied Mathematics and Computing

Research Interests: Machine Learning · Deep Learning · Signal Processing · Time Series Analysis · Healthcare AI · Applied Mathematics

---

*For questions, issues, or collaborations, please open a GitHub Issue or reach out via the repository.*
