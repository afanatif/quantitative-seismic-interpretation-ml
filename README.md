# 🌋 Sub-Seismic Thin Bed Resolution Workbench

An advanced geological workstation and machine learning pipeline designed to predict reservoir properties and resolve thin sandstone beds lying below traditional seismic tuning limits ($\lambda/4$).

---

## 🚀 Key Features

* **Interactive Widess Wedge Simulator:** A real-time geophysical modeling tool. Adjust wavelet frequencies (15–60 Hz) to visualize seismic tuning limits ($\lambda/4$), constructive interference peaks, and amplitude decay trends in the sub-seismic window.
* **Multi-Track Well Log Viewer:** Compare raw seismic traces, advanced attributes (Envelope, Sweetness), inverted acoustic impedance, and Gamma Ray logs side-by-side. 
* **Raw ML Resolution Engine:** Employs LassoCV-selected Random Forest and XGBoost regressors to predict reservoir properties directly from seismic attributes without signal-flattening calibration.
* **Workstation-Grade 3D Grid Map:** Render structural horizons and property maps with fully calibrated coordinate grids (Inlines & Crosslines) and a real-time hover coordinate status bar.
* **Geologist Gallery:** Contains publication-grade QC figures representing seismic sections, well-to-seismic ties, relative acoustic impedance inversions, and thin-bed spectral slices.

---

## 🛠️ Tech Stack & Science

### 1. Attribute Engineering & Thin Bed Detection
Traditional seismic amplitude fails in thin reservoir zones due to wave cancellation. To bypass this, the pipeline extracts:
* **Envelope (Instantaneous Amplitude):** Captures energy boundaries.
* **Instantaneous Frequency:** Captures phase shifts at thin bed boundaries.
* **Sweetness ($\text{Envelope} / \sqrt{\text{Instantaneous Frequency}}$):** Isolates clean, porous sandstone zones from tight shales.

### 2. Machine Learning Architecture
* **Feature Selection:** Iterates through 42+ feature shifts using **LassoCV** to identify optimal attribute patterns.
* **Core Regressors:** Fits **Random Forest** (bagging, robust to noisy logs) and **XGBoost** (boosting, ideal for capturing sharp porosity gradients).
* **Calibration Bypassed:** Operates on raw predictions to preserve high-frequency geological variations and avoid the signal-flattening artifacts of Isotonic regression.

---

## 📂 Project Structure

```text
├── ml_training/
│   ├── train_model.py                # Main ML model training script
│   ├── generate_frontend_data.py     # Generates src/data.js (well logs database)
│   └── generate_grid_predictions.py   # Generates src/grid_data.js (3D grid predictions)
├── training_data/
│   └── build_training_data.py        # Feature extraction & training table builder
├── well_seismic/
│   └── well_seismic_tie.py           # Well-to-seismic tie alignment scripts
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # Main workstation display controller
│   │   ├── ThinBedTab.jsx            # Wedge simulator & log track component
│   │   ├── GridPredictorTab.jsx      # 3D surface map visualizer
│   │   ├── data.js                   # Mapped well database (Git Ignored)
│   │   └── grid_data.js              # Vectorized grid predictions (Git Ignored)
│   └── package.json                  # React + Vite frontend configurations
├── las/                              # High-resolution well logs (.las) (Git Ignored)
├── segy/                             # Raw SEG-Y seismic volumes (Git Ignored)
└── README.md                         # Project documentation
```

---

## 🏃‍♂️ Quick Start Guide

### Step 1: Extract Seismic Attributes & Build Dataset
```bash
python training_data/build_training_data.py
```

### Step 2: Train the ML Model
```bash
python ml_training/train_model.py
```

### Step 3: Compile Predictions & Sync Frontend Database
```bash
python ml_training/generate_frontend_data.py
python ml_training/generate_grid_predictions.py
```

### Step 4: Run the Interactive Workstation Dashboard
Navigate to the frontend folder and spin up the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5174/`** in your browser to start exploring the workstation.

---

## 🔒 Confidentiality & Git Security

To ensure sensitive well locations, log data, and proprietary seismic files are not uploaded to GitHub, a **root `.gitignore`** has been pre-configured. It automatically ignores:
* All raw data folders (`segy/`, `las/`) and extensions (`*.las`, `*.segy`, `*.sgy`)
* All intermediate spreadsheets and predictions (`*.csv`)
* All compiled frontend data files (`data.js`, `grid_data.js`)
* All trained model checkpoints (`*.joblib`)

*Do not force-stage files using `git add -f` to maintain data compliance.*
