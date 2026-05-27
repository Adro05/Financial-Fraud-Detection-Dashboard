# 💳 Financial Fraud Analytics Engine

<div align="center">

### AI-Powered Financial Anomaly Detection using Calibrated Isolation Forest & Advanced Analytics

Analyze transaction datasets intelligently using unsupervised Isolation Forest models, calibrated risk scoring, multidimensional scatter spaces, and dynamic confusion matrix performance matrices.

---

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Analytics-success?style=for-the-badge)

</div>

---

# 🔗 Project Links

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Adro05/Financial-Fraud-Detection-Dashboard)

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-red?style=for-the-badge&logo=streamlit)](https://financial-fraud-detection-dashboard-dy5oa7nk5v4qmnejbjrcyn.streamlit.app/)

---

# 🚀 Features

## 🚨 Unsupervised Anomaly Detection
- Leverages **Isolation Forest** algorithms to isolate anomalies in multidimensional feature spaces.
- Fully capable of **zero-day threat detection** without requiring historic label streaming or supervised training.

---

## 🧪 Synthetic Data Stream Generator
- Provides an **interactive, customizable credit card transaction stream** with selectable sample sizes (up to 3,000 records).
- Generates realistic correlations between transaction values, latent components, and actual fraud classifications for immediate dashboard evaluation.

---

## ⚖️ Calibrated Fraud Risk Scoring
- Extracts raw continuous anomaly scores from isolation trees.
- Scales and calibrates scores linearly into a standardized **0% to 100% Fraud Risk Score** for clear, actionable compliance analysis.

---

## ⚙️ Sidebar Control Center
Dynamic modeling controls allowing immediate, real-time retraining:
- Expected contamination rate sliders (`0.001` to `0.05`).
- Forest estimator sliders (`50` to `300`).
- Seamless toggle between Synthetic Data and Custom CSV uploads.

---

## 📊 Dimensional Visualizer Space
- **2D & 3D Interactive Scatter Spaces**: Map high-dimensional features (e.g. Time, Amount, V1-V3 components) colored by continuous risk gradients.
- **Risk Distribution Histogram**: Interactive histograms tracking frequency densities across marked categories.

---

# 🖥️ Dashboard Preview

The dashboard provides:

- Dynamic glassmorphic metric cards (processed volume, anomaly count, ratio, model accuracy).
- 2D & 3D dimensionality visualizer panes.
- Continuous risk score distribution analysis.
- Live threshold slider filtering anomalies above specified risk percentages.
- Inter-active Confusion Matrix heatmap.
- CSV downloader for suspicious transaction registries.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend logic and computations |
| Streamlit | Web application rendering |
| Scikit-learn | Machine learning training and anomaly detection |
| Isolation Forest | Unsupervised multidimensional feature isolating model |
| Plotly Express | Dynamic 2D/3D Scatter visualizers and risk histograms |
| Plotly Figure Factory | Annotated Confusion Matrix heatmaps |
| Pandas & NumPy | High-performance data cleaning and preprocessing |

---

# 📂 Project Structure

```bash
Financial-Fraud-Detection-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

---

# 🧭 System Architecture

```text
                ┌─────────────────────┐
                │ Transaction Upload/ │
                │  Synthetic Stream   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Robust Preprocessor │
                │  (NaN & Inf Fixes)  │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Attribute Matcher   │
                │ (Amounts / Classes) │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  Isolation Forest   │
                │  (Cached Engine)    │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Risk Score Scale    │
                │ (Calibrated 0-100%) │
                └─────────┬───────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
  ┌─────────────┐ ┌────────────────┐ ┌─────────────────┐
  │ KPI Metrics │ │ Interactive 2D │ │ Confusion Matrix│
  │ (Glassmorphic)│ │ & 3D Plots     │ │ Heatmap (Ground)│
  └──────┬──────┘ └────────┬───────┘ └────────┬────────┘
         │                 │                  │
         └─────────────────┼──────────────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Suspicious Registry │
                 │  & CSV Downloader   │
                 └─────────────────────┘
```

---

# ⚙️ Workflow

## Step 1 — Data Ingestion
User selects the synthetic transaction stream generator or uploads a custom transaction CSV.

---

## Step 2 — Robust Clean-up & Preprocessing
The engine performs a full clean-up: replacing infinite floats with `NaN` and imputing missing data using column-level median values to prevent machine learning crashes.

---

## Step 3 — Isolation Forest Training
The model builds isolation trees over the preprocessed numerical feature space utilizing dynamic contamination settings chosen by the analyst in the sidebar.

---

## Step 4 — Calibration & Anomaly Mapping
Continuous outlier scoring is mapped into a calibrated `0 - 100%` Fraud Risk Score, separating high-probability alerts from standard transactions.

---

## Step 5 — Real-time Visual Decisioning
The dashboard populates the glassmorphic KPI grid, custom plot visualizations, confusion matrices (if ground-truth labels are matched), and provides downloadable filtered compliance reports.

---

# ✨ Core Functionalities

| Functionality | Description |
|---|---|
| Unsupervised Engine | Isolation Forest model isolating anomalies using custom contaminant trees |
| Risk Score Calibration | Continuous anomaly scoring mapped to user-friendly risk levels (0-100%) |
| Synthetic Generator | Dynamic dataset builder simulating 1500+ credit card transactions |
| Adaptive Columns | Automatically scans and matches Amount and Class columns case-insensitively |
| Exporter Registry | Fully filterable registries with direct downloads to CSV |

---

# 🎨 UI Features

- Frosted glass metric dashboards with vivid color accents.
- Responsive translation cards featuring gentle hover movements.
- Full 2D/3D Plotly dark template layouts.
- Side panels managing model hyperparameters dynamically.
- Smooth transitions and clean typographic hierarchies.

---

# ▶️ Run Locally

```bash
# Setup environment and install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

---

# 🌐 Live Demo
https://financial-fraud-detection-dashboard-dy5oa7nk5v4qmnejbjrcyn.streamlit.app/

---

# 📸 Screenshots

## Dashboard Analytics Overview

![Dashboard Metrics](C:/Users/Lenovo/.gemini/antigravity-ide/brain/b78d7b69-889b-408e-9899-ee1cc5649b78/dashboard_metrics.png)

---

# 🔮 Future Improvements

- Transformer-based recurrent neural network (RNN) temporal fraud sequence predictions.
- Integration of Apache Kafka for real-time transaction ingestion.
- Semi-supervised interactive feedback loops allowing fraud analysts to re-label transactions.
- Automated webhook triggers sending real-time slack/email alerts for high-risk flags (>90%).
- Native integration with Plaid and Stripe transactions developer API gateways.

---

# 👨‍💻 Author

### Aadhya Rohatgi

B.Tech Data Science Student  
AI • ML • NLP • Data Science