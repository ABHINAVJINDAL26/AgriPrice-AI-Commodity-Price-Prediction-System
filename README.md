# 🌾 AgriPrice AI — Commodity Price Prediction System

<div align="center">

![SIH 2024](https://img.shields.io/badge/SIH-2024-orange?style=for-the-badge)
![Problem ID](https://img.shields.io/badge/Problem_ID-SIH1647-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-SARIMA%20%7C%20ARIMA%20%7C%20XGBoost-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI/ML based price prediction system for agri-horticultural commodities**  
*Built for Smart India Hackathon 2024 | Ministry of Consumer Affairs, Food and Public Distribution*

</div>

---

> [!NOTE]
> **Demo & Presentation Disclaimer**: This application is currently configured in **Demo Mode** using a high-fidelity simulated historical market dataset (`data/raw_data.csv`) that models actual price spikes, seasonal harvest dips, and festival demands in Rupees per Quintal (e.g., Onion base at Rs. 1,500/quintal). In production, this can be seamlessly integrated with the `data.gov.in` real-time API for live government reporting center feed ingestion.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [ML Models Used](#-ml-models-used)
- [Results & Evaluation](#-results--evaluation)
- [Future Enhancements](#-future-enhancements)
- [Team & Institution](#-team--institution)
- [License](#-license)

---

## 📌 Problem Statement

| Field | Details |
|-------|---------|
| **Problem ID** | SIH1647 |
| **Title** | Development of AI-ML based models for predicting prices of agri-horticultural commodities such as pulses and vegetables (onion, potato, tomato) |
| **Organization** | Ministry of Consumer Affairs, Food and Public Distribution |
| **Department** | Department of Consumer Affairs |
| **Theme** | Agriculture, FoodTech & Rural Development |

### Background

The Department of Consumer Affairs monitors the daily prices of 22 essential food commodities through 550 price reporting centres across the country. The Department also maintains a buffer stock of pulses (gram, tur, urad, moong, and masur) and onion for strategic market interventions to stabilize price volatility.

### Goal

Build an intelligent AI/ML prediction system that forecasts commodity prices 7 to 30 days in advance, enabling the government to make proactive buffer stock decisions, prevent price spikes, and protect consumer and farmer interests.

---

## 🎯 Project Overview

**AgriPrice AI** is an end-to-end machine learning pipeline that:
* 📥 **Ingests** daily price data from government reporting centres.
* 🔄 **Processes** and cleans multi-source government datasets.
* 📊 **Analyzes** historical price trends, seasonality, and market patterns.
* 🤖 **Predicts** commodity prices for the next 7, 14, and 30 days with confidence intervals.
* 🚨 **Alerts** when prices are predicted to cross danger thresholds at both national and state levels.
* 📈 **Visualizes** forecasts through a multi-tab interactive Streamlit dashboard.

---

## ⚙️ How This Project Works

```
GOVERNMENT WEBSITES
(data.gov.in, agmarknet.gov.in)
         |
         v
DATA COLLECTION LAYER (API call or CSV feed)
         |
         v
DATA CLEANING & PREPROCESSING (Grouped by Commodity/State, Outliers removed)
         |
         v
FEATURE ENGINEERING (Lag features, rolling averages, seasonal flags)
         |
         v
MODEL TRAINING (SARIMA, ARIMA, XGBoost trained per commodity)
         |
         v
PRICE PREDICTION & EVALUATION (30-day forecast, confidence bands, MAPEs)
         |
         v
OUTPUTS / DASHBOARD (Interactive charts, state-wise alerts, policy recommendations)
```

---

## 🛠️ Tech Stack

* **Core Language**: Python 3.10+
* **Data Processing & Analytics**: Pandas, NumPy, SciPy
* **Visualization**: Plotly Express, Matplotlib, Seaborn
* **Machine Learning**: Statsmodels (ARIMA, SARIMA), Scikit-Learn, XGBoost
* **Dashboard Interface**: Streamlit
* **Job Scheduler**: Schedule library

---

## 📁 Project Structure

```
AgriPrice-AI/
├── main.py                     # Main pipeline orchestrator & scheduler
├── app.py                      # Streamlit dashboard script
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw_data.csv            # Raw price dataset (statewise/commodity-wise)
│   └── processed_data.csv      # Enriched feature engineered dataset
├── src/
│   ├── generate_data.py        # Simulated market data generator
│   ├── data_preprocessing.py   # Cleaning and lag/rolling feature extraction
│   ├── eda.py                  # Exploratory Data Analysis plotting script
│   ├── model.py                # Model training (ARIMA, SARIMA, XGBoost)
│   └── evaluation.py           # Model evaluation, CI plotting, statewise alerts
├── models/                     # Saved model pickles (.pkl) and registry
├── outputs/                    # Diagnostic plots and policy reports
└── notebook/                   # Jupyter development notebooks
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ABHINAVJINDAL26/AgriPrice-AI-Commodity-Price-Prediction-System.git
   cd AgriPrice-AI-Commodity-Price-Prediction-System
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Activate on Windows:
   .venv\Scripts\activate
   # Activate on Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Demo Data** (Optional):
   ```bash
   python src/generate_data.py
   ```

---

## 🚀 How to Run

* **Run the Pipeline (Preprocessing, Training, Evaluation)**:
  ```bash
  python main.py
  ```
* **Launch the Interactive Dashboard**:
  ```bash
  python main.py dashboard
  ```
* **Start the Daily Automated Pipeline Scheduler (6:00 AM)**:
  ```bash
  python main.py schedule
  ```

---

## 📊 Results & Evaluation

The pipeline outputs comparative performance metrics on the test partition, saving the benchmark to `outputs/model_benchmark.csv`:

* **ARIMA**: Baseline model used as a classical time-series benchmark.
* **SARIMA**: Models seasonality cycles and provides 95% confidence intervals (visualized as shaded bands).
* **XGBoost**: Employs rolling statistical features and lag prices for multi-factor regression, showing high predictive accuracy.

All diagnostics (e.g. Model Comparisons and XGBoost Feature Importances) are updated dynamically on the dashboard.

---

## 🔮 Future Enhancements

* [ ] **Live API Ingestion**: Direct pipeline integration with live `data.gov.in` feeds.
* [ ] **Geospatial Maps**: State-wise visual color-graded pricing maps of India.
* [ ] **Market Sentiment Ingestion**: Analyzing agriculture news headlines to anticipate market shocks.

---

## 👥 Team & Institution

* **Abhinav Jindal** (Team Lead) — *Lead Developer, Pipeline Orchestration, & ML Modeling*
* **Parth Madrewar** — *Frontend Dashboard Developer & Data Analyst*

**Institution**: Lovely Professional University, Phagwara, Jalandhar, Punjab  
**SIH 2024 Participation**

---

## 📄 License

This project is licensed under the MIT License.
