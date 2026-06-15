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

## 🎓 INT312 Course Outcomes Alignment

This project directly demonstrates the practical implementation of the **INT312 (Big Data)** course outcomes:

1. **CO1: Understand The Need And Importance Of Fundamental Concepts And Principles Of Big Data**
   - *Mapping*: AgriPrice AI handles high-velocity and high-volume daily price records across 550+ reporting centers. The necessity of processing high-volume and high-velocity datasets is addressed by orchestrating an ingestion pipeline that cleanses raw tabular streams.
2. **CO2: Analyze Internal Functioning Of Different Modules Of Big Data And Hadoop**
   - *Mapping*: The preprocessing architecture mimics distributed ingestion and storage setups. In production, this scales by storing raw feeds in HDFS and executing MapReduce-based preprocessing steps.
3. **CO3: Illustrate The Big Data Ecosystem And Appreciate Its Key Components**
   - *Mapping*: The system implements a complete analytical ecosystem: Data Collection (ingestion layer), Preprocessing/Feature Engineering (analytical transformation), Model Training & Registry (machine learning layer), and Streamlit (visualization delivery).
4. **CO4: Evaluate How Hadoop Solves Big Data Problems**
   - *Mapping*: Illustrates how parallel processing resolves data bottlenecks. By dividing work into state-wise processing groups and training models concurrently, it mirrors Hadoop's distributed node computations.
5. **CO5: Apply Tools And Techniques To Analyze Big Data**
   - *Mapping*: Applied state-of-the-art big data analysis frameworks: Pandas for large-scale data manipulation, XGBoost for gradient boosted regression, Statsmodels for parametric time-series, and Plotly for interactive dashboards.
6. **CO6: Examine Solution For A Given Problem Using Suitable Big Data Techniques**
   - *Mapping*: Directly solved the Ministry of Consumer Affairs' problem of market volatility. The system leverages predictive analytics to output automated market stabilization recommendations (e.g. NAFED buffer stock release requirements in Metric Tons).

---

## 👥 Team & Institution

* **Abhinav Jindal** (Team Lead) — *Lead Developer, Pipeline Orchestration, & ML Modeling*
* **Parth Madrewar** — *Frontend Dashboard Developer & Data Analyst*

**Institution**: Lovely Professional University, Phagwara, Jalandhar, Punjab  
**SIH 2024 Participation**

---

## 📄 License

This project is licensed under the MIT License.
