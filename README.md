# 🌾 AgriPrice AI — Commodity Price Prediction System

<div align="center">

![SIH 2024](https://img.shields.io/badge/SIH-2024-orange?style=for-the-badge)
![Problem ID](https://img.shields.io/badge/Problem_ID-SIH1647-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![ML](https://img.shields.io/badge/ML-SARIMA%20%7C%20LSTM%20%7C%20XGBoost-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI/ML based price prediction system for agri-horticultural commodities**
*Built for Smart India Hackathon 2024 | Ministry of Consumer Affairs, Food and Public Distribution*

</div>

> [!NOTE]
> **Demo & Presentation Disclaimer**: This application is currently configured in **Demo Mode** using a high-fidelity simulated historical market dataset (`data/raw_data.csv`) that models actual price spikes, seasonal harvest dips, and festival demands in Rupees per Quintal (e.g., Onion base at Rs. 1,500/quintal). In production, this can be seamlessly integrated with the `data.gov.in` real-time API for live government reporting center feed ingestion.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Overview](#-project-overview)
- [How This Project Works](#-how-this-project-works)
- [How to Build This Project](#-how-to-build-this-project)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset Sources](#-dataset-sources)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [ML Models Used](#-ml-models-used)
- [Results & Evaluation](#-results--evaluation)
- [Big Data Concepts Used](#-big-data-concepts-used)
- [Future Enhancements](#-future-enhancements)
- [Team](#-team)

---

## 📌 Problem Statement

| Field | Details |
|-------|---------|
| **Problem ID** | SIH1647 |
| **Title** | Development of AI-ML based models for predicting prices of agri-horticultural commodities such as pulses and vegetables (onion, potato, tomato) |
| **Organization** | Ministry of Consumer Affairs, Food and Public Distribution |
| **Department** | Department of Consumer Affairs |
| **Category** | Software |
| **Theme** | Agriculture, FoodTech & Rural Development |

### Background

The Department of Consumer Affairs monitors the **daily prices of 22 essential food commodities** through **550 price reporting centres** across the country. The Department also maintains buffer stock of pulses (gram, tur, urad, moong, and masur) and onion for strategic market interventions to stabilize price volatility.

Decisions for market interventions such as **release of stocks from the buffer** are taken on the basis of price trends and outlook. Currently, analyses are based on seasonality, historical and emerging trends, market intelligence inputs, crop sowing, and production estimates. ARIMA-based economic models have also been used to examine and forecast prices of pulses.

### Goal

> Build an intelligent AI/ML prediction system that forecasts commodity prices 7 to 30 days in advance, enabling the government to take proactive buffer stock decisions and prevent price spikes.

---

## 🎯 Project Overview

**AgriPrice AI** is an end-to-end machine learning pipeline that:

- 📥 **Ingests** daily price data from 550 government reporting centres
- 🔄 **Processes** and cleans multi-source government datasets
- 📊 **Analyzes** historical price trends, seasonality, and market patterns
- 🤖 **Predicts** commodity prices for the next 7, 14, and 30 days
- 🚨 **Alerts** when prices are predicted to cross danger thresholds
- 📈 **Visualizes** forecasts through an interactive dashboard

### Commodities Covered

| Category | Commodities |
|----------|-------------|
| **Vegetables** | Onion, Potato, Tomato |
| **Pulses** | Gram (Chana), Tur (Arhar), Urad Dal, Moong Dal, Masur Dal |

---

## ⚙️ How This Project Works

Ye section explain karta hai ki ye project **andar se kaise kaam karta hai** — step by step, bilkul simple language mein.

---

### 🔁 Complete Working Flow

```
GOVERNMENT WEBSITES
(data.gov.in, agmarknet.gov.in)
         |
         |  Daily price data fetch hoti hai
         v
DATA COLLECTION LAYER
(Python scripts se API call ya CSV download)
         |
         |  Raw data → data/raw_data.csv mein save hota hai
         v
DATA CLEANING & PREPROCESSING
(Missing values fill, outliers remove, date sort)
         |
         |  Clean data → data/processed_data.csv mein save hota hai
         v
FEATURE ENGINEERING
(Lag features, rolling average, season flag, weather data add)
         |
         |  Naye features data mein add hote hain
         v
EXPLORATORY DATA ANALYSIS (EDA)
(Charts, trends, seasonal patterns dekhe jaate hain)
         |
         |  Insights milte hain — kab price badhti hai, kab girti hai
         v
MODEL TRAINING
(SARIMA / XGBoost / LSTM model price patterns seekhta hai)
         |
         |  Trained model → models/sarima_model.pkl mein save hota hai
         v
PRICE PREDICTION
(Model agle 7 / 14 / 30 din ke prices predict karta hai)
         |
         |  Forecast ready hota hai
         v
EVALUATION
(Actual vs Predicted compare karte hain — MAE, RMSE, MAPE)
         |
         |  Accuracy check hoti hai
         v
OUTPUT / DASHBOARD
(Charts, alerts, buffer stock recommendation)
```

---

### 📖 Step-by-Step Explanation of Each Stage

---

#### Stage 1 — Data Collection (Kahan se aata hai data?)

Har din **data.gov.in** ya **agmarknet.gov.in** se onion, potato, pulses ke prices fetch hote hain. Python script API call karke JSON/CSV data download karti hai aur `data/raw_data.csv` mein store karti hai.

**Raw data ka format:**
```
Date        | Commodity | State       | Market  | Min_Price | Max_Price | Modal_Price
------------|-----------|-------------|---------|-----------|-----------|------------
2023-01-01  | Onion     | Maharashtra | Nashik  | 800       | 1200      | 1000
2023-01-01  | Potato    | UP          | Agra    | 600       | 900       | 750
2023-01-02  | Onion     | Maharashtra | Pune    | 850       | 1300      | 1050
```

---

#### Stage 2 — Data Preprocessing (Data clean karna)

Raw data mein kafi problems hote hain — missing values, wrong prices (outliers), unsorted dates. Preprocessing mein ye sab fix hota hai:

```python
# 1. Missing values fill karo (forward fill)
data = data.ffill()

# 2. Date column proper format mein
data['Date'] = pd.to_datetime(data['Date'])

# 3. Date ke hisaab se sort karo
data = data.sort_values(by='Date')

# 4. Outliers remove karo (IQR method)
Q1 = data['Modal_Price'].quantile(0.25)
Q3 = data['Modal_Price'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Modal_Price'] >= Q1 - 1.5*IQR) &
            (data['Modal_Price'] <= Q3 + 1.5*IQR)]
```

Clean data `data/processed_data.csv` mein save hota hai.

---

#### Stage 3 — Feature Engineering (Model ke liye naye features banana)

Sirf price se model accha predict nahi kar sakta. Isliye hum extra features add karte hain:

| Feature | Kya hai | Kyu zaruri hai |
|---------|---------|----------------|
| `price_lag_7` | 7 din pehle ka price | Price patterns follow karta hai |
| `price_lag_14` | 14 din pehle ka price | 2 hafte ka trend |
| `rolling_mean_7` | Last 7 din ka average | Smoothed trend dikhata hai |
| `rolling_std_7` | Last 7 din ka std deviation | Volatility measure |
| `month` | Mahina (1-12) | Seasonal pattern pakadta hai |
| `is_harvest_season` | 1 ya 0 | Harvest mein prices girte hain |
| `is_festival_season` | 1 ya 0 | Festival mein demand badhti hai |
| `rainfall_mm` | IMD se rainfall data | Zyada baarish = fasal damage = price up |
| `crop_production` | DACFW se production data | Zyada production = price down |

---

#### Stage 4 — Exploratory Data Analysis (Data ko samajhna)

Model banane se pehle data ko samajhna zaroori hai. EDA mein ye dekha jaata hai:

- **Price Trend Chart** — Price pichle saalon mein kaise upar-neeche gayi
- **Seasonal Decomposition** — Trend, Seasonality, aur Noise alag alag dikhna
- **Correlation Matrix** — Kaunse features price se zyada related hain
- **Monthly Boxplot** — Har mahine ki price distribution

**Key Insight — Onion price pattern:**
```
November - January : Prices LOW  (harvest season — supply zyada)
May - September    : Prices HIGH (off season — supply kam)
Festival months    : Prices SPIKE (demand zyada)
```

---

#### Stage 5 — Model Training (ML model ko data sikhana)

SARIMA Model historical price data se train hota hai. Model price ke patterns seekhta hai — trend, seasonality, aur cycles — aur phir future predict karta hai.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    data['Modal_Price'],
    order=(1, 1, 1),              # p=1 (AR), d=1 (differencing), q=1 (MA)
    seasonal_order=(1, 1, 1, 12)  # Yearly seasonal cycle (12 months)
)
sarima_model = model.fit(disp=False)

# Model save karo
import joblib
joblib.dump(sarima_model, 'models/sarima_model.pkl')
```

**SARIMA parameters ka matlab:**
- `p=1` — Ek previous value ka influence use karo
- `d=1` — Data ko stationary banane ke liye ek baar difference karo
- `q=1` — Ek previous prediction error ka use karo
- `s=12` — 12 mahine ka yearly seasonal cycle

---

#### Stage 6 — Prediction (Future prices predict karna)

Trained model se future prices predict hote hain:

```python
sarima_model = joblib.load('models/sarima_model.pkl')

# Agle 30 din predict karo
forecast = sarima_model.forecast(steps=30)

# Example output:
# 2024-02-01: Rs.1850/quintal
# 2024-02-15: Rs.1950/quintal
# 2024-03-01: Rs.2100/quintal  <-- Price badh rahi hai!
```

---

#### Stage 7 — Evaluation (Model kitna accurate hai?)

Actual price aur predicted price compare hoti hai:

```
Actual Price (2024-01-15):    Rs.1900/quintal
Predicted Price (2024-01-15): Rs.1850/quintal
Error:                         Rs.50  (2.6% error = 97.4% accurate)
```

**Evaluation Metrics:**

| Metric | Matlab |
|--------|--------|
| MAE (Mean Absolute Error) | Average mein kitne Rs. ka error hai |
| RMSE (Root Mean Square Error) | Bade errors ko zyada penalize karta hai |
| MAPE (Mean Absolute % Error) | % mein accuracy — e.g. 91% accurate |

---

#### Stage 8 — Buffer Stock Alert (Government ko warning dena)

Agar predicted price ek threshold se upar jaaye, system automatically alert generate karta hai:

```
=====================================================
  PRICE FORECAST REPORT — ONION
=====================================================
  Current Price    : Rs.1800/quintal
  Predicted (30d)  : Rs.2800/quintal
  Expected Change  : +55.6%
=====================================================
  ALERT: BUFFER STOCK RELEASE RECOMMENDED!
  Suggested Action: Release stocks in affected states
=====================================================
```

---

### 🔄 Real-World Use Case Example

**Scenario:** November 2024

1. System ne agmarknet se data fetch kiya — Onion price = Rs.1500/quintal
2. Preprocessing hua — Clean data ready
3. SARIMA model ne predict kiya — January 2025 mein price Rs.3000/quintal hogi
4. Alert generate hua — "Price doubling predicted in 60 days"
5. Government ne action liya — NAFED ka buffer stock release kiya
6. Market mein supply badhi — Price Rs.2000 pe stable raha
7. **Result: Consumer ko Rs.1000/quintal bachaya**

---

## 🏗️ How to Build This Project

Ye section batata hai ki **ye project kaise banaya gaya** — ek developer perspective se, poori detail mein.

---

### Phase 1 — Environment Setup (1-2 din)

#### Step 1.1 — Python Install Karo
```bash
# Python 3.10+ download karo from:
https://www.python.org/downloads/

# Version check karo
python --version
# Output: Python 3.10.x
```

#### Step 1.2 — Virtual Environment Banao
```bash
# Project folder banao
mkdir agriprice-ai
cd agriprice-ai

# Virtual environment banao
python -m venv venv

# Activate karo (Windows)
venv\Scripts\activate

# Activate karo (Mac/Linux)
source venv/bin/activate
```

#### Step 1.3 — Saari Libraries Install Karo
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn statsmodels joblib
pip install xgboost
pip install plotly
pip install requests jupyter
```

---

### Phase 2 — Project Structure Banana (1 din)

Ye folders aur files banao:
```bash
# Folders banao
mkdir data models src notebook outputs

# Source files banao
touch main.py
touch src/data_preprocessing.py
touch src/eda.py
touch src/model.py
touch src/evaluation.py
touch requirements.txt
touch README.md
```

Final structure aise dikhna chahiye:
```
agriprice-ai/
├── main.py
├── requirements.txt
├── data/
├── src/
├── models/
├── notebook/
└── outputs/
```

---

### Phase 3 — Data Collection (2-3 din)

#### Step 3.1 — data.gov.in se Data Download Karo

**Method 1: Direct CSV Download (Easiest)**
1. Jao: `https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi`
2. "Download" button pe click karo
3. CSV file `data/raw_data.csv` mein rename karke rakho

**Method 2: API se Automatically Fetch Karo**

Ek naya file `data_fetch.py` banao aur ye code likho:

```python
import requests
import pandas as pd

API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aae38d976ea657de9a"
BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

def fetch_commodity_data(commodity="Onion", limit=10000):
    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[commodity]": commodity,
        "limit": limit
    }
    response = requests.get(BASE_URL, params=params)
    df = pd.DataFrame(response.json()['records'])
    return df

# Saari commodities fetch karo
commodities = ["Onion", "Potato", "Gram", "Tur", "Urad", "Moong", "Masur"]
all_data = pd.concat([fetch_commodity_data(c) for c in commodities])
all_data.to_csv("data/raw_data.csv", index=False)
print(f"Total records downloaded: {len(all_data)}")
```

```bash
python data_fetch.py
```

#### Step 3.2 — Weather Data Fetch Karo (Free, No Registration)
```python
import requests
import pandas as pd

def fetch_weather(lat=19.9, lon=75.3, start="2015-01-01", end="2024-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min"
    }
    r = requests.get(url, params=params)
    df = pd.DataFrame(r.json()['daily'])
    df.to_csv("data/weather_data.csv", index=False)
    return df

weather = fetch_weather()
print(f"Weather records: {len(weather)}")
```

---

### Phase 4 — Data Preprocessing Likhna (2-3 din)

**`src/data_preprocessing.py` mein ye code likho:**

```python
import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Raw CSV data load karo"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File nahi mili: {file_path}")
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def clean_data(data):
    """Data clean karo — missing values, outliers, sorting"""
    # Missing values fill karo
    data = data.ffill()

    # Date ke hisaab se sort karo
    data = data.sort_values(by='Date').reset_index(drop=True)

    # Outliers remove karo using IQR method
    Q1 = data['Modal_Price'].quantile(0.25)
    Q3 = data['Modal_Price'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[
        (data['Modal_Price'] >= Q1 - 1.5 * IQR) &
        (data['Modal_Price'] <= Q3 + 1.5 * IQR)
    ]
    return data

def add_features(data):
    """ML ke liye naye features add karo"""
    # Lag features — pichle dinon ke prices
    data['price_lag_7']  = data['Modal_Price'].shift(7)
    data['price_lag_14'] = data['Modal_Price'].shift(14)
    data['price_lag_30'] = data['Modal_Price'].shift(30)

    # Rolling statistics — smoothed trends
    data['rolling_mean_7']  = data['Modal_Price'].rolling(7).mean()
    data['rolling_mean_30'] = data['Modal_Price'].rolling(30).mean()
    data['rolling_std_7']   = data['Modal_Price'].rolling(7).std()

    # Date-based features
    data['month']   = data['Date'].dt.month
    data['quarter'] = data['Date'].dt.quarter
    data['year']    = data['Date'].dt.year

    # Season flags
    data['is_harvest_season']  = data['month'].isin([11, 12, 1]).astype(int)
    data['is_festival_season'] = data['month'].isin([10, 3, 4]).astype(int)

    # NaN rows drop karo
    data = data.dropna().reset_index(drop=True)
    return data

def save_data(data, path):
    data.to_csv(path, index=False)
    print(f"Saved: {path} ({len(data)} rows)")

if __name__ == "__main__":
    raw     = load_data("data/raw_data.csv")
    cleaned = clean_data(raw)
    final   = add_features(cleaned)
    save_data(final, "data/processed_data.csv")
    print("Preprocessing complete!")
```

**Run karo:**
```bash
python src/data_preprocessing.py
```

---

### Phase 5 — EDA Likhna (2 din)

**`src/eda.py` mein ye code likho:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

def plot_price_trend(data, commodity="Onion"):
    """Price trend chart banao"""
    plt.figure(figsize=(14, 5))
    plt.plot(data['Date'], data['Modal_Price'], color='green', linewidth=1.5)
    plt.fill_between(data['Date'], data['Min_Price'], data['Max_Price'],
                     alpha=0.2, color='green', label='Min-Max Range')
    plt.title(f'{commodity} Price Trend (Rs/Quintal)')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs/Quintal)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/{commodity}_trend.png')
    plt.show()
    print(f"Chart saved: outputs/{commodity}_trend.png")

def monthly_boxplot(data):
    """Har mahine ki price distribution dikhao"""
    data['month_name'] = data['Date'].dt.strftime('%b')
    month_order = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=data, x='month_name', y='Modal_Price',
                order=month_order, palette='viridis')
    plt.title('Monthly Price Distribution — Seasonal Pattern')
    plt.xlabel('Month')
    plt.ylabel('Price (Rs/Quintal)')
    plt.tight_layout()
    plt.savefig('outputs/monthly_boxplot.png')
    plt.show()

def correlation_heatmap(data):
    """Features ka correlation dikhao"""
    cols = ['Modal_Price', 'price_lag_7', 'price_lag_14',
            'rolling_mean_7', 'rolling_mean_30', 'month']
    corr = data[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("data/processed_data.csv", parse_dates=['Date'])
    plot_price_trend(data)
    monthly_boxplot(data)
    correlation_heatmap(data)
    print("EDA complete!")
```

**Run karo:**
```bash
python src/eda.py
```

---

### Phase 6 — ML Model Banana (3-4 din)

**`src/model.py` mein ye code likho:**

```python
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    """ADF test — data stationary hai ya nahi"""
    result = adfuller(series.dropna())
    print(f"ADF p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("Data is Stationary — model train kar sakte ho")
    else:
        print("Data is NOT Stationary — differencing lagao")

def train_sarima(data, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """SARIMA model train karo"""
    print("SARIMA training shuru...")
    model = SARIMAX(
        data['Modal_Price'],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    print("SARIMA trained successfully!")
    return result

def train_xgboost(data):
    """XGBoost model train karo"""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    features = ['price_lag_7', 'price_lag_14', 'price_lag_30',
                'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
                'month', 'quarter', 'is_harvest_season', 'is_festival_season']

    X = data[features].dropna()
    y = data.loc[X.index, 'Modal_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("XGBoost trained successfully!")
    return model

if __name__ == "__main__":
    data = pd.read_csv("data/processed_data.csv", parse_dates=['Date'], index_col='Date')

    print("--- Stationarity Check ---")
    check_stationarity(data['Modal_Price'])

    print("\n--- Training SARIMA ---")
    sarima = train_sarima(data)
    joblib.dump(sarima, "models/sarima_model.pkl")
    print("Saved: models/sarima_model.pkl")

    print("\n--- Training XGBoost ---")
    xgb_model = train_xgboost(data.reset_index())
    joblib.dump(xgb_model, "models/xgboost_model.pkl")
    print("Saved: models/xgboost_model.pkl")
```

**Run karo:**
```bash
python src/model.py
```

---

### Phase 7 — Evaluation aur Alerts Likhna (2 din)

**`src/evaluation.py` mein ye code likho:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

def plot_forecast(data, forecast, steps=30):
    """Actual + Forecast chart banao"""
    plt.figure(figsize=(14, 6))
    plt.plot(data.index[-90:], data['Modal_Price'][-90:],
             label='Historical', color='blue', linewidth=2)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)
    plt.plot(future_dates, forecast,
             label='Forecast', color='red', linestyle='--', linewidth=2)
    threshold = data['Modal_Price'].mean() * 1.3
    plt.axhline(y=threshold, color='orange', linestyle=':', label=f'Alert Level')
    plt.title('Price Forecast — Next 30 Days')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs/Quintal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/forecast_chart.png')
    plt.show()

def buffer_stock_alert(forecast, current_price, commodity):
    """Buffer stock alert system"""
    max_forecast = max(forecast)
    increase_pct = ((max_forecast - current_price) / current_price) * 100
    print("\n" + "="*50)
    print(f"  PRICE FORECAST REPORT — {commodity.upper()}")
    print("="*50)
    print(f"  Current Price   : Rs.{current_price:.0f}/quintal")
    print(f"  Predicted (30d) : Rs.{max_forecast:.0f}/quintal")
    print(f"  Expected Change : +{increase_pct:.1f}%")
    print("="*50)
    if increase_pct > 20:
        print("  ALERT: BUFFER STOCK RELEASE RECOMMENDED!")
    else:
        print("  STABLE: No intervention needed")
    print("="*50 + "\n")

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    data = pd.read_csv("data/processed_data.csv", parse_dates=['Date'], index_col='Date')
    sarima = joblib.load("models/sarima_model.pkl")

    test = data['Modal_Price'][-30:]
    predictions = sarima.predict(start=len(data)-30, end=len(data)-1)

    mae, rmse, mape = calculate_metrics(test.values, predictions.values)
    print(f"\nModel Performance:")
    print(f"  MAE  = Rs.{mae:.2f}")
    print(f"  RMSE = Rs.{rmse:.2f}")
    print(f"  MAPE = {mape:.2f}%  (Accuracy: {100-mape:.2f}%)")

    forecast = sarima.forecast(steps=30)
    plot_forecast(data, forecast)
    buffer_stock_alert(forecast.values, data['Modal_Price'].iloc[-1], "Onion")
```

**Run karo:**
```bash
python src/evaluation.py
```

---

### Phase 8 — Main Pipeline Banana (1 din)

**`main.py` mein ye code likho:**

```python
"""
AgriPrice AI — Main Pipeline
Ek command mein poora project chalao:  python main.py
"""
import os

def run_pipeline():
    print("\n" + "="*55)
    print("  AgriPrice AI — Starting Full Pipeline")
    print("="*55 + "\n")

    os.makedirs("outputs", exist_ok=True)

    print("Step 1: Data Preprocessing...")
    os.system("python src/data_preprocessing.py")

    print("\nStep 2: Exploratory Data Analysis...")
    os.system("python src/eda.py")

    print("\nStep 3: Training ML Models...")
    os.system("python src/model.py")

    print("\nStep 4: Evaluation & Price Forecast...")
    os.system("python src/evaluation.py")

    print("\n" + "="*55)
    print("  Pipeline Complete!")
    print("  Results saved in outputs/ folder")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_pipeline()
```

**Poora pipeline chalao:**
```bash
python main.py
```

---

### Phase 9 — Testing & Debugging (2-3 din)

Sab kuch test karo:
```bash
# Step by step test karo
python src/data_preprocessing.py  # Check: processed_data.csv bana?
python src/eda.py                  # Check: outputs/ mein charts bane?
python src/model.py                # Check: models/ mein pkl files bane?
python src/evaluation.py          # Check: accuracy print hui? Alert aaya?

# Agar koi error aaye to:
# 1. Error message padho
# 2. Line number dekho
# 3. Code check karo
```

---

### ⏱️ Total Development Timeline

| Phase | Kaam | Kitna Time |
|-------|------|-----------|
| Phase 1 | Environment Setup | 1-2 din |
| Phase 2 | Project Structure | 1 din |
| Phase 3 | Data Collection | 2-3 din |
| Phase 4 | Data Preprocessing | 2-3 din |
| Phase 5 | EDA | 2 din |
| Phase 6 | ML Model Training | 3-4 din |
| Phase 7 | Evaluation & Alerts | 2 din |
| Phase 8 | Main Pipeline | 1 din |
| Phase 9 | Testing & Debugging | 2-3 din |
| **Total** | **Poora Project** | **~2-3 Hafte** |

---

## 🏗️ System Architecture

```
+------------------------------------------------------------------+
|                  DATA SOURCES (Government)                        |
|  data.gov.in | agmarknet.gov.in | fcainfoweb.nic.in | IMD API    |
+-----------------------------+------------------------------------+
                              | Daily Automated Ingestion
                              v
+------------------------------------------------------------------+
|                  DATA INGESTION LAYER                             |
|             Python Scripts + REST API Calls                       |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                  DATA STORAGE LAYER                               |
|        CSV / PostgreSQL / HDFS (Big Data Storage)                |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|               DATA PROCESSING PIPELINE                            |
|  Cleaning → Feature Engineering → EDA → Train/Test Split         |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                   ML MODEL LAYER                                   |
|         SARIMA  |  Facebook Prophet  |  XGBoost  |  LSTM          |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|              EVALUATION & PREDICTION                              |
|        MAE | RMSE | MAPE | 7-day | 14-day | 30-day Forecast      |
+-----------------------------+------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                 DASHBOARD / OUTPUT                                 |
|        Charts + Price Trend + Buffer Stock Alerts                 |
+------------------------------------------------------------------+
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Big Data** | Apache Spark (PySpark) |
| **ML / Statistics** | Statsmodels (SARIMA), Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Storage** | CSV, PostgreSQL |
| **Notebooks** | Jupyter Notebook |
| **Version Control** | Git / GitHub |

---

## 📦 Dataset Sources

All datasets are from **official Government of India portals** — no third-party data.

| # | Dataset | Link | Used For |
|---|---------|------|---------|
| 1 | Daily Mandi Prices | https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi | Main price data |
| 2 | Agmarknet Wholesale | https://agmarknet.gov.in | Arrivals + wholesale |
| 3 | FCA Retail Prices | https://fcainfoweb.nic.in/reports/report_menu_web.aspx | Retail prices |
| 4 | Open-Meteo API | https://archive-api.open-meteo.com/v1/archive | Free weather data |
| 5 | Crop Production | https://eands.dacnet.nic.in | Supply data |
| 6 | All-in-One (Onion) | https://data.mendeley.com/datasets/ds9jmxp9zy/1 | Merged 2014-2024 |

---

## 📁 Project Structure

```
agriprice-ai/
|
+-- README.md                      <- You are here
+-- requirements.txt               <- All Python dependencies
+-- main.py                        <- Main entry point (full pipeline)
|
+-- data/
|   +-- raw_data.csv               <- Raw government data
|   +-- processed_data.csv        <- Cleaned + feature-engineered data
|
+-- src/
|   +-- data_preprocessing.py     <- Data cleaning + feature engineering
|   +-- eda.py                    <- Charts and analysis
|   +-- model.py                  <- SARIMA + XGBoost training
|   +-- evaluation.py             <- Metrics + forecast + alerts
|
+-- models/
|   +-- sarima_model.pkl          <- Saved SARIMA model
|   +-- xgboost_model.pkl        <- Saved XGBoost model
|
+-- outputs/
|   +-- price_trend.png          <- Price trend chart
|   +-- monthly_boxplot.png      <- Seasonal pattern chart
|   +-- correlation_heatmap.png  <- Feature correlation
|   +-- forecast_chart.png       <- 30-day forecast
|
+-- notebook/
    +-- data_preprocessing.ipynb  <- Interactive preprocessing
    +-- exploratory_analysis.ipynb <- EDA notebook
    +-- model_training.ipynb      <- Model training notebook
```

---

## ⚙️ Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/agriprice-ai.git
cd agriprice-ai
```

### Step 2 — Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
Download from: https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi

Place as `data/raw_data.csv`. Required columns:
```
Date, Commodity, State, District, Market, Min_Price, Max_Price, Modal_Price
```

---

## ▶️ How to Run

### Option 1 — Full Pipeline (Sabse Easy)
```bash
python main.py
```
Automatically chalega: Preprocessing → EDA → Training → Evaluation → Forecast

### Option 2 — Modern Dashboard (Recommended)
```bash
streamlit run app.py
```
Ya shortcut command:
```bash
python main.py dashboard
```

### Option 3 — Step by Step
```bash
python src/data_preprocessing.py   # Step 1: Data clean karo
python src/eda.py                  # Step 2: Charts dekho
python src/model.py                # Step 3: Model train karo
python src/evaluation.py          # Step 4: Predict + evaluate
```

### Option 4 — Jupyter Notebooks
```bash
jupyter notebook
```
Notebooks is order mein chalao:
1. `notebook/data_preprocessing.ipynb`
2. `notebook/exploratory_analysis.ipynb`
3. `notebook/model_training.ipynb`

---

## 🤖 ML Models Used

| Model | Best For | Expected Accuracy |
|-------|---------|------------------|
| **SARIMA** | Seasonal + trend data | ~85-90% |
| **XGBoost** | Multi-feature prediction | ~88-92% |
| **Facebook Prophet** | Festival/holiday effects | ~83-88% |
| **LSTM** | Complex long-term patterns | ~87-91% |

---

## 📊 Results & Evaluation

After running `evaluation.py`, output aayega:

```
Model Performance:
  MAE  = Rs.85.42        <- Average Rs. error
  RMSE = Rs.112.33       <- Penalized error
  MAPE = 8.3%            <- 91.7% Accurate

=====================================================
  PRICE FORECAST REPORT — ONION
=====================================================
  Current Price   : Rs.1800/quintal
  Predicted (30d) : Rs.2800/quintal
  Expected Change : +55.6%
=====================================================
  ALERT: BUFFER STOCK RELEASE RECOMMENDED!
=====================================================
```

Charts saved in `outputs/` folder.

---

## 🗄️ Big Data Concepts Used

| Concept | How It's Applied |
|---------|----------------|
| **Volume** | 550 centres x 22 commodities x 365 days = crores of records |
| **Velocity** | Daily automated ingestion pipeline |
| **Variety** | Price + Weather + Crop production + Arrivals |
| **Veracity** | Outlier detection, missing value imputation |
| **Time-Series** | SARIMA, LSTM, Prophet on sequential data |
| **Feature Engineering** | Lag features, rolling stats, seasonal flags |
| **Predictive Analytics** | 7/14/30-day price forecasting |
| **Data Pipeline** | Automated end-to-end ingestion to prediction |

---

## 🔮 Future Enhancements

- [ ] Real-Time Pipeline — Automated daily fetch from govt APIs
- [ ] LSTM Deep Learning — Better long-term forecasting
- [x] Streamlit Dashboard — Interactive web UI with live forecasts
- [ ] All 22 Commodities — Extend beyond onion/potato/pulses
- [ ] Geospatial Heat Maps — State-wise price maps on India map
- [ ] News Sentiment Analysis — Market shock detection from headlines
- [ ] Mobile App — Farmer-facing app for local mandi forecasts

---

## 📦 requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
statsmodels>=0.13.0
joblib>=1.2.0
xgboost>=1.7.0
plotly>=5.11.0
requests>=2.28.0
scipy>=1.9.0
jupyter>=1.0.0
```

---

## 👥 Team

| Name | Role |
|------|------|
| **Abhinav Jindal** (Team Lead) | Lead Developer, Pipeline Orchestration, & ML Modeling |
| **Parth Madrewar** | Frontend Dashboard Developer & Data Analyst |

**Institution:** Lovely Professional University, Phagwara, Jalandhar, Punjab
**SIH 2024 Participation**

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- **Ministry of Consumer Affairs, Food and Public Distribution** — for the problem statement
- **data.gov.in** — Open Government Data Platform for official datasets
- **Agmarknet (DMI, Ministry of Agriculture)** — for daily mandi price data
- **India Meteorological Department (IMD)** — for weather data
- **Smart India Hackathon 2024** — for the platform and opportunity

---

<div align="center">

**Made with heart for Smart India Hackathon 2024**

*"Harnessing AI to stabilize food prices for every Indian household"*

</div>
