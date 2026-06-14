import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BENCHMARK_PATH = os.path.join(OUTPUT_DIR, "model_benchmark.csv")
REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.csv")


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "unknown"


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, parse_dates=["Date"])
    if "Price" not in data.columns and "Modal_Price" in data.columns:
        data["Price"] = data["Modal_Price"]
    if "Price" not in data.columns:
        raise ValueError("Processed data must contain 'Price' column.")
    if "Commodity" not in data.columns:
        data["Commodity"] = "Unknown"

    data = data.dropna(subset=["Date", "Price"]).sort_values(["Commodity", "Date"]).reset_index(drop=True)
    return data


def aggregate_to_commodity_level(data: pd.DataFrame) -> pd.DataFrame:
    if "State" in data.columns:
        # Group by Commodity and Date, average the Price and Volume
        groupby_cols = ["Commodity", "Date"]
        agg_df = data.groupby(groupby_cols, as_index=False).agg({
            "Price": "mean",
            "Volume": "sum",
            "month": "first",
            "quarter": "first",
            "year": "first",
            "is_harvest_season": "first",
            "is_festival_season": "first"
        })
        # Re-calculate lag features on the aggregated price
        enriched_groups = []
        for _, group in agg_df.groupby("Commodity", dropna=False):
            group = group.sort_values("Date").copy()
            for lag in (7, 14, 30):
                group[f"price_lag_{lag}"] = group["Price"].shift(lag)
            group["rolling_mean_7"] = group["Price"].rolling(window=7, min_periods=2).mean()
            group["rolling_mean_30"] = group["Price"].rolling(window=30, min_periods=2).mean()
            group["rolling_std_7"] = group["Price"].rolling(window=7, min_periods=2).std()
            
            feature_cols = ["price_lag_7", "price_lag_14", "price_lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7"]
            for col in feature_cols:
                group[col] = group[col].bfill().ffill()
            enriched_groups.append(group)
            
        return pd.concat(enriched_groups, ignore_index=True)
    return data


def check_stationarity(series: pd.Series) -> float:
    if len(series.dropna()) < 5:
        return np.nan
    result = adfuller(series.dropna())
    return float(result[1])


def train_sarima_model(series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def evaluate_series(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - predicted) / denom)) * 100)
    return float(mae), rmse, mape


def save_model_comparison_plot(commodity: str, results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "RMSE", "MAPE"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f1c40f"]
    
    for i, metric in enumerate(metrics):
        models = list(results.keys())
        vals = [results[m][metric] for m in models]
        bars = axes[i].bar(models, vals, color=colors[:len(models)])
        axes[i].set_title(f"{metric} Comparison (lower is better)")
        axes[i].set_ylabel(metric)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, vals):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (val * 0.02),
                f"{val:.1f}",
                ha="center",
                fontsize=9
            )
            
    commodity_slug = sanitize_name(commodity)
    plt.suptitle(f"Model Comparison for {commodity}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    # Save both per-commodity and generic filename
    plt.savefig(os.path.join(OUTPUT_DIR, f"model_comparison_{commodity_slug}.png"), dpi=150)
    if commodity.lower() == "onion":
        plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved model comparison plot for {commodity}")


def run_training(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benchmark_rows = []
    registry_rows = []

    for commodity, group in data.groupby("Commodity", dropna=False):
        group_sorted = group.sort_values("Date")
        series = group_sorted.set_index("Date")["Price"]
        if len(series) < 15:
            print(f"Skipping {commodity}: not enough observations ({len(series)}).")
            continue

        test_len = min(30, max(1, len(series) // 5))
        train = series.iloc[:-test_len]
        test = series.iloc[-test_len:]

        p_value = check_stationarity(train)
        
        # 1. Train ARIMA
        print(f"Training ARIMA for {commodity}...")
        arima_model = ARIMA(train, order=(1, 1, 1)).fit()
        arima_pred = arima_model.forecast(steps=len(test))
        arima_mae, arima_rmse, arima_mape = evaluate_series(test.values, np.array(arima_pred))

        # 2. Train SARIMA
        print(f"Training SARIMA for {commodity}...")
        sarima_model = train_sarima_model(train)
        sarima_pred = sarima_model.forecast(steps=len(test))
        sarima_mae, sarima_rmse, sarima_mape = evaluate_series(test.values, np.array(sarima_pred))

        # 3. Train XGBoost
        print(f"Training XGBoost for {commodity}...")
        features = [
            "price_lag_7", "price_lag_14", "price_lag_30",
            "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
            "month", "quarter", "year", "is_harvest_season", "is_festival_season"
        ]
        X = group_sorted[features]
        y = group_sorted["Price"]
        X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]
        y_train, y_test = y.iloc[:-test_len], y.iloc[-test_len:]
        
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae, xgb_rmse, xgb_mape = evaluate_series(y_test.values, xgb_pred)

        # 4. Naive Last Value Baseline
        naive_pred = np.repeat(train.iloc[-1], len(test))
        naive_mae, naive_rmse, naive_mape = evaluate_series(test.values, naive_pred)

        # Save model objects
        commodity_slug = sanitize_name(str(commodity))
        sarima_path = os.path.join(MODELS_DIR, f"sarima_{commodity_slug}.pkl")
        arima_path = os.path.join(MODELS_DIR, f"arima_{commodity_slug}.pkl")
        xgb_path = os.path.join(MODELS_DIR, f"xgboost_{commodity_slug}.pkl")
        
        joblib.dump(sarima_model, sarima_path)
        joblib.dump(arima_model, arima_path)
        joblib.dump(xgb_model, xgb_path)

        # Register models
        registry_rows.extend([
            {"Commodity": commodity, "Model": "SARIMA", "ModelPath": sarima_path, "TrainRows": len(train), "TestRows": len(test), "ADF_PValue": p_value},
            {"Commodity": commodity, "Model": "ARIMA", "ModelPath": arima_path, "TrainRows": len(train), "TestRows": len(test), "ADF_PValue": p_value},
            {"Commodity": commodity, "Model": "XGBoost", "ModelPath": xgb_path, "TrainRows": len(train), "TestRows": len(test), "ADF_PValue": p_value}
        ])

        benchmark_rows.extend([
            {"Commodity": commodity, "Model": "SARIMA", "MAE": sarima_mae, "RMSE": sarima_rmse, "MAPE": sarima_mape},
            {"Commodity": commodity, "Model": "ARIMA", "MAE": arima_mae, "RMSE": arima_rmse, "MAPE": arima_mape},
            {"Commodity": commodity, "Model": "XGBoost", "MAE": xgb_mae, "RMSE": xgb_rmse, "MAPE": xgb_mape},
            {"Commodity": commodity, "Model": "Naive_LastValue", "MAE": naive_mae, "RMSE": naive_rmse, "MAPE": naive_mape}
        ])

        # Generate comparison plot
        comparison_results = {
            "ARIMA": {"MAE": arima_mae, "RMSE": arima_rmse, "MAPE": arima_mape},
            "SARIMA": {"MAE": sarima_mae, "RMSE": sarima_rmse, "MAPE": sarima_mape},
            "XGBoost": {"MAE": xgb_mae, "RMSE": xgb_rmse, "MAPE": xgb_mape},
            "Naive": {"MAE": naive_mae, "RMSE": naive_rmse, "MAPE": naive_mape}
        }
        save_model_comparison_plot(str(commodity), comparison_results)

        print(
            f"Trained {commodity}: SARIMA MAPE={sarima_mape:.2f}% | "
            f"ARIMA MAPE={arima_mape:.2f}% | XGBoost MAPE={xgb_mape:.2f}% | Naive MAPE={naive_mape:.2f}%"
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    registry_df = pd.DataFrame(registry_rows)
    return benchmark_df, registry_df


if __name__ == "__main__":
    dataset = load_data(PROCESSED_DATA_PATH)
    agg_dataset = aggregate_to_commodity_level(dataset)
    benchmark_df, registry_df = run_training(agg_dataset)

    if benchmark_df.empty or registry_df.empty:
        raise SystemExit("No models trained. Check data volume and schema.")

    benchmark_df.to_csv(BENCHMARK_PATH, index=False)
    registry_df.to_csv(REGISTRY_PATH, index=False)

    print(f"Saved benchmark report: {BENCHMARK_PATH}")
    print(f"Saved model registry: {REGISTRY_PATH}")
