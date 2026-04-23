import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
REGISTRY_PATH = os.path.join(BASE_DIR, "models", "model_registry.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALERTS_PATH = os.path.join(OUTPUT_DIR, "policy_alerts.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "unknown"


def load_data() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Missing data file: {PROCESSED_DATA_PATH}")
    data = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])
    if "Price" not in data.columns and "Modal_Price" in data.columns:
        data["Price"] = data["Modal_Price"]
    if "Commodity" not in data.columns:
        data["Commodity"] = "Unknown"
    return data.sort_values(["Commodity", "Date"]).reset_index(drop=True)


def load_registry() -> pd.DataFrame:
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(f"Missing model registry: {REGISTRY_PATH}")
    return pd.read_csv(REGISTRY_PATH)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - predicted) / denom)) * 100)
    return float(mae), rmse, mape


def policy_label(change_pct: float) -> str:
    if change_pct >= 20:
        return "RELEASE_BUFFER_STOCK"
    if change_pct >= 10:
        return "PREPARE_INTERVENTION"
    if change_pct <= -10:
        return "LOW_PRICE_MONITOR"
    return "STABLE"


def render_forecast_chart(history: pd.Series, forecast: np.ndarray, commodity: str) -> None:
    plt.figure(figsize=(12, 5))
    hist = history.iloc[-min(90, len(history)):]
    future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=len(forecast))
    plt.plot(hist.index, hist.values, label="Historical", color="blue", linewidth=2)
    plt.plot(future_dates, forecast, label="Forecast", color="red", linestyle="--", linewidth=2)
    plt.title(f"{commodity} - 30 Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_name = f"forecast_{sanitize_name(str(commodity))}.png"
    chart_path = os.path.join(OUTPUT_DIR, chart_name)
    plt.savefig(chart_path)
    plt.close()


def evaluate_models(data: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    alert_rows = []

    for _, row in registry.iterrows():
        commodity = row["Commodity"]
        model_path = row["ModelPath"]
        commodity_data = data[data["Commodity"] == commodity].copy()
        if commodity_data.empty:
            continue

        series = commodity_data.set_index("Date")["Price"].sort_index()
        model = joblib.load(model_path)

        test_len = min(30, max(1, len(series) // 3))
        start_idx = len(series) - test_len
        pred = model.predict(start=start_idx, end=len(series) - 1)
        actual = series.iloc[start_idx:]
        pred_arr = np.array(pred[: len(actual)])

        mae, rmse, mape = compute_metrics(actual.values, pred_arr)

        forecast_steps = 30
        forecast = np.array(model.forecast(steps=forecast_steps))
        current_price = float(series.iloc[-1])
        max_forecast = float(np.max(forecast))
        change_pct = ((max_forecast - current_price) / current_price) * 100 if current_price else 0.0
        action = policy_label(change_pct)

        render_forecast_chart(series, forecast, str(commodity))

        summary_rows.append(
            {
                "Commodity": commodity,
                "Model": row.get("Model", "SARIMA"),
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Accuracy": 100 - mape,
            }
        )

        alert_rows.append(
            {
                "Commodity": commodity,
                "CurrentPrice": current_price,
                "PredictedMax30d": max_forecast,
                "ExpectedChangePct": change_pct,
                "PolicyAction": action,
            }
        )

        print(
            f"{commodity}: Accuracy={100 - mape:.2f}% | "
            f"Expected change={change_pct:+.2f}% | Action={action}"
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(alert_rows)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_data = load_data()
    model_registry = load_registry()
    summary_df, alert_df = evaluate_models(full_data, model_registry)

    if summary_df.empty or alert_df.empty:
        raise SystemExit("No evaluation results generated. Check trained models and data.")

    summary_df.to_csv(SUMMARY_PATH, index=False)
    alert_df.to_csv(ALERTS_PATH, index=False)

    print(f"Saved evaluation summary: {SUMMARY_PATH}")
    print(f"Saved policy alerts: {ALERTS_PATH}")
