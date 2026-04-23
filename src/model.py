import os
import re
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def run_training(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benchmark_rows = []
    registry_rows = []

    for commodity, group in data.groupby("Commodity", dropna=False):
        series = group.sort_values("Date").set_index("Date")["Price"]
        if len(series) < 7:
            print(f"Skipping {commodity}: not enough observations ({len(series)}).")
            continue

        test_len = min(30, max(1, len(series) // 3))
        train = series.iloc[:-test_len] if len(series) > test_len else series.iloc[:-1]
        test = series.iloc[-test_len:] if len(series) > test_len else series.iloc[-1:]

        if len(train) < 3:
            print(f"Skipping {commodity}: insufficient train data ({len(train)}).")
            continue

        p_value = check_stationarity(train)
        model = train_sarima_model(train)
        pred = model.forecast(steps=len(test))

        sarima_mae, sarima_rmse, sarima_mape = evaluate_series(test.values, np.array(pred))

        naive_pred = np.repeat(train.iloc[-1], len(test))
        naive_mae, naive_rmse, naive_mape = evaluate_series(test.values, naive_pred)

        commodity_slug = sanitize_name(str(commodity))
        model_path = os.path.join(MODELS_DIR, f"sarima_{commodity_slug}.pkl")
        joblib.dump(model, model_path)

        registry_rows.append(
            {
                "Commodity": commodity,
                "Model": "SARIMA",
                "ModelPath": model_path,
                "TrainRows": len(train),
                "TestRows": len(test),
                "ADF_PValue": p_value,
            }
        )

        benchmark_rows.extend(
            [
                {
                    "Commodity": commodity,
                    "Model": "SARIMA",
                    "MAE": sarima_mae,
                    "RMSE": sarima_rmse,
                    "MAPE": sarima_mape,
                },
                {
                    "Commodity": commodity,
                    "Model": "Naive_LastValue",
                    "MAE": naive_mae,
                    "RMSE": naive_rmse,
                    "MAPE": naive_mape,
                },
            ]
        )

        print(
            f"Trained {commodity}: SARIMA MAPE={sarima_mape:.2f}% | "
            f"Naive MAPE={naive_mape:.2f}%"
        )

    benchmark_df = pd.DataFrame(benchmark_rows)
    registry_df = pd.DataFrame(registry_rows)
    return benchmark_df, registry_df


if __name__ == "__main__":
    dataset = load_data(PROCESSED_DATA_PATH)
    benchmark_df, registry_df = run_training(dataset)

    if benchmark_df.empty or registry_df.empty:
        raise SystemExit("No models trained. Check data volume and schema.")

    benchmark_df.to_csv(BENCHMARK_PATH, index=False)
    registry_df.to_csv(REGISTRY_PATH, index=False)

    print(f"Saved benchmark report: {BENCHMARK_PATH}")
    print(f"Saved model registry: {REGISTRY_PATH}")
