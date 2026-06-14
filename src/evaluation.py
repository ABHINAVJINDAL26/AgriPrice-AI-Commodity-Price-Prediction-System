import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def aggregate_to_commodity_level(data: pd.DataFrame) -> pd.DataFrame:
    if "State" in data.columns:
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


def render_forecast_chart(history: pd.Series, forecast: np.ndarray, conf_int: pd.DataFrame, commodity: str) -> None:
    plt.figure(figsize=(12, 5))
    hist = history.iloc[-min(90, len(history)):]
    future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=len(forecast))
    
    plt.plot(hist.index, hist.values, label="Historical", color="#2c3e50", linewidth=2)
    plt.plot(future_dates, forecast, label="Forecast", color="#e74c3c", linestyle="--", linewidth=2)
    
    # Shaded confidence band
    plt.fill_between(
        future_dates,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values,
        alpha=0.25,
        color="#e74c3c",
        label="95% Confidence Interval"
    )
    
    threshold = hist.mean() * 1.3
    plt.axhline(y=threshold, color="orange", linestyle=":", linewidth=1.5, label="Alert Threshold (+30%)")
    
    plt.title(f"{commodity} - 30 Day Price Forecast with Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    chart_name = f"forecast_{sanitize_name(str(commodity))}.png"
    chart_path = os.path.join(OUTPUT_DIR, chart_name)
    plt.savefig(chart_path, dpi=150)
    if commodity.lower() == "onion":
        plt.savefig(os.path.join(OUTPUT_DIR, "forecast_with_ci.png"), dpi=150)
    plt.close()
    print(f"Saved chart: {chart_path}")


def plot_feature_importance(xgb_model, feature_names: list, commodity: str):
    importances = xgb_model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(feature_names))]
    
    y_labels = [feature_names[i] for i in sorted_idx]
    y_values = [importances[i] for i in sorted_idx]
    
    plt.barh(y_labels, y_values, color=colors[::-1])
    plt.xlabel("Feature Importance Score")
    plt.title(f"Which Features Drive Price Prediction for {commodity}? (XGBoost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    commodity_slug = sanitize_name(commodity)
    chart_path = os.path.join(OUTPUT_DIR, f"feature_importance_{commodity_slug}.png")
    plt.savefig(chart_path, dpi=150)
    if commodity.lower() == "onion":
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    
    print(f"Saved feature importance chart for {commodity}: {chart_path}")
    print(f"Top feature for {commodity}: {feature_names[sorted_idx[0]]} ({importances[sorted_idx[0]]:.3f})")


def statewise_alert_report(data: pd.DataFrame) -> pd.DataFrame:
    alert_rows = []
    if "State" not in data.columns:
        print("Skipping state-wise alerts: State column not present in data.")
        return pd.DataFrame()
        
    groupby_cols = ["Commodity", "State"]
    for (commodity, state), group in data.groupby(groupby_cols, dropna=False):
        state_data = group.sort_values("Date")
        if len(state_data) < 15:
            continue
        try:
            model = SARIMAX(
                state_data["Price"],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            forecast = model.forecast(steps=30)
            current = float(state_data["Price"].iloc[-1])
            predicted = float(forecast.max())
            change_pct = ((predicted - current) / current) * 100 if current else 0.0
            
            action = policy_label(change_pct)
            
            alert_rows.append({
                "State": state,
                "Commodity": commodity,
                "CurrentPrice": current,
                "PredictedMax30d": predicted,
                "ExpectedChangePct": change_pct,
                "PolicyAction": action
            })
        except Exception as e:
            print(f"Error training state model for {commodity} in {state}: {e}")
            
    alert_df = pd.DataFrame(alert_rows)
    if not alert_df.empty:
        alert_path = os.path.join(OUTPUT_DIR, "state_alerts.csv")
        alert_df.to_csv(alert_path, index=False)
        print(f"Saved state-wise alert report to {alert_path}")
        
        # Print report to stdout
        for comm, group in alert_df.groupby("Commodity"):
            print(f"\n{'='*65}")
            print(f"  STATE-WISE PRICE ALERT REPORT - {str(comm).upper()}")
            print(f"{'='*65}")
            print(f"  {'State':<18} {'Current':>10} {'30-day':>10} {'Change':>8} {'Status':>12}")
            print(f"  {'-'*58}")
            for _, row in group.iterrows():
                chg = row["ExpectedChangePct"]
                status = "ALERT" if chg >= 20 else "WATCH" if chg >= 10 else "STABLE"
                if chg <= -10:
                    status = "MONITOR"
                print(f"  {row['State']:<18} {row['CurrentPrice']:>9.0f}  {row['PredictedMax30d']:>9.0f} "
                      f" {chg:>+6.1f}%  {status}")
            print(f"{'='*65}\n")
            
    return alert_df


def evaluate_models(data: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    alert_rows = []

    for _, row in registry.iterrows():
        commodity = row["Commodity"]
        model_path = row["ModelPath"]
        model_name = row.get("Model", "SARIMA")
        
        commodity_data = data[data["Commodity"] == commodity].copy()
        if commodity_data.empty:
            continue

        series = commodity_data.set_index("Date")["Price"].sort_index()
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        model = joblib.load(model_path)
        test_len = min(30, max(1, len(series) // 5))

        if model_name == "SARIMA":
            forecast_steps = 30
            forecast_obj = model.get_forecast(steps=forecast_steps)
            forecast = np.array(forecast_obj.predicted_mean)
            conf_int = forecast_obj.conf_int()

            render_forecast_chart(series, forecast, conf_int, str(commodity))

            current_price = float(series.iloc[-1])
            max_forecast = float(np.max(forecast))
            change_pct = ((max_forecast - current_price) / current_price) * 100 if current_price else 0.0
            action = policy_label(change_pct)

            # Compute offline evaluation metrics on test split
            start_idx = len(series) - test_len
            pred = model.predict(start=start_idx, end=len(series) - 1)
            actual = series.iloc[start_idx:]
            pred_arr = np.array(pred[: len(actual)])
            mae, rmse, mape = compute_metrics(actual.values, pred_arr)

            summary_rows.append({
                "Commodity": commodity,
                "Model": "SARIMA",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Accuracy": 100 - mape,
            })

            alert_rows.append({
                "Commodity": commodity,
                "CurrentPrice": current_price,
                "PredictedMax30d": max_forecast,
                "ExpectedChangePct": change_pct,
                "PolicyAction": action,
            })

            print(
                f"{commodity} (SARIMA): Accuracy={100 - mape:.2f}% | "
                f"Expected change={change_pct:+.2f}% | Action={action}"
            )
            
        elif model_name == "XGBoost":
            feature_names = [
                "price_lag_7", "price_lag_14", "price_lag_30",
                "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
                "month", "quarter", "year", "is_harvest_season", "is_festival_season"
            ]
            plot_feature_importance(model, feature_names, str(commodity))
            
            # Evaluate on test partition
            features = commodity_data.sort_values("Date")[feature_names]
            X_test = features.iloc[-test_len:]
            actual = series.iloc[-test_len:]
            pred = model.predict(X_test)
            mae, rmse, mape = compute_metrics(actual.values, pred)
            
            summary_rows.append({
                "Commodity": commodity,
                "Model": "XGBoost",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Accuracy": 100 - mape,
            })
            print(f"{commodity} (XGBoost): Accuracy={100 - mape:.2f}%")
            
        elif model_name == "ARIMA":
            # Evaluate on test partition
            start_idx = len(series) - test_len
            pred = model.predict(start=start_idx, end=len(series) - 1)
            actual = series.iloc[start_idx:]
            pred_arr = np.array(pred[: len(actual)])
            mae, rmse, mape = compute_metrics(actual.values, pred_arr)
            
            summary_rows.append({
                "Commodity": commodity,
                "Model": "ARIMA",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Accuracy": 100 - mape,
            })
            print(f"{commodity} (ARIMA): Accuracy={100 - mape:.2f}%")

    return pd.DataFrame(summary_rows), pd.DataFrame(alert_rows)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_data = load_data()
    agg_data = aggregate_to_commodity_level(full_data)
    model_registry = load_registry()
    
    summary_df, alert_df = evaluate_models(agg_data, model_registry)
    statewise_alert_report(full_data)

    if summary_df.empty or alert_df.empty:
        raise SystemExit("No evaluation results generated. Check trained models and data.")

    # Group summary by Commodity and Model to ensure unique entries
    summary_df = summary_df.drop_duplicates(subset=["Commodity", "Model"])
    summary_df.to_csv(SUMMARY_PATH, index=False)
    alert_df.to_csv(ALERTS_PATH, index=False)

    print(f"Saved evaluation summary: {SUMMARY_PATH}")
    print(f"Saved policy alerts: {ALERTS_PATH}")
