import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")


def _normalize_price_column(data: pd.DataFrame) -> pd.DataFrame:
    if "Modal_Price" in data.columns:
        data["Price"] = pd.to_numeric(data["Modal_Price"], errors="coerce")
    elif "Price" in data.columns:
        data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    else:
        raise ValueError("Input data must contain either 'Price' or 'Modal_Price' column.")
    return data


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)
    if "Date" not in data.columns:
        raise ValueError("Input data must contain 'Date' column.")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = _normalize_price_column(data)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["Date", "Price"]).copy()
    if "Commodity" not in data.columns:
        data["Commodity"] = "Unknown"

    cleaned_groups = []
    for _, group in data.groupby("Commodity", dropna=False):
        group = group.sort_values(by="Date").reset_index(drop=True)
        group = group.ffill()

        q1 = group["Price"].quantile(0.25)
        q3 = group["Price"].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            group = group[(group["Price"] >= lower) & (group["Price"] <= upper)]

        cleaned_groups.append(group)

    cleaned = pd.concat(cleaned_groups, ignore_index=True)
    return cleaned.sort_values(["Commodity", "Date"]).reset_index(drop=True)


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "Commodity" not in data.columns:
        data["Commodity"] = "Unknown"

    enriched_groups = []
    for _, group in data.groupby("Commodity", dropna=False):
        group = group.sort_values("Date").copy()

        for lag in (7, 14, 30):
            group[f"price_lag_{lag}"] = group["Price"].shift(lag)

        group["rolling_mean_7"] = group["Price"].rolling(window=7, min_periods=2).mean()
        group["rolling_mean_30"] = group["Price"].rolling(window=30, min_periods=2).mean()
        group["rolling_std_7"] = group["Price"].rolling(window=7, min_periods=2).std()

        group["month"] = group["Date"].dt.month
        group["quarter"] = group["Date"].dt.quarter
        group["year"] = group["Date"].dt.year
        group["is_harvest_season"] = group["month"].isin([11, 12, 1]).astype(int)
        group["is_festival_season"] = group["month"].isin([3, 4, 10]).astype(int)

        feature_cols = [
            "price_lag_7",
            "price_lag_14",
            "price_lag_30",
            "rolling_mean_7",
            "rolling_mean_30",
            "rolling_std_7",
        ]
        for col in feature_cols:
            group[col] = group[col].bfill().ffill()

        enriched_groups.append(group)

    enriched = pd.concat(enriched_groups, ignore_index=True)
    return enriched.sort_values(["Commodity", "Date"]).reset_index(drop=True)


def save_data(data: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved processed data: {output_path} ({len(data)} rows)")


if __name__ == "__main__":
    raw_data = load_data(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)
    final_data = add_features(cleaned_data)
    save_data(final_data, PROCESSED_DATA_PATH)
    print("Preprocessing complete.")
