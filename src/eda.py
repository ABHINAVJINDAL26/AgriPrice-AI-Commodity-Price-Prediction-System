import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, parse_dates=["Date"])
    if "Price" not in data.columns and "Modal_Price" in data.columns:
        data["Price"] = data["Modal_Price"]
        
    if "State" in data.columns:
        agg_cols = {
            "Price": "mean",
            "month": "first"
        }
        for col in ["price_lag_7", "price_lag_14", "price_lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7"]:
            if col in data.columns:
                agg_cols[col] = "mean"
        data = data.groupby(["Commodity", "Date"], as_index=False).agg(agg_cols)
        
    return data


def plot_price_trend(data, commodity="Onion"):
    plt.figure(figsize=(14, 5))
    plt.plot(data["Date"], data["Price"], color="green", linewidth=1.5)
    plt.title(f"{commodity} Price Trend (Rs/Quintal)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{commodity.lower()}_trend.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved chart: {output_path}")


def monthly_boxplot(data):
    data = data.copy()
    data["month_name"] = data["Date"].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=data,
        x="month_name",
        y="Price",
        hue="month_name",
        order=month_order,
        palette="viridis",
        legend=False,
    )
    plt.title("Monthly Price Distribution")
    plt.xlabel("Month")
    plt.ylabel("Price")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "monthly_boxplot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved chart: {output_path}")


def correlation_heatmap(data):
    cols = [
        "Price",
        "price_lag_7",
        "price_lag_14",
        "price_lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "month",
    ]
    available_cols = [col for col in cols if col in data.columns]
    if len(available_cols) < 2:
        print("Skipping correlation heatmap: not enough numeric columns.")
        return

    corr = data[available_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved chart: {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        data = load_data(PROCESSED_DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    plot_price_trend(data)
    monthly_boxplot(data)
    correlation_heatmap(data)
    print("EDA complete.")
