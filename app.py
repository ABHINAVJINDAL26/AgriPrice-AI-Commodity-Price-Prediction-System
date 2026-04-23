import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed_data.csv")
POLICY_ALERTS = os.path.join(BASE_DIR, "outputs", "policy_alerts.csv")
EVALUATION_SUMMARY = os.path.join(BASE_DIR, "outputs", "evaluation_summary.csv")
MODEL_BENCHMARK = os.path.join(BASE_DIR, "outputs", "model_benchmark.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


st.set_page_config(
    page_title="AgriPrice AI Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(circle at 5% 10%, rgba(15,118,110,0.12), transparent 30%),
          radial-gradient(circle at 90% 20%, rgba(217,119,6,0.12), transparent 35%),
          linear-gradient(180deg, #F8FBF6 0%, #F2F7EF 100%);
    }

    .hero {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(6,95,70,0.95), rgba(13,148,136,0.88));
        color: white;
        box-shadow: 0 8px 24px rgba(6,95,70,0.22);
    }

    .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(6,95,70,0.15);
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
        color: #0F172A;
    }

    .small-note {
        font-size: 0.83rem;
        color: #4B5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_pipeline() -> tuple[bool, str]:
    command = [sys.executable, os.path.join(BASE_DIR, "main.py")]
    completed = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return False, completed.stderr + "\n" + completed.stdout
    return True, completed.stdout


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def format_action(action: str) -> str:
    return action.replace("_", " ").title()


with st.sidebar:
    st.header("Control Panel")
    st.caption("Run pipeline and filter commodity insights")

    if st.button("Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running preprocessing, training, evaluation..."):
            ok, logs = run_pipeline()
        if ok:
            st.success("Pipeline completed successfully.")
        else:
            st.error("Pipeline failed. Check logs below.")
        st.text_area("Pipeline Logs", logs, height=220)

    st.markdown("---")
    st.markdown("<p class='small-note'>Tip: Run pipeline whenever new raw data is added.</p>", unsafe_allow_html=True)


processed_df = load_csv(PROCESSED_DATA)
alerts_df = load_csv(POLICY_ALERTS)
summary_df = load_csv(EVALUATION_SUMMARY)
benchmark_df = load_csv(MODEL_BENCHMARK)

st.markdown(
    """
    <div class='hero'>
      <h2 style='margin:0;'>AgriPrice AI Dashboard</h2>
      <p style='margin:0.35rem 0 0 0;'>Commodity forecasting, policy alerts, and model quality in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if processed_df.empty or alerts_df.empty or summary_df.empty:
    st.warning("Pipeline outputs are missing. Use 'Run Full Pipeline' from the sidebar.")
    st.stop()

if "Date" in processed_df.columns:
    processed_df["Date"] = pd.to_datetime(processed_df["Date"], errors="coerce")

if "Commodity" not in processed_df.columns:
    processed_df["Commodity"] = "Unknown"

commodities = sorted(processed_df["Commodity"].dropna().unique().tolist())
selected_commodity = st.sidebar.selectbox("Commodity", commodities, index=0)
commodity_data = processed_df[processed_df["Commodity"] == selected_commodity].copy()

alert_row = alerts_df[alerts_df["Commodity"] == selected_commodity]
summary_row = summary_df[summary_df["Commodity"] == selected_commodity]

current_price = float(alert_row["CurrentPrice"].iloc[0]) if not alert_row.empty else float("nan")
predicted_price = float(alert_row["PredictedMax30d"].iloc[0]) if not alert_row.empty else float("nan")
change_pct = float(alert_row["ExpectedChangePct"].iloc[0]) if not alert_row.empty else float("nan")
policy_action = format_action(alert_row["PolicyAction"].iloc[0]) if not alert_row.empty else "Unknown"
accuracy = float(summary_row["Accuracy"].iloc[0]) if not summary_row.empty else float("nan")

col1, col2, col3, col4 = st.columns(4)
col1.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col1.metric("Current Price", f"Rs.{current_price:,.2f}")
col1.markdown("</div>", unsafe_allow_html=True)

col2.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col2.metric("Predicted Max (30d)", f"Rs.{predicted_price:,.2f}")
col2.markdown("</div>", unsafe_allow_html=True)

col3.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col3.metric("Expected Change", f"{change_pct:+.2f}%")
col3.markdown("</div>", unsafe_allow_html=True)

col4.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col4.metric("Model Accuracy", f"{accuracy:.2f}%")
col4.markdown("</div>", unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.8, 1.2])
with left:
    st.markdown("<div class='section-title'>Price Trend and Rolling Signal</div>", unsafe_allow_html=True)
    if not commodity_data.empty and "Date" in commodity_data.columns and "Price" in commodity_data.columns:
        commodity_data = commodity_data.sort_values("Date")
        commodity_data["RollingMean7"] = commodity_data["Price"].rolling(window=7, min_periods=1).mean()
        trend_fig = px.line(
            commodity_data,
            x="Date",
            y=["Price", "RollingMean7"],
            labels={"value": "Price", "variable": "Series"},
            color_discrete_map={"Price": "#0F766E", "RollingMean7": "#D97706"},
        )
        trend_fig.update_layout(
            margin=dict(l=8, r=8, t=10, b=8),
            legend_title=None,
            plot_bgcolor="rgba(255,255,255,0.85)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("No trend data available for selected commodity.")

with right:
    st.markdown("<div class='section-title'>Policy Recommendation</div>", unsafe_allow_html=True)
    action_color = {
        "Release Buffer Stock": "#B91C1C",
        "Prepare Intervention": "#B45309",
        "Stable": "#0F766E",
        "Low Price Monitor": "#1D4ED8",
    }.get(policy_action, "#334155")

    st.markdown(
        f"""
        <div style='padding:1rem; border-radius:14px; background:white; border-left:6px solid {action_color}; box-shadow:0 3px 10px rgba(0,0,0,0.06)'>
          <div style='font-size:0.85rem; color:#4B5563;'>Recommended Action</div>
          <div style='font-size:1.15rem; font-weight:800; color:{action_color}; margin-top:0.35rem;'>{policy_action}</div>
          <div style='margin-top:0.6rem; color:#334155;'>Use this signal for proactive buffer stock planning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<div class='section-title'>Latest Alert Table</div>", unsafe_allow_html=True)
    alerts_show = alerts_df.copy()
    alerts_show["PolicyAction"] = alerts_show["PolicyAction"].map(format_action)
    st.dataframe(alerts_show, use_container_width=True, hide_index=True)

st.write("")

bench_col1, bench_col2 = st.columns([1.1, 1.1])

with bench_col1:
    st.markdown("<div class='section-title'>Model Benchmark (MAPE)</div>", unsafe_allow_html=True)
    if not benchmark_df.empty and {"Commodity", "Model", "MAPE"}.issubset(benchmark_df.columns):
        bm = benchmark_df.copy()
        bm["Label"] = bm["Commodity"] + " - " + bm["Model"]
        bm_fig = px.bar(
            bm,
            x="Label",
            y="MAPE",
            color="Model",
            color_discrete_sequence=["#0EA5A4", "#F59E0B", "#64748B", "#2563EB"],
        )
        bm_fig.update_layout(
            margin=dict(l=8, r=8, t=10, b=8),
            xaxis_title=None,
            yaxis_title="MAPE (%)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(bm_fig, use_container_width=True)
    else:
        st.info("Benchmark file missing required columns.")

with bench_col2:
    st.markdown("<div class='section-title'>Evaluation Summary</div>", unsafe_allow_html=True)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
