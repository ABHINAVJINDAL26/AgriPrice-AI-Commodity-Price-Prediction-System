import os
import re
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA = os.path.join(BASE_DIR, "data", "processed_data.csv")
POLICY_ALERTS = os.path.join(BASE_DIR, "outputs", "policy_alerts.csv")
STATE_ALERTS = os.path.join(BASE_DIR, "outputs", "state_alerts.csv")
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
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
        color: #0F172A;
        border-bottom: 2px solid rgba(6,95,70,0.15);
        padding-bottom: 0.2rem;
    }

    .small-note {
        font-size: 0.83rem;
        color: #4B5563;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "unknown"


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
state_alerts_df = load_csv(STATE_ALERTS)
summary_df = load_csv(EVALUATION_SUMMARY)
benchmark_df = load_csv(MODEL_BENCHMARK)

st.markdown(
    """
    <div class='hero'>
      <h2 style='margin:0;'>AgriPrice AI Dashboard</h2>
      <p style='margin:0.35rem 0 0 0;'>Commodity forecasting, policy alerts, and model quality in one place.</p>
      <div style='margin-top:0.4rem; padding: 0.3rem 0.6rem; border-radius: 6px; background: rgba(251, 191, 36, 0.2); border: 1px solid rgba(251, 191, 36, 0.4); display: inline-block; font-size:0.83rem; font-weight:600; color:#FCD34D;'>
        ⚠️ Demo Mode: Utilizing simulated historical market datasets (Rs. per Quintal). Production mode connects to data.gov.in real-time API.
      </div>
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
# We look for SARIMA model accuracy in evaluation summary
sarima_summary = summary_df[(summary_df["Commodity"] == selected_commodity) & (summary_df["Model"] == "SARIMA")]

current_price = float(alert_row["CurrentPrice"].iloc[0]) if not alert_row.empty else float("nan")
predicted_price = float(alert_row["PredictedMax30d"].iloc[0]) if not alert_row.empty else float("nan")
change_pct = float(alert_row["ExpectedChangePct"].iloc[0]) if not alert_row.empty else float("nan")
policy_action = format_action(alert_row["PolicyAction"].iloc[0]) if not alert_row.empty else "Unknown"
accuracy = float(sarima_summary["Accuracy"].iloc[0]) if not sarima_summary.empty else float("nan")

# Display key metrics cards at the top
col1, col2, col3, col4 = st.columns(4)
col1.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col1.metric("Current Price (Nat. Avg)", f"Rs.{current_price:,.2f}")
col1.markdown("</div>", unsafe_allow_html=True)

col2.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col2.metric("Predicted Max (30d)", f"Rs.{predicted_price:,.2f}")
col2.markdown("</div>", unsafe_allow_html=True)

col3.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col3.metric("Expected Change", f"{change_pct:+.2f}%")
col3.markdown("</div>", unsafe_allow_html=True)

col4.markdown("<div class='metric-card'>", unsafe_allow_html=True)
col4.metric("Model Accuracy (SARIMA)", f"{accuracy:.2f}%")
col4.markdown("</div>", unsafe_allow_html=True)

st.write("")

# Main tab group
tab1, tab2 = st.tabs(["🌾 Forecast & Alerts", "📊 Model Analytics & Quality"])

with tab1:
    left, right = st.columns([1.8, 1.2])
    with left:
        st.markdown("<div class='section-title'>Price Trend and Rolling Signal (National Average)</div>", unsafe_allow_html=True)
        if not commodity_data.empty and "Date" in commodity_data.columns and "Price" in commodity_data.columns:
            # Average prices across states per date for national average plotting
            plot_df = commodity_data.groupby("Date", as_index=False).agg({"Price": "mean"})
            plot_df = plot_df.sort_values("Date")
            plot_df["RollingMean7"] = plot_df["Price"].rolling(window=7, min_periods=1).mean()
            trend_fig = px.line(
                plot_df,
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

        # Show specific intervention recommendation details
        if "Release" in policy_action:
            rec_text = f"**NAFED buffer stock release recommended** — estimated **{int(abs(change_pct) * 80)} MT** required to stabilize the market in high-growth consumption zones."
            action_desc = "Release of buffer stocks in high price areas."
        elif "Prepare" in policy_action:
            rec_text = "**Prepare Market Intervention** — Price spike risk detected. Consider lowering import duties and monitoring wholesale inventories."
            action_desc = "Initiate close monitoring and prep state distribution channels."
        elif "Low" in policy_action:
            rec_text = "**Procurement Intervention** — Prices are dropping rapidly. Consider starting state-level MSP procurement to protect farmers."
            action_desc = "MSP procurement and state-level warehouse storage support."
        else:
            rec_text = "**Market is Stable** — Retail and wholesale prices are within normal ranges. No immediate buffer stock intervention required."
            action_desc = "Monitor standard daily reporting updates."

        st.markdown(
            f"""
            <div style='padding:1rem; border-radius:14px; background:white; border-left:6px solid {action_color}; box-shadow:0 3px 10px rgba(0,0,0,0.06)'>
              <div style='font-size:0.85rem; color:#4B5563;'>Recommended Action</div>
              <div style='font-size:1.15rem; font-weight:800; color:{action_color}; margin-top:0.35rem;'>{policy_action}</div>
              <div style='margin-top:0.6rem; color:#1E293B; font-size:0.9rem;'>{rec_text}</div>
              <div style='margin-top:0.4rem; color:#64748B; font-size:0.8rem; font-style:italic;'>Details: {action_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("<div class='section-title'>National Policy Alert Summary</div>", unsafe_allow_html=True)
        alerts_show = alerts_df.copy()
        alerts_show["PolicyAction"] = alerts_show["PolicyAction"].map(format_action)
        st.dataframe(alerts_show, use_container_width=True, hide_index=True)

    # State-wise alert report
    if not state_alerts_df.empty:
        st.markdown("<div class='section-title'>State-wise Price Alert Report</div>", unsafe_allow_html=True)
        state_comm_alerts = state_alerts_df[state_alerts_df["Commodity"] == selected_commodity].copy()
        if not state_comm_alerts.empty:
            state_comm_alerts["PolicyAction"] = state_comm_alerts["PolicyAction"].map(format_action)
            # Add alert emojis based on actions
            def action_emoji(act):
                if "Release" in act: return "🚨 " + act
                if "Prepare" in act: return "⚠️ " + act
                if "Low" in act: return "📉 " + act
                return "✅ " + act
            state_comm_alerts["PolicyAction"] = state_comm_alerts["PolicyAction"].apply(action_emoji)
            st.dataframe(
                state_comm_alerts[["State", "CurrentPrice", "PredictedMax30d", "ExpectedChangePct", "PolicyAction"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No state-wise alerts available for selected commodity.")

with tab2:
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
        st.markdown("<div class='section-title'>Model Evaluation Summary Table</div>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Diagnostic Plots
    st.markdown("<div class='section-title'>Visual Model Diagnostics</div>", unsafe_allow_html=True)
    diag_col1, diag_col2 = st.columns(2)
    commodity_slug = sanitize_name(selected_commodity)

    with diag_col1:
        comp_path = os.path.join(OUTPUTS_DIR, f"model_comparison_{commodity_slug}.png")
        if os.path.exists(comp_path):
            st.image(comp_path, use_column_width=True, caption=f"{selected_commodity} - Model Metrics Comparison")
        else:
            st.info("Model comparison plot not found. Run pipeline to generate it.")

    with diag_col2:
        feat_path = os.path.join(OUTPUTS_DIR, f"feature_importance_{commodity_slug}.png")
        if os.path.exists(feat_path):
            st.image(feat_path, use_column_width=True, caption=f"{selected_commodity} - XGBoost Feature Importance")
        else:
            st.info("Feature importance plot not found. Run pipeline to generate it.")

st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
