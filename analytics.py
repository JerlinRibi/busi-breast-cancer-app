# ================= analytics.py =================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta

# =====================================================
# FIXED ABSOLUTE PATH (WORKS FROM ANY FOLDER)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # ✅ ensure 'data/' exists
LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")


def analytics_dashboard():

    # ---------------- AUTO-REFRESH IF NEW PREDICTION ----------------
    if st.session_state.get("new_prediction", False):
        st.session_state.new_prediction = False
        st.experimental_rerun()

    st.title("📊 Hospital Analytics Dashboard")

    # ---------------- DEBUG INFO ----------------
    st.caption(f"📁 Reading file from: {LOG_FILE}")
    st.caption(f"✅ File exists: {os.path.exists(LOG_FILE)}")

    # ---------------- FILE CHECK ----------------
    if not os.path.exists(LOG_FILE):
        st.warning("No prediction data available yet.")
        st.info("👉 Make at least one prediction in Predictor page.")
        return

    try:
        df = pd.read_csv(LOG_FILE)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    if df.empty:
        st.warning("Prediction file is empty.")
        return

    # ---------------- CLEAN DATA ----------------
    if "Time" not in df.columns or "Prediction" not in df.columns:
        st.error("CSV format incorrect. Required columns missing.")
        return

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])

    # =====================================================
    # SYSTEM OVERVIEW
    # =====================================================
    st.markdown("### 🏥 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", len(df))
    col2.metric("Malignant Cases",
                int((df["Prediction"] == "malignant").sum()))
    col3.metric("Normal Cases",
                int((df["Prediction"] == "normal").sum()))

    # =====================================================
    # CHART DATA
    # =====================================================
    counts = df["Prediction"].value_counts()

    c1, c2 = st.columns(2)

    # ---------------- PIE CHART ----------------
    with c1:
        st.subheader("Prediction Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%')
        ax1.set_title("Prediction Share")
        st.pyplot(fig1)

    # ---------------- BAR CHART ----------------
    with c2:
        st.subheader("Class Counts")

        fig2, ax2 = plt.subplots()
        counts.plot(kind="bar", ax=ax2)
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    # =====================================================
    # TREND GRAPH (LAST 30 DAYS + DATE OVERLAY FIXED)
    # =====================================================
    st.subheader("Prediction Trend Over Time (Last 30 Days)")

    # filter last 30 days
    today = datetime.now().date()
    start_date = today - timedelta(days=30)
    df_recent = df[df["Time"].dt.date >= start_date]

    if df_recent.empty:
        st.info("No predictions in the last 30 days.")
    else:
        trend = df_recent.groupby(df_recent["Time"].dt.date)["Prediction"].count()

        fig3, ax3 = plt.subplots()
        ax3.plot(trend.index, trend.values, marker="o")

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Predictions")
        ax3.set_title("Daily Prediction Trend")

        # ✅ Format x-axis dates
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig3)

    # =====================================================
    # TABLE VIEW
    # =====================================================
    st.subheader("📄 Prediction Records")
    st.dataframe(df, use_container_width=True)