# ================= app.py =================

import streamlit as st
from predictor import predictor
from analytics import analytics_dashboard  # ✅ Persistent CSV dashboard
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os  # ✅ Needed for folder creation
from dotenv import load_dotenv  # ✅ For secure login

# ---------------- ENSURE DATA FOLDER EXISTS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # ✅ Auto-create 'data/' if missing

# ---------------- LOAD ENV VARIABLES WITH FALLBACK ----------------
load_dotenv()  # reads .env in project root

ENV_USER = os.getenv("ADMIN_USER")
ENV_PASS = os.getenv("ADMIN_PASS")

# Fallback to default if .env missing or empty
ADMIN_USER = ENV_USER if ENV_USER else "admin"
ADMIN_PASS = ENV_PASS if ENV_PASS else "1234"

# ---------------- DEBUG: CHECK ENV ----------------
st.write("DEBUG:", ADMIN_USER, ADMIN_PASS)  # shows which credentials app is using

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Breast Cancer Detection",
    page_icon="🏥",
    layout="wide"
)

# ---------------- HOSPITAL THEME CSS ----------------
st.markdown("""
<style>

body {
    background-color: #f4f8fb;
}

.main {
    background-color: #f4f8fb;
}

h1, h2, h3 {
    color: #0a4f70;
}

.stButton>button {
    background-color: #0a4f70;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #0c6a94;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.sidebar .sidebar-content {
    background-color: #e6f2f8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOGIN PAGE ----------------
def login():

    st.markdown("# 🏥 Medical Imaging Portal")
    st.markdown("### 🔐 Secure Hospital Login")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):

            if username == ADMIN_USER and password == ADMIN_PASS:
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

# ---------------- LOGOUT ----------------
def logout():
    if st.button("Logout 🔓"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- MAIN APP ----------------
if st.session_state.logged_in:

    # Sidebar Hospital Panel
    with st.sidebar:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2785/2785544.png",
            width=80
        )
        st.success("🟢 Hospital System Active")

        menu = st.radio(
            "Navigation",
            ["🔬 Prediction", "📊 Analytics Dashboard"]
        )

        logout()

    # ---------- PREDICTION PAGE ----------
    if menu == "🔬 Prediction":

        result = predictor()

        # save history automatically
        if result is not None:
            st.session_state.history.append({
                "Prediction": result,
                "Time": datetime.now()
            })

    # ---------- DASHBOARD PAGE ----------
    elif menu == "📊 Analytics Dashboard":
        analytics_dashboard()  # ✅ Persistent CSV version called

else:
    login()