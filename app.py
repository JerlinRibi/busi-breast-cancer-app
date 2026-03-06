# ================= app.py =================

import streamlit as st
from predictor import predictor
from analytics import analytics_dashboard
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from dotenv import load_dotenv

st.set_page_config(
    page_title="AI Breast Ultrasound Cancer Detection System",
    page_icon="🏥",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

try:
    ADMIN_USER = st.secrets["ADMIN_USER"]
    ADMIN_PASS = st.secrets["ADMIN_PASS"]
except Exception:
    load_dotenv()
    ADMIN_USER = os.getenv("ADMIN_USER", "admin")
    ADMIN_PASS = os.getenv("ADMIN_PASS", "1234")

st.markdown("""
<style>

body {
background: linear-gradient(120deg,#d7ecff,#f3f9ff,#ffffff);
}

.main {
background: rgba(255,255,255,0.95);
padding:25px;
border-radius:12px;
box-shadow:0px 5px 15px rgba(0,0,0,0.05);
}

h1, h2, h3 {
color:#0a4f70;
}

.stButton>button {
background-color:#0a4f70;
color:white;
border-radius:10px;
height:3em;
width:100%;
font-weight:bold;
}

.stButton>button:hover {
background-color:#0c6a94;
}

[data-testid="stSidebar"] {
background: linear-gradient(180deg,#e3f2fd,#ffffff);
}

[data-testid="metric-container"] {
background:#ffffff;
border-radius:10px;
padding:15px;
border:1px solid #e6eef5;
box-shadow:0px 2px 6px rgba(0,0,0,0.05);
}

.block-container {
padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background: linear-gradient(90deg,#2E86C1,#1B4F72);
padding:18px;
border-radius:12px;
color:white;
text-align:center;
font-size:30px;
font-weight:bold;
margin-bottom:20px;
">
🩺 AI Breast Ultrasound Cancer Detection System
</div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOGIN FUNCTION ----------------
def login():

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

# ---------------- LOGOUT FUNCTION ----------------
def logout():

    if st.button("Logout 🔓"):

        st.session_state.logged_in = False
        st.rerun()

# =====================================================
# MAIN SYSTEM
# =====================================================

if st.session_state.logged_in:

    with st.sidebar:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/2785/2785544.png",
            width=80
        )

        st.sidebar.title("Hospital Navigation")

        st.success("🟢 AI Diagnostic System Active")

        st.info("Use the menu below to access system modules.")

        menu = st.radio(
            "Navigation",
            ["🔬 Prediction", "📊 Analytics Dashboard"]
        )

        logout()

    # ---------------- PREDICTION PAGE ----------------
    if menu == "🔬 Prediction":

        result = predictor()

        if result is not None:

            st.session_state.history.append({
                "Prediction": result,
                "Time": datetime.now()
            })

    # ---------------- ANALYTICS PAGE ----------------
    elif menu == "📊 Analytics Dashboard":

        analytics_dashboard()

else:

    login()


# ---------------- FOOTER (ADDED) ----------------
st.markdown("---")

st.markdown("""
<center>

AI Breast Ultrasound Cancer Detection System  
Research Prototype • Educational Use Only  

</center>
""", unsafe_allow_html=True)