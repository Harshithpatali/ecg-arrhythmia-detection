import sys
from pathlib import Path

# Add the project root to Python's path so imports work everywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---- Imports ----
from components.sidebar import render_sidebar
from pages.dashboard import render_dashboard
from pages.ecg_analysis import render_ecg_analysis
from pages.waveform import render_waveform          # New import
from pages.about import render_about

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="ECG Arrhythmia AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------
# Sidebar
# ----------------------------------

render_sidebar()

# ----------------------------------
# Navigation
# ----------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "ECG Analysis",
        "Waveform Viewer",          # New option
        "About",
    ],
)

# ----------------------------------
# Routing
# ----------------------------------

if page == "Dashboard":
    render_dashboard()

elif page == "ECG Analysis":
    render_ecg_analysis()

elif page == "Waveform Viewer":      # New route
    render_waveform()

elif page == "About":
    render_about()