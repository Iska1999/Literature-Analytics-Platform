# Directory structure:
# streamlit_app/
# ├─1_field_trend_analysis.py
# └─2_topic_trend_analysis.py

# main_app
import streamlit as st
from pathlib import Path

import os, torch

# point torch.classes.__path__ at its real file
torch.classes.__path__ = [
    os.path.join(torch.__path__[0], torch.classes.__file__)
]




st.set_page_config(page_title="Literature Analytics Platform", layout="wide")

st.title("📊 Literature Analytics Platform")
st.markdown(    """
    Welcome to the Literature Analytics Platform!

    This application explores the potential relationships between trends in scientific
    literature and stock market sector movements. The analysis is recent uptil 05/22/2025.

    **👈 Select an analysis from the sidebar** to view detailed results from:
    - Field Trend BSTS Analysis
    - Topic Trend VAR Analysis
    - Field Trend BSTS Analysis
    - Topic Trend VAR Analysis
    

    """)

st.markdown("---")
st.caption("Developed for the second round technical assessment @Ghamut.")
