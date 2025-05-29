# pages/topic_trend_bsts_analysis.py
import sys
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.streamlit_functions as streamlit_functions
from utils.db_mgr import db_manager

from config import Config

# ----- Streamlit App -----
st.set_page_config(page_title="Topic Trend BSTS Analysis", layout="wide")
st.title("📊 Topic Trend BSTS Analysis")

# Initialize session state for data if not already loaded
if 'topic_data_loaded' not in st.session_state:
    try:
        with st.spinner('Connecting to database and loading data...'):
            # Validate configuration
            Config.validate()
            
            # Fetch from server using db_manager
            st.session_state.topic_metadata = db_manager.read_table('monthly_topic')
            st.session_state.topic_rolling_metadata = db_manager.read_table('rolling_topic')
            st.session_state.topic_words = db_manager.read_table('topic_words')
            st.session_state.topic_marketdata = db_manager.read_table('marketdata')
            st.session_state.topic_data_loaded = True
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}")
        st.info("Please ensure your .env file is properly configured with DB_HOST, DB_USER, and DB_PASSWORD")
        st.stop()
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# ----- User Input -----
selected_monthly = st.selectbox("Select Feature Types", Config.MONTHLY_OPTIONS)
selected_sector = st.selectbox("Select Market Sector", Config.SECTOR_OPTIONS)
selected_sector = selected_sector.lower()

# ----- Analysis Trigger -----
if st.button("🔍 Run Analysis"):
    try:
        with st.spinner('Running BSTS analysis on topic trends...'):
            result = streamlit_functions.trend_analysis_bsts(
                "topic",
                selected_monthly,
                st.session_state.topic_metadata,
                st.session_state.topic_rolling_metadata,
                st.session_state.topic_words,
                st.session_state.topic_marketdata,
                "topic_growth",
                selected_sector
            )
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# ----- Exit Button -----
st.markdown("---")
if st.button("❌ Exit App"):
    # Dispose of database connections
    db_manager.dispose()
    print("Database connections closed.")
    
    st.warning("Exiting the app. You can close the browser tab.")
    components.html("<script>window.close()</script>", height=0)