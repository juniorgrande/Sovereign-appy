import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Sovereign Apex", layout="centered")

# Custom CSS for the Cockpit
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: gold; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #ffd700; color: black; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- BULLETPROOF ENGINE ---
def analyze_authors(df):
    try:
        # Clean the data: Flatten Multi-Index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        recent = df.tail(3)
        # Ensure we have enough data
        if len(recent) < 3:
            return False, 0.0
            
        # Extract scalar values safely
        c1_high = float(recent.iloc[0]['High'])
        c3_low = float(recent.iloc[2]['Low'])
        c3_open = float(recent.iloc[2]['Open'])
        c3_close = float(recent.iloc[2]['Close'])
        c3_high = float(recent.iloc[2]['High'])
        
        # 1. Sakata Gap Logic (Displacement)
        is_fvg = c3_low > c1_high
        
        # 2. Nison Conviction Logic (Body Strength)
        body = abs(c3_close - c3_open)
        total_range = c3_high - c3_low
        conviction = (body / total_range) * 100 if total_range > 0 else 0
        
        return is_fvg, conviction
    except Exception as e:
        st.error(f"Logic Error: {e}")
        return False, 0.0

# --- MOBILE UI ---
st.title("🔱 SOVEREIGN APEX")

if 'profit' not in st.session_state: st.session_state.profit = 0.0

# Goal Tracker
col1, col2 = st.columns(2)
col1.metric("Daily Profit", f"${st.session_state.profit}")
col2.metric("Target", "$10.00")
st.progress(min(st.session_state.profit / 10.0, 1.0))

# Asset Selection
asset = st.selectbox("Select Asset", ["GC=F", "BTC-USD", "CL=F", "EURUSD=X"])

if st.button("INVOKE THE 40 AUTHORS"):
    # Fetch data and handle the multi-index issue immediately
    df = yf.download(asset, period="5d", interval="1h", progress=False)
    
    if not df.empty:
        is_fvg, conviction = analyze_authors(df)
        
        if is_fvg and conviction > 70:
            st.success(f"RANK #1: {asset} DISPLACEMENT")
            st.write(f"**Conviction Score:** {conviction:.1f}%")
            st.info("VERDICT: The Authors see a clear Shove. Targeting the Vacuum.")
        else:
            st.warning("STALEMATE: No valid displacement detected.")
        
        # Simple Chart
        st.line_chart(df['Close'])
    else:
        st.error("No data found. Check symbol or internet.")

if st.button("LOG SUCCESSFUL $10 SHOVE"):
    st.session_state.profit += 10.0
    st.balloons()
    st.rerun()
