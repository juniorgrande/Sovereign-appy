import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Sovereign Apex", layout="centered")

# --- CUSTOM CSS FOR THE "COCKPIT" FEEL ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: gold; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ffd700; color: black; }
    </style>
    """, unsafe_allow_html=True)

# --- AUTHOR LOGIC ENGINE ---
def analyze_authors(df):
    recent = df.tail(3)
    c1, c3 = recent.iloc[0], recent.iloc[2]
    
    # 1. Sakata Gap (Homma)
    is_fvg = c3['Low'] > c1['High']
    # 2. Real Body Strength (Nison)
    body = abs(c3['Close'] - c3['Open'])
    total_range = c3['High'] - c3['Low']
    conviction = (body / total_range) * 100
    
    return is_fvg, conviction

# --- THE MOBILE UI ---
st.title("🔱 SOVEREIGN APEX")
st.subheader("Displacement Engine v2.0")

# 1. PSYCHOLOGY GATEKEEPER
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    with st.form("Login"):
        st.write("Are you following the 'Authors' or Chasing Price?")
        if st.form_submit_button("I AM SOVEREIGN"):
            st.session_state.authenticated = True
            st.rerun()
else:
    # 2. DAILY QUOTA TRACKER
    if 'profit' not in st.session_state: st.session_state.profit = 0.0
    
    col1, col2 = st.columns(2)
    col1.metric("Daily Profit", f"${st.session_state.profit}")
    col2.metric("Target", "$10.00")
    
    progress = min(st.session_state.profit / 10.0, 1.0)
    st.progress(progress)

    # 3. LIVE SCANNER
    asset = st.selectbox("Select Asset", ["GC=F", "BTC-USD", "CL=F", "EURUSD=X"])
    
    if st.button("INVOKE THE 40 AUTHORS"):
        df = yf.download(asset, period="5d", interval="1h")
        is_fvg, conviction = analyze_authors(df)
        
        # Display Results
        if is_fvg and conviction > 70:
            st.success(f"RANK #1: {asset} DISPLACEMENT")
            st.write(f"**Author's Conviction:** {conviction:.1f}%")
            st.info("HOMMA VERDICT: The Sakata Gap is confirmed. The Yang energy is pure.")
        else:
            st.warning("WAIT: Stalemate detected. The Authors advise patience.")
            
        # 4. CHART VISUALIZATION
        st.line_chart(df['Close'].tail(20))

    # 5. LOGGING SUCCESS
    if st.button("LOG $10 SHOVE SUCCESS"):
        st.session_state.profit += 10.0
        st.balloons()
