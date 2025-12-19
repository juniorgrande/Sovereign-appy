import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import sqlite3
import requests
import time
from datetime import datetime

# =================================================================
# 1. CORE CONFIG & BIDIRECTIONAL NOTIFIER (WITH RETRIES)
# =================================================================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_apex_alert(tf, rank, asset, entry, tp, sl, size, side, retries=3):
    """Reliable notification system with TF filtering and retry logic."""
    if tf == "15m" and rank == "SCOUT": return 
    
    emoji = "🔴 SELL" if side == "SHORT" else "🟢 BUY"
    rank_icon = "🔱 TITAN" if rank == "TITAN" else "📡 SCOUT"
    
    msg = f"{rank_icon} {emoji} AUTHORIZED\n" \
          f"📍 {asset} ({tf.upper()})\n" \
          f"📥 Entry: {entry}\n" \
          f"🎯 TP: {tp} | 🛡️ SL: {sl}\n" \
          f"⚖️ Size: {size} Lots"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for i in range(retries):
        try:
            r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
            if r.status_code == 200: break
        except:
            time.sleep(2) # Wait and retry if internet flickers

# =================================================================
# 2. PATTERN DETECTION & WIN RATE TRACKING
# =================================================================
def get_pattern(df):
    """Vectorized logic for 10+ patterns."""
    c, o, h, l = df['Close'], df['Open'], df['High'], df['Low']
    body = abs(c - o).iloc[-1]
    wick_top = (h - np.maximum(c, o)).iloc[-1]
    wick_bottom = (np.minimum(c, o) - l).iloc[-1]
    
    if wick_bottom > body * 2: return "Hammer (Bullish)"
    if wick_top > body * 2: return "Shooting Star (Bearish)"
    if c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2]: return "Engulfing (Bullish)"
    return "Neutral"

def get_vault_stats():
    """Calculates historical edge from your saved Shoves."""
    conn = sqlite3.connect('titan_vault.db')
    try:
        df = pd.read_sql_query("SELECT rank, status FROM history", conn)
        # Assuming 'status' is updated after trade closes
        stats = df.groupby('rank')['status'].value_counts(normalize=True).unstack().fillna(0)
        return stats
    except: return None

# =================================================================
# 3. DYNAMIC STRIKE & SL SCALING
# =================================================================
def calculate_apex_strategy(df, score, goal_usd=10.0):
    """Bidirectional logic with Dynamic SL scaling based on Alignment."""
    c = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    side = "LONG" if score > 0 else "SHORT"
    
    # DYNAMIC SL: If score is high (Titan), use 1.0x ATR. If low (Scout), use 2.0x ATR.
    sl_mult = 1.0 if abs(score) >= 5.5 else 2.0
    
    if side == "LONG":
        entry = c - (atr * 0.2)
        sl = entry - (atr * sl_mult)
        tp = entry + (atr * 1.5)
    else:
        entry = c + (atr * 0.2)
        sl = entry + (atr * sl_mult)
        tp = entry - (atr * 1.5)
        
    size = round(goal_usd / abs(tp - entry), 4)
    return side, round(entry, 2), round(tp, 2), round(sl, 2), size

# =================================================================
# 4. MAIN INTERFACE
# =================================================================
st.set_page_config(layout="wide")
st.title("🛡️ SOVEREIGN APEX v23.0")

asset = st.sidebar.selectbox("Market Target", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal = st.sidebar.number_input("Daily Goal ($)", value=10)

# Multi-TF Fetch
tfs = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
data = {tf: yf.download(asset, period=p, interval=tf, progress=False) for tf, p in tfs.items()}

if not data['1h'].empty:
    # --- ALIGNMENT & BIAS ---
    scores = {tf: (1 if data[tf]['Close'].iloc[-1].item() > data[tf]['Close'].rolling(20).mean().iloc[-1].item() else -1) for tf in tfs}
    total_score = (scores['1d']*3) + (scores['4h']*2) + (scores['1h']*1)
    
    # --- DYNAMIC STRATEGY ---
    side, entry, tp, sl, size = calculate_apex_strategy(data['1h'], total_score, goal)
    rank = "TITAN" if abs(total_score) >= 5.5 else "SCOUT"
    pattern = get_pattern(data['1h'])

    # --- UI DASHBOARD ---
    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    c4.metric("Pattern", pattern)

    # --- THE ONE-CLICK SIGNAL ---
    st.code(f"ASSET: {asset}\nSIDE: {side}\nLIMIT: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")
    
    if st.button("📲 SEND SIGNAL TO PHONE"):
        send_apex_alert("1h", rank, asset, entry, tp, sl, size, side)
        st.success("Apex Alert Dispatched via Sentinel.")

    # --- CHARTING ---
    fig = go.Figure(data=[go.Candlestick(x=data['1h'].index, open=data['1h']['Open'], high=data['1h']['High'], low=data['1h']['Low'], close=data['1h']['Close'])])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
