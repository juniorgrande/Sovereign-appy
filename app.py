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
# 1. CORE CONFIG & TELEGRAM SENTINEL
# =================================================================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_apex_alert(tf, rank, asset, entry, tp, sl, size, side, retries=3):
    """
    Reliable notification system with TF filtering and retry logic.
    """
    if tf == "15m" and rank == "SCOUT": 
        return 
    
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
            time.sleep(2)

# =================================================================
# 2. SAFE VECTOR PATTERN DETECTION
# =================================================================
def get_patterns_safe(df):
    """
    Detect multiple candlestick patterns safely and vectorized.
    Returns a list of detected patterns on the last candle.
    """
    if len(df) < 3:
        return ["Neutral"]

    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']

    body = abs(c - o)
    candle_color = np.where(c >= o, 1, -1)  # 1 = bullish, -1 = bearish
    shade_top = h - np.maximum(c, o)
    shade_bottom = np.minimum(c, o) - l

    last = -1
    prev1 = -2
    prev2 = -3

    patterns = []

    # Single-candle patterns
    if (h.iloc[last] - l.iloc[last]) > 0:
        if shade_bottom.iloc[last] > body.iloc[last] * 2: patterns.append("Hammer (Bullish)")
        if shade_top.iloc[last] > body.iloc[last] * 2: patterns.append("Shooting Star (Bearish)")
        if body.iloc[last] / (h.iloc[last] - l.iloc[last] + 1e-9) > 0.7:
            if candle_color[last] == 1: patterns.append("Marubozu Bullish")
            else: patterns.append("Marubozu Bearish")

    # Two-candle patterns
    if len(df) >= 2:
        if candle_color[prev1] == -1 and candle_color[last] == 1:
            if c.iloc[last] > o.iloc[prev1] and o.iloc[last] < c.iloc[prev1]:
                patterns.append("Bullish Engulfing")
        if candle_color[prev1] == 1 and candle_color[last] == -1:
            if o.iloc[last] < c.iloc[prev1] and c.iloc[last] > o.iloc[prev1]:
                patterns.append("Bearish Engulfing")
        if l.iloc[last] == l.iloc[prev1]: patterns.append("Tweezer Bottom")
        if h.iloc[last] == h.iloc[prev1]: patterns.append("Tweezer Top")

    # Three-candle patterns
    if len(df) >= 3:
        if all(candle_color[[prev2, prev1, last]] == 1) and all(c.iloc[[prev2, prev1, last]] > c.shift(1).iloc[[prev2, prev1, last]]):
            patterns.append("Three White Soldiers")
        if all(candle_color[[prev2, prev1, last]] == -1) and all(c.iloc[[prev2, prev1, last]] < c.shift(1).iloc[[prev2, prev1, last]]):
            patterns.append("Three Black Crows")
        if candle_color[prev2] == -1 and body.iloc[prev1] / body.iloc[prev2] < 0.5 and candle_color[last] == 1:
            patterns.append("Morning Star")
        if candle_color[prev2] == 1 and body.iloc[prev1] / body.iloc[prev2] < 0.5 and candle_color[last] == -1:
            patterns.append("Evening Star")

    if not patterns:
        patterns.append("Neutral")

    return patterns

# =================================================================
# 3. DYNAMIC STRIKE & SL/TP CALCULATION
# =================================================================
def calculate_apex_strategy(df, score, goal_usd=10.0):
    """
    Bidirectional strategy with dynamic SL/TP scaling based on alignment score.
    """
    c = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    side = "LONG" if score > 0 else "SHORT"
    sl_mult = 1.0 if abs(score) >= 5.5 else 2.0

    if side == "LONG":
        entry = c - (atr * 0.2)
        sl = entry - (atr * sl_mult)
        tp = entry + (atr * 1.5)
    else:
        entry = c + (atr * 0.2)
        sl = entry + (atr * sl_mult)
        tp = entry - (atr * 1.5)
        
    size = round(goal_usd / max(abs(tp - entry), 1e-6), 4)
    return side, round(entry, 2), round(tp, 2), round(sl, 2), size

# =================================================================
# 4. STREAMLIT INTERFACE
# =================================================================
st.set_page_config(layout="wide")
st.title("🛡️ SOVEREIGN APEX v24.0 - FINAL SENTINEL")

asset = st.sidebar.selectbox("Market Target", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal = st.sidebar.number_input("Daily Goal ($)", value=10)

# Fetch Multi-TF Data
tfs = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
data = {tf: yf.download(asset, period=p, interval=tf, progress=False) for tf, p in tfs.items()}

if not data['1h'].empty:
    # Alignment Score
    scores = {tf: (1 if data[tf]['Close'].iloc[-1] > data[tf]['Close'].rolling(20).mean().iloc[-1] else -1) for tf in tfs}
    total_score = (scores['1d']*3) + (scores['4h']*2) + (scores['1h']*1)

    # Dynamic Strategy
    side, entry, tp, sl, size = calculate_apex_strategy(data['1h'], total_score, goal)
    rank = "TITAN" if abs(total_score) >= 5.5 else "SCOUT"
    patterns = get_patterns_safe(data['1h'])

    # --- Dashboard ---
    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    c4.metric("Detected Patterns", ", ".join(patterns))

    st.code(f"ASSET: {asset}\nSIDE: {side}\nLIMIT: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    if st.button("📲 SEND SIGNAL TO PHONE"):
        send_apex_alert("1h", rank, asset, entry, tp, sl, size, side)
        st.success("Apex Alert Dispatched via Sentinel.")

    # --- Charting ---
    fig = go.Figure(data=[go.Candlestick(
        x=data['1h'].index, open=data['1h']['Open'], high=data['1h']['High'],
        low=data['1h']['Low'], close=data['1h']['Close'])])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
