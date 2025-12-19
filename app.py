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
# 1. CORE CONFIG & TELEGRAM
# =================================================================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_apex_alert(tf, rank, asset, entry, tp, sl, size, side, retries=3):
    if tf == "15m" and rank == "SCOUT": return
    emoji = "🔴 SELL" if side=="SHORT" else "🟢 BUY"
    rank_icon = "🔱 TITAN" if rank=="TITAN" else "📡 SCOUT"
    msg = f"{rank_icon} {emoji} AUTHORIZED\n📍 {asset} ({tf.upper()})\n📥 Entry: {entry}\n🎯 TP: {tp} | 🛡️ SL: {sl}\n⚖️ Size: {size} Lots"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for _ in range(retries):
        try:
            r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode":"Markdown"}, timeout=5)
            if r.status_code == 200: break
        except: time.sleep(2)

# =================================================================
# 2. PATTERN DETECTION (10+ Patterns) SAFE & VECTORIZED
# =================================================================
def detect_patterns(df):
    if df.empty or len(df) < 3:
        return ["Neutral"]
    
    c, o, h, l = df['Close'], df['Open'], df['High'], df['Low']
    body = (c - o).abs()
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    last_idx = -1
    patterns = []

    # Bullish patterns
    if lower_wick.iloc[last_idx] > body.iloc[last_idx]*2 and upper_wick.iloc[last_idx] < body.iloc[last_idx]*0.5:
        patterns.append("Hammer")
    if c.iloc[last_idx] > o.iloc[last_idx] and c.iloc[last_idx] > c.shift(1).iloc[last_idx] and o.iloc[last_idx] < o.shift(1).iloc[last_idx]:
        patterns.append("Engulfing (Bullish)")
    if len(df) >= 3 and all(c.iloc[-3:] > o.iloc[-3:]) and all(c.iloc[-3:] > c.shift(1).iloc[-3:]):
        patterns.append("Three White Soldiers")

    # Bearish patterns
    if upper_wick.iloc[last_idx] > body.iloc[last_idx]*2 and lower_wick.iloc[last_idx] < body.iloc[last_idx]*0.5:
        patterns.append("Shooting Star")
    if c.iloc[last_idx] < o.iloc[last_idx] and c.iloc[last_idx] < c.shift(1).iloc[last_idx] and o.iloc[last_idx] > o.shift(1).iloc[last_idx]:
        patterns.append("Engulfing (Bearish)")
    if len(df) >= 3 and all(c.iloc[-3:] < o.iloc[-3:]) and all(c.iloc[-3:] < c.shift(1).iloc[-3:]):
        patterns.append("Three Black Crows")
    
    if not patterns:
        patterns = ["Neutral"]
    return patterns

# =================================================================
# 3. SAFE BIAS / SCORE CALCULATION
# =================================================================
def get_bias(df, window=20):
    if df.empty or len(df) < window:
        return 0
    rolling_mean = df['Close'].rolling(window).mean().iloc[-1]
    if pd.isna(rolling_mean):
        return 0
    return 1 if df['Close'].iloc[-1] > rolling_mean else -1

# =================================================================
# 4. STRIKE & SL/TP CALCULATION
# =================================================================
def calculate_strategy(df, score, goal_usd=10.0):
    if df.empty:
        return "LONG", 0, 0, 0, 0
    c = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    side = "LONG" if score >= 0 else "SHORT"
    sl_mult = 1.0 if abs(score) >= 5.5 else 2.0

    if side == "LONG":
        entry = c - (atr*0.2)
        sl = entry - (atr*sl_mult)
        tp = entry + (atr*1.5)
    else:
        entry = c + (atr*0.2)
        sl = entry + (atr*sl_mult)
        tp = entry - (atr*1.5)
    size = round(goal_usd / abs(tp-entry), 4) if abs(tp-entry) > 0 else 0
    return side, round(entry,2), round(tp,2), round(sl,2), size

# =================================================================
# 5. APP INTERFACE
# =================================================================
st.set_page_config(layout="wide", page_title="Sovereign Apex v24.0")
st.title("🛡️ SOVEREIGN APEX v24.0")

asset = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal = st.sidebar.number_input("Daily Goal ($)", value=10)

tfs = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
data = {tf: yf.download(asset, period=p, interval=tf, progress=False) for tf,p in tfs.items()}

if not data['1h'].empty:
    # Bias / Alignment
    scores = {tf: get_bias(data[tf], window=20) for tf in tfs}
    total_score = (scores['1d']*3) + (scores['4h']*2) + (scores['1h']*1)

    # Strategy calculation
    side, entry, tp, sl, size = calculate_strategy(data['1h'], total_score, goal)
    rank = "TITAN" if abs(total_score) >= 5.5 else "SCOUT"
    patterns = detect_patterns(data['1h'])

    # UI Dashboard
    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    c4.metric("Pattern", ", ".join(patterns))

    st.code(f"ASSET: {asset}\nSIDE: {side}\nENTRY: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    if st.button("📲 SEND SIGNAL"):
        send_apex_alert("1h", rank, asset, entry, tp, sl, size, side)
        st.success("Apex Alert Dispatched!")

    # Candlestick Chart
    fig = go.Figure(data=[go.Candlestick(x=data['1h'].index,
                                         open=data['1h']['Open'],
                                         high=data['1h']['High'],
                                         low=data['1h']['Low'],
                                         close=data['1h']['Close'])])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
