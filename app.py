import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import sqlite3
import requests
import time
from datetime import datetime
from binance.client import Client

# =================================================================
# 1. CONFIG & TELEGRAM
# =================================================================
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET"
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def send_apex_alert(tf, rank, asset, entry, tp, sl, size, side, retries=3):
    if tf == "15m" and rank == "SCOUT": return 
    emoji = "🔴 SELL" if side == "SHORT" else "🟢 BUY"
    rank_icon = "🔱 TITAN" if rank == "TITAN" else "📡 SCOUT"
    msg = f"{rank_icon} {emoji} AUTHORIZED\n📍 {asset} ({tf.upper()})\n📥 Entry: {entry}\n🎯 TP: {tp} | 🛡️ SL: {sl}\n⚖️ Size: {size} Lots"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for i in range(retries):
        try:
            r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
            if r.status_code == 200: break
        except:
            time.sleep(2)

# =================================================================
# 2. PATTERN DETECTION
# =================================================================
def get_pattern(df):
    c, o, h, l = df['Close'], df['Open'], df['High'], df['Low']
    body = abs(c - o)
    wick_top = h - np.maximum(c, o)
    wick_bottom = np.minimum(c, o) - l
    pattern = []

    if wick_bottom.iloc[-1] > body.iloc[-1]*2: pattern.append("Hammer (Bullish)")
    if wick_top.iloc[-1] > body.iloc[-1]*2: pattern.append("Shooting Star (Bearish)")
    if len(c) > 2 and (c.iloc[-1] > o.iloc[-1] and c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2]): pattern.append("Engulfing (Bullish)")
    if not pattern: pattern.append("Neutral")
    return ", ".join(pattern)

# =================================================================
# 3. DYNAMIC STRIKE & SL SCALING
# =================================================================
def calculate_apex_strategy(df, score, goal_usd=10.0):
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
    size = round(goal_usd / abs(tp - entry), 4)
    return side, round(entry, 2), round(tp, 2), round(sl, 2), size

# =================================================================
# 4. HYBRID DATA FETCH
# =================================================================
def get_binance_klines(symbol, interval, limit=500):
    klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['Open Time','Open','High','Low','Close','Volume',
                                       'Close Time','Quote Asset Volume','Number of Trades',
                                       'Taker Buy Base','Taker Buy Quote','Ignore'])
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
    df.index = pd.to_datetime(df['Close Time'], unit='ms')
    return df[['Open','High','Low','Close','Volume']]

def fetch_data(symbol, tfs):
    crypto_symbols = ["BTC", "ETH", "BNB", "XRP", "ADA"]
    is_crypto = any([symbol.upper().startswith(c) for c in crypto_symbols])
    data = {}
    if is_crypto:
        interval_map = {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
        for tf in tfs:
            try: data[tf] = get_binance_klines(symbol.replace("-USD","USDT"), interval_map[tf])
            except: data[tf] = pd.DataFrame()
    else:
        for tf, period in tfs.items():
            try: data[tf] = yf.download(symbol, period=period, interval=tf, progress=False)
            except: data[tf] = pd.DataFrame()
    return data

# =================================================================
# 5. SAFE BIAS FUNCTION
# =================================================================
def get_bias(df, window=20):
    if df.empty or len(df) < window:
        return 0
    rolling_mean = df['Close'].rolling(window).mean()
    if pd.isna(rolling_mean.iloc[-1]):
        return 0
    return 1 if df['Close'].iloc[-1] > rolling_mean.iloc[-1] else -1

# =================================================================
# 6. MAIN APP
# =================================================================
st.set_page_config(layout="wide")
st.title("🛡️ SOVEREIGN APEX v23.2 (Binance + Yahoo)")

asset = st.sidebar.text_input("Asset Symbol", "BTC-USD")
goal = st.sidebar.number_input("Daily Goal ($)", value=10)

tfs = {"15m": "5d", "1h": "1mo", "4h": "3mo", "1d": "1y"}
data = fetch_data(asset, tfs)

if '1h' in data and not data['1h'].empty:
    scores = {tf: get_bias(data[tf], window=20) for tf in ['1d','4h','1h']}
    total_score = (scores['1d']*3) + (scores['4h']*2) + (scores['1h']*1)

    side, entry, tp, sl, size = calculate_apex_strategy(data['1h'], total_score, goal)
    rank = "TITAN" if abs(total_score) >= 5.5 else "SCOUT"
    pattern = get_pattern(data['1h'])

    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    c4.metric("Pattern", pattern)

    st.code(f"ASSET: {asset}\nSIDE: {side}\nLIMIT: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    if st.button("📲 SEND SIGNAL TO PHONE"):
        send_apex_alert("1h", rank, asset, entry, tp, sl, size, side)
        st.success("Apex Alert Dispatched via Sentinel.")

    fig = go.Figure(data=[go.Candlestick(
        x=data['1h'].index, open=data['1h']['Open'], high=data['1h']['High'],
        low=data['1h']['Low'], close=data['1h']['Close']
    )])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
