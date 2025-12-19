import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from binance.client import Client

# ===============================
# 1. ALPHA CONFIG - INSERT YOUR API KEYS
# ===============================
BINANCE_API_KEY = "iyLc8pXmz825n2vnm217VwvkZCu5V7N8TzHi8K4bRP6WNBlvFc1qvqfa6NCHnM9b"
BINANCE_API_SECRET = "ftMl1xcvnL6ip5AMcw7q3v2srB7E0vnqpoOgXrBpFxgqtZSxh0hVMc2zpuXFyDKy"
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# ===============================
# 2. PATTERN DETECTION
# ===============================
def detect_patterns(df):
    patterns = []
    c, o, h, l = df['Close'], df['Open'], df['High'], df['Low']
    body = abs(c-o)
    upper_wick = h - np.maximum(c,o)
    lower_wick = np.minimum(c,o) - l
    i=-1
    if lower_wick.iloc[i] > 2*body.iloc[i] and upper_wick.iloc[i]<0.5*body.iloc[i]:
        patterns.append("Hammer (Bullish)")
    if upper_wick.iloc[i] > 2*body.iloc[i] and lower_wick.iloc[i]<0.5*body.iloc[i]:
        patterns.append("Shooting Star (Bearish)")
    if len(df)>1:
        if c.iloc[i] > o.iloc[i] and c.iloc[i] > o.iloc[i-1] and o.iloc[i]<c.iloc[i-1]:
            patterns.append("Engulfing (Bullish)")
        if c.iloc[i] < o.iloc[i] and c.iloc[i] < o.iloc[i-1] and o.iloc[i]>c.iloc[i-1]:
            patterns.append("Engulfing (Bearish)")
    if not patterns:
        patterns.append("Neutral")
    return patterns

# ===============================
# 3. STRIKE CALCULATION
# ===============================
def calculate_strategy(df, score, goal_usd=10.0):
    c = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    side = "LONG" if score>0 else "SHORT"
    sl_mult = 1.0 if abs(score)>=5.5 else 2.0
    if side=="LONG":
        entry = c - (atr*0.2)
        sl = entry - (atr*sl_mult)
        tp = entry + (atr*1.5)
    else:
        entry = c + (atr*0.2)
        sl = entry + (atr*sl_mult)
        tp = entry - (atr*1.5)
    size = round(goal_usd/abs(tp-entry),4)
    return side, round(entry,2), round(tp,2), round(sl,2), size

# ===============================
# 4. ALIGNMENT SCORE
# ===============================
def get_alignment_score(dataframes):
    weights={"15m":0.5,"1h":1,"4h":2,"1d":3}
    score=0
    for tf, df in dataframes.items():
        if len(df)<20: continue
        ma = df['Close'].rolling(20).mean().iloc[-1]
        bias = 1 if df['Close'].iloc[-1]>ma else -1
        score += bias*weights[tf]
    return score

# ===============================
# 5. STREAMLIT APP
# ===============================
st.set_page_config(layout="wide", page_title="🛡️ SOVEREIGN ALPHA APEX")
st.title("🛡️ SOVEREIGN ALPHA APEX v2.0")
st.markdown("Next-gen Alpha Trading Platform - Full Market Dominance Logic")

# Sidebar
with st.sidebar:
    st.header("⚙️ MISSION CONTROL")
    asset = st.selectbox("Target Asset", ["BTCUSDT","ETHUSDT","GC=F","TSLA"])
    goal = st.number_input("Daily Profit Goal ($)", min_value=1, max_value=1000, value=10)
    source = st.radio("Data Source", ["Auto","Yahoo","Binance"])

# ===============================
# 6. DATA FETCH
# ===============================
tfs = {"15m":"5d","1h":"1mo","4h":"3mo","1d":"1y"}
data = {}

if source=="Binance" or (source=="Auto" and asset.endswith("USDT")):
    for tf, period in tfs.items():
        interval_map = {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
        klines = binance_client.get_klines(symbol=asset, interval=interval_map[tf], limit=500)
        df = pd.DataFrame(klines, columns=[
            "Open time","Open","High","Low","Close","Volume","Close time","Quote asset volume",
            "Number of trades","Taker buy base","Taker buy quote","Ignore"])
        df = df[["Open","High","Low","Close"]].astype(float)
        df.index = pd.to_datetime([int(t[0]) for t in klines], unit='ms')
        data[tf] = df
else:
    for tf, period in tfs.items():
        try:
            df = yf.download(asset, period=period, interval=tf, progress=False)
            data[tf] = df
        except:
            data[tf] = pd.DataFrame()

# ===============================
# 7. MAIN LOGIC
# ===============================
if not data['1h'].empty:
    total_score = get_alignment_score(data)
    side, entry, tp, sl, size = calculate_strategy(data['1h'], total_score, goal)
    rank = "TITAN" if abs(total_score)>=5.5 else "SCOUT"
    patterns = detect_patterns(data['1h'])

    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Action", f"{side} @ {entry}")
    col2.metric("Target Profit", tp)
    col3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    col4.metric("Pattern", ", ".join(patterns))

    st.code(f"ASSET: {asset}\nSIDE: {side}\nENTRY: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    # Chart
    fig = go.Figure(data=[go.Candlestick(
        x=data['1h'].index,
        open=data['1h']['Open'],
        high=data['1h']['High'],
        low=data['1h']['Low'],
        close=data['1h']['Close']
    )])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Data not available. Try another asset or wait a few minutes.")

st.markdown("---")
st.markdown("Alpha-conscious app: Multi-timeframe scanning, dynamic strategy, pattern detection, Yahoo + Binance data, fully next-gen.")
