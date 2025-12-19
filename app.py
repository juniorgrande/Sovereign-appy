# ================================
# 🔱 SOVEREIGN APEX — ALPHA ENGINE
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta

# --------------------------------
# CONFIG
# --------------------------------
st.set_page_config(
    page_title="SOVEREIGN APEX • ALPHA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# BINANCE (PUBLIC, NO API KEY)
# --------------------------------
def fetch_binance(symbol, interval, limit=500):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        df = pd.DataFrame(data, columns=[
            "time","Open","High","Low","Close","Volume",
            "_","_","_","_","_","_"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        df = df[["Open","High","Low","Close","Volume"]].astype(float)
        return df
    except:
        return pd.DataFrame()

# --------------------------------
# YAHOO
# --------------------------------
def fetch_yahoo(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        return df.dropna()
    except:
        return pd.DataFrame()

# --------------------------------
# SAFE ATR
# --------------------------------
def atr(df, period=14):
    if len(df) < period:
        return np.nan
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )
    return tr.rolling(period).mean().iloc[-1]

# --------------------------------
# FULL PATTERN SCAN (HISTORY‑BASED)
# --------------------------------
def scan_patterns(df):
    if len(df) < 30:
        return {"Bullish":0, "Bearish":0}

    o,c,h,l = df["Open"], df["Close"], df["High"], df["Low"]
    body = abs(c-o)
    rng = h-l

    hammer = ((l < (np.minimum(o,c) - body*2)))
    shooting = ((h > (np.maximum(o,c) + body*2)))
    engulf_bull = (c > o) & (c.shift() < o.shift())
    engulf_bear = (c < o) & (c.shift() > o.shift())

    bullish = hammer.sum() + engulf_bull.sum()
    bearish = shooting.sum() + engulf_bear.sum()

    return {"Bullish": int(bullish), "Bearish": int(bearish)}

# --------------------------------
# TREND SLOPE (ALPHA CORE)
# --------------------------------
def trend_slope(df):
    if len(df) < 20:
        return 0
    y = df["Close"].values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope

# --------------------------------
# ALPHA PROJECTION ENGINE
# --------------------------------
def alpha_projection(df):
    a = atr(df)
    slope = trend_slope(df)
    patterns = scan_patterns(df)

    bias = 1 if slope > 0 else -1 if slope < 0 else 0
    confidence = min(100, abs(slope) * 10000)

    direction = "BUY" if bias > 0 else "SELL" if bias < 0 else "NEUTRAL"
    projected_range = a * (1.2 + (confidence/100)) if not np.isnan(a) else 0

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "range": round(projected_range, 2),
        "patterns": patterns,
        "atr": a
    }

# --------------------------------
# UI — SIDEBAR
# --------------------------------
st.sidebar.title("🔱 ALPHA CONTROL")
market = st.sidebar.selectbox(
    "Market",
    ["BTCUSDT","ETHUSDT","AAPL","TSLA","GC=F","EURUSD=X"]
)

tf = st.sidebar.selectbox(
    "Timeframe",
    ["15m","1h","4h","1d"]
)

# --------------------------------
# DATA ROUTER
# --------------------------------
if market.endswith("USDT"):
    interval_map = {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
    df = fetch_binance(market, interval_map[tf])
else:
    interval_map = {"15m":"15m","1h":"1h","4h":"1h","1d":"1d"}
    period_map = {"15m":"5d","1h":"1mo","4h":"3mo","1d":"1y"}
    df = fetch_yahoo(market, interval_map[tf], period_map[tf])

# --------------------------------
# HARD SAFETY
# --------------------------------
if df.empty or len(df) < 30:
    st.error("Insufficient data — Alpha stands aside.")
    st.stop()

# --------------------------------
# ALPHA ANALYSIS
# --------------------------------
alpha = alpha_projection(df)

# --------------------------------
# COMMAND CENTER UI
# --------------------------------
st.title("🧠 SOVEREIGN APEX — ALPHA COMMAND")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dominant Direction", alpha["direction"])
c2.metric("Confidence", f"{alpha['confidence']}%")
c3.metric("Projected Range", alpha["range"])
c4.metric("ATR", round(alpha["atr"],2) if alpha["atr"] else "N/A")

# --------------------------------
# PATTERN DOMINANCE
# --------------------------------
st.subheader("📊 Pattern Dominance (Full Chart Scan)")
st.json(alpha["patterns"])

# --------------------------------
# CHART
# --------------------------------
fig = go.Figure(data=[
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )
])

fig.update_layout(
    template="plotly_dark",
    height=520,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# ALPHA STATEMENT
# --------------------------------
st.markdown("""
### 🔱 Alpha Assessment
This system evaluates **structure, volatility, trend force, and historical pattern dominance**  
across the **entire chart**, not isolated candles.

No signal is produced unless **alignment exists**.
Silence is a position.
""")
