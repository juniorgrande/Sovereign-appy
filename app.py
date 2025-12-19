import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import time

# Optional Binance import
try:
    from binance.client import Client
except ImportError:
    Client = None

# ===================== CONFIG =====================
# ---------------- BINANCE API KEYS ----------------
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_SECRET_KEY"

# Telegram (optional)
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# Initialize Binance client if available
binance_client = None
if Client and BINANCE_API_KEY and BINANCE_API_SECRET:
    try:
        binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    except Exception as e:
        binance_client = None
        print(f"Binance init failed: {e}")

# ===================== FETCH DATA =====================
def fetch_data(symbol, interval="1h"):
    df = pd.DataFrame()
    try:
        # Binance for crypto
        if binance_client and ("USDT" in symbol or symbol in ["BTCUSDT","ETHUSDT"]):
            klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=500)
            df = pd.DataFrame(klines, columns=[
                "Open time","Open","High","Low","Close","Volume","Close time",
                "Quote asset volume","Trades","TBB","TBQ","Ignore"])
            df = df.astype({"Open":"float","High":"float","Low":"float","Close":"float"})
            df.index = pd.to_datetime(df["Open time"], unit="ms")
        else:
            # Yahoo Finance fallback
            period_map = {"15m":"5d","1h":"1mo","4h":"3mo","1d":"1y"}
            p = period_map.get(interval,"1mo")
            df = yf.download(symbol, period=p, interval=interval, progress=False)
        df = df[['Open','High','Low','Close']].dropna()
    except Exception as e:
        print(f"Data fetch error for {symbol} {interval}: {e}")
        df = pd.DataFrame()
    return df

# ===================== BIAS =====================
def get_bias(df, window=20):
    if df.empty or 'Close' not in df or len(df['Close'].dropna()) < window:
        return 0
    close_series = df['Close'].dropna().astype(float)
    rolling_mean_series = close_series.rolling(window).mean().dropna()
    if rolling_mean_series.empty:
        return 0
    last_close = float(close_series.iloc[-1])
    last_mean = float(rolling_mean_series.iloc[-1])
    if last_close > last_mean:
        return 1
    elif last_close < last_mean:
        return -1
    else:
        return 0

# ===================== PATTERN DETECTION =====================
def detect_patterns(df):
    patterns = []
    if df.empty or len(df) < 3:
        return ["Neutral"]

    try:
        c = df['Close'].dropna().astype(float)
        o = df['Open'].dropna().astype(float)
        h = df['High'].dropna().astype(float)
        l = df['Low'].dropna().astype(float)

        if len(c) < 3:
            return ["Neutral"]

        body = (c - o).abs()
        upper_wick = (h - np.maximum(c, o)).abs()
        lower_wick = (np.minimum(c, o) - l).abs()
        last = len(c) - 1

        last_body = float(body.iloc[last])
        last_upper = float(upper_wick.iloc[last])
        last_lower = float(lower_wick.iloc[last])

        # Hammer
        if last_lower > 2 * last_body and last_upper < 0.5 * last_body:
            patterns.append("Hammer (Bullish)")
        # Shooting Star
        if last_upper > 2 * last_body and last_lower < 0.5 * last_body:
            patterns.append("Shooting Star (Bearish)")
        # Engulfing
        if last >= 1:
            if float(c.iloc[last]) > float(o.iloc[last]) and \
               float(c.iloc[last]) > float(o.iloc[last-1]) and \
               float(o.iloc[last]) < float(c.iloc[last-1]):
                patterns.append("Engulfing (Bullish)")

    except Exception:
        return ["Neutral"]

    return patterns if patterns else ["Neutral"]

# ===================== STRATEGY CALCULATION =====================
def calculate_strategy(df, score, goal=10):
    if df.empty or len(df) < 14:
        return "LONG", 0, 0, 0, 0

    atr_series = (df['High'] - df['Low']).rolling(14).mean().dropna()
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    c = float(df['Close'].dropna().iloc[-1])

    side = "LONG" if score >= 0 else "SHORT"
    sl_mult = 1.0 if abs(score) >= 5.5 else 2.0

    if side == "LONG":
        entry = float(c - 0.2 * atr)
        sl = float(entry - sl_mult * atr)
        tp = float(entry + 1.5 * atr)
    else:
        entry = float(c + 0.2 * atr)
        sl = float(entry + sl_mult * atr)
        tp = float(entry - 1.5 * atr)

    size = round(goal / abs(tp - entry), 4) if abs(tp - entry) > 1e-8 else 0

    return side, entry, tp, sl, size

# ===================== TELEGRAM ALERT =====================
def send_apex_alert(tf, rank, asset, entry, tp, sl, size, side, retries=3):
    if tf == "15m" and rank == "SCOUT":
        return
    emoji = "🔴 SELL" if side == "SHORT" else "🟢 BUY"
    rank_icon = "🔱 TITAN" if rank=="TITAN" else "📡 SCOUT"
    msg = f"{rank_icon} {emoji} AUTHORIZED\n📍 {asset} ({tf})\n📥 Entry: {entry}\n🎯 TP: {tp} | 🛡️ SL: {sl}\n⚖️ Size: {size} Lots"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for _ in range(retries):
        try:
            r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
            if r.status_code == 200:
                break
        except:
            time.sleep(2)

# ===================== STREAMLIT UI =====================
st.set_page_config(layout="wide", page_title="Alpha Apex Trading")
st.title("🛡️ ALPHA APEX TRADER — ERROR-FREE VERSION")

asset = st.sidebar.text_input("Market Target", "BTC-USD")
goal = st.sidebar.number_input("Daily Goal ($)", value=10)

intervals = {"15m":"15m","1h":"1h","4h":"4h","1d":"1d"}
data = {tf: fetch_data(asset, interval=tf) for tf in intervals.values()}

if not data["1h"].empty:
    scores = {}
    for tf in intervals.values():
        try:
            scores[tf] = get_bias(data[tf], window=20)
        except Exception:
            scores[tf] = 0

    total_score = scores.get("1d",0)*3 + scores.get("4h",0)*2 + scores.get("1h",0)*1

    side, entry, tp, sl, size = calculate_strategy(data["1h"], total_score, goal)
    rank = "TITAN" if abs(total_score) >= 5.5 else "SCOUT"
    patterns = detect_patterns(data["1h"])

    st.subheader(f"🎯 Tactical {side} Strike ({rank})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Dynamic SL", sl, delta="-Tight" if rank=="TITAN" else "+Wide")
    c4.metric("Pattern", ", ".join(patterns))

    st.code(f"ASSET: {asset}\nSIDE: {side}\nLIMIT: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    if st.button("📲 SEND SIGNAL TO TELEGRAM"):
        send_apex_alert("1h", rank, asset, entry, tp, sl, size, side)
        st.success("✅ Alert Sent")

    fig = go.Figure(data=[go.Candlestick(
        x=data["1h"].index,
        open=data["1h"]["Open"],
        high=data["1h"]["High"],
        low=data["1h"]["Low"],
        close=data["1h"]["Close"]
    )])
    fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for 1H timeframe. Check your symbol or API connection.")
