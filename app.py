import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
import time

# ===============================
# 1. USER INPUT / API KEYS
# ===============================
st.set_page_config(layout="wide", page_title="Sovereign Apex Alpha v4.0")
st.title("🛡️ SOVEREIGN APEX: NEXT-GEN TRADING DASHBOARD")

# Assets & goal
asset = st.sidebar.selectbox("Target Asset", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])  
goal_usd = st.sidebar.number_input("Daily Profit Goal ($)", value=10, step=5)
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000, step=100)

BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = st.secrets.get("BINANCE_API_SECRET", "")

# ===============================
# 2. DATA FETCHING FUNCTIONS
# ===============================
def fetch_with_retry(func, *args, retries=3, delay=2, **kwargs):
    for _ in range(retries):
        try:
            df = func(*args, **kwargs)
        except Exception:
            df = None
        if df is not None and not df.empty:
            return df
        time.sleep(delay)
    return None

def fetch_binance(symbol, interval="1h", limit=500):
    # Import Binance client lazily to avoid top-level ImportError in environments without the package
    try:
        from binance.client import Client as BinanceClient
    except Exception:
        return None
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return None
    try:
        client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['time','Open','High','Low','Close','Volume','close_time',
                                           'quote_av','trades','tb_base_av','tb_quote_av','ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        df = df[['time','Open','High','Low','Close','Volume']]
        return df.dropna()
    except Exception:
        return None

def fetch_coinbase(symbol, granularity=3600):
    try:
        url = f"https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        data = requests.get(url, timeout=5).json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=['time','Low','High','Open','Close','Volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time','Open','High','Low','Close','Volume']]
        return df.dropna()
    except Exception:
        return None

def fetch_yahoo(symbol, period="1mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # detect the time column robustly
        if 'Datetime' in df.columns:
            time_col = 'Datetime'
        elif 'Date' in df.columns:
            time_col = 'Date'
        else:
            time_col = df.columns[0]
        df = df.rename(columns={time_col: 'time'})
        # Ensure required columns exist
        required = ['time','Open','High','Low','Close','Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None
        return df[['time','Open','High','Low','Close','Volume']].dropna()
    except Exception:
        return None

def get_data(asset):
    df = None
    # Crypto mapping for Binance
    if "USD" in asset:  # treat as crypto or USD-denominated
        binance_map = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
        symbol_binance = binance_map.get(asset, asset.replace("-", ""))
        # try Binance (only if client available and keys set)
        df = fetch_with_retry(fetch_binance, symbol_binance)
        if df is not None:
            return df
        # try Coinbase
        df = fetch_with_retry(fetch_coinbase, asset)
        if df is not None:
            return df
        # fallback to yfinance
        df = fetch_with_retry(fetch_yahoo, asset)
        if df is not None:
            return df
    else:
        df = fetch_with_retry(fetch_yahoo, asset)
    return df

# ===============================
# 3. NEXT-GEN CANDLESTICK PATTERNS
# ===============================
def detect_patterns(df):
    # Return list of (pattern_name, confidence) and net bias string
    if df is None or len(df) < 3:
        return [("Neutral", 50)], "Neutral"

    c = df['Close'].values
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    n = len(df)
    last = n - 1

    body = np.abs(c - o)
    shadow_top = h - np.maximum(c, o)
    shadow_bottom = np.minimum(c, o) - l

    patterns = []

    sma20 = df['Close'].rolling(20, min_periods=1).mean().iloc[-1]
    bias = "Bullish" if c[last] > sma20 else "Bearish"

    # Engulfing (bullish if prior candle opposite and body engulfs)
    if n >= 2:
        if (c[last] > o[last]) and (o[last] < c[last-1]) and (c[last] > o[last-1]):
            patterns.append(("Engulfing (Bullish)", 70 if bias == "Bullish" else 50))

    # Hammer
    if shadow_bottom[last] > 2 * body[last] and shadow_top[last] < 0.5 * body[last]:
        patterns.append(("Hammer (Bullish)", 75 if bias == "Bullish" else 55))

    # Shooting Star
    if shadow_top[last] > 2 * body[last] and shadow_bottom[last] < 0.5 * body[last]:
        patterns.append(("Shooting Star (Bearish)", 70 if bias == "Bearish" else 50))

    # Three White Soldiers / Three Black Crows (use last 3 candles)
    if n >= 3:
        last3_c = c[last-2:last+1]
        last3_o = o[last-2:last+1]
        if np.all(last3_c > last3_o):
            patterns.append(("Three White Soldiers", 80))
        if np.all(last3_c < last3_o):
            patterns.append(("Three Black Crows", 80))

    if not patterns:
        patterns.append(("Neutral", 50))

    return patterns, bias

# ===============================
# 4. ATR + ORDER-BLOCK STRATEGY
# ===============================
def alpha_strategy(df, goal_usd=10, account_balance=1000):
    if df is None or len(df) < 3:
        return "NEUTRAL", 0, 0, 0, 0, [("Neutral",50)], "Neutral"

    # Calculate True Range (vectorized)
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = df['TR'].rolling(14, min_periods=1).mean().iloc[-1]
    atr = max(atr, 1e-6)

    sma_1h = df['Close'].rolling(20, min_periods=1).mean().iloc[-1]
    sma_4h = df['Close'].rolling(80, min_periods=1).mean().iloc[-1]
    bias_mult = 1 if sma_1h > sma_4h else -1

    patterns, pattern_bias = detect_patterns(df)
    confidence = max([p[1] for p in patterns]) if patterns else 50

    last_close = df['Close'].iloc[-1]
    swing_high = df['High'].iloc[-5:-1].max() if len(df) >= 5 else last_close
    swing_low = df['Low'].iloc[-5:-1].min() if len(df) >= 5 else last_close

    if bias_mult > 0:
        side = "LONG"
        entry = last_close - 0.2 * atr
        sl = swing_low - 0.5 * atr
        tp = entry + 1.5 * atr
    else:
        side = "SHORT"
        entry = last_close + 0.2 * atr
        sl = swing_high + 0.5 * atr
        tp = entry - 1.5 * atr

    # Risk-per-trade in dollars
    risk_per_trade = 0.01 * max(account_balance, 1)
    denom = max(abs(entry - sl), 1e-6)
    size = round(risk_per_trade / denom, 4)

    # Adjust TP if low confidence
    if confidence >= 70:
        tp_adjusted = tp
    else:
        tp_adjusted = entry + (tp - entry) * 0.7

    return side, round(entry, 2), round(tp_adjusted, 2), round(sl, 2), size, patterns, pattern_bias

# ===============================
# 5. MAIN APP LOGIC
# ===============================

df = get_data(asset)

if df is None or df.empty:
    st.error("No data available for this asset from Binance, Coinbase, or Yahoo.\n"
             "If you do not have API keys for Binance, the app will fallback to public data (yfinance).")
else:
    side, entry, tp, sl, size, patterns, bias = alpha_strategy(df, goal_usd, account_balance)

    st.subheader(f"🎯 Alpha Strike: {side} ({bias})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Stop Loss", sl)
    c4.metric("Pattern Confidence", ", ".join([f"{p[0]}({p[1]}%)" for p in patterns]))

    st.code(f"ASSET: {asset}\nSIDE: {side}\nENTRY: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    # Plot
    fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'])])
    # Only add lines when numeric and finite
    try:
        if np.isfinite(entry):
            fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
    except Exception:
        pass
    try:
        if np.isfinite(tp):
            fig.add_hline(y=tp, line_color="green", annotation_text="TP")
    except Exception:
        pass
    try:
        if np.isfinite(sl):
            fig.add_hline(y=sl, line_color="red", annotation_text="SL")
    except Exception:
        pass

    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
