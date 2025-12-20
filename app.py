import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from binance.client import Client as BinanceClient
import requests

# ===============================
# 1. USER INPUT / API KEYS
# ===============================
st.set_page_config(layout="wide", page_title="Sovereign Apex Alpha v3.0")
st.title("🛡️ SOVEREIGN APEX: ALPHA TRADING COMMANDER (Error-Free)")

# Assets & goal
asset = st.sidebar.selectbox("Target Asset", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal_usd = st.sidebar.number_input("Daily Profit Goal ($)", value=10, step=5)
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000, step=100)

# Binance API
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = st.secrets.get("BINANCE_API_SECRET", "")

# ===============================
# 2. DATA FETCH FUNCTIONS
# ===============================
def fetch_yahoo(symbol, period="1mo", interval="1h"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df = df[['Datetime' if 'Datetime' in df.columns else 'Date','Open','High','Low','Close','Volume']]
        df.rename(columns={'Datetime':'time','Date':'time'}, inplace=True)
        df = df.dropna()
        return df
    except Exception as e:
        return None

def fetch_binance(symbol, interval="1h", limit=500):
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return None
    try:
        client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['time','Open','High','Low','Close','Volume','close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        df = df[['time','Open','High','Low','Close','Volume']]
        df = df.dropna()
        return df
    except:
        return None

def fetch_coinbase(symbol, granularity=3600):
    try:
        url = f"https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        data = requests.get(url, timeout=5).json()
        df = pd.DataFrame(data, columns=['time','Low','High','Open','Close','Volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time','Open','High','Low','Close','Volume']]
        df = df.dropna()
        return df
    except:
        return None

def get_data(asset):
    df = None
    if "USD" in asset:  # crypto
        symbol_binance = asset.replace("-", "")
        df = fetch_binance(symbol_binance)
        if df is None:
            df = fetch_coinbase(asset)
    else:
        df = fetch_yahoo(asset)
    return df

# ===============================
# 3. NEXT-GEN PATTERN DETECTION
# ===============================
def detect_patterns_nextgen(df):
    patterns = []
    if len(df) < 5:
        return [("Neutral", 50)], "Neutral"
    
    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']
    
    body = abs(c - o)
    shadow_top = h - np.maximum(c, o)
    shadow_bottom = np.minimum(c, o) - l
    
    last = -1
    prev = -2
    prev2 = -3
    prev3 = -4
    prev4 = -5
    
    sma20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else c.iloc[-1]
    bias = "Bullish" if c[last] > sma20 else "Bearish"
    
    # Bullish Engulfing
    if len(df) >= 2 and c[last] > o[last] and o[last] < c[prev] and c[last] > o[prev]:
        confidence = 70 if bias == "Bullish" else 50
        patterns.append(("Engulfing (Bullish)", confidence))
        
    # Hammer
    if shadow_bottom[last] > 2*body[last] and shadow_top[last] < 0.5*body[last]:
        confidence = 75 if bias == "Bullish" else 55
        patterns.append(("Hammer (Bullish)", confidence))
        
    # Shooting Star
    if shadow_top[last] > 2*body[last] and shadow_bottom[last] < 0.5*body[last]:
        confidence = 70 if bias == "Bearish" else 50
        patterns.append(("Shooting Star (Bearish)", confidence))
    
    # Three White Soldiers / Black Crows
    if len(df) >= 5:
        # bullish
        if all(c[last-2:last+1] > o[last-2:last+1]):
            patterns.append(("Three White Soldiers", 80))
        # bearish
        if all(c[last-2:last+1] < o[last-2:last+1]):
            patterns.append(("Three Black Crows", 80))
    
    if not patterns:
        patterns.append(("Neutral", 50))
    
    return patterns, bias

# ===============================
# 4. NEXT-GEN STRATEGY
# ===============================
def alpha_strategy_nextgen(df, goal_usd=10, account_balance=1000):
    if len(df) < 5:
        return "NEUTRAL", 0, 0, 0, 0, [("Neutral",50)], "Neutral"
    
    df['TR'] = df[['High','Low','Close']].apply(lambda x: max(x['High']-x['Low'], abs(x['High']-x['Close']), abs(x['Low']-x['Close'])), axis=1)
    atr = df['TR'].rolling(14).mean().iloc[-1] if len(df) >= 14 else max(df['High'] - df['Low'])
    atr = max(atr, 1e-6)  # avoid division by zero
    
    # Multi-timeframe SMA bias
    sma_1h = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['Close'].iloc[-1]
    sma_4h = df['Close'].rolling(80).mean().iloc[-1] if len(df) >= 80 else df['Close'].iloc[-1]
    bias = 1 if sma_1h > sma_4h else -1
    
    patterns, pattern_bias = detect_patterns_nextgen(df)
    confidence = max([p[1] for p in patterns])
    
    last_close = df['Close'].iloc[-1]
    swing_high = df['High'].iloc[-5:-1].max() if len(df) >= 5 else last_close
    swing_low = df['Low'].iloc[-5:-1].min() if len(df) >= 5 else last_close
    
    if bias > 0:
        side = "LONG"
        entry = last_close - 0.2*atr
        sl = swing_low - 0.5*atr
        tp = entry + 1.5*atr
    else:
        side = "SHORT"
        entry = last_close + 0.2*atr
        sl = swing_high + 0.5*atr
        tp = entry - 1.5*atr
    
    risk_per_trade = 0.01 * max(account_balance, 1)  # avoid zero
    size = round(risk_per_trade / max(abs(entry - sl), 1e-6), 4)
    
    tp_adjusted = tp if confidence >= 70 else entry + (tp - entry) * 0.7
    
    return side, round(entry,2), round(tp_adjusted,2), round(sl,2), size, patterns, pattern_bias

# ===============================
# 5. MAIN APP
# ===============================
df = get_data(asset)

if df is None or df.empty:
    st.error("No data available for this asset from Binance, Coinbase, or Yahoo.")
else:
    side, entry, tp, sl, size, patterns, bias = alpha_strategy_nextgen(df, goal_usd, account_balance)
    
    st.subheader(f"🎯 Alpha Strike: {side} ({bias})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Stop Loss", sl)
    c4.metric("Pattern Confidence", ", ".join([f"{p[0]}({p[1]}%)" for p in patterns]))
    
    st.code(f"ASSET: {asset}\nSIDE: {side}\nENTRY: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")
    
    # Candlestick chart
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
        fig.add_hline(y=tp, line_color="green", annotation_text="TP")
        fig.add_hline(y=sl, line_color="red", annotation_text="SL")
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
