import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sovereign Apex",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# NLTK SAFE LOAD
# =========================
@st.cache_resource
def load_sentiment():
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None

sia = load_sentiment()

# =========================
# SAFE DATA LOADER
# =========================
@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

# =========================
# TIMEFRAME MAP
# =========================
TF_MAP = {
    "15m": ("60d", "15m"),
    "30m": ("90d", "30m"),
    "1h": ("180d", "1h"),
    "4h": ("1y", "4h"),
    "1d": ("1y", "1d"),
    "1wk": ("5y", "1wk"),
    "1mo": ("10y", "1mo")
}

# =========================
# AUTHOR CONSENSUS ENGINE
# =========================
def author_consensus(df):
    if len(df) < 50:
        return 0.0, []

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    scores = []
    reasons = []

    # Trend (Murphy)
    ma50 = close.rolling(50).mean()
    if close.iloc[-1] > ma50.iloc[-1]:
        scores.append(1)
        reasons.append("Trend bullish (Murphy)")

    # Candlestick psychology (Nison)
    body = abs(close.iloc[-1] - close.iloc[-2])
    wick = high.iloc[-1] - close.iloc[-1]
    if wick > body * 0.5:
        scores.append(-1)
        reasons.append("Overhead wick (Nison)")

    # Structure (Chan)
    if close.iloc[-1] > close.iloc[-10:-1].max():
        scores.append(1)
        reasons.append("Structure breakout (Chan)")

    # Volatility expansion (Wyckoff)
    atr = (high - low).rolling(14).mean()
    if atr.iloc[-1] > atr.iloc[-5]:
        scores.append(1)
        reasons.append("Expansion phase (Wyckoff)")

    # Exhaustion
    if close.iloc[-1] < close.iloc[-3]:
        scores.append(-1)
        reasons.append("Momentum stall")

    score = np.mean(scores) if scores else 0
    return score, reasons

# =========================
# PATTERN REPETITION
# =========================
def pattern_repetition(df, lookback=150):
    if len(df) < lookback:
        return False
    recent = df.iloc[-5:]["Close"].pct_change().round(3).values
    past = df.iloc[:-5]["Close"].pct_change().round(3).values
    for i in range(len(past) - 5):
        if np.allclose(recent, past[i:i+5], atol=0.002):
            return True
    return False

# =========================
# MONTE CARLO
# =========================
def monte_carlo(df, sims=1000):
    if len(df) < 50:
        return 0.5
    r = df["Close"].pct_change().dropna()
    try:
        params = t.fit(r)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

# =========================
# NEWS (SAFE)
# =========================
def get_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        clean = []
        seen = set()
        for n in news:
            title = n.get("title", "")
            if title and title not in seen:
                seen.add(title)
                score = sia.polarity_scores(title)["compound"] if sia else 0
                clean.append((title, score))
        return clean[:5]
    except:
        return []

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("🔱 Control Panel")

asset = st.sidebar.selectbox(
    "Asset",
    ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "EURUSD=X", "GC=F"]
)

tf = st.sidebar.selectbox(
    "Timeframe",
    list(TF_MAP.keys())
)

mode = st.sidebar.radio(
    "View",
    ["Chart", "Watchlist", "Dashboard"]
)

# =========================
# DATA LOAD
# =========================
period, interval = TF_MAP[tf]
df = load_data(asset, period, interval)

# =========================
# MAIN VIEW
# =========================
if mode == "Chart":
    st.subheader(f"{asset} · {tf}")

    if df.empty:
        st.warning("No data available.")
    else:
        fig = go.Figure()
        fig.add_candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            dragmode=False,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        score, reasons = author_consensus(df)
        repeated = pattern_repetition(df)
        mc = monte_carlo(df)

        st.subheader("📢 Notifications")
        if score > 0.4:
            st.success(f"High-rank setup · MC {mc*100:.1f}%")
        elif score < -0.4:
            st.error("High risk · Overhead pressure")
        else:
            st.info("Neutral / wait")

        if repeated:
            st.warning("Pattern repetition detected")

        with st.expander("Author Consensus"):
            for r in reasons:
                st.write("•", r)

elif mode == "Watchlist":
    st.subheader("📋 Watchlist")
    st.write("Multi-asset scanning coming next (engine ready).")

elif mode == "Dashboard":
    st.subheader("📊 Dashboard")
    st.write("Performance, rankings, and statistics will aggregate here.")

# =========================
# NEWS PANEL
# =========================
st.sidebar.subheader("📰 News")
for title, score in get_news(asset):
    icon = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
    st.sidebar.write(f"{icon} {title}")
