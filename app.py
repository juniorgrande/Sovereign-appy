import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -------------------------------
# Initialize NLTK Sentiment
# -------------------------------
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None

sia = load_nltk()

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Sovereign Apex MT-Style Dashboard", layout="wide")

# -------------------------------
# State Management
# -------------------------------
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0
if 'notifications' not in st.session_state: st.session_state.notifications = []

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_data(ticker, interval="1h", period="5d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def monte_carlo_prob(df, sims=1000):
    if df.empty or 'Close' not in df.columns or len(df) < 20: 
        return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

def analyze_displacement(df):
    if df.empty or len(df) < 3: return False, 0.0
    try:
        c1, c3 = df.iloc[-3], df.iloc[-1]
        is_fvg = float(c3['Low']) > float(c1['High'])
        body = abs(float(c3['Close']) - float(c3['Open']))
        total = float(c3['High']) - float(c3['Low'])
        conviction = (body / total * 100) if total > 0 else 0
        return is_fvg, conviction
    except:
        return False, 0.0

def get_sentiment_score(ticker):
    if sia is None: return 0, "Neutral"
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news[:10]
        if not news_items: return 0, "Neutral"
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news_items]
        avg_score = np.mean(scores)
        if avg_score > 0.05: label = "BULLISH"
        elif avg_score < -0.05: label = "BEARISH"
        else: label = "NEUTRAL"
        return avg_score, label
    except:
        return 0, "No Data"

def calculate_accuracy():
    if st.session_state.total_scans == 0: return 0.0
    return (len(st.session_state.shove_history) / st.session_state.total_scans) * 100

# -------------------------------
# Layout: MetaTrader-style
# -------------------------------
col_left, col_right = st.columns([1, 3])

# 1️⃣ LEFT: Watchlist + Notifications + Dashboard
with col_left:
    st.subheader("📋 Watchlist")
    assets = ["GC=F","CL=F","BTC-USD","ETH-USD","EURUSD=X","GBPUSD=X","AAPL","TSLA","MSFT"]
    selected_asset = st.selectbox("Select Asset", assets)

    st.subheader("🔔 Notifications")
    if st.session_state.notifications:
        for note in st.session_state.notifications[-10:]:
            st.info(note)
    else:
        st.caption("No notifications yet.")

    st.subheader("📈 Dashboard Metrics")
    st.metric("Win Rate", f"{calculate_accuracy():.1f}%")
    st.metric("Total Gains", f"${len(st.session_state.shove_history)*10}")
    st.metric("Total Scans", st.session_state.total_scans)

    with st.expander("📓 Shove History"):
        if st.session_state.shove_history:
            st.table(pd.DataFrame(st.session_state.shove_history))
        else:
            st.caption("No trades logged yet.")

    # Log a $10 success
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({
            "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "Asset": selected_asset,
            "Profit": "$10"
        })
        st.toast("Victory Recorded.")
        st.rerun()

# 2️⃣ RIGHT: Charts + Radar + News
with col_right:
    st.subheader(f"📊 {selected_asset} Live Chart")
    df_chart = fetch_data(selected_asset, "15m", "2d")
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close']
        )])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market data not available.")

    st.subheader("🛰 Market Radar (Multi-Timeframe)")
    if st.button("RUN SCAN"):
        timeframes = ["15m","30m","1h","4h"]
        radar_results = []
        for tf in timeframes:
            df_tf = fetch_data(selected_asset, tf, "3d")
            fvg, conv = analyze_displacement(df_tf)
            prob = monte_carlo_prob(df_tf)
            if fvg and conv > 60:
                radar_results.append({
                    "Timeframe": tf,
                    "Conviction": f"{conv:.1f}%",
                    "MC Probability": f"{prob*100:.1f}%"
                })
                st.session_state.total_scans += 1
                st.session_state.notifications.append(
                    f"High-rank setup detected on {selected_asset} ({tf}) with {conv:.1f}% conviction"
                )
        if radar_results:
            st.table(pd.DataFrame(radar_results))
        else:
            st.info("No high-quality setups detected.")

    st.subheader("📰 News Sentiment")
    avg_score, sentiment = get_sentiment_score(selected_asset)
    st.markdown(f"**Current Sentiment**: {sentiment} ({avg_score:.2f})")
    try:
        news_items = yf.Ticker(selected_asset).news[:5]
        for n in news_items:
            score = sia.polarity_scores(n['title'])['compound'] if sia else 0
            emoji = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
            st.write(f"{emoji} **{n['title']}**")
            st.caption(f"Source: {n['publisher']} | Score: {score:.2f}")
    except:
        st.warning("No news available.")
