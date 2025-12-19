import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ===============================
# NLTK Initialization
# ===============================
@st.cache_resource
def load_sia():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None

sia = load_sia()

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Sovereign Apex vFinal", layout="wide")
st.title("🔱 SOVEREIGN APEX: ULTIMATE DASHBOARD")

# ===============================
# STATE MANAGEMENT
# ===============================
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []

if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# ===============================
# WATCHLIST & SETTINGS
# ===============================
assets = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD",
    "GC=F", "CL=F", "AAPL", "TSLA", "MSFT", "SI=F"
]

timeframes = ["15m", "30m", "1h", "4h"]

sel_asset = st.sidebar.selectbox("Select Asset", assets)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2)

# ===============================
# DATA FETCHING & SAFE UTILS
# ===============================
@st.cache_data
def fetch_data(symbol, interval, period="5d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
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
    c1, c3 = df.iloc[-3], df.iloc[-1]
    is_fvg = float(c3['Low']) > float(c1['High'])
    body = abs(float(c3['Close']) - float(c3['Open']))
    total = float(c3['High']) - float(c3['Low'])
    conviction = (body / total * 100) if total > 0 else 0
    return is_fvg, conviction

def detect_exhaustion(df):
    if df.empty or len(df) < 1: return False
    c = df.iloc[-1]
    body = abs(c['Close'] - c['Open'])
    upper_wick = c['High'] - max(c['Close'], c['Open'])
    return upper_wick > (body * 0.4)

# ===============================
# SENTIMENT ANALYSIS
# ===============================
def sentiment_score(ticker):
    if not sia:
        return 0, "Neutral"
    try:
        ticker_data = yf.Ticker(ticker)
        news = ticker_data.news[:10]
        if not news: return 0, "Neutral"
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news]
        avg = np.mean(scores)
        if avg > 0.05: label = "BULLISH"
        elif avg < -0.05: label = "BEARISH"
        else: label = "NEUTRAL"
        return avg, label
    except:
        return 0, "No Data"

# ===============================
# DASHBOARD METRICS
# ===============================
def calculate_accuracy():
    wins = len(st.session_state.shove_history)
    scans = st.session_state.total_scans
    return (wins / scans * 100) if scans > 0 else 0

acc_col1, acc_col2, acc_col3 = st.columns(3)
acc_col1.metric("Win Rate", f"{calculate_accuracy():.1f}%")
acc_col2.metric("Total Gains", f"${len(st.session_state.shove_history)*10}")
acc_col3.metric("Total Scans", st.session_state.total_scans)

# ===============================
# MARKET RADAR
# ===============================
st.subheader("📡 Market Radar & Notifications")
radar_results = []

if st.button("🛰 Scan All Timeframes"):
    for tf in timeframes:
        df_tf = fetch_data(sel_asset, tf)
        fvg, conv = analyze_displacement(df_tf)
        exhaustion = detect_exhaustion(df_tf)
        prob = monte_carlo_prob(df_tf)
        # High-rank conditions
        if fvg and conv > 60 and not exhaustion and prob > 0.8:
            radar_results.append({
                "TF": tf,
                "Conviction": f"{conv:.1f}%",
                "MC Prob": f"{prob*100:.1f}%",
                "Rank": "Gold"
            })
            st.session_state.notifications.append(f"🔥 High-rank setup {sel_asset} on {tf}")
        elif fvg:
            radar_results.append({
                "TF": tf,
                "Conviction": f"{conv:.1f}%",
                "MC Prob": f"{prob*100:.1f}%",
                "Rank": "Red"
            })

    st.session_state.total_scans += 1

if radar_results:
    st.table(pd.DataFrame(radar_results))
else:
    st.info("No high-quality shoves detected this scan.")

# ===============================
# LIVE CHARTS
# ===============================
st.subheader(f"📊 {sel_asset} Live Chart")
df_chart = fetch_data(sel_asset, "15m", "2d")
if not df_chart.empty:
    fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                         open=df_chart['Open'],
                                         high=df_chart['High'],
                                         low=df_chart['Low'],
                                         close=df_chart['Close'])])
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Chart data unavailable.")

# ===============================
# NEWS PANEL
# ===============================
st.subheader("📰 News & Sentiment")
avg_sent, label = sentiment_score(sel_asset)
st.write(f"Sentiment: **{label}** ({avg_sent:.2f})")
try:
    ticker_news = yf.Ticker(sel_asset).news[:5]
    for n in ticker_news:
        s = sia.polarity_scores(n['title'])['compound'] if sia else 0
        s_label = "🟢" if s > 0 else "🔴" if s < 0 else "⚪"
        st.write(f"{s_label} {n['title']} - Source: {n['publisher']}")
except:
    st.warning("News feed unavailable.")

# ===============================
# LOG SUCCESSFUL $10 SHOVE
# ===============================
with st.sidebar:
    st.subheader("📈 Mission Control")
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory recorded.")

# ===============================
# SHOVE HISTORY & NOTIFICATIONS
# ===============================
st.subheader("📓 Shove History")
if st.session_state.shove_history:
    st.table(pd.DataFrame(st.session_state.shove_history))
else:
    st.caption("No trades logged yet.")

st.subheader("🔔 Notifications")
if st.session_state.notifications:
    for note in st.session_state.notifications[-10:]:
        st.info(note)
else:
    st.caption("No notifications yet.")
