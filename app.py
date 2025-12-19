    import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----------------- NLTK -----------------
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_nltk()

# ----------------- APP CONFIG -----------------
st.set_page_config(page_title="Sovereign Apex Fortified", layout="wide")

# ----------------- STATE -----------------
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

# ----------------- SAFE DATA FETCH -----------------
def fetch_safe_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 5: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for col in ['Open','High','Low','Close']:
            if col not in df.columns:
                return pd.DataFrame()
        return df
    except:
        return pd.DataFrame()

# ----------------- MONTE CARLO PROBABILITY -----------------
def monte_carlo_prob(df, sims=1000):
    if df.empty or len(df) < 20: return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p > 0.01 else 0 for p in paths])
    except:
        return 0.5

# ----------------- FVG / DISPLACEMENT -----------------
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

# ----------------- SENTIMENT -----------------
def get_sentiment_score(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = getattr(ticker, "news", [])[:10]
        if not news: return 0, "NEUTRAL"
        scores = []
        for n in news:
            title = n.get('title', '')
            if title and sia:
                scores.append(sia.polarity_scores(title)['compound'])
        if not scores: return 0, "NEUTRAL"
        avg_score = np.mean(scores)
        if avg_score > 0.05: sentiment = "BULLISH"
        elif avg_score < -0.05: sentiment = "BEARISH"
        else: sentiment = "NEUTRAL"
        return avg_score, sentiment
    except:
        return 0, "NEUTRAL"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Control Panel")
    assets = ["GC=F", "CL=F", "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "AAPL", "TSLA"]
    sel_asset = st.selectbox("Select Asset", assets)
    
    periods = {"1 Month":"1mo", "3 Month":"3mo", "6 Month":"6mo", "1 Year":"1y", "Max":"max"}
    selected_period_label = st.selectbox("Select Period", list(periods.keys()))
    selected_period = periods[selected_period_label]
    
    intervals = {"15m":"15m", "1h":"1h", "4h":"4h", "1d":"1d"}
    selected_interval_label = st.selectbox("Select Interval", list(intervals.keys()))
    selected_interval = intervals[selected_interval_label]

    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Recorded.")
        st.rerun()

# ----------------- DASHBOARD -----------------
st.title("🔱 SOVEREIGN APEX DASHBOARD")

# Leaderboard
st.subheader("🏆 Leaderboard")
if st.session_state.shove_history:
    df_hist = pd.DataFrame(st.session_state.shove_history)
    leaderboard = df_hist.groupby("Asset").size().reset_index(name="Wins").sort_values(by="Wins", ascending=False)
    st.table(leaderboard)
else:
    st.info("No trades logged yet.")

# ----------------- CHARTS -----------------
st.subheader(f"📈 Chart: {sel_asset}")
df_chart = fetch_safe_data(sel_asset, period=selected_period, interval=selected_interval)
if not df_chart.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close']
    )])
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No chart data available for this asset/interval.")

# ----------------- NOTIFICATIONS -----------------
st.subheader("🔔 Notifications / High-Quality Trades")
fvg, conv = analyze_displacement(df_chart)
prob = monte_carlo_prob(df_chart)
msg = []
if fvg and conv > 60 and prob > 0.8:
    msg.append(f"🏅 HIGH RANK TRADE! Conviction: {conv:.1f}%, Monte Carlo Prob: {prob*100:.1f}%")
elif fvg:
    msg.append(f"🔹 Moderate Trade. Conviction: {conv:.1f}%")
else:
    msg.append("No high-quality trades detected.")
for m in msg:
    st.info(m)

# ----------------- NEWS -----------------
st.subheader("📰 News / Sentiment Analysis")
score, sentiment_label = get_sentiment_score(sel_asset)
color = "green" if sentiment_label=="BULLISH" else "red" if sentiment_label=="BEARISH" else "gray"
st.markdown(f"**Current Sentiment:** :{color}[{sentiment_label}] ({score:.2f})")

ticker = yf.Ticker(sel_asset)
ticker_news = getattr(ticker, "news", [])[:5]
for n in ticker_news:
    title = n.get('title', '')
    publisher = n.get('publisher','Unknown')
    if title:
        s = sia.polarity_scores(title)['compound'] if sia else 0
        s_label = "🟢" if s>0 else "🔴" if s<0 else "⚪"
        st.write(f"{s_label} **{title}**")
        st.caption(f"Source: {publisher} | Score: {s:.2f}")
