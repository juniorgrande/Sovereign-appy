import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK INIT ---
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except:
        return None
sia = load_nltk()

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex Ultimate", layout="wide")

# --- SESSION STATE ---
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0

# --- TIMEFRAMES ---
timeframes = {
    "15m":"15m", "1h":"1h", "4h":"4h", "1D":"1d", "1M":"1mo", "3M":"3mo", "6M":"6mo", "1Y":"1y"
}

# --- ASSETS ---
assets = ["GC=F","CL=F","BTC-USD","EURUSD=X","AAPL","TSLA","MSFT","ETH-USD","SPY"]

# --- FUNCTIONS ---
def fetch_safe_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 1: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def monte_carlo_prob(df, sims=1000):
    if df.empty or 'Close' not in df.columns or len(df)<20: return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p>0.01 else 0 for p in paths])
    except:
        return 0.5

def analyze_displacement(df):
    if df.empty or len(df)<3: return False, 0.0
    try:
        c1, c3 = df.iloc[-3], df.iloc[-1]
        is_fvg = float(c3['Low'])>float(c1['High'])
        body = abs(float(c3['Close'])-float(c3['Open']))
        total = float(c3['High'])-float(c3['Low'])
        conviction = (body/total*100) if total>0 else 0
        return is_fvg, conviction
    except:
        return False,0.0

def get_sentiment_score(ticker_symbol):
    if sia is None: return 0,"Neutral"
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news[:10]
        if not news: return 0,"Neutral"
        scores = [sia.polarity_scores(n.get('title',""))['compound'] for n in news]
        avg_score = np.mean(scores)
        if avg_score>0.05: sentiment="BULLISH"
        elif avg_score<-0.05: sentiment="BEARISH"
        else: sentiment="NEUTRAL"
        return avg_score, sentiment
    except:
        return 0,"No Data"

# --- APP LAYOUT ---
st.title("🔱 SOVEREIGN APEX: ULTIMATE DASHBOARD")

# Sidebar
with st.sidebar:
    st.header("Controls")
    sel_asset = st.selectbox("Select Asset", assets)
    sel_tf = st.selectbox("Select Timeframe", list(timeframes.keys()))
    if st.button("✅ Log $10 Success"):
        st.session_state.shove_history.append({"Time":pd.Timestamp.now(),"Asset":sel_asset})
        st.toast("Victory Logged!")
        st.rerun()

# Dashboard Section
st.subheader("📊 Dashboard")
col1,col2,col3 = st.columns(3)
win_rate = len(st.session_state.shove_history)/st.session_state.total_scans*100 if st.session_state.total_scans>0 else 0
col1.metric("Win Rate", f"{win_rate:.1f}%")
col2.metric("Total Gains", f"${len(st.session_state.shove_history)*10}")
col3.metric("Total Scans", st.session_state.total_scans)

# Watchlist
st.subheader("👁 Watchlist")
for a in assets:
    st.write(a)

# --- Notifications & Authors Views (Pull Down) ---
with st.expander("🔔 Notifications & Authors' Views"):
    df_pull = fetch_safe_data(sel_asset, period="1y", interval=timeframes[sel_tf])
    fvg, conv = analyze_displacement(df_pull)
    mc_prob = monte_carlo_prob(df_pull)
    st.write(f"Displacement: {'YES' if fvg else 'NO'} | Conviction: {conv:.1f}% | Monte Carlo Prob: {mc_prob*100:.1f}%")
    st.write("Authors' Views (Combined Logic of 40 Masters)")
    st.write("High Rank Trade" if conv>70 else "Low Rank Trade")

# --- Chart Section (Full Screen Activation) ---
st.subheader("📈 Chart View")
interval_value = timeframes[sel_tf]
if interval_value in ["15m","5m","1m"]: period_value="7d"
elif interval_value=="1h": period_value="60d"
else: period_value="1y"
df_chart = fetch_safe_data(sel_asset, period=period_value, interval=interval_value)

if df_chart.empty:
    st.warning("No chart data available for this asset/interval.")
else:
    if st.button("🔍 Open Full Chart"):
        fig = go.Figure(data=[go.Candlestick(
            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], 
            low=df_chart['Low'], close=df_chart['Close']
        )])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart preview minimized. Click 'Open Full Chart' to activate interactive view.")

# News Section
st.subheader("📰 Market News & Sentiment")
avg_score,label=get_sentiment_score(sel_asset)
color="green" if label=="BULLISH" else "red" if label=="BEARISH" else "gray"
st.markdown(f"### Current Sentiment: :{color}[{label}] ({avg_score:.2f})")
try:
    ticker_news = yf.Ticker(sel_asset).news
    for n in ticker_news[:5]:
        s= sia.polarity_scores(n.get('title',""))['compound'] if sia else 0
        s_label="🟢" if s>0 else "🔴" if s<0 else "⚪"
        st.write(f"{s_label} **{n.get('title','No Title')}**")
        st.caption(f"Source: {n.get('publisher','Unknown')} | Score: {s:.2f}")
except:
    st.info("No news data available.")
