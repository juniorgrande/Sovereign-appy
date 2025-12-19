import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK SENTIMENT INIT ---
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_nltk()

# --- APP CONFIG ---
st.set_page_config(page_title="Sovereign Apex v5", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

# --- UTILITY FUNCTIONS ---
@st.cache_data
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def monte_carlo_prob(df, sims=1000):
    if df.empty or len(df) < 20: 
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

def get_sentiment_score(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news[:10]
        if not news: return 0, "Neutral"
        scores = [sia.polarity_scores(n['title'])['compound'] for n in news if sia]
        avg_score = np.mean(scores) if scores else 0
        if avg_score > 0.05: sentiment = "BULLISH"
        elif avg_score < -0.05: sentiment = "BEARISH"
        else: sentiment = "NEUTRAL"
        return avg_score, sentiment
    except:
        return 0, "No Data"

# --- DASHBOARD LAYOUT ---
assets = ["GC=F", "CL=F", "BTC-USD", "EURUSD=X", "AAPL", "TSLA", "ETH-USD", "MSFT", "NVDA"]
timeframes = ["15m", "30m", "1h", "4h", "1d", "1mo", "3mo", "6mo", "1y"]

with st.sidebar:
    st.header("Controls & Watchlist")
    sel_asset = st.selectbox("Select Asset", assets)
    sel_tf = st.selectbox("Select Timeframe", timeframes)
    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Recorded.")
        st.rerun()
    st.subheader("Watchlist")
    for a in assets:
        st.write(a)

# --- MAIN TABS ---
tab_dash, tab_chart, tab_news, tab_watchlist = st.tabs(["Dashboard", "Chart View", "News Feed", "Watchlist"])

# --- DASHBOARD TAB ---
with tab_dash:
    st.subheader("📊 Leaderboard & Notifications")
    df_asset = fetch_data(sel_asset, interval=sel_tf)
    fvg, conv = analyze_displacement(df_asset)
    prob = monte_carlo_prob(df_asset)
    rank_color = "gold" if conv > 70 else "red"
    st.markdown(f"**Ranked Trade for {sel_asset}:** <span style='color:{rank_color}'>Conviction {conv:.1f}% | Monte Carlo Prob {prob*100:.1f}%</span>", unsafe_allow_html=True)
    
    # Display all authors' logic combined (simulated)
    st.subheader("📌 Combined Master Views")
    masters_logic = [
        "Nisson: Candlestick structure", "Homa: Market context", "Murph: Trend strength", "Chan: Market profile",
        "... other 36 masters combined logic ..."
    ]
    for l in masters_logic:
        st.write(f"• {l}")

    st.subheader("🏆 Shove History")
    if st.session_state.shove_history:
        st.table(pd.DataFrame(st.session_state.shove_history))
    else:
        st.caption("No successful shoves logged yet.")

# --- CHART TAB ---
with tab_chart:
    st.subheader(f"📈 Full Chart: {sel_asset} ({sel_tf})")
    if not df_asset.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_asset.index, open=df_asset['Open'],
                                             high=df_asset['High'], low=df_asset['Low'], close=df_asset['Close'])])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No chart data available for this asset/timeframe.")

# --- NEWS TAB ---
with tab_news:
    st.subheader(f"📰 Latest News: {sel_asset}")
    score, label = get_sentiment_score(sel_asset)
    st.markdown(f"**Current Sentiment:** {label} ({score:.2f})")
    ticker_news = yf.Ticker(sel_asset).news
    for n in ticker_news[:5]:
        s = sia.polarity_scores(n['title'])['compound'] if sia else 0
        s_label = "🟢" if s > 0 else "🔴" if s < 0 else "⚪"
        st.write(f"{s_label} **{n['title']}**")
        st.caption(f"Source: {n['publisher']} | Score: {s:.2f}")

# --- WATCHLIST TAB ---
with tab_watchlist:
    st.subheader("💼 Watchlist Overview")
    table_data = []
    for a in assets:
        df_a = fetch_data(a, interval="1d")
        price = df_a['Close'].iloc[-1] if not df_a.empty else None
        table_data.append({"Asset": a, "Last Close": price})
    st.table(pd.DataFrame(table_data))
