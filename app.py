import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- NLTK SENTIMENT ANALYSIS ---
@st.cache_resource
def load_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error(f"NLTK Load Error: {e}")
        return None

sia = load_nltk()

# --- CONFIG ---
st.set_page_config(page_title="Sovereign Apex Hybrid", layout="wide")

# --- STATE MANAGEMENT ---
if 'shove_history' not in st.session_state: st.session_state.shove_history = []
if 'total_scans' not in st.session_state: st.session_state.total_scans = 0

MARKETAUX_API_KEY = "YOUR_MARKETAUX_API_KEY"  # Add your API key here

# --- FETCH NEWS ---
def fetch_marketaux_news(symbol, limit=8):
    try:
        url = (
            f"https://api.marketaux.com/v1/news/all?"
            f"symbols={symbol}&filter_entities=true&language=en&api_token={MARKETAUX_API_KEY}&limit={limit}"
        )
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        return []
    except Exception:
        return []

def get_combined_news(symbol):
    seen = set()
    combined = []
    # YFinance news
    try:
        yf_ticker = yf.Ticker(symbol)
        yf_news = yf_ticker.news
        for n in yf_news[:10]:
            title = n.get("title", "")
            if title and title not in seen:
                seen.add(title)
                combined.append({"title": title, "publisher": n.get("publisher","Yahoo")})
    except:
        pass
    # Marketaux news
    news_api_items = fetch_marketaux_news(symbol, limit=10)
    for item in news_api_items:
        title = item.get("title", "")
        if title and title not in seen:
            seen.add(title)
            combined.append({"title": title, "publisher": item.get("source","Marketaux")})
    return combined

# --- CORE FUNCTIONS ---
def fetch_safe_data(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    assets = ["GC=F","CL=F","BTC-USD","ETH-USD","EURUSD=X","GBPUSD=X","AAPL","TSLA"]
    sel_asset = st.selectbox("Select Asset", assets)

    timeframes = ["15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]
    sel_interval = st.selectbox("Select Time Frame", timeframes)

    if st.button("✅ LOG $10 SUCCESS"):
        st.session_state.shove_history.append({"Time": pd.Timestamp.now(), "Asset": sel_asset})
        st.toast("Victory Recorded.")
        st.rerun()

    st.subheader("📡 Notifications & Authors' Views")
    tfs = ["15m","30m","1h","4h","1d"]
    notif_results = []
    for tf in tfs:
        data = fetch_safe_data(sel_asset, interval=tf)
        fvg, conv = analyze_displacement(data)
        prob = monte_carlo_prob(data)
        authors_rating = conv*0.7 + prob*30  # simplified example
        if fvg and authors_rating>70:
            notif_results.append({"TF": tf,"Conviction":f"{conv:.1f}%","MC Prob":f"{prob*100:.1f}%","AuthorRank":f"{authors_rating:.1f}"})
    if notif_results: st.table(pd.DataFrame(notif_results))
    else: st.info("No high-quality setups detected.")

# --- MAIN CHART ---
st.subheader(f"📈 Chart: {sel_asset} ({sel_interval})")
df_chart = fetch_safe_data(sel_asset, interval=sel_interval, period="1y")
if not df_chart.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df_chart.index,
        open=df_chart['Open'],
        high=df_chart['High'],
        low=df_chart['Low'],
        close=df_chart['Close'],
        increasing_line_color='gold',
        decreasing_line_color='red'
    )])
    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=0,r=0,b=0,t=0),
        dragmode=False  # disable zoom while scrolling
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Chart data unavailable for this asset.")

# --- NEWS SECTION ---
st.subheader("📰 Hybrid News Feed")
news_items = get_combined_news(sel_asset)
if news_items:
    for n in news_items[:10]:
        s = sia.polarity_scores(n['title'])['compound'] if sia else 0
        s_label = "🟢" if s>0 else "🔴" if s<0 else "⚪"
        st.write(f"{s_label} **{n['title']}**")
        st.caption(f"Source: {n['publisher']} | Sentiment: {s:.2f}")
else:
    st.info("No news available currently.")
