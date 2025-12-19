import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import t

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Sovereign Apex v5.0", layout="wide")

# ---------------- STATE ----------------
if 'shove_history' not in st.session_state:
    st.session_state.shove_history = []

if 'notifications' not in st.session_state:
    st.session_state.notifications = []

assets = ["GC=F","CL=F","BTC-USD","ETH-USD","EURUSD=X","GBPUSD=X","AAPL","TSLA","MSFT"]

timeframes = ["15m","30m","1h","4h","1d"]

# ---------------- FUNCTIONS ----------------
def fetch_data(ticker, interval):
    try:
        df = yf.download(ticker, period="10d", interval=interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return pd.DataFrame()

def analyze_displacement(df):
    if df.empty or len(df)<3: return False, 0.0
    c1, c3 = df.iloc[-3], df.iloc[-1]
    is_fvg = float(c3['Low']) > float(c1['High'])
    body = abs(float(c3['Close']) - float(c3['Open']))
    total = float(c3['High']) - float(c3['Low'])
    conviction = (body/total*100) if total>0 else 0
    return is_fvg, conviction

def monte_carlo_prob(df, sims=1000):
    if df.empty or len(df)<20: return 0.5
    returns = df['Close'].pct_change().dropna()
    try:
        params = t.fit(returns)
        paths = [t.rvs(*params, size=10).sum() for _ in range(sims)]
        return np.mean([1 if p>0.01 else 0 for p in paths])
    except:
        return 0.5

def generate_author_notifications(df):
    # Placeholder for combined 40 authors' logic
    # Returns highest ranked setup notification if exists
    if df.empty: return None
    fvg, conv = analyze_displacement(df)
    prob = monte_carlo_prob(df)
    if fvg and conv>60 and prob>0.8:
        return f"🔥 HIGH-RANK TRADE DETECTED | Conviction {conv:.1f}%, Prob {prob*100:.1f}%"
    elif fvg and conv>40:
        return f"⚠️ LOW-RANK TRADE | Conviction {conv:.1f}%"
    else:
        return None

# ---------------- TABS ----------------
tab_chart, tab_watchlist, tab_news = st.tabs(["Charts","Watchlist","News"])

# ---------------- CHARTS ----------------
with tab_chart:
    st.subheader("📊 Asset Chart")
    sel_asset = st.selectbox("Select Asset", assets)
    sel_tf = st.selectbox("Select Timeframe", timeframes)

    df_chart = fetch_data(sel_asset, sel_tf)
    if not df_chart.empty:
        fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                             open=df_chart['Open'],
                                             high=df_chart['High'],
                                             low=df_chart['Low'],
                                             close=df_chart['Close'])])
        fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

        # Author notifications
        note = generate_author_notifications(df_chart)
        if note:
            st.info(note)
            st.session_state.notifications.append(note)
    else:
        st.warning("No data available for selected asset/timeframe.")

# ---------------- WATCHLIST ----------------
with tab_watchlist:
    st.subheader("📋 Watchlist Overview")
    watchlist_data = []
    for a in assets:
        df_w = fetch_data(a, "1h")
        fvg, conv = analyze_displacement(df_w)
        prob = monte_carlo_prob(df_w)
        watchlist_data.append({
            "Asset": a,
            "Conviction": f"{conv:.1f}%",
            "MonteCarloProb": f"{prob*100:.1f}%",
            "FVG": "Yes" if fvg else "No"
        })
    st.table(pd.DataFrame(watchlist_data))

# ---------------- NEWS ----------------
with tab_news:
    st.subheader("📰 Latest News")
    for a in assets[:5]:
        try:
            news_items = yf.Ticker(a).news[:5]
            for n in news_items:
                st.write(f"**{n['title']}** | Source: {n['publisher']}")
        except:
            st.write(f"No news available for {a}")

# ---------------- DASHBOARD / NOTIFICATIONS ----------------
st.sidebar.subheader("🚀 Dashboard / Notifications")
st.sidebar.metric("Total $10 Gains", f"${len(st.session_state.shove_history)*10}")
if st.sidebar.button("✅ Log $10 Success"):
    st.session_state.shove_history.append({"Asset": sel_asset})
st.sidebar.subheader("Notifications")
for n in reversed(st.session_state.notifications[-10:]):  # show last 10
    st.sidebar.info(n)
