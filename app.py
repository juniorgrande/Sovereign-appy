"""
Sovereign Apex — improved data-source handling, CSV uploader, connectivity checks, and clear errors.
Replace your existing app.py with this file.

Features added:
- Sidebar data source selector: Auto / yfinance / coinbase / binance
- Local CSV upload (highest priority)
- Connectivity test button (quick checks for Binance, Coinbase, yfinance)
- Robust lazy import/use of python-binance (won't crash if package or keys missing)
- Fallback logic:
    * Auto: try Binance -> if Binance works use Binance (then yfinance fallback)
            if Binance fails, try Coinbase -> if works use Coinbase (then yfinance fallback)
            finally fallback to yfinance
    * If user explicitly picks 'binance' or 'coinbase' we honor that preference but still
      fall back to yfinance if the chosen provider fails.
- Clear informative messages on the UI about which source was used and any failures.
- Defensive coding to avoid exceptions when fields are missing or values are non-finite.
"""
import io
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from typing import Optional, Tuple

st.set_page_config(layout="wide", page_title="Sovereign Apex Alpha v4.1")
st.title("🛡️ SOVEREIGN APEX: NEXT-GEN TRADING DASHBOARD (improved data handling)")

# -----------------------
# Sidebar / user inputs
# -----------------------
asset = st.sidebar.selectbox("Target Asset", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal_usd = st.sidebar.number_input("Daily Profit Goal ($)", value=10, step=5)
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000, step=100)

# Data source options
data_source_choice = st.sidebar.selectbox(
    "Preferred Data Source",
    ["Auto (recommended)", "yfinance", "coinbase", "binance"]
)

# CSV uploader (highest priority)
st.sidebar.write("Upload local CSV (optional) — must contain time/Open/High/Low/Close/Volume)")
csv_upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Binance keys (optional)
st.sidebar.markdown("Binance API keys (optional; needed only if you want Binance live REST):")
BINANCE_API_KEY = st.sidebar.text_input("BINANCE_API_KEY", type="password", value=st.secrets.get("BINANCE_API_KEY", ""))
BINANCE_API_SECRET = st.sidebar.text_input("BINANCE_API_SECRET", type="password", value=st.secrets.get("BINANCE_API_SECRET", ""))

# Connectivity test
if st.sidebar.button("Run connectivity test"):
    test = st.sidebar.empty()
    with test.container():
        st.write("Running quick connectivity checks (this may take a few seconds)...")
        b_ok, b_msg = False, ""
        try:
            from binance.client import Client as _BinanceClient  # lazy import test
            if BINANCE_API_KEY and BINANCE_API_SECRET:
                try:
                    c = _BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
                    # quick ping and sample ticker
                    ping = c.ping()
                    t = c.get_symbol_ticker(symbol="BTCUSDT")
                    b_ok, b_msg = True, f"Binance reachable (sample ticker ok)."
                except Exception as e:
                    b_ok, b_msg = False, f"Binance client installed but ping/ticker failed: {e}"
            else:
                b_ok, b_msg = False, "Binance client available but API keys not provided."
        except Exception:
            b_ok, b_msg = False, "python-binance not installed."

        # Coinbase test
        try:
            url = "https://api.pro.coinbase.com/products/BTC-USD/candles?granularity=3600"
            r = requests.get(url, timeout=6)
            if r.status_code == 200 and r.json():
                c_ok, c_msg = True, "Coinbase public candles reachable."
            else:
                c_ok, c_msg = False, f"Coinbase returned status {r.status_code} or empty data."
        except Exception as e:
            c_ok, c_msg = False, f"Coinbase request failed: {e}"

        # yfinance test
        try:
            df_test = yf.download("BTC-USD", period="5d", interval="1h", progress=False)
            if df_test is None or df_test.empty:
                y_ok, y_msg = False, "yfinance returned empty data (possible network or yfinance issue)."
            else:
                y_ok, y_msg = True, "yfinance download OK."
        except Exception as e:
            y_ok, y_msg = False, f"yfinance error: {e}"

        st.write("Binance:", "OK" if b_ok else "FAIL", "-", b_msg)
        st.write("Coinbase:", "OK" if c_ok else "FAIL", "-", c_msg)
        st.write("yfinance:", "OK" if y_ok else "FAIL", "-", y_msg)

# -----------------------
# Data fetching helpers
# -----------------------
def parse_uploaded_csv(uploaded_file: io.BytesIO) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, parse_dates=True)
        # try to detect time column
        time_cols = [c for c in df.columns if c.lower() in ("time", "date", "datetime")]
        if time_cols:
            df = df.rename(columns={time_cols[0]: "time"})
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            # attempt to parse index
            try:
                df.index = pd.to_datetime(df.index)
                df = df.reset_index().rename(columns={"index": "time"})
            except Exception:
                pass
        # ensure required columns present
        required = ["time", "Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            # try case-insensitive matching
            cols = {c.lower(): c for c in df.columns}
            mapped = {}
            for req in required:
                if req.lower() in cols:
                    mapped[req] = cols[req.lower()]
            if len(mapped) >= 6:
                df = df.rename(columns=mapped)
            else:
                st.sidebar.error(f"Uploaded CSV missing required columns: {required}")
                return None
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        return df
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded CSV: {e}")
        return None

def fetch_yahoo(symbol: str, period: str = "1mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # robust column detection
        time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
        df = df.rename(columns={time_col: "time"})
        required = ["time", "Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return None

def fetch_coinbase(symbol: str, granularity: int = 3600) -> Optional[pd.DataFrame]:
    # Coinbase expects symbols like BTC-USD
    try:
        url = f"https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        # Coinbase returns [time, low, high, open, close, volume] rows
        df = pd.DataFrame(data, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        # Coinbase candles are returned in reverse chronological order; sort ascending
        df = df.sort_values(by="time").reset_index(drop=True)
        return df
    except Exception:
        return None

def fetch_binance(symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    # Lazy import python-binance to avoid ImportError at top-level
    try:
        from binance.client import Client as BinanceClient
    except Exception:
        return None
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return None
    try:
        client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            return None
        df = pd.DataFrame(klines, columns=[
            "time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return None

def choose_data_source_and_fetch(asset: str, choice: str, uploaded_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Returns (df, used_source) with used_source in {"uploaded","binance","coinbase","yfinance","none"}
    Behavior:
      - If uploaded_df present -> use it
      - If choice == "Auto": try Binance first; if works use binance (and yfinance as fallback if needed),
        if Binance fails try Coinbase then yfinance.
      - If choice == "binance": try Binance then yfinance
      - If choice == "coinbase": try Coinbase then yfinance
      - If choice == "yfinance": only yfinance
    """
    if uploaded_df is not None:
        return uploaded_df, "uploaded (local CSV)"

    # Helper: attempt binance only for USD-denominated crypto forms (or if user explicitly picks)
    is_crypto_like = "USD" in asset or asset.endswith("USDT") or asset.endswith("-USD")

    # Auto flow
    if choice == "Auto (recommended)":
        # Try Binance first (only for crypto-like)
        if is_crypto_like:
            bin_map = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
            symbol_bin = bin_map.get(asset, asset.replace("-", ""))
            df = fetch_binance(symbol_bin)
            if df is not None and not df.empty:
                return df, "binance"
            # if binance fails, try coinbase
            df = fetch_coinbase(asset)
            if df is not None and not df.empty:
                return df, "coinbase"
            # fallback to yfinance
            df = fetch_yahoo(asset, period="1mo", interval="1h")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"
        else:
            # non-crypto: yfinance only
            df = fetch_yahoo(asset, period="6mo", interval="1d")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"

    # Explicit choices
    if choice == "binance":
        if is_crypto_like:
            bin_map = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
            symbol_bin = bin_map.get(asset, asset.replace("-", ""))
            df = fetch_binance(symbol_bin)
            if df is not None and not df.empty:
                return df, "binance"
            # fallback to yfinance
            df = fetch_yahoo(asset, period="1mo", interval="1h")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"
        else:
            # binance not applicable, use yfinance
            df = fetch_yahoo(asset, period="6mo", interval="1d")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"

    if choice == "coinbase":
        df = fetch_coinbase(asset)
        if df is not None and not df.empty:
            return df, "coinbase"
        # fallback to yfinance
        df = fetch_yahoo(asset, period="1mo", interval="1h")
        if df is not None and not df.empty:
            return df, "yfinance"
        return None, "none"

    if choice == "yfinance":
        df = fetch_yahoo(asset, period="6mo", interval="1d") if not ("USD" in asset) else fetch_yahoo(asset, period="1mo", interval="1h")
        if df is not None and not df.empty:
            return df, "yfinance"
        return None, "none"

    return None, "none"

# -----------------------
# Trading logic (unchanged core, defensive)
# -----------------------
def detect_patterns(df):
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
    if n >= 2:
        if (c[last] > o[last]) and (o[last] < c[last-1]) and (c[last] > o[last-1]):
            patterns.append(("Engulfing (Bullish)", 70 if bias == "Bullish" else 50))
    if shadow_bottom[last] > 2 * body[last] and shadow_top[last] < 0.5 * body[last]:
        patterns.append(("Hammer (Bullish)", 75 if bias == "Bullish" else 55))
    if shadow_top[last] > 2 * body[last] and shadow_bottom[last] < 0.5 * body[last]:
        patterns.append(("Shooting Star (Bearish)", 70 if bias == "Bearish" else 50))
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

def alpha_strategy(df, goal_usd=10, account_balance=1000):
    if df is None or len(df) < 3:
        return "NEUTRAL", 0, 0, 0, 0, [("Neutral",50)], "Neutral"
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
    risk_per_trade = 0.01 * max(account_balance, 1)
    denom = max(abs(entry - sl), 1e-6)
    size = round(risk_per_trade / denom, 4)
    tp_adjusted = tp if confidence >= 70 else entry + (tp - entry) * 0.7
    return side, round(entry, 4), round(tp_adjusted, 4), round(sl, 4), size, patterns, pattern_bias

# -----------------------
# Main app execution
# -----------------------
uploaded_df = parse_uploaded_csv(csv_upload) if csv_upload else None
df, used_source = choose_data_source_and_fetch(asset, data_source_choice, uploaded_df)

if used_source == "none" or df is None or df.empty:
    st.error("No data available for the selected asset from the chosen sources. Try a different source or upload a CSV.")
    st.info("Sources attempted. If you want to test without APIs, upload a CSV file using the sidebar.")
else:
    st.success(f"Data source used: {used_source} — rows: {len(df)}")
    # show a small preview in sidebar
    with st.expander("Data preview (last 5 rows)"):
        st.write(df.tail().reset_index(drop=True))

    # Call strategy
    side, entry, tp, sl, size, patterns, bias = alpha_strategy(df, goal_usd, account_balance)

    # Display metrics
    st.subheader(f"🎯 Alpha Strike: {side} ({bias})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action", f"{side} @ {entry}")
    c2.metric("Target Profit", tp)
    c3.metric("Stop Loss", sl)
    c4.metric("Pattern Confidence", ", ".join([f"{p[0]}({p[1]}%)" for p in patterns]))

    st.code(f"ASSET: {asset}\nSOURCE: {used_source}\nSIDE: {side}\nENTRY: {entry}\nSL: {sl}\nTP: {tp}\nLOTS: {size}")

    # Plot candlesticks
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
    except Exception as e:
        st.error(f"Plotting failed: {e}")
        fig = None

    # Add horizontal lines defensively
    if fig is not None:
        try:
            if pd.notna(entry) and np.isfinite(entry):
                fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
        except Exception:
            pass
        try:
            if pd.notna(tp) and np.isfinite(tp):
                fig.add_hline(y=tp, line_color="green", annotation_text="TP")
        except Exception:
            pass
        try:
            if pd.notna(sl) and np.isfinite(sl):
                fig.add_hline(y=sl, line_color="red", annotation_text="SL")
        except Exception:
            pass
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
