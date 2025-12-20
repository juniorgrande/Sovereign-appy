"""
Sovereign Apex — improved data-source handling, CSV uploader, connectivity checks, and clear errors.

This version fixes a runtime error caused by ambiguous pandas Series comparisons
(ensures SMA/ATR variables are scalars) and makes numeric conversions robust
for uploaded CSVs. It also allows Binance API keys to be set in-code (DEFAULT_*)
and still supports entering them in the sidebar (sidebar overrides in-code defaults).

Replace your existing app.py with this file.
"""
import io
import os
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from typing import Optional, Tuple

st.set_page_config(layout="wide", page_title="Sovereign Apex Alpha v4.2")
st.title("🛡️ SOVEREIGN APEX: NEXT-GEN TRADING DASHBOARD (data handling + fixes)")

# -----------------------
# Optional in-file defaults (you can edit these directly if you prefer hardcoding)
# -----------------------
# If you want to hardcode Binance keys in the file, put them here.
# Otherwise leave blank and enter keys via the sidebar or streamlit secrets.
DEFAULT_BINANCE_KEY = ""
DEFAULT_BINANCE_SECRET = ""

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

# Binance keys (optional) — sidebar overrides DEFAULT_*
st.sidebar.markdown("Binance API keys (optional; needed only for Binance REST):")
bin_key_input = st.sidebar.text_input("BINANCE_API_KEY (sidebar)", type="password", value="")
bin_secret_input = st.sidebar.text_input("BINANCE_API_SECRET (sidebar)", type="password", value="")

# Determine BINANCE keys: priority -> sidebar inputs -> streamlit secrets -> DEFAULT constants -> env
BINANCE_API_KEY = bin_key_input.strip() or st.secrets.get("BINANCE_API_KEY", "") or os.environ.get("BINANCE_API_KEY", "") or DEFAULT_BINANCE_KEY
BINANCE_API_SECRET = bin_secret_input.strip() or st.secrets.get("BINANCE_API_SECRET", "") or os.environ.get("BINANCE_API_SECRET", "") or DEFAULT_BINANCE_SECRET

# Connectivity test
if st.sidebar.button("Run connectivity test"):
    test = st.sidebar.empty()
    with test.container():
        st.write("Running quick connectivity checks (this may take a few seconds)...")
        # Binance test
        try:
            from binance.client import Client as _BinanceClient  # lazy import test
            if BINANCE_API_KEY and BINANCE_API_SECRET:
                try:
                    c = _BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
                    _ = c.ping()
                    _ = c.get_symbol_ticker(symbol="BTCUSDT")
                    b_ok, b_msg = True, "Binance reachable (ping & ticker OK)."
                except Exception as e:
                    b_ok, b_msg = False, f"Binance client installed but ping/ticker failed: {e}"
            else:
                b_ok, b_msg = False, "python-binance installed but keys not provided."
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
        df = pd.read_csv(uploaded_file)
        # try to detect time column (several common names)
        time_cols = [c for c in df.columns if c.lower() in ("time", "date", "datetime", "timestamp")]
        if time_cols:
            df = df.rename(columns={time_cols[0]: "time"})
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            # attempt to parse index as time
            try:
                df.index = pd.to_datetime(df.index)
                df = df.reset_index().rename(columns={"index": "time"})
            except Exception:
                pass
        # Normalize column names and ensure required columns exist
        col_map = {c.lower(): c for c in df.columns}
        required_lower = {"time", "open", "high", "low", "close", "volume"}
        if not required_lower.issubset(set(col_map.keys())):
            st.sidebar.error(f"Uploaded CSV missing required columns: time/Open/High/Low/Close/Volume (case-insensitive).")
            return None
        # rename to canonical names
        df = df.rename(columns={col_map["time"]: "time",
                                col_map["open"]: "Open",
                                col_map["high"]: "High",
                                col_map["low"]: "Low",
                                col_map["close"]: "Close",
                                col_map["volume"]: "Volume"})
        # convert numeric columns robustly
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        # Drop rows where price columns are NaN
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        # Ensure ascending time order
        df = df.sort_values("time").reset_index(drop=True)
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
        # convert numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        required = ["time", "Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return None
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None

def fetch_coinbase(symbol: str, granularity: int = 3600) -> Optional[pd.DataFrame]:
    try:
        url = f"https://api.pro.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        # rename to canonical and convert numeric
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None

def fetch_binance(symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
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
        # convert numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None

def choose_data_source_and_fetch(asset: str, choice: str, uploaded_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], str]:
    if uploaded_df is not None:
        return uploaded_df, "uploaded (local CSV)"

    is_crypto_like = "USD" in asset or asset.endswith("USDT") or asset.endswith("-USD")

    # Auto: try Binance -> Coinbase -> yfinance (for crypto-like), else yfinance
    if choice == "Auto (recommended)":
        if is_crypto_like:
            bin_map = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
            symbol_bin = bin_map.get(asset, asset.replace("-", ""))
            df = fetch_binance(symbol_bin)
            if df is not None and not df.empty:
                return df, "binance"
            df = fetch_coinbase(asset)
            if df is not None and not df.empty:
                return df, "coinbase"
            df = fetch_yahoo(asset, period="1mo", interval="1h")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"
        else:
            df = fetch_yahoo(asset, period="6mo", interval="1d")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"

    # Explicit preferences
    if choice == "binance":
        if is_crypto_like:
            bin_map = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
            symbol_bin = bin_map.get(asset, asset.replace("-", ""))
            df = fetch_binance(symbol_bin)
            if df is not None and not df.empty:
                return df, "binance"
            df = fetch_yahoo(asset, period="1mo", interval="1h")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"
        else:
            df = fetch_yahoo(asset, period="6mo", interval="1d")
            if df is not None and not df.empty:
                return df, "yfinance"
            return None, "none"

    if choice == "coinbase":
        df = fetch_coinbase(asset)
        if df is not None and not df.empty:
            return df, "coinbase"
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
# Trading logic (defensive & fixes applied)
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
    sma20_val = df['Close'].rolling(20, min_periods=1).mean().iloc[-1]
    try:
        sma20 = float(sma20_val)
    except Exception:
        sma20 = float(df['Close'].iloc[-1])
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
    # Ensure numeric Close column
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    # Compute TR robustly
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    df['TR'] = tr_df.max(axis=1)
    atr_val = df['TR'].rolling(14, min_periods=1).mean().iloc[-1]
    try:
        atr = float(atr_val)
    except Exception:
        atr = float((df['High'] - df['Low']).iloc[-1]) if len(df) >= 1 else 1e-6
    atr = max(atr, 1e-6)
    # compute SMAs as floats (defensive)
    try:
        sma_1h_val = df['Close'].rolling(20, min_periods=1).mean().iloc[-1]
        sma_1h = float(sma_1h_val)
    except Exception:
        sma_1h = float(df['Close'].iloc[-1])
    try:
        sma_4h_val = df['Close'].rolling(80, min_periods=1).mean().iloc[-1]
        sma_4h = float(sma_4h_val)
    except Exception:
        sma_4h = float(df['Close'].iloc[-1])
    bias_mult = 1 if sma_1h > sma_4h else -1
    patterns, pattern_bias = detect_patterns(df)
    confidence = max([p[1] for p in patterns]) if patterns else 50
    last_close = float(df['Close'].iloc[-1])
    if len(df) >= 5:
        try:
            swing_high = float(df['High'].iloc[-5:-1].max())
        except Exception:
            swing_high = last_close
        try:
            swing_low = float(df['Low'].iloc[-5:-1].min())
        except Exception:
            swing_low = last_close
    else:
        swing_high = last_close
        swing_low = last_close
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
    # Risk-per-trade (dollars)
    risk_per_trade = 0.01 * max(account_balance, 1)
    denom = max(abs(entry - sl), 1e-6)
    size = round(risk_per_trade / denom, 4)
    tp_adjusted = tp if confidence >= 70 else entry + (tp - entry) * 0.7
    # Return rounded values
    try:
        return side, round(float(entry), 6), round(float(tp_adjusted), 6), round(float(sl), 6), size, patterns, pattern_bias
    except Exception:
        return side, entry, tp_adjusted, sl, size, patterns, pattern_bias

# -----------------------
# Main app execution
# -----------------------
uploaded_df = parse_uploaded_csv(csv_upload) if csv_upload else None
df, used_source = choose_data_source_and_fetch(asset, data_source_choice, uploaded_df)

if used_source == "none" or df is None or df.empty:
    st.error("No data available for the selected asset from the chosen sources. Try a different source or upload a CSV.")
    st.info("If you want to test without APIs: upload a CSV file using the sidebar.")
else:
    st.success(f"Data source used: {used_source} — rows: {len(df)}")
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
