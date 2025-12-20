"""
Sovereign Apex — Robust version that avoids ambiguous Series comparisons.

Replace your existing app.py with this file. It:
 - Removes duplicate columns
 - Normalizes and enforces numeric types for price columns
 - Extracts scalar values from rolling/statistics before comparisons
 - Provides CSV upload, data-source selector and connectivity test
 - Allows Binance keys set in-file (DEFAULT_) or via sidebar/streamlit secrets/env
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
from typing import Optional, Tuple, Any

st.set_page_config(layout="wide", page_title="Sovereign Apex Alpha (stable)")
st.title("🛡️ SOVEREIGN APEX — Stable Data Handling")

# -----------------------
# Optional in-file defaults (edit here if you want)
# -----------------------
DEFAULT_BINANCE_KEY = ""
DEFAULT_BINANCE_SECRET = ""

# -----------------------
# Sidebar / user inputs
# -----------------------
asset = st.sidebar.selectbox("Target Asset", ["BTC-USD", "ETH-USD", "GC=F", "TSLA"])
goal_usd = st.sidebar.number_input("Daily Profit Goal ($)", value=10, step=5)
account_balance = st.sidebar.number_input("Account Balance ($)", value=1000, step=100)

data_source_choice = st.sidebar.selectbox(
    "Preferred Data Source",
    ["Auto (recommended)", "yfinance", "coinbase", "binance"]
)

st.sidebar.write("Upload local CSV (optional) — must contain time/Open/High/Low/Close/Volume)")
csv_upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("Binance API keys (optional):")
bin_key_input = st.sidebar.text_input("BINANCE_API_KEY (sidebar)", type="password", value="")
bin_secret_input = st.sidebar.text_input("BINANCE_API_SECRET (sidebar)", type="password", value="")

BINANCE_API_KEY = bin_key_input.strip() or st.secrets.get("BINANCE_API_KEY", "") or os.environ.get("BINANCE_API_KEY", "") or DEFAULT_BINANCE_KEY
BINANCE_API_SECRET = bin_secret_input.strip() or st.secrets.get("BINANCE_API_SECRET", "") or os.environ.get("BINANCE_API_SECRET", "") or DEFAULT_BINANCE_SECRET

# -----------------------
# Utility helpers
# -----------------------
def safe_scalar(value: Any, fallback: float = 0.0) -> float:
    """
    Convert a pandas Series/ndarray/scalar to a float scalar by taking the last element when appropriate.
    """
    try:
        if isinstance(value, (pd.Series, np.ndarray, list)):
            if len(value) == 0:
                return float(fallback)
            # take the last non-nan element if possible
            arr = pd.Series(value).dropna()
            if arr.empty:
                return float(fallback)
            return float(arr.iloc[-1])
        return float(value)
    except Exception:
        return float(fallback)

def ensure_single_close_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has canonical columns and 'Close' is a single series.
    Remove duplicate columns, do case-insensitive mapping if needed.
    """
    # drop exact-duplicate column names (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # case-insensitive mapping to canonical names
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for std in ["time", "open", "high", "low", "close", "volume"]:
        if std in cols_lower:
            orig = cols_lower[std]
            mapping[orig] = std.capitalize() if std != "time" else "time"
    if mapping:
        df = df.rename(columns=mapping)

    # If 'Close' still missing, try common capitalizations
    if "Close" not in df.columns:
        for cand in ["close", "CLOSE", "Close"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Close"})
                break

    # Ensure numeric for price columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure time column exists and is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        # try index
        try:
            df = df.reset_index().rename(columns={"index": "time"})
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        except Exception:
            pass

    # Keep only required columns if available
    available = [c for c in ["time","Open","High","Low","Close","Volume"] if c in df.columns]
    if len(available) >= 4:  # require at least OHLC
        df = df[available].copy()
    # sort ascending by time if present
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df

# -----------------------
# Data fetchers
# -----------------------
def parse_uploaded_csv(uploaded_file: io.BytesIO) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, engine="python")
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
            return None
    df = ensure_single_close_series(df)
    # minimal validation
    if "Close" not in df.columns or ("Open" not in df.columns and "High" not in df.columns):
        st.sidebar.error("Uploaded CSV doesn't contain required OHLC columns.")
        return None
    return df

def fetch_yahoo(symbol: str, period: str = "1mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df = ensure_single_close_series(df)
        if "Close" not in df.columns:
            return None
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
        df = ensure_single_close_series(df)
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
        df = ensure_single_close_series(df)
        return df
    except Exception:
        return None

def choose_data_source_and_fetch(asset: str, choice: str, uploaded_df: Optional[pd.DataFrame]) -> Tuple[Optional[pd.DataFrame], str]:
    if uploaded_df is not None:
        return uploaded_df, "uploaded (local CSV)"
    is_crypto_like = "USD" in asset or asset.endswith("USDT") or asset.endswith("-USD")
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
    if choice == "binance":
        if is_crypto_like:
            symbol_bin = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}.get(asset, asset.replace("-", ""))
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
# Trading logic with safe scalar extraction
# -----------------------
def detect_patterns(df: pd.DataFrame):
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
    sma20_val = df['Close'].rolling(20, min_periods=1).mean()
    sma20 = safe_scalar(sma20_val)
    bias = "Bullish" if c[last] > sma20 else "Bearish"
    if n >= 2 and (c[last] > o[last]) and (o[last] < c[last-1]) and (c[last] > o[last-1]):
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

def alpha_strategy(df: pd.DataFrame, goal_usd=10, account_balance=1000):
    if df is None or len(df) < 3:
        return "NEUTRAL", 0.0, 0.0, 0.0, 0.0, [("Neutral",50)], "Neutral"
    # defensive: ensure numeric and single-close
    df = ensure_single_close_series(df)
    for col in ["Close", "High", "Low", "Open"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # compute TR and ATR
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    df['TR'] = tr_df.max(axis=1)
    atr_rolling = df['TR'].rolling(14, min_periods=1).mean()
    atr = safe_scalar(atr_rolling, fallback=1e-6)
    atr = max(atr, 1e-6)
    # compute SMAs and convert to scalar floats
    sma_1h_roll = df['Close'].rolling(20, min_periods=1).mean()
    sma_4h_roll = df['Close'].rolling(80, min_periods=1).mean()
    sma_1h = safe_scalar(sma_1h_roll, fallback=df['Close'].iloc[-1])
    sma_4h = safe_scalar(sma_4h_roll, fallback=df['Close'].iloc[-1])
    bias_mult = 1 if sma_1h > sma_4h else -1
    patterns, pattern_bias = detect_patterns(df)
    confidence = max([p[1] for p in patterns]) if patterns else 50
    last_close = safe_scalar(df['Close'].iloc[-1], fallback=0.0)
    # swings
    if len(df) >= 5:
        swing_high = safe_scalar(df['High'].iloc[-5:-1].max(), fallback=last_close)
        swing_low = safe_scalar(df['Low'].iloc[-5:-1].min(), fallback=last_close)
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
    risk_per_trade = 0.01 * max(account_balance, 1)
    denom = max(abs(entry - sl), 1e-6)
    size = round(risk_per_trade / denom, 6)
    tp_adjusted = tp if confidence >= 70 else entry + (tp - entry) * 0.7
    # final formatting
    return side, float(round(entry, 8)), float(round(tp_adjusted, 8)), float(round(sl, 8)), size, patterns, pattern_bias

# -----------------------
# Main execution with debug-on-error
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
    # try running strategy and show debug info if error occurs
    try:
        side, entry, tp, sl, size, patterns, bias = alpha_strategy(df, goal_usd, account_balance)
    except Exception as e:
        # show extra debug info to help identify underlying cause
        st.error("Strategy calculation failed — showing debug info to help diagnose.")
        st.exception(e)
        try:
            # useful debug values
            df_info = {
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(3).to_dict(orient="list")
            }
            st.json(df_info)
            # compute problematic values defensively
            try:
                sma1 = df['Close'].rolling(20, min_periods=1).mean()
                sma4 = df['Close'].rolling(80, min_periods=1).mean()
                st.write("sma_1h tail:", sma1.tail(3).tolist())
                st.write("sma_4h tail:", sma4.tail(3).tolist())
            except Exception as e2:
                st.write("Error while computing SMAs for debug:", e2)
        except Exception:
            pass
        st.stop()
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
            open=df.get('Open', df['Close']),
            high=df.get('High', df['Close']),
            low=df.get('Low', df['Close']),
            close=df['Close']
        )])
    except Exception as e:
        st.error(f"Plotting failed: {e}")
        fig = None
    if fig is not None:
        try:
            if np.isfinite(entry):
                fig.add_hline(y=entry, line_color="yellow", annotation_text="ENTRY")
        except Exception:
            pass
        try:
            if np.isfinite(tp):
                fig.add_hline(y=tp, line_color="green", annotation_text="TP")
        except Exception:
            pass
        try:
            if np.isfinite(sl):
                fig.add_hline(y=sl, line_color="red", annotation_text="SL")
        except Exception:
            pass
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
