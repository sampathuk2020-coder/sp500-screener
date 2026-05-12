import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# SETTINGS
# -----------------------------
PE_LIMIT = 27
VOLUME_SPIKE_MULTIPLIER = 2
RSI_THRESHOLD = 50
RSI_PERIOD = 14

# -----------------------------
# LOAD S&P 500 TICKERS
# -----------------------------
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(sp500_url)[0]
tickers = sp500_table['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]

# -----------------------------
# RSI FUNCTION
# -----------------------------
def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=close.index).rolling(period).mean()
    loss = pd.Series(loss, index=close.index).rolling(period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -----------------------------
# CANDLE PATTERNS
# -----------------------------

def is_hammer(row):
    body = abs(row['Close'] - row['Open'])
    range_ = row['High'] - row['Low']
    lower_wick = min(row['Open'], row['Close']) - row['Low']
    upper_wick = row['High'] - max(row['Open'], row['Close'])

    if range_ == 0:
        return False

    return (
        lower_wick > 2 * body and
        upper_wick < body and
        body / range_ < 0.4
    )

def is_bullish_engulfing(prev, curr):
    return (
        prev['Close'] < prev['Open'] and  # previous red candle
        curr['Close'] > curr['Open'] and  # current green candle
        curr['Close'] > prev['Open'] and
        curr['Open'] < prev['Close']
    )

# -----------------------------
# SCAN
# -----------------------------
results = []

print(f"Scanning {len(tickers)} S&P 500 stocks...\n")

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")

        if hist.empty or len(hist) < 30:
            continue

        info = stock.info
        pe = info.get("trailingPE", None)

        if pe is None:
            continue

        # Volume spike
        current_vol = hist['Volume'].iloc[-1]
        avg_vol_20 = hist['Volume'].rolling(20).mean().iloc[-1]
        volume_spike = current_vol > (VOLUME_SPIKE_MULTIPLIER * avg_vol_20)

        # RSI
        rsi = calculate_rsi(hist['Close'], RSI_PERIOD).iloc[-1]

        # Candle patterns (last 2 candles)
        last = hist.iloc[-1]
        prev = hist.iloc[-2]

        hammer = is_hammer(last)
        engulfing = is_bullish_engulfing(prev, last)

        # FILTERS
        if pe < PE_LIMIT and volume_spike and rsi > RSI_THRESHOLD:

            results.append({
                "Ticker": ticker,
                "P/E": round(pe, 2),
                "RSI": round(rsi, 2),
                "Volume Spike": volume_spike,
                "Hammer": hammer,
                "Bullish Engulfing": engulfing
            })

            print(f"Match: {ticker} | Hammer={hammer} | Engulfing={engulfing}")

    except Exception:
        continue

# -----------------------------
# OUTPUT
# -----------------------------
df = pd.DataFrame(results)

if not df.empty:
    df = df.sort_values(by="RSI", ascending=False)

    print("\n=== FINAL RESULTS ===")
    print(df)

    filename = f"sp500_screener_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved: {filename}")
else:
    print("No matches found.")
