#Max Profit Calculation Validation 

import numpy as np
import pandas as pd
from yfinance import Ticker  # pip install yfinance

# Prompt user for a stock ticker symbol
def user_input_ticker():
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
    return ticker_symbol

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker_symbol, period='1y', interval='1d'):
    ticker = Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        print("No data found for the given ticker or period.")
        exit()
    # Ensure the index name is 'Date' so reset_index yields a 'Date' column
    if hist.index.name != 'Date':
        hist = hist.copy()
        hist.index.name = 'Date'
    return hist

# ---------------- Existing logic below ----------------

def calculate_max_profit(data):
    """Calculate maximum profit possible from historical data."""
    min_price = float('inf')
    max_profit = 0
    buy_date = None
    sell_date = None

    data_reset = data.reset_index()

    for i, row in data_reset.iterrows():
        if row['Close'] < min_price:
            min_price = row['Close']
            temp_buy_date = row['Date']

        profit = row['Close'] - min_price
        if profit > max_profit:
            max_profit = profit
            buy_date = temp_buy_date
            sell_date = row['Date']

    return buy_date, sell_date, float(max_profit)

def brute_force_max_profit(data):
    dr = data.reset_index()
    max_profit = 0.0
    buy_date = None
    sell_date = None
    n = len(dr)
    for i in range(n):
        for j in range(i + 1, n):
            profit = float(dr.loc[j, 'Close'] - dr.loc[i, 'Close'])
            if profit > max_profit:
                max_profit = profit
                buy_date = dr.loc[i, 'Date']
                sell_date = dr.loc[j, 'Date']
    return buy_date, sell_date, float(max_profit)

def vectorized_max_profit(data):
    dr = data.reset_index()
    prices = dr['Close'].to_numpy(dtype=float)
    if prices.size == 0:
        return None, None, 0.0
    cummin_prices = np.minimum.accumulate(prices)
    profits = prices - cummin_prices
    j = int(np.argmax(profits))
    max_profit = float(profits[j])
    if max_profit <= 0:
        return None, None, 0.0
    i = int(np.where(prices[:j+1] == cummin_prices[j])[0][0])
    buy_date = dr.loc[i, 'Date']
    sell_date = dr.loc[j, 'Date']
    return buy_date, sell_date, float(max_profit)

def _tuple_equal(a, b, tol=1e-9):
    da1, ds1, p1 = a
    da2, ds2, p2 = b
    return (da1 == da2) and (ds1 == ds2) and (abs(float(p1) - float(p2)) <= tol)

def validate_calculate_max_profit(data):
    r_single = calculate_max_profit(data)
    r_brute  = brute_force_max_profit(data)
    r_vect   = vectorized_max_profit(data)
    all_equal = _tuple_equal(r_single, r_brute) and _tuple_equal(r_single, r_vect)
    return {
        'single_pass': r_single,
        'brute_force': r_brute,
        'vectorized': r_vect,
        'all_equal': all_equal
    }

def validate_first_five(data):
    # Ensure chronological order, then take the first 5 rows
    df5 = data.sort_index().head(5)

    # 1) Preview table of the first 5 rows
    print("First 5 rows (Table):")
    # Only show the Close column; the Date index (tz-aware) will be printed as the row index
    print(df5[['Close']].to_string())

    # 2) Run the existing validators on the first 5 rows
    r_single = calculate_max_profit(df5)
    r_brute  = brute_force_max_profit(df5)
    r_vect   = vectorized_max_profit(df5)

    # Build a small table for the three methods on first 5 rows
    res5_df = pd.DataFrame(
        [
            {"Method": "Single-pass", "Buy": r_single[0], "Sell": r_single[1], "Profit": float(r_single[2])},
            {"Method": "Brute-force", "Buy": r_brute[0],  "Sell": r_brute[1],  "Profit": float(r_brute[2])},
            {"Method": "Vectorized",  "Buy": r_vect[0],   "Sell": r_vect[1],   "Profit": float(r_vect[2])},
        ],
        columns=["Method", "Buy", "Sell", "Profit"]
    )

    print("\nFirst 5 rows validation (Table):")
    print(res5_df.to_string(index=False))

    # 3) Summary line in the requested style
    bf_matches_vect = _tuple_equal(r_brute, r_vect)
    print(
        "Results: The Validation methods for the single-pass, brute force and vectorized " +
        ("match on the first 5 rows" if bf_matches_vect else "do NOT match on the first 5 rows")
    )




if __name__ == "__main__":
    symbol = user_input_ticker()
    df = fetch_stock_data(symbol, period='1y', interval='1d')
    results = validate_calculate_max_profit(df)

    # Keep prior overall summary line
    bf_matches_vect = _tuple_equal(results['brute_force'], results['vectorized'])
    print("Results: The Validation methods for the single-pass, brute force and vectorized " +
          ("match" if bf_matches_vect else "do NOT match"))

    # New: first-5 validation output
    print()
    validate_first_five(df)

