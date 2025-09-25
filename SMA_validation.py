from yfinance import Ticker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import mplfinance as mpf

# Prompt user for a stock ticker symbol
def user_input_ticker():
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
    return ticker_symbol

# Fetch historical stock data using yfinance
def fetch_stock_data(ticker_symbol, period='3y', interval='1d'):
    ticker = Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    return hist

# Display a line chart of closing prices
def display_line_chart_for_close_prices(data, title="Stock Close Prices"):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Display a candlestick chart using mplfinance
def display_in_candlesticks(data, title="CandleSticks Chart"):
    mpf.plot(data, type='candle', style='yahoo', volume=True)


# O(n)
def calculate_sma(data, windows=[20, 50, 200]):
    """
    Compute SMA for multiple window sizes using sliding window trick.

    Args:
        data: DataFrame with 'Close' column
        windows: list of integers (window sizes)

    Returns:
        dict: {window_size: list_of_sma_values}
    """
    close_prices = data['Close']
    prices = list(close_prices)  # Ensure it's a list
    n = len(close_prices)
    sma_dict = {}

    for k in windows:
        if k > n:
            sma_dict[k] = []  # Window larger than data → empty
            continue

        result = []
        # Calculate SMA only when a full window of k prices is available
        for i in range(k - 1, n):
            window = prices[i - k + 1:i + 1]
            if len(window) == k:  # Ensure full window
                window_sum = sum(window)
                result.append(round(window_sum / k, 2))

        sma_dict[k] = result

    return sma_dict


# Validate manual SMA calculation against pandas rolling mean
def validate_sma(data, window_size=[20, 50, 200]):
    manual_sma = calculate_sma(data, window_size)
    # Convert np.float64 to Python float for comparison
    manual_sma = {window: [float(val) for val in vals] for window, vals in manual_sma.items()}
    rolling_sma = {
        window: data['Close'].rolling(window=window).mean().dropna().round(2).tolist()
        for window in window_size
    }
    rolling_sma = {window: [float(val) for val in vals] for window, vals in rolling_sma.items()}

    all_ok = True
    for window in window_size:
        m = manual_sma[window]
        r = rolling_sma[window]

        # ✅ Use tolerance instead of strict equality
        if not np.allclose(m, r, rtol=1e-5, atol=0.01):
            diffs = [(i, m[i], r[i]) for i in range(min(len(m), len(r))) if not np.isclose(m[i], r[i], atol=0.01)]
            print(f"❌ Window {window} differs at {len(diffs)} positions, e.g.: {diffs[:5]}")
            all_ok = False
        else:
            print(f"✅ Window {window} matches within tolerance.")

    return {window: np.allclose(m, r, rtol=1e-5, atol=0.01) for window in window_size}



# Calculate moving averages using pandas rolling for chart display
def calculate_moving_averages(data, windows=[20, 50, 200]):
    moving_averages = {}
    for window in windows:
        moving_averages[f"MA_{window}"] = data['Close'].rolling(window=window).mean()
    return pd.DataFrame(moving_averages)

# Display line chart with close price and SMAs
def display_close_and_sma_chart(data, ma_data, title="Stock Close Price with SMAs"):
    plt.figure(figsize=(14, 7))
    # Plot Close Price
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    # Plot each SMA with different color
    colors = { "MA_20": "green", "MA_50": "red", "MA_200": "orange" }
    for column in ma_data.columns:
        plt.plot(ma_data.index, ma_data[column], label=column, color=colors.get(column, "gray"))
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run SMA validation and chart display
def test_run():
    ticker_symbol = user_input_ticker()
    stock_data = fetch_stock_data(ticker_symbol)
    print("Starting date of fetched data:", stock_data.index[0])
    ma_data = calculate_moving_averages(stock_data)
    print(stock_data.tail())

    # Validate SMA calculation before displaying charts
    validation_results = validate_sma(stock_data)
    if all(validation_results.values()):
        chart_type = input("Choose chart type: (1) Line Chart with SMA, (2) Candlestick Chart with SMA: ")
        if chart_type == "1":
            display_close_and_sma_chart(stock_data, ma_data, title=f"{ticker_symbol} Close Prices")
        elif chart_type == "2":
            # Overlay SMAs on candlestick chart
            addplots = [mpf.make_addplot(ma_data[col]) for col in ma_data.columns]
            mpf.plot(stock_data, type='candle', style='yahoo', volume=True, addplot=addplots, title=f"{ticker_symbol} Candlestick with SMAs")
        else:
            print("Invalid choice. Showing line chart by default.")
            display_close_and_sma_chart(stock_data, ma_data, title=f"{ticker_symbol} Close Prices")
    else:
        print("SMA validation failed. Please check the calculations.")

# Run the analysis if this file is executed directly
if __name__ == "__main__":
    test_run()
