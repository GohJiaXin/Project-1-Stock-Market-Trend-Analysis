from yfinance import Ticker
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import mplfinance as mpf


def user_input_ticker():
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").upper()
    return ticker_symbol

def fetch_stock_data(ticker_symbol, period='3y', interval='1d'):
    ticker = Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    return hist

def display_line_chart_for_close_prices(data, title="Stock Close Prices"):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def display_in_candlesticks(data, title="CandleSticks Chart"):
    mpf.plot(data, type='candle', style='yahoo', volume=True)

def calculate_sma(data, window_size=[20, 50, 200]):
    close_prices = data['Close']
    moving_averages = {window: [] for window in window_size}
    for window in window_size:
        for i in range(len(close_prices) - window + 1):
            window_data = close_prices.iloc[i : i + window]
            window_average = round(window_data.mean(), 2)
            moving_averages[window].append(window_average)
    return moving_averages

def validate_sma(data, window_size=[20, 50, 200]):
    manual_sma = calculate_sma(data, window_size)
    # Convert np.float64 to Python float
    manual_sma = {window: [float(val) for val in vals] for window, vals in manual_sma.items()}
    rolling_sma = {window: data['Close'].rolling(window=window).mean().dropna().round(2).tolist() for window in window_size}
    rolling_sma = {window: [float(val) for val in vals] for window, vals in rolling_sma.items()}
    for window in window_size:
        print(f"Manual SMA (first 5) for window {window}:", manual_sma[window][:5])
        print(f"Rolling SMA (first 5) for window {window}:", rolling_sma[window][:5])
        print(f"Validation (first 5) for window {window}:",
        manual_sma[window][:5] == rolling_sma[window][:5])
        print("\n")

    if manual_sma == rolling_sma:
        return True

def calculate_moving_averages(data, windows=[20, 50, 200]):
    moving_averages = {}
    for window in windows:
        moving_averages[f"MA_{window}"] = data['Close'].rolling(window=window).mean()
    return pd.DataFrame(moving_averages)

def display_close_and_sma_chart(data, ma_data, title="Stock Close Price with SMAs"):
    plt.figure(figsize=(14, 7))
    
    # Plot Close Price
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    # Plot each SMA
    colors = { "MA_20": "green", "MA_50": "red", "MA_200": "orange" }
    for column in ma_data.columns:
        plt.plot(ma_data.index, ma_data[column], label=column, color=colors.get(column, "gray"))
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_run():
    ticker_symbol = user_input_ticker()
    stock_data = fetch_stock_data(ticker_symbol)
    print("Starting date of fetched data:", stock_data.index[0])
    ma_data = calculate_moving_averages(stock_data)
    print(stock_data.tail())

    
    if validate_sma(stock_data) is True:
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
    
if __name__ == "__main__":
    test_run()