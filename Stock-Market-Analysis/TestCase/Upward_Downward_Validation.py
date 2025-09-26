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
def fetch_stock_data(ticker_symbol, period='1y', interval='1d'):
    ticker = Ticker(ticker_symbol)
    hist = ticker.history(period=period, interval=interval)
    if hist.empty:
        print("No data found for the given ticker or period.")
        exit()
    return hist

# Validate trends using pandas diff (per-day movement)
def validate_trend(df):
    df['Upward_Trend'] = df['Close'].diff() > 0  # True if today's close > yesterday's
    df['Downward_Trend'] = df['Close'].diff() < 0  # True if today's close < yesterday's
    return df

# Manually count upward/downward movements and longest streaks
# O(n)
def manual_trend(df):
    close_prices = df['Close'].tolist()
    upward_trends = 0
    downward_trends = 0
    longest_streak_up = 0
    current_streak_up = 0
    longest_streak_down = 0
    current_streak_down = 0
    up_streak_start = 0
    up_streak_end = 0
    down_streak_start = 0
    down_streak_end = 0
    current_up_start = 0
    current_down_start = 0

    for i in range(1, len(close_prices)):
        # Compare prices with previous day
        if close_prices[i] > close_prices[i-1]:
            # Current price > yesterday's price = Up ++
            upward_trends += 1
            if current_streak_up == 0:
                current_up_start = i - 1
            current_streak_up += 1
            if current_streak_up > longest_streak_up:
                longest_streak_up = current_streak_up
                up_streak_start = current_up_start
                up_streak_end = i
        else:
            current_streak_up = 0
        if close_prices[i] < close_prices[i-1]:
            downward_trends += 1
            if current_streak_down == 0:
                current_down_start = i - 1
            current_streak_down += 1
            if current_streak_down > longest_streak_down:
                longest_streak_down = current_streak_down
                down_streak_start = current_down_start
                down_streak_end = i
        else:
            current_streak_down = 0
    return upward_trends, downward_trends, longest_streak_up, longest_streak_down, up_streak_start, up_streak_end, down_streak_start, down_streak_end

# Main function to run trend analysis and validation
def test_run():
    ticker_symbol = user_input_ticker()
    data = fetch_stock_data(ticker_symbol)
    print(f"Fetched {len(data)} rows of data for {ticker_symbol}.")
    
    # Manual calculation (matches validate_trend_v2 logic)
    manual_upward, manual_downward, longest_up, longest_down, up_streak_start, up_streak_end, down_streak_start, down_streak_end = manual_trend(data)
    
    # Validate using pandas diff
    validated_data = validate_trend(data.copy())
    validated_upward = validated_data['Upward_Trend'].sum()
    validated_downward = validated_data['Downward_Trend'].sum()
    
    print(f"Manual Upward Trends: {manual_upward}, Downward Trends: {manual_downward}")
    print(f"Validated Upward Trends: {validated_upward}, Downward Trends: {validated_downward}")
    
    print("Upward trends match:", manual_upward == validated_upward)
    print("Downward trends match:", manual_downward == validated_downward)
    print(f"Longest Upward Streak: {longest_up} days")
    print(f"Longest Downward Streak: {longest_down} days")
    print(validated_data[['Close', 'Upward_Trend', 'Downward_Trend']].tail(30))

    # Check if all validations passed
    if manual_upward == validated_upward and manual_downward == validated_downward:
        print("All trend validations passed!")

        # Prepare masks for up/down days
        up_mask = validated_data['Upward_Trend']
        down_mask = validated_data['Downward_Trend']

        # Prepare marker data for mplfinance addplot
        up_marker = np.where(up_mask, validated_data['High'] + 1, np.nan)  # Place marker above candle
        down_marker = np.where(down_mask, validated_data['Low'] - 1, np.nan)  # Place marker below candle

        # Create addplot objects for markers
        ap_up = mpf.make_addplot(up_marker, type='scatter', markersize=50, marker='^', color='green', panel=0)
        ap_down = mpf.make_addplot(down_marker, type='scatter', markersize=50, marker='v', color='red', panel=0)

        # Prepare data for longest streak lines
        up_streak_data = np.full(len(validated_data), np.nan)
        down_streak_data = np.full(len(validated_data), np.nan)
        
        # Fill data for longest upward streak
        if longest_up > 0:
            up_streak_data[up_streak_start:up_streak_end + 1] = validated_data['Close'].iloc[up_streak_start:up_streak_end + 1]
        
        # Fill data for longest downward streak
        if longest_down > 0:
            down_streak_data[down_streak_start:down_streak_end + 1] = validated_data['Close'].iloc[down_streak_start:down_streak_end + 1]

        # Create addplot objects for streak lines
        ap_up_streak = mpf.make_addplot(up_streak_data, type='line', color='green', width=2, panel=0, label=f'Longest Upward Streak ({longest_up} days)')
        ap_down_streak = mpf.make_addplot(down_streak_data, type='line', color='red', width=2, panel=0, label=f'Longest Downward Streak ({longest_down} days)')

        # Plot candlestick chart with trend markers and streak lines
        mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc)

        mpf.plot(
            validated_data,
            type='candle',
            style=s,
            title=f"{ticker_symbol} Candlestick Chart with Up/Down Trend Markers and Longest Streaks",
            ylabel='Price',
            volume=True,
            show_nontrading=False,
            addplot=[ap_up, ap_down, ap_up_streak, ap_down_streak],
            figscale=1.2,
            figratio=(16, 9),
            main_panel=0,
            volume_panel=1
        )

# Run the analysis if this file is executed directly
if __name__ == "__main__":

    test_run()
