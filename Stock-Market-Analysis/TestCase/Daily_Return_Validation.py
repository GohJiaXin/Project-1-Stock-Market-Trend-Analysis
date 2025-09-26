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

# Prompt user for their buy price
def user_buy_price():
    try:
        buy_price = float(input("Enter your buy price: "))
        print(f"Your buy price is set at: {buy_price}")
        return buy_price
    except ValueError:
        print("Invalid input. Please enter a numeric value for the buy price.")
        return user_buy_price() 

# Calculate daily returns based on user buy price
def return_for_user_price(df, buy_price):
    df['Daily_Return'] = (df['Close'] - buy_price) / buy_price
    df['Percentage(%)'] = df['Daily_Return'] * 100  # Add percentage column

    return df

# Validate daily returns using pandas pct_change
def validate_daily_returns(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Percentage(%)'] = df['Daily_Return'] * 100  # Add percentage column
    return df

# Manually calculate daily returns
# O(n)
def manual_daily_returns(df):
    close_prices = df['Close'].tolist()
    daily_returns = [None]  # First day has no previous day to compare
    for i in range(1, len(close_prices)):
        r_t = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
        daily_returns.append(round(r_t, 6))  # Round to 6 decimal places
    df['Manual_Daily_Return'] = daily_returns
    df['Percentage(%)'] = df['Manual_Daily_Return'] * 100  # Add percentage column
    return df

def test_run():
    ticker_symbol = user_input_ticker()
    data = fetch_stock_data(ticker_symbol)
    print(f"Fetched {len(data)} rows of data for {ticker_symbol}.")

    # Extra feature: Ask user if they want to input their buy price
    while True:
        user_choice =  input("Do you want to enter your buy price? (y/n): ").lower()
        if user_choice =='y':

            buy_price = user_buy_price()
            user_data = return_for_user_price(data.copy(), buy_price)
            latest_row = user_data.iloc[-1]
            print("Latest day price and return based on your buy price:")
            print(latest_row[['Close', 'Daily_Return', 'Percentage(%)']].to_string())
            break

        elif user_choice =='n':
            # Validate daily returns using pandas pct_change
            validated_data = validate_daily_returns(data.copy())
            print("First 5 rows with Daily Returns (Table):")
            print(validated_data[['Close', 'Daily_Return', 'Percentage(%)']].head().to_string())

            # Validate daily returns using manual calculation
            manual_data = manual_daily_returns(data.copy())
            print("First 5 rows with Manual Daily Returns (Table):")
            print(manual_data[['Close', 'Manual_Daily_Return', 'Percentage(%)']].head().to_string())

            if validated_data['Daily_Return'].round(6).equals(manual_data['Manual_Daily_Return'].round(6)):
                print("Validation successful: Both methods yield the same daily returns.")
            else:
                print("Validation failed: The daily returns do not match.")
            
            break

        else:
            print("Invalid choice. Please enter 'y' or 'n'.")
            


# Run the analysis if this file is executed directly
if __name__ == "__main__":
    test_run()
