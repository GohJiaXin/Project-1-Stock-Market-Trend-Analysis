import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from io import BytesIO
import seaborn as sns

def validate_with_yf(ticker: str) -> bool:
    """Validate stock symbol directly with yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1mo")
        return not hist.empty
    except Exception:
        return False

def download_stock_data(ticker: str, years: int = 3, save_path: str = None):
    """Download historical stock data from Yahoo Finance."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"Data saved to {save_path}")
    
    return df

def clean_csv(filepath: str):
    """Clean CSV file by removing invalid rows."""
    df = pd.read_csv(filepath, on_bad_lines='skip')
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_cols)
    df.to_csv(filepath, index=False)
    print(f"Cleaned file saved back to {filepath}")
    return df

def select_and_analyze_csv(base_dir: str):
    """Let the user pick a CSV file if multiple exist in the directory."""
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    
    if not csv_files:
        st.warning("No CSV files found in the directory.")
        return None, None
    
    if len(csv_files) > 1:
        st.subheader("Select CSV File")
        selected_file = st.selectbox("Choose a CSV file to analyze:", csv_files)
        filepath = os.path.join(base_dir, selected_file)
        st.info(f"You selected: {selected_file}")
    else:
        filepath = os.path.join(base_dir, csv_files[0])
        st.success(f"Found 1 CSV file: {csv_files[0]}")
    
    st.subheader("📊 Select Analysis Functionality")
    analysis_options = ["SMA", "Upward and Downward Runs", "Daily Returns", 
                       "Max Profit Calculations", "Line Chart", "Candlestick Chart", 
                       "Seaborn SMA Chart", "Seaborn Runs Chart"]
    selected_analysis = st.multiselect(
        "Choose one or more analyses to run:", analysis_options
    )
    
    return filepath, selected_analysis

# Analysis functions
def display_line_chart_for_close_prices(data, title="Stock Close Prices"):
    """Display line chart for close prices."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig

def display_candlesticks(data, title="CandleSticks Chart"):
    """Display candlestick chart."""
    # Create a subset of data for better performance with mplfinance
    plot_data = data.tail(100)  # Show only last 100 days for better performance
    
    fig, axes = mpf.plot(plot_data, type='candle', style='yahoo', volume=True, 
                        returnfig=True, figsize=(14, 7))
    axes[0].set_title(title)
    return fig

def calculate_sma(data, window_size=[20, 50, 200]):
    """Calculate Simple Moving Average manually."""
    close_prices = data['Close']
    moving_averages = {window: [] for window in window_size}
    
    for window in window_size:
        for i in range(len(close_prices) - window + 1):
            window_data = close_prices.iloc[i : i + window]
            window_average = round(window_data.mean(), 2)
            moving_averages[window].append(window_average)
    
    return moving_averages

def validate_sma(data, window_size=[20, 50, 200]):
    """Validate manual SMA calculation against pandas rolling mean."""
    manual_sma = calculate_sma(data, window_size)
    manual_sma = {window: [float(val) for val in vals] for window, vals in manual_sma.items()}
    
    rolling_sma = {window: data['Close'].rolling(window=window).mean().dropna().round(2).tolist() 
                  for window in window_size}
    rolling_sma = {window: [float(val) for val in vals] for window, vals in rolling_sma.items()}
    
    validation_results = {}
    for window in window_size:
        validation_results[window] = manual_sma[window][:5] == rolling_sma[window][:5]
    
    return validation_results

def calculate_moving_averages(data, windows=[20, 50, 200]):
    """Calculate moving averages using pandas rolling mean."""
    moving_averages = {}
    for window in windows:
        moving_averages[f"MA_{window}"] = data['Close'].rolling(window=window).mean()
    return pd.DataFrame(moving_averages)

def display_close_and_sma_chart(data, ma_data, title="Stock Close Price with SMAs"):
    """Display close price with SMA lines."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    colors = {"MA_20": "green", "MA_50": "red", "MA_200": "orange"}
    for column in ma_data.columns:
        ax.plot(ma_data.index, ma_data[column], label=column, color=colors.get(column, "gray"))
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig

def analyze_upward_downward_runs(data):
    """Analyze upward and downward runs in stock prices."""
    data = data.copy()
    data['Price_Change'] = data['Close'].diff()
    data['Direction'] = data['Price_Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    current_run = 0
    current_direction = 0
    runs = []
    
    for i, row in data.iterrows():
        if i == 0:
            continue
        if row['Direction'] == current_direction:
            current_run += 1
        else:
            if current_run > 0:
                runs.append((current_direction, current_run))
            current_run = 1
            current_direction = row['Direction']
    
    if current_run > 0:
        runs.append((current_direction, current_run))
    
    upward_runs = [length for direction, length in runs if direction == 1]
    downward_runs = [length for direction, length in runs if direction == -1]
    
    return upward_runs, downward_runs

def calculate_max_profit(data):
    """Calculate maximum profit possible from historical data."""
    min_price = float('inf')
    max_profit = 0
    buy_date = None
    sell_date = None
    
    # Reset index to work with iterrows properly
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
    
    return buy_date, sell_date, max_profit

# New Seaborn-based plotting functions
def compute_sma(series, window):
    """Compute Simple Moving Average for a series."""
    return series.rolling(window=window).mean()

def plot_close_sma(df, window, title=None):
    """Plot closing price vs SMA using Seaborn."""
    if title is None:
        title = f"Closing Price vs SMA ({window})"
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df, x=df.index, y="Close", label="Close", linewidth=2, ax=ax)
    sns.lineplot(data=df, x=df.index, y=compute_sma(df["Close"], window),
                 label=f"SMA ({window})", linewidth=2, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_runs(df):
    """Plot runs with up/down markers using Seaborn."""
    # Use a subset of data for better performance
    plot_data = df.tail(100).copy()
    
    d = plot_data["Close"]
    up_mask = d.diff() > 0
    down_mask = d.diff() < 0

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(x=d.index, y=d.values, label="Close", linewidth=2, ax=ax)
    sns.scatterplot(x=d.index[up_mask], y=d[up_mask], marker="^",
                    color="green", s=60, label="Up days", ax=ax)
    sns.scatterplot(x=d.index[down_mask], y=d[down_mask], marker="v",
                    color="red", s=60, label="Down days", ax=ax)

    ax.set_title("Closing Price with Up/Down Markers (Last 100 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    return fig

def main():
    st.title("📈 Stock Data Downloader & Analyzer")
    st.write("This application downloads historical stock data from Yahoo Finance and provides analysis tools.")
    
    # Initialize session state
    if 'valid_symbol' not in st.session_state:
        st.session_state.valid_symbol = False
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""
    if 'data_downloaded' not in st.session_state:
        st.session_state.data_downloaded = False
    
    # Define base directory
    base_dir = r"C:\Users\Goh Jia Xin\Downloads\Project-1-Stock-Market-Trend-Analysis-main"
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Download Data", "Analyze Data"])
    
    with tab1:
        st.header("Download Stock Data")
        
        # Step 1: User enters stock ticker
        ticker = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):", 
                              value=st.session_state.ticker).upper()
        
        if st.button("Check Symbol"):
            if not isinstance(ticker, str) or ticker.strip() == "":
                st.error("Please enter a valid string as stock symbol.")
            else:
                st.session_state.ticker = ticker
                if validate_with_yf(ticker):
                    st.session_state.valid_symbol = True
                    st.success(f"'{ticker}' is a valid stock symbol!")
                else:
                    st.session_state.valid_symbol = False
                    st.error(f"'{ticker}' does not exist on Yahoo Finance. Please try again.")
        
        # Step 2: If valid ticker, ask for years (min 3)
        if st.session_state.valid_symbol:
            years = st.number_input(
                "Enter number of years of data to download (minimum 3):",
                min_value=3, max_value=10, value=3, step=1
            )
            
            filename = os.path.join(base_dir, f"{st.session_state.ticker}_{years}Y.csv")
            
            if st.button("Download Stock Data"):
                with st.spinner(f"Downloading {years} years of data for {st.session_state.ticker}..."):
                    try:
                        stock_df = download_stock_data(st.session_state.ticker, years=years, save_path=filename)
                        stock_df = clean_csv(filename)
                        st.success("Data downloaded and cleaned successfully!")
                        st.session_state.data_downloaded = True
                        
                        st.subheader("Sample Data:")
                        st.dataframe(stock_df.head())
                        
                        st.subheader("Basic Statistics:")
                        st.dataframe(stock_df.describe())
                        
                        st.info(f"Data saved to: {filename}")
                        
                        with open(filename, "rb") as file:
                            st.download_button(
                                label="Download CSV",
                                data=file,
                                file_name=f"{st.session_state.ticker}_{years}Y.csv",
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error downloading data: {str(e)}")
    
    with tab2:
        st.header("Analyze Stock Data")
        
        # Check if there are CSV files to analyze
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.warning("No CSV files found. Please download data first using the 'Download Data' tab.")
        else:
            # Use the select_and_analyze_csv function
            filepath, selected_analysis = select_and_analyze_csv(base_dir)
            
            if filepath and selected_analysis:
                # Load the data
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                st.success(f"Loaded data from {os.path.basename(filepath)}")
                st.dataframe(df.head())
                
                # Perform selected analyses
                if "Line Chart" in selected_analysis:
                    st.subheader("Line Chart - Close Prices")
                    fig = display_line_chart_for_close_prices(df)
                    st.pyplot(fig)
                
                if "Candlestick Chart" in selected_analysis:
                    st.subheader("Candlestick Chart")
                    fig = display_candlesticks(df)
                    st.pyplot(fig)
                
                if "SMA" in selected_analysis:
                    st.subheader("Simple Moving Average (SMA) Analysis")
                    
                    # Calculate SMAs
                    ma_data = calculate_moving_averages(df)
                    
                    # Plot SMAs
                    fig = display_close_and_sma_chart(df, ma_data)
                    st.pyplot(fig)
                    
                    # SMA crossover signals
                    df_sma = df.join(ma_data)
                    df_sma['Signal'] = 0
                    df_sma.loc[df_sma['MA_20'] > df_sma['MA_50'], 'Signal'] = 1  # Buy signal
                    df_sma.loc[df_sma['MA_20'] < df_sma['MA_50'], 'Signal'] = -1  # Sell signal
                    
                    st.write("SMA Crossover Signals:")
                    st.dataframe(df_sma[['Close', 'MA_20', 'MA_50', 'Signal']].tail(10))
                    
                    # Validate SMA calculation
                    st.write("SMA Validation (first 5 values for each window):")
                    validation_results = validate_sma(df)
                    for window, is_valid in validation_results.items():
                        st.write(f"Window {window}: {'Valid' if is_valid else 'Invalid'}")
                
                if "Seaborn SMA Chart" in selected_analysis:
                    st.subheader("Seaborn SMA Chart")
                    
                    # Let user select SMA window
                    sma_window = st.slider("Select SMA Window:", min_value=5, max_value=100, value=20, step=5)
                    
                    # Plot using Seaborn
                    fig = plot_close_sma(df, sma_window)
                    st.pyplot(fig)
                
                if "Seaborn Runs Chart" in selected_analysis:
                    st.subheader("Seaborn Runs Chart - Up/Down Days")
                    
                    # Plot using Seaborn
                    fig = plot_runs(df)
                    st.pyplot(fig)
                
                if "Daily Returns" in selected_analysis:
                    st.subheader("Daily Returns Analysis")
                    
                    # Calculate daily returns
                    df['Daily_Return'] = df['Close'].pct_change() * 100
                    
                    # Plot daily returns
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(df.index, df['Daily_Return'], label='Daily Return', color='purple')
                    ax.set_title("Daily Returns")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Return (%)')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
                    
                    # Statistics
                    st.write("Daily Returns Statistics:")
                    st.write(f"Average Daily Return: {df['Daily_Return'].mean():.2f}%")
                    st.write(f"Standard Deviation: {df['Daily_Return'].std():.2f}%")
                    st.write(f"Maximum Daily Gain: {df['Daily_Return'].max():.2f}%")
                    st.write(f"Maximum Daily Loss: {df['Daily_Return'].min():.2f}%")
                
                if "Upward and Downward Runs" in selected_analysis:
                    st.subheader("Upward and Downward Runs Analysis")
                    
                    # Analyze runs
                    upward_runs, downward_runs = analyze_upward_downward_runs(df.reset_index())
                    
                    st.write("Upward Runs Analysis:")
                    if upward_runs:
                        st.write(f"Longest upward run: {max(upward_runs)} days")
                        st.write(f"Average upward run: {sum(upward_runs)/len(upward_runs):.2f} days")
                        st.write(f"Total upward runs: {len(upward_runs)}")
                    else:
                        st.write("No upward runs found")
                    
                    st.write("Downward Runs Analysis:")
                    if downward_runs:
                        st.write(f"Longest downward run: {max(downward_runs)} days")
                        st.write(f"Average downward run: {sum(downward_runs)/len(downward_runs):.2f} days")
                        st.write(f"Total downward runs: {len(downward_runs)}")
                    else:
                        st.write("No downward runs found")
                
                if "Max Profit Calculations" in selected_analysis:
                    st.subheader("Maximum Profit Calculation")
                    
                    # Find the best buy and sell points
                    buy_date, sell_date, max_profit = calculate_max_profit(df)
                    
                    if max_profit > 0:
                        buy_price = df.loc[df.index == buy_date, 'Close'].iloc[0]
                        sell_price = df.loc[df.index == sell_date, 'Close'].iloc[0]
                        
                        st.write(f"Best buy date: {buy_date.strftime('%Y-%m-%d')} at ${buy_price:.2f}")
                        st.write(f"Best sell date: {sell_date.strftime('%Y-%m-%d')} at ${sell_price:.2f}")
                        st.write(f"Maximum profit: ${max_profit:.2f} per share ({max_profit/buy_price*100:.2f}% return)")
                    else:
                        st.write("No profitable trading opportunity found in this period")

if __name__ == "__main__":
    main()
