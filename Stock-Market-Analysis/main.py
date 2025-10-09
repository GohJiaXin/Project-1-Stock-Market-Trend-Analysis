# Import necessary libraries
import streamlit as st  # For creating web application interface
import yfinance as yf  # For downloading stock data from Yahoo Finance
import pandas as pd  # For data manipulation and analysis
from datetime import datetime, timedelta  # For handling dates and time periods
import os  # For operating system interactions (file paths, directories)
import matplotlib.pyplot as plt  # For creating static visualizations
import mplfinance as mpf  # For creating financial charts (candlesticks)
import numpy as np  # For numerical operations
from io import BytesIO  # For handling file input/output operations
import seaborn as sns  # For enhanced data visualization

def validate_with_yf(ticker: str) -> bool:
    """Validate stock symbol directly with yfinance.
    
    Args:
        ticker (str): Stock symbol to validate
        
    Returns:
        bool: True if valid stock symbol, False otherwise
    """
    try:
        # Create ticker object and try to download 1 month of historical data
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1mo")
        # Return True if data is not empty (valid symbol)
        return not hist.empty
    except Exception:
        # Return False if any error occurs (invalid symbol)
        return False

def download_stock_data(ticker: str, years: int = 3, save_path: str = None):
    """Download historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock symbol to download data for
        years (int): Number of years of historical data to download
        save_path (str): File path where data should be saved (optional)
        
    Returns:
        pd.DataFrame: DataFrame containing stock data with columns: Date, Open, High, Low, Close, Volume
    """
    # Calculate date range for data download
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)
    
    # Download data using yfinance with daily interval
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    df = df.reset_index()  # Move Date from index to column
    df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime format
    
    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
        df.to_csv(save_path, index=False, encoding='utf-8')  # Save as CSV
        print(f"Data saved to {save_path}")
    
    return df

def clean_csv(filepath: str):
    """Clean CSV file by removing invalid rows and handling missing data.
    
    Args:
        filepath (str): Path to the CSV file to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Read CSV, skipping bad lines that can't be parsed
    df = pd.read_csv(filepath, on_bad_lines='skip')
    # Remove rows with missing dates
    df = df.dropna(subset=['Date'])
    # Convert Date column to datetime, coercing errors to NaT
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Remove rows where date conversion failed
    df = df.dropna(subset=['Date'])
    
    # Define numeric columns that need cleaning
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, setting invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing numeric values
    df = df.dropna(subset=numeric_cols)
    # Save cleaned data back to file
    df.to_csv(filepath, index=False)
    print(f"Cleaned file saved back to {filepath}")
    return df

def select_and_analyze_csv(base_dir: str, tab_type="basic"):
    """Let the user pick a CSV file if multiple exist in the directory.
    
    Args:
        base_dir (str): Directory path to search for CSV files
        tab_type (str): Type of analysis tab ("basic", "advanced")
        
    Returns:
        tuple: (filepath, selected_analysis) - Path to selected file and list of analysis options chosen
    """
    # Find all CSV files in directory
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    
    # Handle case where no CSV files found
    if not csv_files:
        st.warning("No CSV files found in the directory.")
        return None, None
    
    # If multiple CSV files, let user choose one
    if len(csv_files) > 1:
        st.subheader("Select CSV File")
        selected_file = st.selectbox("Choose a CSV file to analyze:", csv_files, key=f"select_{tab_type}")
        filepath = os.path.join(base_dir, selected_file)
        st.info(f"You selected: {selected_file}")
    else:
        # If only one file, use it automatically
        filepath = os.path.join(base_dir, csv_files[0])
        st.success(f"Found 1 CSV file: {csv_files[0]}")
    
    # Different analysis options based on tab type
    if tab_type == "basic":
        analysis_options = ["SMA", "Upward and Downward Runs", "Daily Returns", 
                           "Line Chart", "Candlestick Chart", "Seaborn SMA Chart", "Seaborn Runs Chart"]
    elif tab_type == "advanced":
        analysis_options = ["Trend Analysis", "Max Profit Calculations","Multiple Transactions Max Profit", "Daily Returns Analysis", "Manual SMA Calculations & Validation"]
    
    # Multi-select widget for analysis choices
    selected_analysis = st.multiselect(
        "Choose one or more analyses to run:", analysis_options, key=f"analysis_{tab_type}"
    )
    
    return filepath, selected_analysis

# Analysis functions

def display_line_chart_for_close_prices(data, title="Stock Close Prices"):
    """Display line chart for close prices.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        title (str): Chart title
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot close prices as blue line
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    # Set chart properties
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()  # Show legend
    ax.grid(True)  # Add grid lines
    return fig

def display_candlesticks(data, title="CandleSticks Chart"):
    """Display candlestick chart.
    
    Args:
        data (pd.DataFrame): Stock data with OHLC prices
        title (str): Chart title
        
    Returns:
        tuple: (fig, axes) - Figure and axes objects from mplfinance
    """
    # Use only last 100 days for better performance and clarity
    plot_data = data.tail(100)
    
    # Create candlestick chart with volume subplot
    fig, axes = mpf.plot(plot_data, type='candle', style='yahoo', volume=True, 
                        returnfig=True, figsize=(14, 7))
    axes[0].set_title(title)  # Set title on main axis
    return fig

def calculate_sma(data, windows=[20, 50, 200]):
    """
    SMA calculation using cumulative sum (O(n) per window).

    Args:
        data (pd.DataFrame): DataFrame with a 'Close' column
        windows (list[int]): List of window sizes (e.g., [20, 50, 200])

    Returns:
        dict[int, list[float]]: Mapping window size -> SMA values
    """
    close_prices = data['Close'].tolist()
    n = len(close_prices)
    sma_dict = {}

    for k in windows:
        if k > n or k <= 0:
            sma_dict[k] = []
            continue

        # Compute cumulative sum array
        cumsum = [0.0]
        for price in close_prices:
            cumsum.append(cumsum[-1] + price)
        # cumsum[i] now holds sum of first i elements (exclusive)

        # Compute SMA using difference of cumsum
        sma = []
        for i in range(k, n + 1):
            window_sum = cumsum[i] - cumsum[i - k]
            sma.append(round(window_sum / k, 2))

        sma_dict[k] = sma

    return sma_dict

def validate_sma(data, window_size=[20, 50, 200]):
    """Validate manual SMA calculation against pandas rolling mean with tolerance."""
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

def calculate_moving_averages(data, windows=[20, 50, 200]):
    """Calculate moving averages using pandas rolling mean.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        windows (list): List of window sizes for moving averages
        
    Returns:
        pd.DataFrame: DataFrame containing moving average columns
    """
    moving_averages = {}
    for window in windows:
        # Calculate moving average for each window size
        moving_averages[f"MA_{window}"] = data['Close'].rolling(window=window).mean()
    return pd.DataFrame(moving_averages)

def display_close_and_sma_chart(data, ma_data, title="Stock Close Price with SMAs"):
    """Display close price with SMA lines.
    
    Args:
        data (pd.DataFrame): Original stock data
        ma_data (pd.DataFrame): Moving averages data
        title (str): Chart title
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot close price
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    
    # Define colors for different moving averages
    colors = {"MA_20": "green", "MA_50": "red", "MA_200": "orange"}
    # Plot each moving average
    for column in ma_data.columns:
        ax.plot(ma_data.index, ma_data[column], label=column, color=colors.get(column, "gray"))
    
    # Set chart properties
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig

def analyze_upward_downward_runs(data):
    """Analyze upward and downward runs in stock prices.
    A 'run' is a sequence of consecutive days with price moving in same direction.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        
    Returns:
        tuple: (upward_runs, downward_runs) - Lists containing lengths of runs
    """
    data = data.copy()  # Avoid modifying original data
    # Calculate daily price changes
    data['Price_Change'] = data['Close'].diff()
    # Direction: 1=up, -1=down, 0=no change
    data['Direction'] = data['Price_Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Initialize variables for tracking runs
    current_run = 0  # Length of current run
    current_direction = 0  # Direction of current run
    runs = []  # List to store all runs
    
    # Iterate through data to identify runs
    for i, row in data.iterrows():
        if i == 0:  # Skip first row (no previous price)
            continue
        if row['Direction'] == current_direction:
            # Continue current run
            current_run += 1
        else:
            # End of current run, start new one
            if current_run > 0:
                runs.append((current_direction, current_run))
            current_run = 1
            current_direction = row['Direction']
    
    # Don't forget the last run
    if current_run > 0:
        runs.append((current_direction, current_run))
    
    # Separate upward and downward runs
    upward_runs = [length for direction, length in runs if direction == 1]
    downward_runs = [length for direction, length in runs if direction == -1]
    
    return upward_runs, downward_runs

# Max Profit Calculation Functions

def single_pass_max_profit(data):
    """Calculate maximum profit possible from historical data (best buy/sell points).
    Implements algorithm to find maximum difference with constraint that buy must come before sell.
    
    Args:
        data (pd.DataFrame): Stock data with Date and Close prices
        
    Returns:
        tuple: (buy_date, sell_date, max_profit) - Optimal trading dates and profit
    """
    min_price = float('inf')  # Initialize with very high value
    max_profit = 0  # Track maximum profit found
    buy_date = None  # Best buy date
    sell_date = None  # Best sell date
    
    # Reset index to ensure proper iteration
    data_reset = data.reset_index()
    
    # Iterate through data to find optimal buy/sell points
    for i, row in data_reset.iterrows():
        # Update minimum price encountered so far
        if row['Close'] < min_price:
            min_price = row['Close']
            temp_buy_date = row['Date']  # Temporary buy date candidate
        
        # Calculate profit if selling at current price
        profit = row['Close'] - min_price
        # Update maximum profit if current profit is better
        if profit > max_profit:
            max_profit = profit
            buy_date = temp_buy_date  # Finalize buy date
            sell_date = row['Date']  # Set sell date
    
    return buy_date, sell_date, max_profit

def brute_force_max_profit(data):
    """Calculate maximum profit using brute force approach (O(n^2) complexity).
    
    Args:
        data (pd.DataFrame): Stock data with Date and Close prices
        
    Returns:
        tuple: (buy_date, sell_date, max_profit) - Optimal trading dates and profit
    """
    dr = data.reset_index()
    max_profit = 0.0
    buy_date = None
    sell_date = None
    n = len(dr)
    
    # Check all possible buy/sell pairs
    for i in range(n):
        for j in range(i + 1, n):
            profit = float(dr.loc[j, 'Close'] - dr.loc[i, 'Close'])
            if profit > max_profit:
                max_profit = profit
                buy_date = dr.loc[i, 'Date']
                sell_date = dr.loc[j, 'Date']
                
    return buy_date, sell_date, float(max_profit)

def vectorized_max_profit(data):
    """Calculate maximum profit using vectorized approach (O(n) complexity).
    
    Args:
        data (pd.DataFrame): Stock data with Date and Close prices
        
    Returns:
        tuple: (buy_date, sell_date, max_profit) - Optimal trading dates and profit
    """
    dr = data.reset_index()
    prices = dr['Close'].to_numpy(dtype=float)
    
    if prices.size == 0:
        return None, None, 0.0
        
    # Use cumulative minimum to track best buy price up to each point
    cummin_prices = np.minimum.accumulate(prices)
    profits = prices - cummin_prices
    j = int(np.argmax(profits))  # Find best sell index
    max_profit = float(profits[j])
    
    if max_profit <= 0:
        return None, None, 0.0
        
    # Find corresponding buy index
    i = int(np.where(prices[:j+1] == cummin_prices[j])[0][0])
    buy_date = dr.loc[i, 'Date']
    sell_date = dr.loc[j, 'Date']
    
    return buy_date, sell_date, float(max_profit)

def _tuple_equal(a, b, tol=1e-9):
    """Helper function to compare max profit results with tolerance.
    
    Args:
        a, b: Tuples of (buy_date, sell_date, profit)
        tol: Tolerance for floating point comparison
        
    Returns:
        bool: True if results are effectively equal
    """
    da1, ds1, p1 = a
    da2, ds2, p2 = b
    return (da1 == da2) and (ds1 == ds2) and (abs(float(p1) - float(p2)) <= tol)

def validate_max_profit_calculations(data):
    """Validate all max profit calculation methods against each other.
    
    Args:
        data (pd.DataFrame): Stock data with Date and Close prices
        
    Returns:
        dict: Results from all methods and validation status
    """
    r_single = single_pass_max_profit(data)
    r_brute  = brute_force_max_profit(data)
    r_vect   = vectorized_max_profit(data)
    
    all_equal = _tuple_equal(r_single, r_brute) and _tuple_equal(r_single, r_vect)
    
    return {
        'single_pass': r_single,
        'brute_force': r_brute,
        'vectorized': r_vect,
        'all_equal': all_equal
    }

def max_profit_multiple_transactions(data):
    """Calculate maximum profit with multiple transactions allowed (Buy-Sell Stock II).
    
    Strategy: Buy whenever price increases from previous day (sum all positive differences).
    
    Args:
        data (pd.DataFrame): Stock data with Date and Close prices
        
    Returns:
        tuple: (total_profit, transactions) - Total profit and list of buy/sell transactions
    """
    dr = data.reset_index()
    prices = dr['Close'].tolist()
    n = len(prices)
    
    total_profit = 0.0
    transactions = []  # List to store (buy_date, sell_date, profit) tuples
    buy_date = None
    hold_price = None
    
    for i in range(1, n):
        # If price increases from previous day and we're not holding
        if prices[i] > prices[i-1]:
            if hold_price is None:  # Buy if not already holding
                buy_date = dr.loc[i-1, 'Date']
                hold_price = prices[i-1]
                # st.write(f"Buy on {buy_date.strftime('%Y-%m-%d')} at ${hold_price:.2f}")
        
        # If price decreases or it's the last day and we're holding, sell
        if hold_price is not None and (prices[i] < prices[i-1] or i == n-1):
            sell_date = dr.loc[i-1, 'Date'] if prices[i] < prices[i-1] else dr.loc[i, 'Date']
            sell_price = prices[i-1] if prices[i] < prices[i-1] else prices[i]
            profit = sell_price - hold_price
            
            if profit > 0:  # Only add profitable transactions
                total_profit += profit
                transactions.append((buy_date, sell_date, profit))
                # st.write(f"Sell on {sell_date.strftime('%Y-%m-%d')} at ${sell_price:.2f}, Profit: ${profit:.2f}")
            
            hold_price = None  # Reset holding status
    
    return total_profit, transactions

def max_profit_multiple_transactions_simple(data):
    """Simplified version: Sum all positive daily differences.
    
    This is the most efficient O(n) solution for multiple transactions.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        
    Returns:
        float: Total maximum profit achievable
    """
    dr = data.reset_index()
    prices = dr['Close'].tolist()
    total_profit = 0.0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            total_profit += prices[i] - prices[i-1]
    
    return total_profit

def validate_multiple_transactions(data):
    """Validate multiple transactions profit calculation.
    
    Args:
        data (pd.DataFrame): Stock data with Close prices
        
    Returns:
        dict: Results from both methods
    """
    profit_complex, transactions = max_profit_multiple_transactions(data)
    profit_simple = max_profit_multiple_transactions_simple(data)
    
    return {
        'complex_method': {
            'total_profit': profit_complex,
            'transactions': transactions,
            'num_transactions': len(transactions)
        },
        'simple_method': {
            'total_profit': profit_simple
        },
        'validation_passed': abs(profit_complex - profit_simple) < 1e-9
    }
# Trend Analysis Functions

def validate_trend(df):
    """Add upward and downward trend columns to DataFrame using pandas diff.
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        pd.DataFrame: DataFrame with added trend columns
    """
    df = df.copy()
    df['Upward_Trend'] = df['Close'].diff() > 0  # True if today's close > yesterday's
    df['Downward_Trend'] = df['Close'].diff() < 0  # True if today's close < yesterday's
    return df

def manual_trend(df):
    """Manually count upward/downward movements and longest streaks (O(n) complexity).
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        tuple: Various trend statistics and streak information
    """
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
            
    return (upward_trends, downward_trends, longest_streak_up, longest_streak_down, 
            up_streak_start, up_streak_end, down_streak_start, down_streak_end)

def plot_trend_candlestick(validated_data, manual_results, ticker_symbol):
    """Plot candlestick chart with trend markers and longest streaks.
    
    Args:
        validated_data (pd.DataFrame): DataFrame with trend columns
        manual_results (tuple): Results from manual_trend function
        ticker_symbol (str): Stock symbol for chart title
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Extract manual trend results
    upward_trends, downward_trends, longest_up, longest_down, up_streak_start, up_streak_end, down_streak_start, down_streak_end = manual_results
    
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

    fig, axes = mpf.plot(
        validated_data,
        type='candle',
        style=s,
        title=f"Candlestick Chart with Up/Down Trend Markers and Longest Streaks",
        ylabel='Price',
        volume=True,
        show_nontrading=False,
        addplot=[ap_up, ap_down, ap_up_streak, ap_down_streak],
        returnfig=True,
        figsize=(14, 10)
    )
    
    return fig

def run_trend_analysis(df, ticker_symbol):
    """Run comprehensive trend analysis and validation.
    
    Args:
        df (pd.DataFrame): Stock data
        ticker_symbol (str): Stock symbol for display purposes
        
    Returns:
        dict: Analysis results and validation status
    """
    # Manual calculation
    manual_upward, manual_downward, longest_up, longest_down, up_streak_start, up_streak_end, down_streak_start, down_streak_end = manual_trend(df)
    
    # Validate using pandas diff
    validated_data = validate_trend(df.copy())
    validated_upward = int(validated_data['Upward_Trend'].sum())
    validated_downward = int(validated_data['Downward_Trend'].sum())
    
    # Check if validations passed
    upward_match = manual_upward == validated_upward
    downward_match = manual_downward == validated_downward
    all_validations_passed = upward_match and downward_match
    
    return {
        'manual_upward': manual_upward,
        'manual_downward': manual_downward,
        'validated_upward': validated_upward,
        'validated_downward': validated_downward,
        'longest_up_streak': longest_up,
        'longest_down_streak': longest_down,
        'upward_match': upward_match,
        'downward_match': downward_match,
        'all_validations_passed': all_validations_passed,
        'validated_data': validated_data,
        'manual_results': (manual_upward, manual_downward, longest_up, longest_down, 
                          up_streak_start, up_streak_end, down_streak_start, down_streak_end)
    }

# New Seaborn-based plotting functions

def compute_sma(series, window):
    """Compute Simple Moving Average for a series using pandas rolling.
    
    Args:
        series (pd.Series): Price data series
        window (int): Window size for moving average
        
    Returns:
        pd.Series: Moving average values
    """
    return series.rolling(window=window).mean()

def plot_close_sma(df, window, title=None):
    """Plot closing price vs SMA using Seaborn for enhanced visuals.
    
    Args:
        df (pd.DataFrame): Stock data
        window (int): SMA window size
        title (str): Custom chart title (optional)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if title is None:
        title = f"Closing Price vs SMA ({window})"
    
    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot close price using seaborn
    sns.lineplot(data=df, x=df.index, y="Close", label="Close", linewidth=2, ax=ax)
    # Plot SMA using seaborn
    sns.lineplot(data=df, x=df.index, y=compute_sma(df["Close"], window),
                 label=f"SMA ({window})", linewidth=2, ax=ax)
    # Set chart properties
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()  # Improve layout
    return fig

def plot_runs(df):
    """Plot runs with up/down markers using Seaborn.
    Shows price movement with visual indicators for upward/downward days.
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Use subset for better performance and clarity
    plot_data = df.tail(100).copy()
    
    d = plot_data["Close"]
    # Identify upward and downward moving days
    up_mask = d.diff() > 0  # True for days with price increase
    down_mask = d.diff() < 0  # True for days with price decrease

    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot closing price line
    sns.lineplot(x=d.index, y=d.values, label="Close", linewidth=2, ax=ax)
    # Mark upward days with green triangles
    sns.scatterplot(x=d.index[up_mask], y=d[up_mask], marker="^",
                    color="green", s=60, label="Up days", ax=ax)
    # Mark downward days with red triangles
    sns.scatterplot(x=d.index[down_mask], y=d[down_mask], marker="v",
                    color="red", s=60, label="Down days", ax=ax)

    # Set chart properties
    ax.set_title("Closing Price with Up/Down Markers (Last 100 Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    return fig


# Daily Returns Analysis Functions

def return_for_user_price(df, buy_price):
    """Calculate daily returns based on user's buy price.
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        buy_price (float): User's purchase price
        
    Returns:
        pd.DataFrame: DataFrame with added return columns
    """
    df = df.copy()
    # Calculate return relative to user's buy price
    df['Daily_Return'] = (df['Close'] - buy_price) / buy_price
    df['Percentage(%)'] = df['Daily_Return'] * 100  # Add percentage column
    return df

def validate_daily_returns(df):
    """Calculate daily returns using pandas pct_change (day-to-day returns).
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        pd.DataFrame: DataFrame with added return columns
    """
    df = df.copy()
    # Calculate day-to-day percentage change
    df['Daily_Return'] = df['Close'].pct_change()
    df['Percentage(%)'] = df['Daily_Return'] * 100  # Add percentage column
    return df

def manual_daily_returns(df):
    """Manually calculate daily returns (O(n) complexity).
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        pd.DataFrame: DataFrame with added manual return columns
    """
    df = df.copy()
    close_prices = df['Close'].tolist()
    daily_returns = [None]  # First day has no previous day to compare
    
    # Calculate returns for each day compared to previous day
    for i in range(1, len(close_prices)):
        r_t = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
        daily_returns.append(round(r_t, 6))  # Round to 6 decimal places
    
    df['Manual_Daily_Return'] = daily_returns
    df['Percentage(%)'] = df['Manual_Daily_Return'] * 100  # Add percentage column
    return df

def run_daily_returns_analysis(df):
    """Run comprehensive daily returns analysis and validation.
    
    Args:
        df (pd.DataFrame): Stock data with Close prices
        
    Returns:
        dict: Analysis results and validation status
    """
    # Calculate returns using different methods
    validated_data = validate_daily_returns(df.copy())
    manual_data = manual_daily_returns(df.copy())
    
    # Check if validations passed (compare manual vs pandas)
    validation_passed = validated_data['Daily_Return'].round(6).equals(
        manual_data['Manual_Daily_Return'].round(6)
    )
    
    return {
        'validated_data': validated_data,
        'manual_data': manual_data,
        'validation_passed': validation_passed,
        'latest_validated': validated_data.iloc[-1][['Close', 'Daily_Return', 'Percentage(%)']],
        'latest_manual': manual_data.iloc[-1][['Close', 'Manual_Daily_Return', 'Percentage(%)']]
    }
def main():
    """Main function to run the Streamlit application."""
    st.title("📈 Stock Data Downloader & Analyzer")
    st.write("This application downloads historical stock data from Yahoo Finance and provides analysis tools.")
    
    # Initialize session state variables to preserve state across reruns
    if 'valid_symbol' not in st.session_state:
        st.session_state.valid_symbol = False  # Track if current symbol is valid
    if 'ticker' not in st.session_state:
        st.session_state.ticker = ""  # Store current ticker symbol
    if 'data_downloaded' not in st.session_state:
        st.session_state.data_downloaded = False  # Track if data was downloaded
    
    # Define base directory for storing CSV files
    base_dir = r"C:\Users\Goh Jia Xin\Downloads\Project-1-Stock-Market-Trend-Analysis-main (1)\Project-1-Stock-Market-Trend-Analysis-main"

    
    # Create tabs for different functionalities 
    tab1, tab2, tab3 = st.tabs(["Download Data", "Basic Analysis", "Advanced Analysis"])
    
    with tab1:
        st.header("Download Stock Data")
        
        # Step 1: User enters stock ticker symbol
        ticker = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):", 
                              value=st.session_state.ticker, key="download_ticker").upper()  # Convert to uppercase
        
        # Button to validate stock symbol
        if st.button("Check Symbol", key="check_symbol"):
            if not isinstance(ticker, str) or ticker.strip() == "":
                st.error("Please enter a valid string as stock symbol.")
            else:
                st.session_state.ticker = ticker  # Store in session state
                if validate_with_yf(ticker):
                    st.session_state.valid_symbol = True
                    st.success(f"'{ticker}' is a valid stock symbol!")
                else:
                    st.session_state.valid_symbol = False
                    st.error(f"'{ticker}' does not exist on Yahoo Finance. Please try again.")
        
        # Step 2: If valid ticker, allow user to specify years of data
        if st.session_state.valid_symbol:
            years = st.number_input(
                "Enter number of years of data to download (minimum 3):",
                min_value=3, max_value=10, value=3, step=1, key="download_years"
            )
            
            # Define filename for saving data
            filename = os.path.join(base_dir, f"{st.session_state.ticker}_{years}Y.csv")
            
            # Button to initiate data download
            if st.button("Download Stock Data", key="download_data"):
                with st.spinner(f"Downloading {years} years of data for {st.session_state.ticker}..."):
                    try:
                        # Download and clean data
                        stock_df = download_stock_data(st.session_state.ticker, years=years, save_path=filename)
                        stock_df = clean_csv(filename)
                        st.success("Data downloaded and cleaned successfully!")
                        st.session_state.data_downloaded = True
                        
                        # Display sample data
                        st.subheader("Sample Data:")
                        st.dataframe(stock_df.head())
                        
                        # Display basic statistics
                        st.subheader("Basic Statistics:")
                        st.dataframe(stock_df.describe())
                        
                        st.info(f"Data saved to: {filename}")
                        
                        # Provide download button for the CSV file
                        with open(filename, "rb") as file:
                            st.download_button(
                                label="Download CSV",
                                data=file,
                                file_name=f"{st.session_state.ticker}_{years}Y.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                    except Exception as e:
                        st.error(f"Error downloading data: {str(e)}")
    
    with tab2:
        st.header("Basic Stock Analysis")
        st.write("Perform basic technical analysis including moving averages, runs analysis, and chart visualizations.")
        
        # Check if there are CSV files available for analysis
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.warning("No CSV files found. Please download data first using the 'Download Data' tab.")
        else:
            # Let user select file and analysis options
            filepath, selected_analysis = select_and_analyze_csv(base_dir, "basic")
            
            if filepath and selected_analysis:
                # Load and prepare data for analysis
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
                df.set_index('Date', inplace=True)  # Set date as index
                
                st.success(f"Loaded data from {os.path.basename(filepath)}")
                st.dataframe(df.head())  # Show first few rows
                
                # Perform selected analyses based on user choices
                
                if "Line Chart" in selected_analysis:
                    st.subheader("Line Chart - Close Prices")
                    fig = display_line_chart_for_close_prices(df)
                    st.pyplot(fig)  # Display chart in Streamlit
                
                if "Candlestick Chart" in selected_analysis:
                    st.subheader("Candlestick Chart")
                    fig = display_candlesticks(df)
                    st.pyplot(fig)
                
                if "SMA" in selected_analysis:
                    st.subheader("Simple Moving Average (SMA) Analysis")
                    
                    # Calculate moving averages
                    ma_data = calculate_moving_averages(df)
                    
                    # Plot SMAs with close price
                    fig = display_close_and_sma_chart(df, ma_data)
                    st.pyplot(fig)
                    
                    # Generate trading signals based on SMA crossovers
                    df_sma = df.join(ma_data)  # Combine with original data
                    df_sma['Signal'] = 0  # Initialize signals
                    # Buy signal when short-term MA crosses above medium-term MA
                    df_sma.loc[df_sma['MA_20'] > df_sma['MA_50'], 'Signal'] = 1
                    # Sell signal when short-term MA crosses below medium-term MA  
                    df_sma.loc[df_sma['MA_20'] < df_sma['MA_50'], 'Signal'] = -1
                    
                    st.write("SMA Crossover Signals:")
                    st.dataframe(df_sma[['Close', 'MA_20', 'MA_50', 'Signal']].tail(10))
                    
                    # Validate SMA calculations
                    st.write("SMA Validation (comparing manual vs pandas calculation):")
                    validation_results = validate_sma(df)
                    for window, is_valid in validation_results.items():
                        st.write(f"Window {window}: {'✓ Valid' if is_valid else '✗ Invalid'}")
                    
                    # Display sample SMA values from manual calculation
                    st.write("Sample SMA Values (Manual Calculation):")
                    manual_sma = calculate_sma(df)
                    for window, values in manual_sma.items():
                        if values:  # Only show if there are values
                            st.write(f"Window {window}: First 5 values: {values[:5]}")
                
                if "Seaborn SMA Chart" in selected_analysis:
                    st.subheader("Seaborn SMA Chart")
                    
                    # Let user select SMA window size
                    sma_window = st.slider("Select SMA Window:", min_value=5, max_value=100, value=20, step=5, key="seaborn_sma")
                    
                    # Create and display Seaborn chart
                    fig = plot_close_sma(df, sma_window)
                    st.pyplot(fig)
                
                if "Seaborn Runs Chart" in selected_analysis:
                    st.subheader("Seaborn Runs Chart - Up/Down Days")
                    
                    # Create and display runs visualization
                    fig = plot_runs(df)
                    st.pyplot(fig)
                
                if "Daily Returns" in selected_analysis:
                    st.subheader("Daily Returns Analysis")
                    
                    # Calculate daily percentage returns
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
                    
                    # Display return statistics
                    st.write("Daily Returns Statistics:")
                    st.write(f"Average Daily Return: {df['Daily_Return'].mean():.2f}%")
                    st.write(f"Standard Deviation: {df['Daily_Return'].std():.2f}%")
                    st.write(f"Maximum Daily Gain: {df['Daily_Return'].max():.2f}%")
                    st.write(f"Maximum Daily Loss: {df['Daily_Return'].min():.2f}%")
                
                if "Upward and Downward Runs" in selected_analysis:
                    st.subheader("Upward and Downward Runs Analysis")
                    
                    # Analyze consecutive price movements
                    upward_runs, downward_runs = analyze_upward_downward_runs(df.reset_index())
                    
                    # Create a DataFrame to display runs analysis in table format
                    runs_data = []
                    
                    # Add upward runs statistics
                    if upward_runs:
                        runs_data.append({
                            "Run Type": "Upward",
                            "Total Runs": len(upward_runs),
                            "Longest Run (days)": max(upward_runs),
                            "Average Run (days)": round(sum(upward_runs)/len(upward_runs), 2),
                            "Shortest Run (days)": min(upward_runs),
                            "Total Days in Runs": sum(upward_runs)
                        })
                    else:
                        runs_data.append({
                            "Run Type": "Upward",
                            "Total Runs": 0,
                            "Longest Run (days)": "N/A",
                            "Average Run (days)": "N/A",
                            "Shortest Run (days)": "N/A",
                            "Total Days in Runs": 0
                        })
                    
                    # Add downward runs statistics
                    if downward_runs:
                        runs_data.append({
                            "Run Type": "Downward",
                            "Total Runs": len(downward_runs),
                            "Longest Run (days)": max(downward_runs),
                            "Average Run (days)": round(sum(downward_runs)/len(downward_runs), 2),
                            "Shortest Run (days)": min(downward_runs),
                            "Total Days in Runs": sum(downward_runs)
                        })
                    else:
                        runs_data.append({
                            "Run Type": "Downward",
                            "Total Runs": 0,
                            "Longest Run (days)": "N/A",
                            "Average Run (days)": "N/A",
                            "Shortest Run (days)": "N/A",
                            "Total Days in Runs": 0
                        })
                    
                    # Convert to DataFrame and display
                    runs_df = pd.DataFrame(runs_data)
                    st.dataframe(runs_df, use_container_width=True)
                    
                    # Additional detailed view of individual runs (optional)
                    st.subheader("Detailed Run Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Upward Runs Distribution**")
                        if upward_runs:
                            # Create frequency table for upward runs
                            upward_counts = pd.Series(upward_runs).value_counts().sort_index()
                            upward_dist_df = pd.DataFrame({
                                'Run Length (days)': upward_counts.index,
                                'Frequency': upward_counts.values
                            })
                            st.dataframe(upward_dist_df, use_container_width=True)
                            
                            # Display as bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            upward_dist_df.plot(kind='bar', x='Run Length (days)', y='Frequency', 
                                              ax=ax, color='green', alpha=0.7)
                            ax.set_title('Distribution of Upward Run Lengths')
                            ax.set_ylabel('Frequency')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.write("No upward runs found")
                    
                    with col2:
                        st.write("**Downward Runs Distribution**")
                        if downward_runs:
                            # Create frequency table for downward runs
                            downward_counts = pd.Series(downward_runs).value_counts().sort_index()
                            downward_dist_df = pd.DataFrame({
                                'Run Length (days)': downward_counts.index,
                                'Frequency': downward_counts.values
                            })
                            st.dataframe(downward_dist_df, use_container_width=True)
                            
                            # Display as bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            downward_dist_df.plot(kind='bar', x='Run Length (days)', y='Frequency', 
                                                ax=ax, color='red', alpha=0.7)
                            ax.set_title('Distribution of Downward Run Lengths')
                            ax.set_ylabel('Frequency')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.write("No downward runs found")
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    if upward_runs and downward_runs:
                        total_runs = len(upward_runs) + len(downward_runs)
                        upward_percentage = (len(upward_runs) / total_runs) * 100
                        downward_percentage = (len(downward_runs) / total_runs) * 100
                        
                        summary_data = {
                            "Metric": ["Total Runs", "Upward Runs Percentage", "Downward Runs Percentage", 
                                      "Overall Average Run Length", "Market Bias (Up/Down Ratio)"],
                            "Value": [total_runs, 
                                    f"{upward_percentage:.1f}%", 
                                    f"{downward_percentage:.1f}%",
                                    f"{(sum(upward_runs) + sum(downward_runs)) / total_runs:.2f} days",
                                    f"{len(upward_runs)/len(downward_runs):.2f}" if downward_runs else "Infinite"]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
    
    with tab3:
        st.header("Advanced Analysis")
        st.write("Comprehensive analysis including trend analysis, streak identification, and maximum profit calculations.")
        
        # Check if there are CSV files available for analysis
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.warning("No CSV files found. Please download data first using the 'Download Data' tab.")
        else:
            # Let user select file and analysis options
            filepath, selected_analysis = select_and_analyze_csv(base_dir, "advanced")
            
            if filepath and selected_analysis:
                # Load and prepare data for analysis
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
                df.set_index('Date', inplace=True)  # Set date as index
                
                st.success(f"Loaded data from {os.path.basename(filepath)}")
                
                # TREND ANALYSIS SECTION
                if "Trend Analysis" in selected_analysis:
                    st.subheader("📊 Trend Analysis and Validation")
                    
                    # Extract ticker symbol from filename for display
                    ticker_symbol = os.path.basename(filepath).split('_')[0]
                    
                    # Run trend analysis
                    trend_results = run_trend_analysis(df, ticker_symbol)
                    
                    # Display trend statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Manual Calculation Results:**")
                        st.write(f"Upward Trends: {trend_results['manual_upward']}")
                        st.write(f"Downward Trends: {trend_results['manual_downward']}")
                        st.write(f"Longest Upward Streak: {trend_results['longest_up_streak']} days")
                        st.write(f"Longest Downward Streak: {trend_results['longest_down_streak']} days")
                    
                    with col2:
                        st.write("**Pandas Validation Results:**")
                        st.write(f"Upward Trends: {trend_results['validated_upward']}")
                        st.write(f"Downward Trends: {trend_results['validated_downward']}")
                    
                    # Display validation results
                    st.write("**Validation Results:**")
                    if trend_results['upward_match']:
                        st.success("✅ Upward trends match!")
                    else:
                        st.error("❌ Upward trends don't match!")
                    
                    if trend_results['downward_match']:
                        st.success("✅ Downward trends match!")
                    else:
                        st.error("❌ Downward trends don't match!")
                    
                    if trend_results['all_validations_passed']:
                        st.success("🎉 All trend validations passed!")
                        
                        # Display trend candlestick chart
                        st.subheader("Trend Candlestick Chart")
                        fig = plot_trend_candlestick(
                            trend_results['validated_data'], 
                            trend_results['manual_results'], 
                            ticker_symbol
                        )
                        st.pyplot(fig)
                        
                        # Display recent trend data
                        st.write("**Recent Trend Data (Last 30 days):**")
                        st.dataframe(trend_results['validated_data'][['Close', 'Upward_Trend', 'Downward_Trend']].tail(30))
                    else:
                        st.warning("Trend validation failed. Please check the data.")
                #MULTIPLE TRANSACTIONS MAX PROFIT
                if "Multiple Transactions Max Profit" in selected_analysis:
                    st.subheader("💰 Multiple Transactions Max Profit (Buy-Sell Stock II)")
    
                    st.info("""
                    **Strategy**: Buy whenever the price increases from the previous day and sell when it decreases.
                    This captures all upward movements in the stock price.
                    """)
    
                    # Run the analysis
                    validation_results = validate_multiple_transactions(df)
    
                    # Display results
                    col1, col2 = st.columns(2)
    
                    with col1:
                        st.write("**Complex Method (with transaction details):**")
                        st.write(f"Total Profit: ${validation_results['complex_method']['total_profit']:.2f}")
                        st.write(f"Number of Transactions: {validation_results['complex_method']['num_transactions']}")
    
                    with col2:
                        st.write("**Simple Method (sum of positive differences):**")
                        st.write(f"Total Profit: ${validation_results['simple_method']['total_profit']:.2f}")
                    # Validation result
                    if validation_results['validation_passed']:
                        st.success("✅ Both methods produce identical results!")
                    else:
                        st.error("❌ Methods produced different results!")    
                    # Display transactions if any
                    if validation_results['complex_method']['transactions']:
                        st.subheader("Transaction Details")
        
                        transactions_data = []
                        for i, (buy_date, sell_date, profit) in enumerate(validation_results['complex_method']['transactions'], 1):
                            buy_price = df.loc[df.index == buy_date, 'Close'].iloc[0]
                            sell_price = df.loc[df.index == sell_date, 'Close'].iloc[0]
            
                            transactions_data.append({
                                'Transaction': i,
                                'Buy Date': buy_date.strftime('%Y-%m-%d'),
                                'Buy Price': f"${buy_price:.2f}",
                                'Sell Date': sell_date.strftime('%Y-%m-%d'),
                                'Sell Price': f"${sell_price:.2f}",
                                'Profit': f"${profit:.2f}",
                                'Return %': f"{(profit/buy_price)*100:.2f}%"
                            })
        
                        transactions_df = pd.DataFrame(transactions_data)
                        st.dataframe(transactions_df, use_container_width=True)
                        # Visualization
                        st.subheader("Transaction Visualization")
                        fig, ax = plt.subplots(figsize=(14, 7))
        
                        # Plot price
                        ax.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
        
                        # Mark buy and sell points
                        for buy_date, sell_date, profit in validation_results['complex_method']['transactions']:
                            buy_price = df.loc[df.index == buy_date, 'Close'].iloc[0]
                            sell_price = df.loc[df.index == sell_date, 'Close'].iloc[0]
            
                            ax.scatter(buy_date, buy_price, color='green', s=100, zorder=5, marker='^')
                            ax.scatter(sell_date, sell_price, color='red', s=100, zorder=5, marker='v')
                            # Add arrows connecting buy to sell
                            ax.annotate('', xy=(sell_date, sell_price), xytext=(buy_date, buy_price),
                            arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.7))
        
                        ax.set_title("Multiple Transactions Buy/Sell Points")
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    else: 
                        st.warning("No profitable transactions found in this period.")
                # MAX PROFIT ANALYSIS SECTION  
                if "Max Profit Calculations" in selected_analysis:
                    st.subheader("💰 Maximum Profit Analysis")
                    
                    # Let user choose which algorithm to use
                    algorithm_choice = st.selectbox(
                        "Select profit calculation algorithm:",
                        ["Single Pass (O(n))", "Vectorized (O(n))", "Brute Force (O(n²))", "Validate All Methods"],
                        key="profit_algorithm"
                    )
                    
                    if algorithm_choice == "Single Pass (O(n))":
                        # Use the original single pass algorithm
                        buy_date, sell_date, max_profit = single_pass_max_profit(df)
                        method_name = "Single Pass Algorithm"
                        
                    elif algorithm_choice == "Vectorized (O(n))":
                        # Use the vectorized numpy algorithm
                        buy_date, sell_date, max_profit = vectorized_max_profit(df)
                        method_name = "Vectorized Algorithm"
                        
                    elif algorithm_choice == "Brute Force (O(n²))":
                        # Use brute force (warning for large datasets)
                        if len(df) > 1000:
                            st.warning("Brute force algorithm may be slow for large datasets. Consider using a different method.")
                        
                        buy_date, sell_date, max_profit = brute_force_max_profit(df)
                        method_name = "Brute Force Algorithm"
                        
                    else:  # Validate All Methods
                        st.subheader("Algorithm Validation Results")
                        validation_results = validate_max_profit_calculations(df)
                        
                        # Display results from all methods
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Single Pass (O(n))**")
                            bd, sd, profit = validation_results['single_pass']
                            if bd:
                                st.write(f"Buy: {bd.strftime('%Y-%m-%d')}")
                                st.write(f"Sell: {sd.strftime('%Y-%m-%d')}")
                                st.write(f"Profit: ${profit:.2f}")
                        
                        with col2:
                            st.write("**Vectorized (O(n))**")
                            bd, sd, profit = validation_results['vectorized']
                            if bd:
                                st.write(f"Buy: {bd.strftime('%Y-%m-%d')}")
                                st.write(f"Sell: {sd.strftime('%Y-%m-%d')}")
                                st.write(f"Profit: ${profit:.2f}")
                        
                        with col3:
                            st.write("**Brute Force (O(n²))**")
                            bd, sd, profit = validation_results['brute_force']
                            if bd:
                                st.write(f"Buy: {bd.strftime('%Y-%m-%d')}")
                                st.write(f"Sell: {sd.strftime('%Y-%m-%d')}")
                                st.write(f"Profit: ${profit:.2f}")
                        
                        # Show validation result
                        if validation_results['all_equal']:
                            st.success("✅ All algorithms produced identical results!")
                        else:
                            st.error("❌ Algorithms produced different results!")
                        
                        # For the main display, use vectorized results
                        buy_date, sell_date, max_profit = validation_results['vectorized']
                        method_name = "Vectorized Algorithm (Validation Mode)"
                    
                    # Display results (unless we're in validation mode and results differ)
                    if algorithm_choice != "Validate All Methods" or (algorithm_choice == "Validate All Methods" and validation_results['all_equal']):
                        if max_profit > 0:
                            # Get actual prices at buy/sell dates
                            buy_price = df.loc[df.index == buy_date, 'Close'].iloc[0]
                            sell_price = df.loc[df.index == sell_date, 'Close'].iloc[0]
                            
                            # Display results
                            st.write(f"**Method used:** {method_name}")
                            st.write(f"Best buy date: {buy_date.strftime('%Y-%m-%d')} at ${buy_price:.2f}")
                            st.write(f"Best sell date: {sell_date.strftime('%Y-%m-%d')} at ${sell_price:.2f}")
                            st.write(f"Maximum profit: ${max_profit:.2f} per share ({max_profit/buy_price*100:.2f}% return)")
                            
                            # Visualize the optimal buy/sell points
                            fig, ax = plt.subplots(figsize=(14, 7))
                            ax.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.7)
                            ax.scatter([buy_date, sell_date], [buy_price, sell_price], 
                                     color=['green', 'red'], s=100, zorder=5)
                            ax.annotate('Buy', xy=(buy_date, buy_price), xytext=(buy_date, buy_price*0.95),
                                       arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
                            ax.annotate('Sell', xy=(sell_date, sell_price), xytext=(sell_date, sell_price*1.05),
                                       arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12)
                            ax.set_title(f"Optimal Buy/Sell Points\nProfit: ${max_profit:.2f} ({max_profit/buy_price*100:.1f}%)")
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Price')
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.write("No profitable trading opportunity found in this period")

                # DAILY RETURNS ANALYSIS SECTION
                if "Daily Returns Analysis" in selected_analysis:
                    st.subheader("📊 Daily Returns Analysis")
                    
                    # Option for user to input their buy price
                    user_choice = st.radio(
                        "Choose analysis type:",
                        ["Standard Daily Returns (Day-to-Day)", "Returns Based on My Buy Price"],
                        key="returns_choice"
                    )
                    
                    if user_choice == "Returns Based on My Buy Price":
                        # User inputs their purchase price
                        buy_price = st.number_input(
                            "Enter your buy price:",
                            min_value=0.01,
                            value=float(df['Close'].iloc[0]),  # Default to first available price
                            step=0.01,
                            key="user_buy_price"
                        )
                        
                        if buy_price > 0:
                            # Calculate returns based on user's buy price
                            user_return_df = return_for_user_price(df.copy(), buy_price)
                            
                            # Display latest return information
                            latest_row = user_return_df.iloc[-1]
                            st.write("**Latest Return Based on Your Buy Price:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${latest_row['Close']:.2f}")
                            with col2:
                                st.metric("Your Buy Price", f"${buy_price:.2f}")
                            with col3:
                                return_pct = latest_row['Percentage(%)']
                                st.metric("Total Return", f"{return_pct:.2f}%")
                            
                            # Plot cumulative returns
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(user_return_df.index, user_return_df['Percentage(%)'], 
                                   color='purple', linewidth=2)
                            ax.set_title(f"Cumulative Returns Based on ${buy_price:.2f} Buy Price")
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Return (%)")
                            ax.grid(True, alpha=0.3)
                            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                            st.pyplot(fig)
                            
                            # Show recent data
                            st.write("**Recent Returns Data:**")
                            st.dataframe(user_return_df[['Close', 'Daily_Return', 'Percentage(%)']].tail(10))
                    
                    else:  # Standard Daily Returns
                        # Run validation analysis
                        returns_results = run_daily_returns_analysis(df)
                        
                        # Display validation results
                        st.write("**Daily Returns Validation Results:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Pandas pct_change() Results (First 5 rows):**")
                            st.dataframe(returns_results['validated_data'][['Close', 'Daily_Return', 'Percentage(%)']].head())
                        
                        with col2:
                            st.write("**Manual Calculation Results (First 5 rows):**")
                            st.dataframe(returns_results['manual_data'][['Close', 'Manual_Daily_Return', 'Percentage(%)']].head())
                        
                        # Show validation status
                        if returns_results['validation_passed']:
                            st.success("✅ Daily returns validation passed! Both methods yield identical results.")
                        else:
                            st.error("❌ Daily returns validation failed! Methods produced different results.")
                        
                        # Display latest values comparison
                        st.write("**Latest Values Comparison:**")
                        comparison_data = {
                            'Method': ['Pandas pct_change', 'Manual Calculation'],
                            'Close Price': [returns_results['latest_validated']['Close'], 
                                          returns_results['latest_manual']['Close']],
                            'Daily Return': [returns_results['latest_validated']['Daily_Return'], 
                                           returns_results['latest_manual']['Manual_Daily_Return']],
                            'Percentage': [returns_results['latest_validated']['Percentage(%)'], 
                                         returns_results['latest_manual']['Percentage(%)']]
                        }
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        
                        # Plot daily returns
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # Plot 1: Daily returns over time
                        ax1.plot(returns_results['validated_data'].index, 
                                returns_results['validated_data']['Daily_Return'] * 100, 
                                color='blue', alpha=0.7, linewidth=1)
                        ax1.set_title("Daily Returns Over Time")
                        ax1.set_ylabel("Daily Return (%)")
                        ax1.grid(True, alpha=0.3)
                        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                        
                        # Plot 2: Histogram of daily returns
                        daily_returns = returns_results['validated_data']['Daily_Return'].dropna() * 100
                        ax2.hist(daily_returns, bins=50, color='green', alpha=0.7, edgecolor='black')
                        ax2.set_title("Distribution of Daily Returns")
                        ax2.set_xlabel("Daily Return (%)")
                        ax2.set_ylabel("Frequency")
                        ax2.grid(True, alpha=0.3)
                        ax2.axvline(x=daily_returns.mean(), color='red', linestyle='--', 
                                   label=f'Mean: {daily_returns.mean():.2f}%')
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display statistics
                        st.write("**Daily Returns Statistics:**")
                        stats_data = {
                            'Statistic': ['Mean', 'Standard Deviation', 'Maximum Gain', 'Maximum Loss', 
                                         'Positive Days', 'Negative Days', 'Zero Change Days'],
                            'Value': [f"{daily_returns.mean():.4f}%",
                                    f"{daily_returns.std():.4f}%",
                                    f"{daily_returns.max():.4f}%",
                                    f"{daily_returns.min():.4f}%",
                                    f"{(daily_returns > 0).sum()} days",
                                    f"{(daily_returns < 0).sum()} days", 
                                    f"{(daily_returns == 0).sum()} days"]
                        }
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)            
                 # MANUAL SMA CALCULATIONS & VALIDATION SECTION
            if "Manual SMA Calculations & Validation" in selected_analysis:
                st.subheader("📊 Manual SMA Calculations & Validation")
                
                # Let user select SMA window sizes
                st.write("**Select SMA Window Sizes:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    window1 = st.number_input("Window 1", min_value=5, max_value=500, value=20, key="sma_window1")
                with col2:
                    window2 = st.number_input("Window 2", min_value=5, max_value=500, value=50, key="sma_window2")
                with col3:
                    window3 = st.number_input("Window 3", min_value=5, max_value=500, value=200, key="sma_window3")
                
                windows = [window1, window2, window3]
                
                # Run SMA validation
                st.write("**SMA Validation Results:**")
                validation_results = validate_sma(df, windows)
                
                # Display validation status for each window
                validation_passed = True
                for window, is_valid in validation_results.items():
                    if is_valid:
                        st.success(f"✅ Window {window}: Manual SMA matches pandas rolling mean within tolerance.")
                    else:
                        st.error(f"❌ Window {window}: Manual SMA differs from pandas rolling mean.")
                        validation_passed = False
                
                if validation_passed:
                    st.success("🎉 All SMA validations passed! Manual calculation matches pandas rolling mean.")
                else:
                    st.warning("Some SMA validations failed. Check the calculations.")
                
                # Display sample SMA values from manual calculation
                st.write("**Sample SMA Values (Manual O(n) Calculation):**")
                manual_sma = calculate_sma(df, windows)
                
                for window, values in manual_sma.items():
                    if values:  # Only show if there are values
                        st.write(f"**Window {window}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"First 5 values: {values[:5]}")
                        with col2:
                            st.write(f"Last 5 values: {values[-5:]}")
                    else:
                        st.write(f"**Window {window}:** No data (window larger than dataset)")
                
                # Display chart with SMAs
                st.write("**Chart with SMAs:**")
                ma_data = calculate_moving_averages(df, windows)
                
                # Create the chart
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=2)
                
                # Define colors for different moving averages
                colors = ["green", "red", "orange", "purple", "brown"]
                for i, column in enumerate(ma_data.columns):
                    if i < len(colors):
                        ax.plot(ma_data.index, ma_data[column], label=column, color=colors[i], linewidth=2)
                    else:
                        ax.plot(ma_data.index, ma_data[column], label=column, linewidth=2)
                
                ax.set_title(f"Stock Close Price with SMAs (Manual Validation: {'PASSED' if validation_passed else 'FAILED'})")
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Performance comparison
                st.write("**Performance Information:**")
                st.info("""
                - **Manual O(n) Algorithm**: Processes each window separately with O(n) complexity
                - **Pandas Rolling**: Uses optimized rolling window calculations
                - Both methods should produce identical results when validated correctly
                """)
            
        
# Standard Python idiom to run the main function when script is executed directly
if __name__ == "__main__":
    main()
