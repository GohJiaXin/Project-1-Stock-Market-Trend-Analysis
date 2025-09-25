# Import necessary libraries
import streamlit as st  # For creating web application interface
import yfinance as yf  # For downloading stock data from Yahoo Finance
import pandas as pd  # For data manipulation and analysis
from datetime import datetime, timedelta  # For handling dates and time periods
import os  # For operating system interactions (file paths, directories)
import matplotlib.pyplot as plt  # For creating static visualizations
import mplfinance as mpf  # For creating financial charts (candlesticks)
import numpy as np  # For numerical operations (though not heavily used here)
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

def select_and_analyze_csv(base_dir: str):
    """Let the user pick a CSV file if multiple exist in the directory.
    
    Args:
        base_dir (str): Directory path to search for CSV files
        
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
        selected_file = st.selectbox("Choose a CSV file to analyze:", csv_files)
        filepath = os.path.join(base_dir, selected_file)
        st.info(f"You selected: {selected_file}")
    else:
        # If only one file, use it automatically
        filepath = os.path.join(base_dir, csv_files[0])
        st.success(f"Found 1 CSV file: {csv_files[0]}")
    
    # Analysis options for user to choose from
    st.subheader("📊 Select Analysis Functionality")
    analysis_options = ["SMA", "Upward and Downward Runs", "Daily Returns", 
                       "Max Profit Calculations", "Line Chart", "Candlestick Chart", 
                       "Seaborn SMA Chart", "Seaborn Runs Chart"]
    
    # Multi-select widget for analysis choices
    selected_analysis = st.multiselect(
        "Choose one or more analyses to run:", analysis_options
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

def calculate_max_profit(data):
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
    base_dir = r"C:\Users\desmo\OneDrive\Documents\SIT AC in Fintech\INF1002 (Programming Fundamentals)\Stock Market Analysis"
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Download Data", "Analyze Data"])
    
    with tab1:
        st.header("Download Stock Data")
        
        # Step 1: User enters stock ticker symbol
        ticker = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):", 
                              value=st.session_state.ticker).upper()  # Convert to uppercase
        
        # Button to validate stock symbol
        if st.button("Check Symbol"):
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
                min_value=3, max_value=10, value=3, step=1
            )
            
            # Define filename for saving data
            filename = os.path.join(base_dir, f"{st.session_state.ticker}_{years}Y.csv")
            
            # Button to initiate data download
            if st.button("Download Stock Data"):
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
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error downloading data: {str(e)}")
    
    with tab2:
        st.header("Analyze Stock Data")
        
        # Check if there are CSV files available for analysis
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.warning("No CSV files found. Please download data first using the 'Download Data' tab.")
        else:
            # Let user select file and analysis options
            filepath, selected_analysis = select_and_analyze_csv(base_dir)
            
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
                    
                    # # Validate SMA calculations
                    # st.subheader("📊 SMA Validation Results")
                    # validation_results = validate_sma(df)
                    # for window, is_valid in validation_results.items():
                    #     st.write(f"Window {window}: {'✓ Valid' if is_valid else '✗ Invalid'}")
                    
                    # # Display sample SMA values from manual calculation
                    # st.write("Sample SMA Values (Manual Calculation):")
                    # manual_sma = calculate_sma(df)
                    # for window, values in manual_sma.items():
                    #     if values:  # Only show if there are values
                    #         st.write(f"Window {window}: First 5 values: {values[:5]}")
                
                    # Validate SMA calculations
                    st.subheader("📊 SMA Validation Results")
                    validation_results = validate_sma(df)

                    # Turn results into a DataFrame for a clean table without index
                    validation_df = pd.DataFrame({
                        "Window": list(validation_results.keys()),
                        "Validation": ["✅ Valid" if v else "❎ Invalid" for v in validation_results.values()]
                    })  
                    
                    validation_df["Window"] = validation_df["Window"].astype(str)  # Ensure Window column is string for better display

                    # Display the table with left-aligned "Window" column and no index
                    st.dataframe(
                        validation_df.style.set_properties(subset=["Window"], **{"text-align": "left"}),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Display sample SMA values (manual calculation)
                    st.subheader("🔍 Sample SMA Values (Manual Calculation)")
                    manual_sma = calculate_sma(df)

                    # Prepare table with first 5 values for each window without index
                    sample_data = {
                        "Window": [],
                        "First 5 SMA Values": []
                    }

                    for window, values in manual_sma.items():
                        if values:  # Only show if there are values
                            sample_data["Window"].append(window)
                            sample_data["First 5 SMA Values"].append(values[:5])

                    sample_df = pd.DataFrame(sample_data) 
                    sample_df["Window"] = sample_df["Window"].astype(str)  # Ensure Window column is string for better display
                    
                    # Left-align "Window" column and hide index
                    st.dataframe(
                        sample_df.style.set_properties(subset=["Window"], **{"text-align": "left"}),
                        use_container_width=True,
                        hide_index=True
                    )

                if "Seaborn SMA Chart" in selected_analysis:
                    st.subheader("Seaborn SMA Chart")
                    
                    # Let user select SMA window size
                    sma_window = st.slider("Select SMA Window:", min_value=5, max_value=100, value=20, step=5)
                    
                    # Create and display Seaborn chart
                    fig = plot_close_sma(df, sma_window)
                    st.pyplot(fig)
                
                if "Seaborn Runs Chart" in selected_analysis:
                    st.subheader("Seaborn Runs Chart - Up/Down Days")
                    
                    # Create and display runs visualization
                    fig = plot_runs(df)
                    st.pyplot(fig)
                
                if "Daily Returns" in selected_analysis:
                    st.subheader("📊 Daily Returns Statistics")
                    
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

                    # More user interface friednly metrics display
                    col1, col2, col3, col4 = st.columns(4)
                    avg_return = df['Daily_Return'].mean()
                    std_return = df['Daily_Return'].std()
                    max_gain = df['Daily_Return'].max()
                    max_loss = df['Daily_Return'].min()
                    col1.metric("Average Daily Return", f"{avg_return:.2f}%", delta=None)
                    col2.metric("Std Dev", f"{std_return:.2f}%")
                    col3.metric("Max Gain", f"{max_gain:.2f}%")
                    col4.metric("Max Loss", f"{max_loss:.2f}%")
                
                if "Upward and Downward Runs" in selected_analysis:
                    st.subheader("Upward and Downward Runs Analysis")
                    
                    # Analyze consecutive price movements
                    upward_runs, downward_runs = analyze_upward_downward_runs(df.reset_index())
                    
                    # Display upward runs analysis
                    st.write("📈 Upward Runs Analysis")
                    
                    if upward_runs:
                        up_df = pd.DataFrame({
                        "Metric": ["🔹 Longest run", "🔹 Average run", "🔹 Total upward runs"],
                        "Value": [f"{max(upward_runs)} days",
                                f"{sum(upward_runs)/len(upward_runs):.2f} days",
                                len(upward_runs)]
                    })
                        st.dataframe(
                            up_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No upward runs found")
                    # Display downward runs analysis
                    st.write("📉 Downward Runs Analysis")
                    if downward_runs:
                        down_df = pd.DataFrame({
                        "Metric": ["🔹 Longest run", "🔹 Average run", "🔹 Total downward runs"],
                        "Value": [f"{max(downward_runs)} days",
                                f"{sum(downward_runs)/len(downward_runs):.2f} days",
                                len(downward_runs)]
                    })
                        st.dataframe(
                            down_df,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No downward runs found")
                
                if "Max Profit Calculations" in selected_analysis:
                    st.subheader("Maximum Profit Calculation")
                    
                    # Find optimal buy/sell points for maximum profit
                    buy_date, sell_date, max_profit = calculate_max_profit(df)
                    
                    if max_profit > 0:
                        # Get actual prices at buy/sell dates
                        buy_price = df.loc[df.index == buy_date, 'Close'].iloc[0]
                        sell_price = df.loc[df.index == sell_date, 'Close'].iloc[0]
                        
                        # Display results
                        st.write(f"Best buy date: {buy_date.strftime('%Y-%m-%d')} at ${buy_price:.2f}")
                        st.write(f"Best sell date: {sell_date.strftime('%Y-%m-%d')} at ${sell_price:.2f}")
                        st.write(f"Maximum profit: ${max_profit:.2f} per share ({max_profit/buy_price*100:.2f}% return)")
                    else:
                        st.write("No profitable trading opportunity found in this period")

# Standard Python idiom to run the main function when script is executed directly
if __name__ == "__main__":
    main()
