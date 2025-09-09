import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def validate_with_yf(ticker: str) -> bool:
    """
    Validate stock symbol directly with yfinance.
    Returns True if the ticker has historical data, False otherwise.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1mo")  # Fetch small period for validation
        return not hist.empty
    except Exception:
        return False

def download_stock_data(ticker: str, years: int = 3, save_path: str = None):
    """
    Download historical stock data from Yahoo Finance.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

    # Reset index so Date is a column
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])

    # Save to CSV if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f" Data saved to {save_path}")

    return df

def clean_csv(filepath: str):
    """
    Removes invalid rows (like ,AA,AA,AA,AA,AA) from a CSV file
    and rewrites the cleaned content into the same file.
    """
    # Read CSV while skipping bad lines
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Drop rows where 'Date' is missing
    df = df.dropna(subset=['Date'])

    # Ensure Date column is proper datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where Date could not be parsed
    df = df.dropna(subset=['Date'])

    # Enforce correct dtypes for numeric columns
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs in numeric columns
    df = df.dropna(subset=numeric_cols)

    # Save back to the same file
    df.to_csv(filepath, index=False)

    print(f" Cleaned file saved back to {filepath}")
    return df

def select_and_analyze_csv(base_dir: str):
    """
    Let the user pick a CSV file if multiple exist in the directory.
    If only one file exists, skip selection and ask what analysis to run.
    Returns the filepath and selected analysis options.
    """
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

    if not csv_files:
        st.warning(" No CSV files found in the directory.")
        return None, None

    # Case 1: Multiple CSVs → let user choose
    if len(csv_files) > 1:
        st.subheader(" Select CSV File")
        selected_file = st.selectbox("Choose a CSV file to analyze:", csv_files)
        filepath = os.path.join(base_dir, selected_file)
        st.info(f" You selected: {selected_file}")
        
        # Ask what analysis to run after selecting file
        st.subheader("📊 Select Analysis Functionality")
        analysis_options = ["SMA", "Upward and Downward Runs", "Daily Returns", "Max Profit Calculations"]
        selected_analysis = st.multiselect(
            "Choose one or more analyses to run:",
            analysis_options
        )
        
        return filepath, selected_analysis

    # Case 2: Only one CSV → ask what analysis to run
    elif len(csv_files) == 1:
        filepath = os.path.join(base_dir, csv_files[0])
        st.success(f" Found 1 CSV file: {csv_files[0]}")

        st.subheader("📊 Select Analysis Functionality")
        analysis_options = ["SMA", "Upward and Downward Runs", "Daily Returns", "Max Profit Calculations"]
        selected_analysis = st.multiselect(
            "Choose one or more analyses to run:",
            analysis_options
        )

        return filepath, selected_analysis

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
    base_dir = r"C:\Users\Goh Jia Xin\OneDrive - Singapore Polytechnic\Desktop\Project-1-Stock-Market-Trend-Analysis-main"
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Download Data", "Analyze Data"])
    
    with tab1:
        st.header("Download Stock Data")
        
        # Step 1: User enters stock
        ticker = st.text_input("Enter stock symbol (e.g., AAPL, TSLA, MSFT):", value=st.session_state.ticker).upper()

        if st.button("Check Symbol"):
            if not isinstance(ticker, str) or ticker.strip() == "":
                st.error(" Please enter a valid string as stock symbol.")
            else:
                st.session_state.ticker = ticker
                if validate_with_yf(ticker):
                    st.session_state.valid_symbol = True
                    st.success(f" '{ticker}' is a valid stock symbol!")
                else:
                    st.session_state.valid_symbol = False
                    st.error(f" '{ticker}' does not exist on Yahoo Finance. Please try again.")

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

                        # Clean the CSV immediately after download
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

        # Reset button
        if st.session_state.valid_symbol:
            if st.button("Try a different symbol"):
                st.session_state.valid_symbol = False
                st.session_state.ticker = ""
                st.session_state.data_downloaded = False
                st.experimental_rerun()
    
    with tab2:
        st.header("Analyze Stock Data")
        
        # Check if there are CSV files to analyze
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        
        if not csv_files:
            st.warning("No CSV files found. Please download data first using the 'Download Data' tab.")
        else:
            # Use the select_and_analyze_csv function
            filepath, selected_analysis = select_and_analyze_csv(base_dir)
            
            # if filepath and selected_analysis:
            #     # Load the data
            #     df = pd.read_csv(filepath)
            #     df['Date'] = pd.to_datetime(df['Date'])
                
            #     st.success(f"Loaded data from {os.path.basename(filepath)}")
            #     st.dataframe(df.head())
                
            #     # Perform selected analyses
            #     if "SMA" in selected_analysis:
            #         st.subheader("Simple Moving Average (SMA) Analysis")
                    
            #         # Calculate SMAs
            #         df['SMA_20'] = df['Close'].rolling(window=20).mean()
            #         df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    
            #         # Plot SMAs
            #         st.line_chart(df.set_index('Date')[['Close', 'SMA_20', 'SMA_50']])
                    
            #         # SMA crossover signals
            #         df['Signal'] = 0
            #         df['Signal'][df['SMA_20'] > df['SMA_50']] = 1  # Buy signal
            #         df['Signal'][df['SMA_20'] < df['SMA_50']] = -1  # Sell signal
                    
            #         st.write("SMA Crossover Signals:")
            #         st.dataframe(df[['Date', 'Close', 'SMA_20', 'SMA_50', 'Signal']].tail(10))
                
            #     if "Daily Returns" in selected_analysis:
            #         st.subheader("Daily Returns Analysis")
                    
            #         # Calculate daily returns
            #         df['Daily_Return'] = df['Close'].pct_change() * 100
                    
            #         # Plot daily returns
            #         st.line_chart(df.set_index('Date')['Daily_Return'])
                    
            #         # Statistics
            #         st.write("Daily Returns Statistics:")
            #         st.write(f"Average Daily Return: {df['Daily_Return'].mean():.2f}%")
            #         st.write(f"Standard Deviation: {df['Daily_Return'].std():.2f}%")
            #         st.write(f"Maximum Daily Gain: {df['Daily_Return'].max():.2f}%")
            #         st.write(f"Maximum Daily Loss: {df['Daily_Return'].min():.2f}%")
                
            #     if "Upward and Downward Runs" in selected_analysis:
            #         st.subheader("Upward and Downward Runs Analysis")
                    
            #         # Identify runs
            #         df['Price_Change'] = df['Close'].diff()
            #         df['Direction'] = df['Price_Change'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                    
            #         # Calculate run lengths
            #         current_run = 0
            #         current_direction = 0
            #         runs = []
                    
            #         for i, row in df.iterrows():
            #             if i == 0:
            #                 continue
                            
            #             if row['Direction'] == current_direction:
            #                 current_run += 1
            #             else:
            #                 if current_run > 0:
            #                     runs.append((current_direction, current_run))
            #                 current_run = 1
            #                 current_direction = row['Direction']
                    
            #         if current_run > 0:
            #             runs.append((current_direction, current_run))
                    
            #         # Analyze runs
            #         upward_runs = [length for direction, length in runs if direction == 1]
            #         downward_runs = [length for direction, length in runs if direction == -1]
                    
            #         st.write("Upward Runs Analysis:")
            #         if upward_runs:
            #             st.write(f"Longest upward run: {max(upward_runs)} days")
            #             st.write(f"Average upward run: {sum(upward_runs)/len(upward_runs):.2f} days")
            #         else:
            #             st.write("No upward runs found")
                    
            #         st.write("Downward Runs Analysis:")
            #         if downward_runs:
            #             st.write(f"Longest downward run: {max(downward_runs)} days")
            #             st.write(f"Average downward run: {sum(downward_runs)/len(downward_runs):.2f} days")
            #         else:
            #             st.write("No downward runs found")
                
            #     if "Max Profit Calculations" in selected_analysis:
            #         st.subheader("Maximum Profit Calculation")
                    
            #         # Find the best buy and sell points
            #         min_price = float('inf')
            #         max_profit = 0
            #         buy_date = None
            #         sell_date = None
                    
            #         for i, row in df.iterrows():
            #             if row['Close'] < min_price:
            #                 min_price = row['Close']
            #                 temp_buy_date = row['Date']
                        
            #             profit = row['Close'] - min_price
            #             if profit > max_profit:
            #                 max_profit = profit
            #                 buy_date = temp_buy_date
            #                 sell_date = row['Date']
                    
            #         if max_profit > 0:
            #             st.write(f"Best buy date: {buy_date.strftime('%Y-%m-%d')} at ${min_price:.2f}")
            #             st.write(f"Best sell date: {sell_date.strftime('%Y-%m-%d')} at ${df[df['Date'] == sell_date]['Close'].iloc[0]:.2f}")
            #             st.write(f"Maximum profit: ${max_profit:.2f} per share ({max_profit/min_price*100:.2f}% return)")
            #         else:
            #             st.write("No profitable trading opportunity found in this period")

if __name__ == "__main__":
    main()
