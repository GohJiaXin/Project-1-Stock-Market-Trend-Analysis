import yfinance as yf
def download_stock(ticker: str, start: str, end: str, filename: str):
    """
    Download daily stock data from Yahoo Finance and save as CSV.
    Example: download_stock("AAPL", "2023-01-01", "2025-12-31", "apple.csv")
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj_Close",
        "Volume": "Volume"
    })
    df.to_csv(filename)
    print(f"✅ Saved {ticker} data to {filename}")

if __name__ == "__main__":
    # Example: Apple for the past 3 years
    download_stock("AAPL", "2023-01-01", "2025-12-31", "apple.csv")
