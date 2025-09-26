import unittest  # Import Python's built-in unit testing framework
import pandas as pd  # Import pandas for data handling
from unittest.mock import patch  # Import patch to mock functions during tests
from SMA_validation import calculate_sma, fetch_stock_data, validate_sma  # Import functions to be tested


# Unit test class for stock analysis functions
class TestStockAnalysis(unittest.TestCase):
    def setUp(self):
        # Prepare sample stock data to use in tests
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')  # 10 daily dates
        self.sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],  # Close prices
            'Open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],  # Open prices
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # High prices
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],  # Low prices
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]  # Volume
        }, index=dates)  # Set dates as index

    def test_calculate_sma_valid_windows(self):
        """Test SMA calculation for valid window sizes."""
        windows = [3, 5]  # Define windows to test
        result = calculate_sma(self.sample_data, windows)  # Calculate SMA
        expected_sma_3 = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]  # Expected 3-day SMA
        expected_sma_5 = [102.0, 103.0, 104.0, 105.0, 106.0, 107.0]  # Expected 5-day SMA
        self.assertEqual(result[3], expected_sma_3)  # Assert SMA matches expected
        self.assertEqual(result[5], expected_sma_5)

    def test_calculate_sma_window_larger_than_data(self):
        """Test SMA calculation when window size exceeds data length."""
        windows = [20]  # Window bigger than data length
        result = calculate_sma(self.sample_data, windows)  # Calculate SMA
        self.assertEqual(result[20], [], "SMA for window larger than data should return empty list")  # Check empty result

    def test_calculate_sma_single_value(self):
        """Test SMA calculation with a window size of 1."""
        windows = [1]  # Window of size 1
        result = calculate_sma(self.sample_data, windows)  # Calculate SMA
        expected = [float(x) for x in self.sample_data['Close']]  # SMA with window=1 is original prices
        self.assertEqual(result[1], expected, "SMA with window=1 should match original prices")  # Assert match

    def test_validate_sma(self):
        """Test SMA validation against pandas rolling mean."""
        windows = [3, 5]  # Test windows
        result = validate_sma(self.sample_data, windows)  # Validate SMA
        self.assertTrue(result, "Manual SMA should match pandas rolling mean")  # Assert True

    @patch('yfinance.Ticker')  # Mock Ticker class to avoid real API calls
    def test_fetch_stock_data(self, mock_ticker):
        """Test fetching stock data with mocked yfinance."""
        # Create fake stock data
        mock_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        mock_ticker.return_value.history.return_value = mock_hist  # Mock history method

        result = fetch_stock_data('AAPL', period='3d', interval='1d')  # Call function
        self.assertEqual(len(result), 3, "Fetched data should have 3 rows")  # Check row count
        # Ensure all required columns exist
        self.assertTrue(all(col in result.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume']),
                        "Data should contain required columns")
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex), "Index should be DatetimeIndex")  # Check index type

    def test_calculate_sma_empty_data(self):
        """Test SMA calculation with empty data."""
        empty_data = pd.DataFrame({'Close': []})  # Empty DataFrame
        windows = [3, 5]
        result = calculate_sma(empty_data, windows)  # Calculate SMA
        self.assertEqual(result[3], [], "SMA for empty data should return empty list")
        self.assertEqual(result[5], [], "SMA for empty data should return empty list")

# Run all unit tests when script is executed
if __name__ == '__main__':
    unittest.main()
