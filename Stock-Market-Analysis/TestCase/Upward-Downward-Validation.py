from unittest import TestCase, main  # Import base class for tests and test runner
from unittest.mock import patch, MagicMock  # Import patch to mock functions and MagicMock objects
import pandas as pd  # Import pandas for handling dataframes

# Import functions to test from your main module
from Upward_Downward_squidward import user_input_ticker, fetch_stock_data, validate_trend, manual_trend, test_run

# Define a test class inheriting from TestCase
class TestStockTrendAnalysis(TestCase):
    def setUp(self):
        # Prepare sample stock data to use in multiple tests
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')  # Create 10 consecutive dates
        self.sample_data = pd.DataFrame({
            'Close': [100, 101, 100, 99, 98, 97, 98, 99, 100, 99],  # Sample closing prices
            'Open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],  # Sample opening prices
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # High prices
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],  # Low prices
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]  # Trading volume
        }, index=dates)  # Assign dates as DataFrame index

    def test_validate_trend(self):
        """Test validate_trend function correctly flags upward/downward movements."""
        validated = validate_trend(self.sample_data.copy())  # Run trend validation
        # Expected results for upward trend by comparing the sample close prices
        expected_up = [False, True, False, False, False, False, True, True, True, False]
        # Expected results for downward trend by comparing the sample close prices
        expected_down = [False, False, True, True, True, True, False, False, False, True]
        # Check that the function returns correct upward trend flags
        self.assertListEqual(validated['Upward_Trend'].tolist(), expected_up)
        # Check that the function returns correct downward trend flags
        self.assertListEqual(validated['Downward_Trend'].tolist(), expected_down)

    def test_manual_trend(self):
        """Test manual_trend function calculates correct trend counts and streaks."""
        # Run manual trend analysis
        upward, downward, long_up, long_down, up_start, up_end, down_start, down_end = manual_trend(self.sample_data)
        # Assert number of upward moves
        self.assertEqual(upward, 4)  # Up moves: 100->101, 97->98, 98->99, 99->100
        # Assert number of downward moves
        self.assertEqual(downward, 5)  # Down moves: 101->100, 100->99, 99->98, 98->97, 100->99
        # Assert length of longest upward streak
        self.assertEqual(long_up, 3)  # Longest up streak: 97->98->99->100 (indices 5-8)
        # Assert length of longest downward streak
        self.assertEqual(long_down, 4)  # Longest down streak: 101->100->99->98->97 (indices 1-5)
        # Assert start and end indices of longest upward streak
        self.assertEqual(up_start, 5)
        self.assertEqual(up_end, 8)
        # Assert start and end indices of longest downward streak
        self.assertEqual(down_start, 1)
        self.assertEqual(down_end, 5)

    @patch('builtins.input', return_value='aapl')  # Mock input() to always return 'aapl'
    def test_user_input_ticker(self, mock_input):
        """Test user_input_ticker returns uppercase ticker symbol."""
        ticker = user_input_ticker()  # Call the function
        self.assertEqual(ticker, 'AAPL')  # Should convert input to uppercase

    @patch('yfinance.Ticker')  # Mock Ticker class from yfinance
    def test_fetch_stock_data(self, mock_ticker):
        """Test fetch_stock_data returns correct DataFrame structure."""
        # Prepare fake historical data
        mock_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        mock_ticker.return_value.history.return_value = mock_hist  # Mock history() to return fake data

        result = fetch_stock_data('AAPL', period='3d', interval='1d')  # Call function
        self.assertEqual(len(result), 3)  # Check number of rows
        # Ensure all required columns exist
        self.assertTrue(all(col in result.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume']))
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))  # Index should be datetime

    @patch('builtins.input', return_value='aapl')  # Mock input for ticker
    @patch('Upward_Downward_squidward.Ticker')  # Mock Ticker for yfinance
    @patch('mplfinance.plot')  # Mock plotting function to avoid actual plotting
    @patch('builtins.print')  # Mock print to capture print outputs
    def test_test_run(self, mock_print, mock_plot, mock_ticker, mock_input):
        """Test test_run function with mocked dependencies."""
        mock_hist = self.sample_data  # Use sample data as historical data
        mock_ticker.return_value.history.return_value = mock_hist  # Mock history() return

        test_run()  # Run the full analysis function

        mock_plot.assert_called_once()  # Check that plotting was called once

        # Capture printed messages
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        # Check that key messages are printed
        self.assertIn('Fetched 10 rows of data for AAPL.', calls)
        self.assertIn('All trend validations passed!', calls)

# Run all tests if script is executed directly
if __name__ == '__main__':
    main()
