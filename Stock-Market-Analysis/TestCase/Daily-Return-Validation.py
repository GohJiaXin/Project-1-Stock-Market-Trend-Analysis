from unittest import TestCase, main
from unittest.mock import patch  # Used to mock functions or objects during tests
import pandas as pd  # Used for handling data frames

# Import the functions to be tested from daily_return_validation modules
from daily_return_validation import (
    user_input_ticker,
    fetch_stock_data,
    user_buy_price,
    return_for_user_price,
    validate_daily_returns,
    manual_daily_returns,
    test_run
)

# Define a test class that inherits from unittest.TestCase
class TestDailyReturnsAnalysis(TestCase):
    # setUp() runs before each test to initialize data
    def setUp(self):
        # Create a sample DataFrame representing stock data for 5 days
        dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        self.sample_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105],  # Closing prices
            'Open': [99, 101, 100, 102, 104],    # Opening prices
            'High': [101, 103, 102, 104, 106],   # Highest price of the day
            'Low': [98, 100, 99, 101, 103],      # Lowest price of the day
            'Volume': [1000, 1100, 1200, 1300, 1400]  # Trading volume
         })

    # Test user input for ticker symbol
    def test_user_input_ticker(self):
        # Patch the built-in input() to return 'aapl' automatically
        with patch('builtins.input', return_value='aapl'):
            ticker = user_input_ticker()  # Call the function
            self.assertEqual(ticker, 'AAPL')  # Check that it converts to uppercase

    # Test fetching stock data using yfinance
    @patch('yfinance.Ticker')  # Mock the Ticker class so no real HTTP request occurs
    def test_fetch_stock_data(self, mock_ticker):
        # Create a fake historical data DataFrame to return
        mock_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        
        # Set the mock to return this DataFrame when history() is called
        mock_ticker.return_value.history.return_value = mock_hist

        # Call the function
        result = fetch_stock_data('AAPL', period='3d', interval='1d')
        
        # Assertions to check if the data is returned correctly
        self.assertEqual(len(result), 3)  # Check number of rows
        self.assertTrue(all(col in result.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume']))  # Columns exist
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))  # Index is datetime

    # Test fetching stock data when the ticker is invalid (empty DataFrame)
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_empty(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()  # Return empty
        # Patch print and sys.exit to check messages instead of stopping the test
        with patch('builtins.print') as mock_print, patch('sys.exit') as mock_exit:
            fetch_stock_data('INVALID', period='1y', interval='1d')
            # Check that the correct message was printed
            mock_print.assert_called_with("No data found for the given ticker or period.")
            # Check that sys.exit was called
            mock_exit.assert_called_once()

    # Test user input for buy price with valid and invalid entries
    @patch('builtins.input', side_effect=['100', 'invalid', '200'])
    @patch('builtins.print')
    def test_user_buy_price(self, mock_print, mock_input):
        buy_price = user_buy_price()  # First input is '100'
        self.assertEqual(buy_price, 100.0)
        buy_price = user_buy_price()  # Second input is invalid, then '200'
        self.assertEqual(buy_price, 200.0)
        # Check that the error message was printed for invalid input
        mock_print.assert_any_call("Invalid input. Please enter a numeric value for the buy price.")

    # Test return calculation for a user-defined buy price
    def test_return_for_user_price(self):
        buy_price = 100.0
        result = return_for_user_price(self.sample_data.copy(), buy_price)
        # Calculate expected daily returns manually
        expected_daily_returns = [(100-100)/100, (102-100)/100, (101-100)/100, (103-100)/100, (105-100)/100]
        # Convert to percentages and round to 4 decimals
        expected_percentages = [round(x * 100, 4) for x in expected_daily_returns]
        # Round the function outputs to 4 decimals to avoid floating-point issues
        result['Percentage(%)'] = [round(x, 4) for x in result['Percentage(%)']]
        result['Daily_Return'] = [round(x, 4) for x in result['Daily_Return']]
        # Assertions
        self.assertListEqual(result['Daily_Return'].tolist(), [round(x, 4) for x in expected_daily_returns])
        self.assertListEqual(result['Percentage(%)'].tolist(), expected_percentages)

    # Test automatic daily returns calculation
    def test_validate_daily_returns(self):
        result = validate_daily_returns(self.sample_data.copy())
        # Round results to 4 decimals
        result_list = [None if pd.isna(x) else round(x, 4) for x in result['Daily_Return']]
        expected_daily_returns = [None, (102-100)/100, (101-102)/102, (103-101)/101, (105-103)/103]
        expected_daily_returns = [None if x is None else round(x, 4) for x in expected_daily_returns]
        result_percentages = [None if pd.isna(x) else round(x, 4) for x in result['Percentage(%)']]
        expected_percentages = [None if x is None else round(x * 100, 4) for x in expected_daily_returns]
        # Assertions
        self.assertListEqual(result_list, expected_daily_returns)
        self.assertListEqual(result_percentages, expected_percentages)

    # Test manual daily returns calculation
    def test_manual_daily_returns(self):
        result = manual_daily_returns(self.sample_data.copy())
        # Check value in the loop and replace NaN with None, round valid numbers to 4 decimal places, and return as a list.
        result_list = [None if pd.isna(x) else round(x, 4) for x in result['Manual_Daily_Return']]
        expected_daily_returns = [None, (102-100)/100, (101-102)/102, (103-101)/101, (105-103)/103]
        expected_daily_returns = [None if x is None else round(x, 4) for x in expected_daily_returns]
        result_percentages = [None if pd.isna(x) else round(x, 4) for x in result['Percentage(%)']]
        expected_percentages = [None if x is None else round(x * 100, 4) for x in expected_daily_returns]
        # Assertions
        self.assertListEqual(result_list, expected_daily_returns)
        self.assertListEqual(result_percentages, expected_percentages)

    # Test running the main program with a buy price
    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'y', '100'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    # mock sys.exit to prevent exiting the test runner
    # mock input provides fake user inputs so the function runs automatically in a test.
    # mock yfinance to avoid real API calls
    def test_test_run_with_buy_price(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        mock_ticker_history.return_value = self.sample_data  # Mock fetched data
        test_run()  # Run the main function
        # Collect printed strings for assertion
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        self.assertIn(f"Fetched {len(self.sample_data)} rows of data for AAPL.", calls)
        self.assertIn('Your buy price is set at: 100.0', calls)
        self.assertIn('Latest day price and return based on your buy price:', calls)

    # Test running the main program without entering a buy price
    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'n'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    def test_test_run_without_buy_price(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        mock_ticker_history.return_value = self.sample_data
        test_run()  # Run the main function
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        self.assertIn(f"Fetched {len(self.sample_data)} rows of data for AAPL.", calls)
        self.assertIn('First 5 rows with Daily Returns (Table):', calls)
        self.assertIn('First 5 rows with Manual Daily Returns (Table):', calls)
        self.assertIn('Validation successful: Both methods yield the same daily returns.', calls)

    # Test handling invalid input choice in main program
    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'x', 'y', '100'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    def test_test_run_invalid_choice(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        mock_ticker_history.return_value = self.sample_data
        test_run()
        calls = [call[0][0] for call in mock_print.call_args_list if isinstance(call[0][0], str)]
        self.assertIn('Invalid choice. Please enter \'y\' or \'n\'.', calls)
        self.assertIn('Your buy price is set at: 100.0', calls)

# Run the tests if the script is executed directly
if __name__ == '__main__':
    main()
