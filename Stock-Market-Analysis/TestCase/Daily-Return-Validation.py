from unittest import TestCase, main
from unittest.mock import patch
import pandas as pd

# Import functions under test
from daily_return_validation import (
    user_input_ticker,
    fetch_stock_data,
    user_buy_price,
    return_for_user_price,
    validate_daily_returns,
    manual_daily_returns,
    test_run
)

class TestDailyReturnsAnalysis(TestCase):
    """
    Comprehensive test suite for the daily_return_validation module.
    Covers both functional correctness and robustness against edge cases.
    """

    # -------------------------------------------------------------------------
    # Setup common test fixture
    # -------------------------------------------------------------------------
    def setUp(self):
        """
        Create a standard sample DataFrame that mimics real stock data.
        Executed before every individual test to ensure isolation.
        """
        dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        self.sample_data = pd.DataFrame({
            'Close': [100, 102, 101, 103, 105],
            'Open': [99, 101, 100, 102, 104],
            'High': [101, 103, 102, 104, 106],
            'Low':  [98, 100,  99, 101, 103],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

    # -------------------------------------------------------------------------
    # Basic input-handling tests
    # -------------------------------------------------------------------------
    def test_user_input_ticker(self):
        """Ensure ticker input is captured and normalized to uppercase."""
        with patch('builtins.input', return_value='aapl'):
            ticker = user_input_ticker()
            self.assertEqual(ticker, 'AAPL')

    # -------------------------------------------------------------------------
    # Data-fetching tests (mocking yfinance to avoid real HTTP calls)
    # -------------------------------------------------------------------------
    @patch('yfinance.Ticker')
    def test_fetch_stock_data(self, mock_ticker):
        """
        Verify fetch_stock_data returns a valid DataFrame when yfinance succeeds.
        """
        mock_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low':  [98,  99, 100],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        mock_ticker.return_value.history.return_value = mock_hist

        result = fetch_stock_data('AAPL', period='3d', interval='1d')
        # Check shape and columns
        self.assertEqual(len(result), 3)
        self.assertTrue(all(col in result.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume']))
        self.assertTrue(isinstance(result.index, pd.DatetimeIndex))

    @patch('yfinance.Ticker')
    def test_fetch_stock_data_empty(self, mock_ticker):
        """
        Confirm that when yfinance returns an empty DataFrame,
        the function prints a warning and exits gracefully.
        """
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        with patch('builtins.print') as mock_print, patch('sys.exit') as mock_exit:
            fetch_stock_data('INVALID', period='1y', interval='1d')
            mock_print.assert_called_with("No data found for the given ticker or period.")
            mock_exit.assert_called_once()

    # -------------------------------------------------------------------------
    # Buy-price input validation
    # -------------------------------------------------------------------------
    @patch('builtins.input', side_effect=['100', 'invalid', '200'])
    @patch('builtins.print')
    def test_user_buy_price(self, mock_print, mock_input):
        """
        Ensure user_buy_price correctly handles valid numeric input,
        retries on invalid input, and prints error messages properly.
        """
        self.assertEqual(user_buy_price(), 100.0)
        self.assertEqual(user_buy_price(), 200.0)
        mock_print.assert_any_call("Invalid input. Please enter a numeric value for the buy price.")

    # -------------------------------------------------------------------------
    # Return calculation tests
    # -------------------------------------------------------------------------
    def test_return_for_user_price(self):
        """
        Validate that returns relative to the user’s buy price are computed correctly.
        """
        buy_price = 100.0
        result = return_for_user_price(self.sample_data.copy(), buy_price)

        # Manual expected computation for verification
        expected_daily_returns = [(p - buy_price) / buy_price for p in self.sample_data['Close']]
        expected_percentages = [round(x * 100, 4) for x in expected_daily_returns]

        # Round for floating-point consistency
        result['Daily_Return'] = [round(x, 4) for x in result['Daily_Return']]
        result['Percentage(%)'] = [round(x, 4) for x in result['Percentage(%)']]

        self.assertListEqual(result['Daily_Return'].tolist(), [round(x, 4) for x in expected_daily_returns])
        self.assertListEqual(result['Percentage(%)'].tolist(), expected_percentages)

    # -------------------------------------------------------------------------
    # Validation: automatic (pandas) vs. manual (loop) daily returns
    # -------------------------------------------------------------------------
    def test_validate_daily_returns(self):
        """
        Compare expected vs. actual daily returns computed using pandas diff().
        Uses assertAlmostEqual for floating-point tolerance.
        """
        result = validate_daily_returns(self.sample_data.copy())
        expected = [None, (102-100)/100, (101-102)/102, (103-101)/101, (105-103)/103]

        # Compare raw daily returns
        for a, b in zip(result['Daily_Return'], expected):
            if pd.isna(a) or b is None:
                continue
            self.assertAlmostEqual(a, b, places=4)

        # Compare percentage returns (scaled by 100)
        expected_percentages = [None if b is None else b * 100 for b in expected]
        for a, b in zip(result['Percentage(%)'], expected_percentages):
            if pd.isna(a) or b is None:
                continue
            self.assertAlmostEqual(a, b, places=3)

    def test_manual_daily_returns(self):
        """
        Validate manual loop-based return calculations against expected values.
        Ensures the custom algorithm matches pandas results numerically.
        """
        result = manual_daily_returns(self.sample_data.copy())
        expected = [None, (102-100)/100, (101-102)/102, (103-101)/101, (105-103)/103]

        for a, b in zip(result['Manual_Daily_Return'], expected):
            if pd.isna(a) or b is None:
                continue
            self.assertAlmostEqual(a, b, places=4)

        expected_percentages = [None if b is None else b * 100 for b in expected]
        for a, b in zip(result['Percentage(%)'], expected_percentages):
            if pd.isna(a) or b is None:
                continue
            self.assertAlmostEqual(a, b, places=3)

    # -------------------------------------------------------------------------
    # Integration tests for the main execution flow (test_run)
    # -------------------------------------------------------------------------
    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'y', '100'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    def test_test_run_with_buy_price(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        """
        Simulate full execution where user enters 'y' to use a buy price.
        Checks for correct output messages and logic flow.
        """
        mock_ticker_history.return_value = self.sample_data
        test_run()
        calls = [c[0][0] for c in mock_print.call_args_list if isinstance(c[0][0], str)]
        self.assertIn(f"Fetched {len(self.sample_data)} rows of data for AAPL.", calls)
        self.assertIn('Your buy price is set at: 100.0', calls)
        self.assertIn('Latest day price and return based on your buy price:', calls)

    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'n'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    def test_test_run_without_buy_price(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        """
        Simulate execution where user declines to input a buy price.
        Ensures both auto and manual validation outputs appear.
        """
        mock_ticker_history.return_value = self.sample_data
        test_run()
        calls = [c[0][0] for c in mock_print.call_args_list if isinstance(c[0][0], str)]
        self.assertIn('First 5 rows with Daily Returns (Table):', calls)
        self.assertIn('First 5 rows with Manual Daily Returns (Table):', calls)
        self.assertIn('Validation successful: Both methods yield the same daily returns.', calls)

    @patch('sys.exit')
    @patch('builtins.input', side_effect=['aapl', 'x', 'y', '100'])
    @patch('yfinance.Ticker.history')
    @patch('builtins.print')
    def test_test_run_invalid_choice(self, mock_print, mock_ticker_history, mock_input, mock_exit):
        """
        Test invalid user response in main flow ('x' instead of 'y'/'n').
        Confirms prompt repeats and program still proceeds correctly.
        """
        mock_ticker_history.return_value = self.sample_data
        test_run()
        calls = [c[0][0] for c in mock_print.call_args_list if isinstance(c[0][0], str)]
        self.assertIn("Invalid choice. Please enter 'y' or 'n'.", calls)
        self.assertIn('Your buy price is set at: 100.0', calls)

    # -------------------------------------------------------------------------
    # Additional edge-case scenarios
    # -------------------------------------------------------------------------
    def test_rising_streak(self):
        """Positive consecutive closes should yield increasing daily returns."""
        df = pd.DataFrame({'Close': [100, 102, 104]})
        result = manual_daily_returns(df)
        self.assertAlmostEqual(result['Manual_Daily_Return'][2], 0.0196, places=4)

    def test_falling_streak(self):
        """Falling close prices must produce negative returns."""
        df = pd.DataFrame({'Close': [105, 103, 100]})
        result = validate_daily_returns(df)
        self.assertLess(result['Daily_Return'].iloc[2], 0)

    def test_flat_day(self):
        """Constant close values should give zero (or NaN) returns."""
        df = pd.DataFrame({'Close': [100, 100, 100]})
        res = validate_daily_returns(df)
        self.assertTrue(all(x == 0 or pd.isna(x) for x in res['Daily_Return']))

    def test_nan_input(self):
        """NaN in 'Close' column should propagate as NaN in computed returns."""
        df = pd.DataFrame({'Close': [100, None, 102]})
        out = manual_daily_returns(df)
        self.assertTrue(pd.isna(out['Manual_Daily_Return'][1]))

    def test_empty_input(self):
        """Empty DataFrame input should yield an empty or all-NaN output."""
        df = pd.DataFrame({'Close': []})
        out = validate_daily_returns(df)
        self.assertTrue(out.empty or out['Daily_Return'].isna().all())


# -------------------------------------------------------------------------
# Run the test suite
# -------------------------------------------------------------------------
if __name__ == '__main__':
    main()
