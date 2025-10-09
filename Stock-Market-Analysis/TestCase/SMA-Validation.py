import unittest
import pandas as pd
from unittest.mock import patch
from SMA_validation import calculate_sma, fetch_stock_data, validate_sma


class TestSMAEdgeCases(unittest.TestCase):
    """Unified test suite covering core and edge cases for SMA computation and validation."""

    def setUp(self):
        """Prepare reusable sample data."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        self.sample_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'Open': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)

    # -------------------------------------------------------------------------
    # CORE & EDGE CASES FOR calculate_sma
    # -------------------------------------------------------------------------
    def test_sma_all_edge_cases(self):
        """Test calculate_sma() under normal, empty, oversized, and single-value conditions."""
        # Case 1: Normal increasing prices (expected linear growth)
        result = calculate_sma(self.sample_data, [3, 5])
        expected_sma_3 = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
        expected_sma_5 = [102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        self.assertEqual(result[3], expected_sma_3, "3-day SMA should match expected values")
        self.assertEqual(result[5], expected_sma_5, "5-day SMA should match expected values")

        # Case 2: Window larger than dataset length
        oversized = calculate_sma(self.sample_data, [20])
        self.assertEqual(oversized[20], [], "Window larger than data should return empty list")

        # Case 3: Window = 1 (should match original Close prices)
        single_window = calculate_sma(self.sample_data, [1])
        expected_single = [float(x) for x in self.sample_data['Close']]
        self.assertEqual(single_window[1], expected_single, "Window=1 should reproduce Close prices")

        # Case 4: Empty DataFrame input
        empty_data = pd.DataFrame({'Close': []})
        empty_result = calculate_sma(empty_data, [3, 5])
        self.assertEqual(empty_result[3], [], "Empty data should yield empty SMA list")
        self.assertEqual(empty_result[5], [], "Empty data should yield empty SMA list")

        # Case 5: Constant price series (SMA should equal that constant)
        const_data = pd.DataFrame({'Close': [100, 100, 100, 100, 100]})
        const_result = calculate_sma(const_data, [3])
        self.assertTrue(all(v == 100.0 for v in const_result[3]),
                        "Constant price series should produce identical SMA values")

    # -------------------------------------------------------------------------
    # VALIDATION AGAINST PANDAS
    # -------------------------------------------------------------------------
    def test_validate_sma_against_pandas(self):
        """Validate manual SMA output against pandas rolling mean."""
        result = validate_sma(self.sample_data, [3, 5])
        # validate_sma() returns dict of window:bool -> all should be True
        self.assertTrue(all(result.values()), "Manual SMA should match pandas rolling mean within tolerance")

    # -------------------------------------------------------------------------
    # MOCKED DATA FETCHING (non-SMA but required for coverage)
    # -------------------------------------------------------------------------
    @patch('yfinance.Ticker')
    def test_fetch_stock_data_mocked(self, mock_ticker):
        """Ensure fetch_stock_data() works correctly with mocked yfinance call."""
        fake_hist = pd.DataFrame({
            'Close': [100, 101, 102],
            'Open': [99, 100, 101],
            'High': [101, 102, 103],
            'Low': [98, 99, 100],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))
        mock_ticker.return_value.history.return_value = fake_hist

        result = fetch_stock_data('AAPL', period='3d', interval='1d')
        self.assertEqual(len(result), 3, "Fetched data should have 3 rows")
        self.assertTrue(all(col in result.columns for col in ['Close', 'Open', 'High', 'Low', 'Volume']),
                        "Fetched data should include required OHLCV columns")
        self.assertIsInstance(result.index, pd.DatetimeIndex,
                              "Index should be a pandas DatetimeIndex")


if __name__ == '__main__':
    unittest.main()
