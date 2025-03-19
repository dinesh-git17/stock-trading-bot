from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_extraction.ohlc_data_retriever import (
    fetch_ohlc_data,
    fetch_stock_data,
    fetch_stock_data_alpha_vantage,
    fetch_stock_data_yahoo,
)


@pytest.fixture
def mock_yf_ticker():
    """Fixture to mock yfinance Ticker calls."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = MagicMock(empty=False)
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@patch("requests.get")
def test_fetch_stock_data_alpha_vantage(mock_requests):
    """Test Alpha Vantage stock data fetching."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Time Series (Daily)": {
            "2024-03-18": {
                "1. open": "100",
                "2. high": "110",
                "3. low": "95",
                "4. close": "105",
                "5. adjusted close": "104",
                "6. volume": "1000000",
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_requests.return_value = mock_response

    df = fetch_stock_data_alpha_vantage("AAPL")
    assert df is not None, "DataFrame should not be None"
    assert "open" in df.columns, "Open column should exist"


@patch("yfinance.Ticker")
def test_fetch_stock_data_yahoo(mock_ticker):
    """Test Yahoo Finance stock data fetching."""
    mock_instance = MagicMock()

    # ✅ Create a mock DataFrame with test data
    mock_df = pd.DataFrame(
        {
            "Open": [150.0],
            "High": [155.0],
            "Low": [149.0],
            "Close": [153.0],
            "Volume": [1000000],
        },
        index=pd.to_datetime(["2024-03-18"]),
    )

    mock_instance.history.return_value = mock_df  # ✅ Assign a proper DataFrame
    mock_ticker.return_value = mock_instance

    df = fetch_stock_data_yahoo("AAPL")

    assert df is not None, "DataFrame should not be None"
    assert not df.empty, "DataFrame should not be empty"
    assert "open" in df.columns, "Open column should exist"
    assert "high" in df.columns, "High column should exist"
    assert "low" in df.columns, "Low column should exist"
    assert "close" in df.columns, "Close column should exist"
    assert "volume" in df.columns, "Volume column should exist"


@patch("src.data_extraction.ohlc_data_retriever.fetch_stock_data_alpha_vantage")
@patch("src.data_extraction.ohlc_data_retriever.fetch_stock_data_yahoo")
def test_fetch_stock_data(mock_yahoo, mock_alpha):
    """Test fallback mechanism in fetch_stock_data."""
    mock_alpha.return_value = None  # Simulate Alpha Vantage failure
    mock_yahoo.return_value = MagicMock(empty=False)  # Yahoo succeeds

    result = fetch_stock_data("AAPL")
    assert result == "AAPL", "Should return valid ticker after fetching data"


@patch("src.data_extraction.ohlc_data_retriever.save_stock_data")
@patch("src.data_extraction.ohlc_data_retriever.fetch_stock_data")
def test_fetch_ohlc_data(mock_fetch_stock, mock_save_stock):
    """Test OHLC data fetching pipeline."""
    mock_fetch_stock.return_value = "AAPL"
    mock_save_stock.return_value = True

    fetch_ohlc_data(["AAPL", "GOOGL", "MSFT"])
    assert mock_fetch_stock.call_count > 0, "fetch_stock_data should be called"
