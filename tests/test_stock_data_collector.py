from unittest.mock import MagicMock, patch

import pytest

from src.data_extraction.stock_data_collector import (
    fetch_most_active_stocks,
    fetch_stock_list,
    validate_ticker,
)


@pytest.fixture
def mock_yf_ticker():
    """Fixture to mock yfinance Ticker calls."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.history.return_value = MagicMock(empty=False)
        mock_ticker.return_value = mock_instance
        yield mock_ticker


@patch("src.data_extraction.stock_data_collector.urlopen")
def test_fetch_stock_list(mock_urlopen):
    """Test fetching stock list from Yahoo Finance sources."""
    mock_response = MagicMock()
    mock_response.read.return_value = (
        b"<table><tr><th>Symbol</th></tr><tr><td>AAPL</td></tr></table>"
    )
    mock_urlopen.return_value = mock_response

    stock_list = fetch_stock_list("https://mock-url.com", "Mock Source")
    assert stock_list == {"AAPL"}, "Stock list should contain AAPL"


@patch("yfinance.Ticker")
def test_validate_ticker(mock_ticker):
    """Test ticker validation function."""
    mock_instance = MagicMock()
    mock_instance.history.return_value = MagicMock(empty=False)
    mock_ticker.return_value = mock_instance

    assert validate_ticker("AAPL") == "AAPL", "Ticker AAPL should be valid"
    mock_instance.history.return_value = MagicMock(empty=True)
    assert validate_ticker("INVALID") is None, "Invalid ticker should return None"


@patch("src.data_extraction.stock_data_collector.fetch_stock_list")
@patch("src.data_extraction.stock_data_collector.validate_ticker")
def test_fetch_most_active_stocks(mock_validate_ticker, mock_fetch_stock_list):
    """Test fetching and validation of most active stocks."""
    mock_fetch_stock_list.return_value = {"AAPL", "GOOGL", "MSFT"}
    mock_validate_ticker.side_effect = (
        lambda ticker: ticker
    )  # Returns same ticker if valid

    result = fetch_most_active_stocks()
    result_tickers = [item[0] for item in result]

    assert set(result_tickers) == {
        "AAPL",
        "GOOGL",
        "MSFT",
    }, "Should return valid tickers"
    assert len(result_tickers) > 0, "Should return at least one ticker"
