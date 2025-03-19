from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from rich.console import Console

from src.data_extraction.news_sentiment import (
    convert_datetime_format,
    fetch_news_from_google_rss,
    fetch_news_from_newsapi,
    fetch_news_from_reddit,
    fetch_news_from_yahoo,
    insert_news_sentiment_data,
)

console = Console()

# ✅ Mock API Response for NewsAPI
mock_newsapi_response = {
    "status": "ok",
    "articles": [
        {
            "source": {"name": "CNN"},
            "publishedAt": "2025-03-18T14:17:31Z",
            "title": "Tesla Stock Drops Again",
            "description": "Tesla stock is down 5% after weak sales in China.",
            "url": "https://cnn.com/tesla-drop",
        }
    ],
}


@pytest.fixture
def mock_requests_get():
    """Mocks the requests.get call."""
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_newsapi_response
        yield mock_get


# ✅ Test NewsAPI Fetch Function
def test_fetch_news_from_newsapi(mock_requests_get):
    ticker = "TSLA"
    df = fetch_news_from_newsapi(ticker)

    assert not df.empty
    assert df["ticker"][0] == ticker
    assert df["source_name"][0] == "CNN"
    assert df["title"][0] == "Tesla Stock Drops Again"
    assert df["url"][0] == "https://cnn.com/tesla-drop"
    console.print("[green]✅ Test Passed: fetch_news_from_newsapi()[/green]")


# ✅ Test Yahoo Finance Fetch Function
@pytest.fixture
def mock_yahoo_response():
    """Mocks Yahoo Finance response."""
    return [
        {
            "published": "Wed, 19 Mar 2025 14:55:00 +0000",
            "title": "Tesla Hits Record Low",
            "summary": "Tesla stock hits record low due to market downturn.",
            "link": "https://yahoo.com/tesla-low",
        }
    ]


@patch("yahoo_fin.news.get_yf_rss")
def test_fetch_news_from_yahoo(mock_get_yf_rss, mock_yahoo_response):
    mock_get_yf_rss.return_value = mock_yahoo_response
    ticker = "TSLA"
    df = fetch_news_from_yahoo(ticker)

    assert not df.empty
    assert df["ticker"][0] == ticker
    assert df["title"][0] == "Tesla Hits Record Low"
    assert df["url"][0] == "https://yahoo.com/tesla-low"
    console.print("[green]✅ Test Passed: fetch_news_from_yahoo()[/green]")


# ✅ Test Google RSS Fetch Function
@patch("feedparser.parse")
def test_fetch_news_from_google_rss(mock_feedparser_parse):
    mock_feedparser_parse.return_value.entries = [
        {
            "title": "Google RSS Tesla News",
            "summary": "Tesla stock moves higher.",
            "link": "https://news.google.com/rss/tesla",
            "published": "2025-03-18T12:00:00Z",
        }
    ]
    ticker = "TSLA"
    df = fetch_news_from_google_rss(ticker)

    assert not df.empty
    assert df["ticker"][0] == ticker
    assert df["title"][0] == "Google RSS Tesla News"
    console.print("[green]✅ Test Passed: fetch_news_from_google_rss()[/green]")


# ✅ Test Reddit Fetch Function
@patch("praw.Reddit")
def test_fetch_news_from_reddit(mock_reddit):
    mock_reddit.return_value.subreddit.return_value.search.return_value = [
        MagicMock(title="Reddit Tesla Discussion", url="https://reddit.com/tesla")
    ]
    ticker = "TSLA"
    df = fetch_news_from_reddit(ticker)

    assert not df.empty
    assert df["ticker"][0] == ticker
    assert df["title"][0] == "Reddit Tesla Discussion"
    console.print("[green]✅ Test Passed: fetch_news_from_reddit()[/green]")


# ✅ Test Date Format Conversion
def test_convert_datetime_format():
    df = pd.DataFrame(
        {
            "published_at": [
                "Wed, 19 Mar 2025 14:55:00 +0000",
                "2025-03-19T19:09:21Z",
                "Thu, 13 Mar 2025 13:44:00 GMT",
            ]
        }
    )

    df = convert_datetime_format(df)

    assert df["published_at"][0] == "2025-03-19 14:55:00"
    assert df["published_at"][1] == "2025-03-19 19:09:21"
    assert df["published_at"][2] == "2025-03-13 13:44:00"
    console.print("[green]✅ Test Passed: convert_datetime_format()[/green]")


# ✅ Test Database Insert Function (Mocking Database Calls)
@patch("src.tools.utils.get_database_engine")
def test_insert_news_sentiment_data(mock_get_db_engine):
    """Test inserting news sentiment data into the database."""
    engine_mock = MagicMock()
    mock_get_db_engine.return_value = engine_mock
    mock_conn = engine_mock.raw_connection.return_value
    mock_cursor = mock_conn.cursor.return_value

    df = pd.DataFrame(
        {
            "ticker": ["TSLA"],
            "published_at": ["2025-03-19 14:55:00"],
            "source_name": ["CNN"],
            "title": ["Tesla Drops Again"],
            "description": ["Tesla stock drops due to weak sales"],
            "url": ["https://cnn.com/tesla-drop"],
            "sentiment_score": [0.5],
            "created_at": ["2025-03-19T19:25:12Z"],
        }
    )

    insert_news_sentiment_data(df)

    # Assert copy_from() was called instead of execute()
    mock_cursor.copy_from.assert_called()
    mock_conn.commit.assert_called()
    console.print("[green]✅ Test Passed: insert_news_sentiment_data()[/green]")
