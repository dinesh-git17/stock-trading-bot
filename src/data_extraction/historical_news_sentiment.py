import datetime
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from webdriver_manager.chrome import ChromeDriverManager

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run without opening a browser
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())


def fetch_yahoo_finance_news(stock_symbol, start_date, end_date):
    """
    Fetches historical news from Yahoo Finance using Selenium.
    """
    base_url = f"https://finance.yahoo.com/quote/{stock_symbol}/news?p={stock_symbol}"
    news_data = []

    driver = webdriver.Chrome(service=service, options=options)
    driver.get(base_url)
    time.sleep(3)  # Allow page to load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    articles = soup.find_all("li", class_="js-stream-content")

    for article in articles:
        headline_tag = article.find("h3")
        if not headline_tag:
            continue

        headline = headline_tag.get_text()
        link_tag = headline_tag.find("a")
        link = f"https://finance.yahoo.com{link_tag['href']}" if link_tag else None
        news_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")  # Approximate

        # Sentiment analysis
        sentiment = analyzer.polarity_scores(headline)["compound"]

        news_data.append(
            {
                "date": news_date.strftime("%Y-%m-%d"),
                "source": "Yahoo Finance",
                "headline": headline,
                "sentiment_score": sentiment,
                "link": link,
            }
        )

    return news_data


def fetch_reuters_news(query, start_date, end_date):
    """
    Fetches historical news from Reuters by searching their archive.
    """
    base_url = f"https://www.reuters.com/site-search/?query={query}&dateRange=pastYear"
    news_data = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch Reuters news for {query}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article")

    for article in articles:
        headline_tag = article.find("h3")
        if not headline_tag:
            continue

        headline = headline_tag.get_text()
        link_tag = article.find("a")
        link = f"https://www.reuters.com{link_tag['href']}" if link_tag else None
        date_tag = article.find("time")

        if not date_tag:
            continue

        news_date = date_tag["datetime"].split("T")[0]

        # Filter by date
        if start_date <= news_date <= end_date:
            sentiment = analyzer.polarity_scores(headline)["compound"]

            news_data.append(
                {
                    "date": news_date,
                    "source": "Reuters",
                    "headline": headline,
                    "sentiment_score": sentiment,
                    "link": link,
                }
            )

    return news_data


def fetch_google_news_rss(query, start_date, end_date):
    """
    Uses Google News RSS to get historical news.
    """
    base_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    news_data = []

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to fetch Google News for {query}")
        return []

    soup = BeautifulSoup(response.text, "xml")
    articles = soup.find_all("item")

    for article in articles:
        headline = article.find("title").text
        link = article.find("link").text
        pub_date = article.find("pubDate").text

        # Convert pub_date to YYYY-MM-DD format
        news_date = datetime.datetime.strptime(
            pub_date, "%a, %d %b %Y %H:%M:%S GMT"
        ).strftime("%Y-%m-%d")

        # Filter by date
        if start_date <= news_date <= end_date:
            sentiment = analyzer.polarity_scores(headline)["compound"]

            news_data.append(
                {
                    "date": news_date,
                    "source": "Google News",
                    "headline": headline,
                    "sentiment_score": sentiment,
                    "link": link,
                }
            )

    return news_data


def save_news_data(news_data, filename="historical_news_sentiment.csv"):
    """
    Saves collected news data to a CSV file.
    """
    df = pd.DataFrame(news_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} news articles to {filename}")


def main():
    stock_symbol = "AAPL"  # Example stock
    query = "Apple stock"
    start_date = "2024-01-01"
    end_date = "2024-03-10"

    print("Fetching Yahoo Finance news...")
    yahoo_news = fetch_yahoo_finance_news(stock_symbol, start_date, end_date)

    print("Fetching Reuters news...")
    reuters_news = fetch_reuters_news(query, start_date, end_date)

    print("Fetching Google News RSS...")
    google_news = fetch_google_news_rss(query, start_date, end_date)

    all_news = yahoo_news + reuters_news + google_news
    save_news_data(all_news)


if __name__ == "__main__":
    main()
