# ============================================================
# Project: Financial news sentiment and S&P 500 returns
# Author: Leonardo / ChatGPT
# Description:
#   1. Downloads S&P 500 data from Yahoo Finance.
#   2. Downloads financial news from GDELT DOC API.
#   3. Computes FinBERT sentiment scores for news titles.
#   4. Aligns news sentiment to the next S&P 500 trading day.
#   5. Builds final daily dataset for time series modelling.
# ============================================================

# pip install pandas numpy requests yfinance transformers torch tqdm scikit-learn statsmodels

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import yfinance as yf

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# 1. CONFIGURATION
# ============================================================

START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Important: GDELT requires OR expressions to be wrapped in parentheses.
# Keep this query stable across the whole period for methodological consistency.
GDELT_QUERY = '("stock market" OR "Wall Street" OR "Federal Reserve" OR "inflation" OR "recession")'

# Recommended for daily analysis.
# Do NOT use parallel requests unless you accept a higher risk of HTTP 429.
MAX_RECORDS_PER_DAY = 50
REQUEST_SLEEP_SECONDS = 2.5
MAX_RETRIES = 4

# FinBERT can be slow on CPU. Batch size 16/32 usually works well.
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 16

# Output folders/files
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SP500_FILE = RAW_DIR / "sp500_raw.csv"
NEWS_RAW_FILE = RAW_DIR / "gdelt_financial_news_raw.csv"
NEWS_SENTIMENT_FILE = PROCESSED_DIR / "gdelt_financial_news_with_finbert_sentiment.csv"
FINAL_DATASET_FILE = PROCESSED_DIR / "sp500_returns_finbert_sentiment_dataset.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. S&P 500 DATA
# ============================================================

def download_sp500(start_date: str, end_date: str) -> pd.DataFrame:
    print("Downloading S&P 500 data...")

    sp500 = yf.download(
        "^GSPC",
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
    )

    if sp500.empty:
        raise RuntimeError("No S&P 500 data was downloaded. Check yfinance connection.")

    sp500 = sp500.reset_index()
    sp500 = sp500[["Date", "Close"]]
    sp500.columns = ["date", "sp500_close"]

    sp500["date"] = pd.to_datetime(sp500["date"]).dt.date
    sp500 = sp500.sort_values("date")

    # Log returns
    sp500["log_return"] = np.log(sp500["sp500_close"] / sp500["sp500_close"].shift(1))

    # Return-based volatility proxies
    sp500["abs_return"] = sp500["log_return"].abs()
    sp500["squared_return"] = sp500["log_return"] ** 2

    sp500 = sp500.dropna().reset_index(drop=True)
    sp500.to_csv(SP500_FILE, index=False)

    print("S&P 500 preview:")
    print(sp500.head())
    print(f"S&P 500 rows: {len(sp500)}")
    print(f"Saved to: {SP500_FILE}")

    return sp500


# ============================================================
# 3. GDELT NEWS DOWNLOAD
# ============================================================

def fetch_gdelt_news_for_day(date: str, query: str, max_records: int = 50) -> pd.DataFrame:
    """
    Downloads GDELT news for one calendar day.

    Why daily requests?
    - The final dataset is daily.
    - Daily requests avoid monthly top-250 truncation that can miss quiet days.
    - The article timestamp is preserved and later aligned to the next trading day.

    Why no threading?
    - GDELT can return HTTP 429 Too Many Requests.
    - Sequential requests with retry/backoff are more reproducible.
    """

    start_dt = pd.to_datetime(date).strftime("%Y%m%d000000")
    end_dt = pd.to_datetime(date).strftime("%Y%m%d235959")

    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "maxrecords": max_records,
        "sort": "datedesc",
        "sourcelang": "English",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(base_url, params=params, timeout=40)

            # Too many requests: wait longer and retry.
            if response.status_code == 429:
                wait = 20 * attempt + random.uniform(0, 5)
                print(f"{date}: HTTP 429 Too Many Requests. Retry in {wait:.1f}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"{date}: status code {response.status_code}")
                return pd.DataFrame()

            if not response.text.strip():
                print(f"{date}: empty response from GDELT")
                return pd.DataFrame()

            try:
                data = response.json()
            except Exception:
                print(f"{date}: non-JSON response from GDELT")
                print(response.text[:300])
                return pd.DataFrame()

            if "articles" not in data:
                # GDELT sometimes returns text errors, but if JSON has no articles,
                # we simply treat it as no news for that day.
                return pd.DataFrame()

            articles = data["articles"]
            if len(articles) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(articles)

            keep_cols = ["title", "seendate", "domain", "url", "sourcecountry", "language"]
            keep_cols = [col for col in keep_cols if col in df.columns]

            return df[keep_cols]

        except requests.exceptions.RequestException as e:
            wait = 10 * attempt + random.uniform(0, 3)
            print(f"{date}: request error: {e}. Retry in {wait:.1f}s...")
            time.sleep(wait)

    print(f"{date}: failed after {MAX_RETRIES} retries")
    return pd.DataFrame()


def download_gdelt_news(start_date: str, end_date: str, query: str) -> pd.DataFrame:
    """
    Downloads or loads cached GDELT news.
    """

    if NEWS_RAW_FILE.exists():
        print(f"Loading cached news from: {NEWS_RAW_FILE}")
        news = pd.read_csv(NEWS_RAW_FILE)
        print(f"Cached news rows: {len(news)}")
        return news

    print("Downloading GDELT financial news...")

    all_days = pd.date_range(start_date, end_date, freq="D")
    news_list = []

    for day in tqdm(all_days):
        day_str = day.strftime("%Y-%m-%d")

        daily_news = fetch_gdelt_news_for_day(
            date=day_str,
            query=query,
            max_records=MAX_RECORDS_PER_DAY,
        )

        if not daily_news.empty:
            news_list.append(daily_news)

        # Important: avoid rate limiting.
        time.sleep(REQUEST_SLEEP_SECONDS + random.uniform(0, 0.5))

    if len(news_list) == 0:
        raise RuntimeError("No GDELT news downloaded. Try a simpler query or a shorter date range.")

    news = pd.concat(news_list, ignore_index=True)

    # Basic cleaning
    news = news.drop_duplicates(subset=["url"])
    news = news.dropna(subset=["title", "seendate"])

    news["datetime_utc"] = pd.to_datetime(news["seendate"], errors="coerce", utc=True)
    news = news.dropna(subset=["datetime_utc"])

    # Keep only English when the column exists.
    if "language" in news.columns:
        news = news[news["language"].str.lower().eq("english") | news["language"].isna()]

    news = news.sort_values("datetime_utc").reset_index(drop=True)
    news.to_csv(NEWS_RAW_FILE, index=False)

    print("News preview:")
    print(news.head())
    print(f"News rows: {len(news)}")
    print(f"Saved to: {NEWS_RAW_FILE}")

    return news


# ============================================================
# 4. FINBERT SENTIMENT
# ============================================================

def compute_finbert_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """
    Computes FinBERT sentiment scores for news titles.

    Main score:
        sentiment_score = P(positive) - P(negative)

    For volatility modelling, p_negative is especially important.
    """

    if NEWS_SENTIMENT_FILE.exists():
        print(f"Loading cached FinBERT sentiment from: {NEWS_SENTIMENT_FILE}")
        news_sent = pd.read_csv(NEWS_SENTIMENT_FILE)
        news_sent["datetime_utc"] = pd.to_datetime(news_sent["datetime_utc"], errors="coerce", utc=True)
        print(f"Cached sentiment rows: {len(news_sent)}")
        return news_sent

    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"FinBERT device: {device}")

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f"FinBERT labels: {id2label}")

    titles = news["title"].astype(str).tolist()

    p_positive_all = []
    p_negative_all = []
    p_neutral_all = []

    print("Computing FinBERT sentiment...")

    for start in tqdm(range(0, len(titles), FINBERT_BATCH_SIZE)):
        batch_texts = titles[start:start + FINBERT_BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

        for row in probs:
            result = {}
            for i, p in enumerate(row):
                label = id2label[i]
                result[label] = float(p)

            p_positive_all.append(result.get("positive", 0.0))
            p_negative_all.append(result.get("negative", 0.0))
            p_neutral_all.append(result.get("neutral", 0.0))

    news_sent = news.copy()
    news_sent["p_positive"] = p_positive_all
    news_sent["p_negative"] = p_negative_all
    news_sent["p_neutral"] = p_neutral_all
    news_sent["sentiment_score"] = news_sent["p_positive"] - news_sent["p_negative"]

    news_sent.to_csv(NEWS_SENTIMENT_FILE, index=False)

    print("Sentiment preview:")
    print(news_sent[["datetime_utc", "title", "p_positive", "p_negative", "p_neutral", "sentiment_score"]].head())
    print(f"Saved to: {NEWS_SENTIMENT_FILE}")

    return news_sent


# ============================================================
# 5. TEMPORAL ALIGNMENT AND DAILY DATASET
# ============================================================

def build_final_dataset(sp500: pd.DataFrame, news_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns each news item to the next available S&P 500 trading day.

    This avoids look-ahead bias:
    - News from calendar day t is used for the next trading session.
    - Friday/weekend news is mapped to Monday or the next trading day.
    - allow_exact_matches=False ensures the same calendar date is not used.
    """

    print("Aligning news sentiment to next trading day...")

    news_sent = news_sent.copy()
    news_sent["datetime_utc"] = pd.to_datetime(news_sent["datetime_utc"], errors="coerce", utc=True)
    news_sent = news_sent.dropna(subset=["datetime_utc"])
    news_sent["news_date"] = news_sent["datetime_utc"].dt.date

    trading_days = pd.DataFrame({
        "market_date": pd.to_datetime(sp500["date"])
    }).sort_values("market_date")

    news_sent["news_date_dt"] = pd.to_datetime(news_sent["news_date"])
    news_sent = news_sent.sort_values("news_date_dt")

    news_aligned = pd.merge_asof(
        news_sent,
        trading_days,
        left_on="news_date_dt",
        right_on="market_date",
        direction="forward",
        allow_exact_matches=False,
    )

    news_aligned = news_aligned.dropna(subset=["market_date"])
    news_aligned["market_date"] = news_aligned["market_date"].dt.date

    print("Alignment preview:")
    print(news_aligned[["news_date", "market_date", "title", "sentiment_score"]].head())

    # Aggregate all available news assigned to each trading day.
    daily_sentiment = news_aligned.groupby("market_date").agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_median=("sentiment_score", "median"),
        negative_mean=("p_negative", "mean"),
        positive_mean=("p_positive", "mean"),
        neutral_mean=("p_neutral", "mean"),
        news_volume=("title", "count"),
    ).reset_index()

    daily_sentiment = daily_sentiment.rename(columns={"market_date": "date"})

    sp500 = sp500.copy()
    sp500["date"] = pd.to_datetime(sp500["date"]).dt.date
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"]).dt.date

    dataset = sp500.merge(daily_sentiment, on="date", how="left")

    # Neutral fill for days with no mapped news.
    fill_zero_cols = [
        "sentiment_mean",
        "sentiment_median",
        "negative_mean",
        "positive_mean",
        "neutral_mean",
        "news_volume",
    ]

    for col in fill_zero_cols:
        dataset[col] = dataset[col].fillna(0)

    dataset["log_news_volume"] = np.log1p(dataset["news_volume"])

    # Useful lag variables for initial modelling.
    dataset["abs_return_lag1"] = dataset["abs_return"].shift(1)
    dataset["squared_return_lag1"] = dataset["squared_return"].shift(1)
    dataset["sentiment_mean_lag1"] = dataset["sentiment_mean"].shift(1)
    dataset["negative_mean_lag1"] = dataset["negative_mean"].shift(1)
    dataset["log_news_volume_lag1"] = dataset["log_news_volume"].shift(1)

    dataset.to_csv(FINAL_DATASET_FILE, index=False)

    print("Final dataset preview:")
    print(dataset.head())
    print(f"Final dataset rows: {len(dataset)}")
    print(f"Saved to: {FINAL_DATASET_FILE}")

    return dataset


# ============================================================
# 6. OPTIONAL BASELINE REGRESSION
# ============================================================

def run_baseline_regression(dataset: pd.DataFrame) -> None:
    """
    Simple first check:
        abs_return_t ~ abs_return_lag1 + negative_mean_lag1 + sentiment_mean_lag1 + log_news_volume_lag1
    """

    try:
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels is not installed. Skipping baseline regression.")
        return

    cols = [
        "abs_return",
        "abs_return_lag1",
        "negative_mean_lag1",
        "sentiment_mean_lag1",
        "log_news_volume_lag1",
    ]

    df = dataset[cols].dropna().copy()

    if len(df) < 30:
        print("Not enough observations for baseline regression.")
        return

    y = df["abs_return"]
    X = df[[
        "abs_return_lag1",
        "negative_mean_lag1",
        "sentiment_mean_lag1",
        "log_news_volume_lag1",
    ]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    print("\nBaseline OLS with HAC standard errors:")
    print(model.summary())


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    sp500_df = download_sp500(START_DATE, END_DATE)
    news_df = download_gdelt_news(START_DATE, END_DATE, GDELT_QUERY)
    news_sentiment_df = compute_finbert_sentiment(news_df)
    final_dataset = build_final_dataset(sp500_df, news_sentiment_df)
    run_baseline_regression(final_dataset)

    print("\nDone.")
    print(f"Raw news: {NEWS_RAW_FILE}")
    print(f"News with sentiment: {NEWS_SENTIMENT_FILE}")
    print(f"Final dataset: {FINAL_DATASET_FILE}")
