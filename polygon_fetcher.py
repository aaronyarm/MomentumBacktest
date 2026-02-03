"""
Polygon.io API Fetcher with rate limiting and caching.
"""
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import config


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.timestamps = deque()

    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        now = time.time()

        # Remove timestamps older than the period
        while self.timestamps and now - self.timestamps[0] > self.period:
            self.timestamps.popleft()

        # If at limit, wait for the oldest call to expire
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0]) + 0.1
            if sleep_time > 0:
                print(f"Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        self.timestamps.append(time.time())


class PolygonFetcher:
    """
    Fetches market data from Polygon.io with rate limiting.
    Supports aggregates (bars), ticker details, and reference data.
    Uses in-memory caching for speed optimization.
    """

    # Class-level cache shared across instances for session persistence
    _shared_cache: Dict[str, Any] = {}

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.POLYGON_API_KEY
        self.base_url = config.POLYGON_BASE_URL
        self.rate_limiter = RateLimiter(
            config.RATE_LIMIT_CALLS,
            config.RATE_LIMIT_PERIOD
        )
        # Use shared cache for persistence across backtests
        self.cache = PolygonFetcher._shared_cache

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited API request."""
        self.rate_limiter.wait_if_needed()

        params = params or {}
        params["apiKey"] = self.api_key

        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"results": [], "error": str(e)}

    def get_aggregates(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str,
        to_date: str,
        adjusted: bool = True,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch aggregate bars for a ticker.

        Args:
            ticker: Stock symbol
            multiplier: Size of the timespan multiplier
            timespan: minute, hour, day, week, month, quarter, year
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Adjust for splits
            limit: Max results

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"agg_{ticker}_{multiplier}_{timespan}_{from_date}_{to_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit
        }

        data = self._make_request(endpoint, params)

        if "results" not in data or not data["results"]:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])

        # Polygon API returns abbreviated column names - rename them properly
        column_map = {
            "v": "volume",
            "vw": "vwap",
            "o": "open",
            "c": "close",
            "h": "high",
            "l": "low",
            "t": "timestamp",
            "n": "transactions"
        }
        df = df.rename(columns=column_map)

        # Ensure timestamp column exists and convert
        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        # Select columns that exist
        available_cols = [c for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"] if c in df.columns]
        df = df[available_cols]

        self.cache[cache_key] = df
        return df

    def get_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data."""
        return self.get_aggregates(ticker, 1, "day", from_date, to_date)

    def get_minute_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        multiplier: int = 1
    ) -> pd.DataFrame:
        """Fetch minute-level OHLCV data."""
        return self.get_aggregates(ticker, multiplier, "minute", from_date, to_date)

    def get_ticker_details(self, ticker: str) -> Dict:
        """Fetch ticker details including float and sector."""
        cache_key = f"details_{ticker}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        endpoint = f"/v3/reference/tickers/{ticker}"
        data = self._make_request(endpoint)

        result = data.get("results", {})
        self.cache[cache_key] = result
        return result

    def get_ticker_float(self, ticker: str) -> Optional[float]:
        """Get the float (shares outstanding) for a ticker."""
        details = self.get_ticker_details(ticker)
        return details.get("share_class_shares_outstanding")

    def get_ticker_sector(self, ticker: str) -> str:
        """Get the sector/industry for a ticker."""
        details = self.get_ticker_details(ticker)
        return details.get("sic_description", "Unknown")

    def get_previous_close(self, ticker: str) -> Dict:
        """Get previous day's close data."""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        data = self._make_request(endpoint)

        if "results" in data and data["results"]:
            return data["results"][0]
        return {}

    def get_grouped_daily(self, date: str) -> pd.DataFrame:
        """
        Get grouped daily bars for all tickers on a date.
        Useful for screening across the market.
        """
        endpoint = f"/v2/aggs/grouped/locale/us/market/stocks/{date}"
        data = self._make_request(endpoint)

        if "results" not in data or not data["results"]:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        return df

    def get_market_status(self) -> Dict:
        """Check if market is open."""
        endpoint = "/v1/marketstatus/now"
        return self._make_request(endpoint)

    def clear_cache(self):
        """Clear the data cache."""
        self.cache = {}


class DataManager:
    """
    High-level data management with preloading and alignment.
    """

    def __init__(self, fetcher: PolygonFetcher = None):
        self.fetcher = fetcher or PolygonFetcher()
        self.daily_data: Dict[str, pd.DataFrame] = {}
        self.minute_data: Dict[str, pd.DataFrame] = {}
        self.ticker_info: Dict[str, Dict] = {}

    def _fetch_ticker_data(self, ticker: str, from_date: str, to_date: str, include_minute: bool) -> Dict:
        """Fetch all data for a single ticker (helper for parallel execution)."""
        result = {"ticker": ticker, "daily": None, "minute": None, "info": None}

        try:
            df = self.fetcher.get_daily_bars(ticker, from_date, to_date)
            if not df.empty:
                result["daily"] = df
                result["info"] = {
                    "float": self.fetcher.get_ticker_float(ticker),
                    "sector": self.fetcher.get_ticker_sector(ticker)
                }

            if include_minute:
                minute_df = self.fetcher.get_minute_bars(ticker, from_date, to_date)
                if not minute_df.empty:
                    result["minute"] = minute_df
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        return result

    def preload_tickers(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
        include_minute: bool = False
    ):
        """Preload data for multiple tickers with parallel fetching for speed."""
        from tqdm import tqdm

        print(f"Preloading data for {len(tickers)} tickers...")

        # Use parallel fetching for speed (respecting rate limits)
        # Limit concurrent requests based on rate limit
        max_workers = min(3, config.RATE_LIMIT_CALLS)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_ticker_data, ticker, from_date, to_date, include_minute): ticker
                for ticker in tickers
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
                result = future.result()
                ticker = result["ticker"]

                if result["daily"] is not None:
                    self.daily_data[ticker] = result["daily"]
                if result["info"] is not None:
                    self.ticker_info[ticker] = result["info"]
                if result["minute"] is not None:
                    self.minute_data[ticker] = result["minute"]

    def get_daily(self, ticker: str) -> pd.DataFrame:
        """Get daily data for a ticker."""
        return self.daily_data.get(ticker, pd.DataFrame())

    def get_minute(self, ticker: str) -> pd.DataFrame:
        """Get minute data for a ticker."""
        return self.minute_data.get(ticker, pd.DataFrame())

    def get_info(self, ticker: str) -> Dict:
        """Get ticker info (float, sector)."""
        return self.ticker_info.get(ticker, {})
