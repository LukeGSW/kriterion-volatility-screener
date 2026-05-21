"""
data_fetcher.py — EODHD API client for the Kriterion Quant Volatility Screener.

Handles:
  - Universe retrieval via EODHD Screener endpoint (All-In-One plan).
  - Bulk OHLCV historical data (parallel, with retry/backoff).
  - Upcoming earnings calendar for a list of tickers.

No options data is fetched. Liquidità delle option chain è responsabilità
del trader in fase di esecuzione.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://eodhd.com/api"
REQUEST_TIMEOUT = 30          # seconds per HTTP call
MAX_RETRIES = 3
RETRY_BASE_BACKOFF = 2.0      # seconds (exponential: 2, 4, 8)
SCREENER_PAGE_SIZE = 100      # Max results per screener page

# EODHD exchange codes we accept as "US market"
US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "BATS", "ARCA", "NYSE ARCA",
                "NYSE MKT", "OTC", "OTCBB", "US", "PINK"}

# Instrument types to include
VALID_TYPES = {"Common Stock", "ETF"}


# ── Custom exceptions ─────────────────────────────────────────────────────────
class EODHDError(Exception):
    """Raised when an EODHD API call fails after all retries."""


# ── Low-level HTTP ────────────────────────────────────────────────────────────
def _get_json(url: str, params: dict, max_retries: int = MAX_RETRIES) -> object:
    """
    Perform a GET request that returns JSON, with exponential-backoff retry.

    Raises EODHDError after exhausting retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429:
                wait = RETRY_BASE_BACKOFF ** (attempt + 1)
                logger.warning(
                    f"Rate limited (429). Waiting {wait:.0f}s "
                    f"[attempt {attempt + 1}/{max_retries}]"
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise EODHDError(f"HTTP error: {e}") from e
            time.sleep(RETRY_BASE_BACKOFF)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise EODHDError(f"Request failed: {e}") from e
            wait = RETRY_BASE_BACKOFF * (attempt + 1)
            logger.warning(f"Request error, waiting {wait}s: {e}")
            time.sleep(wait)

    raise EODHDError(f"Failed to GET {url} after {max_retries} attempts")


# ── EODHD Client ──────────────────────────────────────────────────────────────
class EODHDClient:
    """
    Thin client wrapping the EODHD All-In-One API.

    Usage
    -----
    client = EODHDClient(api_token=os.environ["EODHD_API_KEY"])
    universe = client.get_universe()
    ohlcv    = client.get_bulk_ohlcv(universe["ticker"].tolist(), "2021-01-01", "2024-12-31")
    earnings = client.get_upcoming_earnings(universe["ticker"].tolist())
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("EODHD API token is required.")
        self.api_token = api_token

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _params(self, **extra) -> dict:
        """Base params dict (api_token + fmt=json) merged with extras."""
        return {"api_token": self.api_token, "fmt": "json", **extra}

    # ── Universe / Screener ───────────────────────────────────────────────────
    def get_universe(
        self,
        min_market_cap: float = 2_000_000_000,
    ) -> pd.DataFrame:
        """
        Retrieve all US large/mid-cap tickers via the EODHD Screener.

        Filters applied:
          - market_capitalization >= min_market_cap (default $2B)
          - exchange in US_EXCHANGES
          - type in VALID_TYPES (Common Stock, ETF)

        Handles screener pagination automatically.

        Returns
        -------
        pd.DataFrame with columns: ticker, exchange, name, type, market_cap
        """
        url = f"{BASE_URL}/screener"
        filters = json.dumps([
            ["market_capitalization", ">=", min_market_cap],
        ])

        all_records: List[dict] = []
        offset = 0

        while True:
            params = {
                "api_token": self.api_token,
                "filters": filters,
                "limit": SCREENER_PAGE_SIZE,
                "offset": offset,
                "sort": "market_capitalization.desc",
                "fmt": "json",
            }

            try:
                data = _get_json(url, params)
            except EODHDError as e:
                logger.error(f"Screener fetch failed at offset {offset}: {e}")
                break

            # Screener response: {"total": N, "data": [...]}
            if not isinstance(data, dict):
                logger.error(f"Unexpected screener response type: {type(data)}")
                break

            records = data.get("data", [])
            total = int(data.get("total", 0))

            if not records:
                break

            all_records.extend(records)
            logger.info(
                f"Screener: fetched {len(all_records)}/{total} tickers "
                f"(offset={offset})"
            )

            if len(all_records) >= total:
                break

            offset += SCREENER_PAGE_SIZE
            time.sleep(0.25)  # polite pacing

        if not all_records:
            logger.warning("Screener returned 0 records.")
            return pd.DataFrame(columns=["ticker", "exchange", "name", "type", "market_cap"])

        df = pd.DataFrame(all_records)

        # Normalize column names (EODHD returns lowercase keys)
        col_rename = {
            "code": "ticker",
            "exchange": "exchange",
            "name": "name",
            "type": "type",
            "market_capitalization": "market_cap",
        }
        df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

        # ── Post-filter in Python ─────────────────────────────────────────────
        # Exchange: keep US markets only
        if "exchange" in df.columns:
            df = df[df["exchange"].str.upper().isin(
                {e.upper() for e in US_EXCHANGES}
            )].copy()

        # Type: Common Stock and ETF only
        if "type" in df.columns:
            df = df[df["type"].isin(VALID_TYPES)].copy()

        # Market cap: numeric check (screener may return nulls)
        if "market_cap" in df.columns:
            df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
            df = df[df["market_cap"] >= min_market_cap].copy()

        # Drop tickers with missing essential fields
        essential = [c for c in ["ticker", "exchange"] if c in df.columns]
        df = df.dropna(subset=essential)
        df = df[df["ticker"].str.strip() != ""]

        logger.info(
            f"Universe after filters: {len(df)} tickers "
            f"(exchange in US_EXCHANGES, type in Common Stock/ETF, "
            f"market_cap >= ${min_market_cap:,.0f})"
        )

        keep_cols = [c for c in ["ticker", "exchange", "name", "type", "market_cap"]
                     if c in df.columns]
        return df[keep_cols].reset_index(drop=True)

    # ── Single-ticker OHLCV ───────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted OHLCV for one US ticker.

        EODHD uses the "{TICKER}.US" symbol format for all US-listed securities.

        Returns
        -------
        pd.DataFrame with columns: date, open, high, low, close,
        adjusted_close, volume — sorted ascending by date.
        Returns an empty DataFrame on error.
        """
        symbol = f"{ticker}.US"
        url = f"{BASE_URL}/eod/{symbol}"

        params = self._params()
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        params["adjusted_close"] = "true"

        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.debug(f"OHLCV fetch failed for {ticker}: {e}")
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        col_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjusted_close": "adjusted_close",
            "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "adjusted_close"])
        df = df.sort_values("date").reset_index(drop=True)

        return df

    # ── Bulk OHLCV (parallel) ─────────────────────────────────────────────────
    def get_bulk_ohlcv(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
        max_workers: int = 5,
        inter_request_delay: float = 0.1,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for many tickers in parallel using a thread pool.

        Parameters
        ----------
        tickers : list of str
        from_date / to_date : "YYYY-MM-DD" strings
        max_workers : concurrent threads (keep ≤ 5 to respect EODHD rate limits)
        inter_request_delay : polite sleep between spawning tasks

        Returns
        -------
        dict: {ticker: pd.DataFrame}
        """
        results: Dict[str, pd.DataFrame] = {}
        total = len(tickers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}
            for ticker in tickers:
                fut = executor.submit(self.get_ohlcv, ticker, from_date, to_date)
                future_to_ticker[fut] = ticker
                time.sleep(inter_request_delay)

            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    results[ticker] = df
                except Exception as e:
                    logger.warning(f"Unexpected error for {ticker}: {e}")
                    results[ticker] = pd.DataFrame()

                completed += 1
                if completed % 100 == 0 or completed == total:
                    success = sum(1 for d in results.values() if not d.empty)
                    logger.info(
                        f"OHLCV progress: {completed}/{total} fetched "
                        f"({success} with data)"
                    )

        return results

    # ── Earnings calendar ─────────────────────────────────────────────────────
    def get_upcoming_earnings(
        self,
        tickers: List[str],
        days_ahead: int = 90,
    ) -> Dict[str, Optional[str]]:
        """
        Fetch next upcoming earnings date for a list of US tickers.

        Uses the EODHD /api/calendar/earnings endpoint with a date range
        of [today, today + days_ahead].

        Returns
        -------
        dict: {ticker: "YYYY-MM-DD"} or {ticker: None} if no earnings found.
        """
        if not tickers:
            return {}

        today = datetime.utcnow().date()
        future_date = today + timedelta(days=days_ahead)

        # EODHD requires symbols with exchange suffix: "AAPL.US,MSFT.US"
        # Process in batches of 50 to avoid excessively long query strings
        BATCH_SIZE = 50
        earnings_map: Dict[str, Optional[str]] = {t: None for t in tickers}

        batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            symbols_str = ",".join(f"{t}.US" for t in batch)
            url = f"{BASE_URL}/calendar/earnings"
            params = self._params(
                **{
                    "from": today.isoformat(),
                    "to": future_date.isoformat(),
                    "symbols": symbols_str,
                }
            )

            try:
                data = _get_json(url, params)
            except EODHDError as e:
                logger.warning(f"Earnings batch {batch_idx + 1} failed: {e}")
                continue

            # Response: {"earnings": [{"code": "AAPL.US", "report_date": "...", ...}]}
            if not isinstance(data, dict):
                continue

            earnings_list = data.get("earnings", [])
            for entry in earnings_list:
                code = entry.get("code", "")
                # Strip exchange suffix
                ticker = code.split(".")[0].upper()
                report_date = entry.get("report_date") or entry.get("date")

                if ticker in earnings_map and report_date:
                    existing = earnings_map[ticker]
                    # Keep the nearest upcoming date
                    if existing is None or str(report_date) < str(existing):
                        earnings_map[ticker] = str(report_date)

            time.sleep(0.3)  # pacing between batches
            logger.info(
                f"Earnings calendar: processed batch "
                f"{batch_idx + 1}/{len(batches)}"
            )

        found = sum(1 for v in earnings_map.values() if v is not None)
        logger.info(f"Earnings dates found: {found}/{len(tickers)} tickers")
        return earnings_map
