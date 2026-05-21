"""
data_fetcher.py — EODHD API client for the Kriterion Quant Volatility Screener.

Handles:
  - Universe retrieval via EODHD Screener (primary) con auto-detection unità
    market_capitalization e filtro country-based + fallback exchange-symbol-list.
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
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL            = "https://eodhd.com/api"
REQUEST_TIMEOUT     = 30
MAX_RETRIES         = 3
RETRY_BASE_BACKOFF  = 2.0
SCREENER_PAGE_SIZE  = 100

# Stringa per la normalizzazione degli exchange US.
# EODHD può restituire: "NYSE", "NASDAQ", "NASDAQ GS", "NasdaqGS",
# "NYSE MKT", "AMEX", "BATS", "US", ecc.
# Usiamo contains case-insensitive su parole chiave invece di exact-match.
US_EXCHANGE_KEYWORDS = ["nyse", "nasdaq", "amex", "bats", "arca", "otc", "pink", "us"]

# Country field che EODHD può restituire per titoli USA
US_COUNTRIES = {"USA", "US", "United States", "united states", "us", "usa"}

# Tipo strumento (campo "type" nel response EODHD)
VALID_TYPES = {"Common Stock", "ETF"}

# Soglie market_cap da provare in auto-detection (ordine: USD, migliaia, milioni)
# EODHD All-In-One restituisce market_capitalization in USD nella risposta screener.
# Il filtro però potrebbe interpretare valori enormi come overflow → proviamo le tre scale.
_CAP_PROBES = [
    2_000_000_000,   # USD puri (standard EODHD All-In-One)
    2_000_000,       # migliaia di USD
    2_000,           # milioni di USD
]
_CAP_PROBE_PLAUSIBLE_MIN = 300    # almeno 300 ticker US con market_cap > $2B
_CAP_PROBE_PLAUSIBLE_MAX = 5_000  # sanity cap


# ── Custom exception ──────────────────────────────────────────────────────────
class EODHDError(Exception):
    """Raised when an EODHD API call fails after all retries."""


# ── Low-level HTTP ────────────────────────────────────────────────────────────
def _get_json(url: str, params: dict, max_retries: int = MAX_RETRIES) -> object:
    """GET request con exponential-backoff retry. Raise EODHDError on failure."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = RETRY_BASE_BACKOFF ** (attempt + 1)
                logger.warning(f"Rate limit 429 — attendo {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.warning(f"JSON decode error attempt {attempt+1}: {e}")
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise EODHDError(f"HTTP error: {e}") from e
            time.sleep(RETRY_BASE_BACKOFF)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise EODHDError(f"Request failed: {e}") from e
            wait = RETRY_BASE_BACKOFF * (attempt + 1)
            logger.warning(f"Request error, wait {wait}s: {e}")
            time.sleep(wait)
    raise EODHDError(f"Failed GET {url} after {max_retries} attempts")


# ── Helpers per classificazione US ───────────────────────────────────────────
def _is_us_exchange(exchange_str: str) -> bool:
    """
    Restituisce True se la stringa exchange appartiene a un mercato US.
    Usa contains case-insensitive su keyword invece di exact-match,
    perché EODHD restituisce valori eterogenei ("NASDAQ GS", "NasdaqGS", ecc.).
    """
    if not exchange_str or not isinstance(exchange_str, str):
        return False
    exc_lower = exchange_str.lower().strip()
    return any(kw in exc_lower for kw in US_EXCHANGE_KEYWORDS)


def _is_us_country(country_str: str) -> bool:
    """Restituisce True se il campo country indica USA."""
    if not country_str or not isinstance(country_str, str):
        return False
    return country_str.strip() in US_COUNTRIES


# ── EODHD Client ──────────────────────────────────────────────────────────────
class EODHDClient:
    """
    Client EODHD All-In-One.

    Usage
    -----
    client   = EODHDClient(api_token=os.environ["EODHD_API_KEY"])
    universe = client.get_universe()
    ohlcv    = client.get_bulk_ohlcv(universe["ticker"].tolist(), "2021-01-01", "2024-12-31")
    earnings = client.get_upcoming_earnings(universe["ticker"].tolist())
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("EODHD API token is required.")
        self.api_token = api_token
        self._detected_cap_threshold: Optional[int] = None  # cache auto-detection

    def _p(self, **extra) -> dict:
        """Base params dict."""
        return {"api_token": self.api_token, "fmt": "json", **extra}

    # ─────────────────────────────────────────────────────────────────────────
    # Screener — auto-detection soglia market_cap
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_cap_threshold(self) -> Tuple[int, int]:
        """
        Determina l'unità di misura di market_capitalization nel filtro screener
        EODHD provando tre soglie diverse.

        Returns (threshold, total_count) con threshold plausibile,
        oppure (2_000_000_000, 0) come fallback senza filtro.
        """
        if self._detected_cap_threshold is not None:
            return self._detected_cap_threshold, -1

        url = f"{BASE_URL}/screener"
        for threshold in _CAP_PROBES:
            filters = json.dumps([["market_capitalization", ">=", threshold]])
            params  = {
                "api_token": self.api_token,
                "filters":   filters,
                "limit":     1,
                "fmt":       "json",
            }
            try:
                data  = _get_json(url, params)
                total = int(data.get("total", 0)) if isinstance(data, dict) else 0
                logger.info(
                    f"Cap probe threshold={threshold:>15,} → total screener={total}"
                )
                if _CAP_PROBE_PLAUSIBLE_MIN <= total <= _CAP_PROBE_PLAUSIBLE_MAX:
                    logger.info(f"Auto-detected market_cap threshold: {threshold:,}")
                    self._detected_cap_threshold = threshold
                    return threshold, total
            except EODHDError as e:
                logger.warning(f"Cap probe {threshold}: {e}")
            time.sleep(0.5)

        # Nessuna soglia plausibile → usa default USD e procede senza filtro efficace
        logger.warning(
            "Auto-detection market_cap fallita. "
            "Uso threshold=2_000_000_000 e applico fallback exchange-list."
        )
        self._detected_cap_threshold = 2_000_000_000
        return 2_000_000_000, 0

    # ─────────────────────────────────────────────────────────────────────────
    # Screener — fetch paginato
    # ─────────────────────────────────────────────────────────────────────────
    def _fetch_screener_pages(
        self,
        threshold: int,
        min_market_cap_usd: float,
    ) -> List[dict]:
        """
        Scarica tutte le pagine dello screener per il threshold dato.
        Restituisce lista grezza di record dict.
        """
        url    = f"{BASE_URL}/screener"
        filters = json.dumps([["market_capitalization", ">=", threshold]])

        all_records: List[dict] = []
        offset = 0

        while True:
            params = {
                "api_token": self.api_token,
                "filters":   filters,
                "limit":     SCREENER_PAGE_SIZE,
                "offset":    offset,
                "sort":      "market_capitalization.desc",
                "fmt":       "json",
            }
            try:
                data = _get_json(url, params)
            except EODHDError as e:
                logger.error(f"Screener page offset={offset}: {e}")
                break

            if not isinstance(data, dict):
                logger.error(f"Risposta screener non è dict: {type(data)}")
                break

            records = data.get("data", [])
            total   = int(data.get("total", 0))

            if not records:
                break

            all_records.extend(records)
            logger.info(
                f"Screener page offset={offset}: "
                f"+{len(records)} record ({len(all_records)}/{total} totali)"
            )

            if len(all_records) >= total or total == 0:
                break

            offset += SCREENER_PAGE_SIZE
            time.sleep(0.25)

        return all_records

    # ─────────────────────────────────────────────────────────────────────────
    # Exchange-symbol-list — fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _fetch_exchange_symbol_list(self) -> pd.DataFrame:
        """
        Recupera la lista completa dei ticker US da /api/exchange-symbol-list/US.
        Endpoint affidabile, restituisce TUTTI i titoli listati su mercati US.
        Non include market cap → verrà filtrata solo per tipo.

        Returns DataFrame con colonne: ticker, exchange, name, type.
        market_cap sarà NaN (non disponibile da questo endpoint).
        """
        url    = f"{BASE_URL}/exchange-symbol-list/US"
        params = {"api_token": self.api_token, "fmt": "json"}

        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.error(f"Exchange-symbol-list fallita: {e}")
            return pd.DataFrame()

        if not isinstance(data, list) or not data:
            logger.error(f"Exchange-symbol-list: risposta inattesa {type(data)}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info(f"Exchange-symbol-list US: {len(df)} ticker totali")

        # Normalizza colonne
        col_map = {"Code": "ticker", "Exchange": "exchange", "Name": "name", "Type": "type"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # Alcune versioni API usano lowercase
        col_map_lower = {"code": "ticker", "exchange": "exchange", "name": "name", "type": "type"}
        df = df.rename(columns={k: v for k, v in col_map_lower.items() if k in df.columns})

        # Filtra tipo
        if "type" in df.columns:
            df = df[df["type"].isin(VALID_TYPES)].copy()
            logger.info(f"Dopo filtro tipo (Common Stock/ETF): {len(df)} ticker")

        df["market_cap"] = float("nan")  # non disponibile da questo endpoint

        keep = [c for c in ["ticker", "exchange", "name", "type", "market_cap"] if c in df.columns]
        df   = df[keep].dropna(subset=["ticker"])
        df   = df[df["ticker"].str.strip() != ""]
        return df.reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────
    # get_universe — metodo pubblico principale
    # ─────────────────────────────────────────────────────────────────────────
    def get_universe(
        self,
        min_market_cap: float = 2_000_000_000,
    ) -> pd.DataFrame:
        """
        Recupera l'universo investibile US (Common Stock + ETF, market_cap >= $2B).

        Strategia:
          1. Auto-detect unità market_cap nel filtro screener EODHD.
          2. Fetch paginato dallo screener.
          3. Post-filtra per US (country USA o exchange US-based) e tipo.
          4. Se risultato < 50 ticker, fallback su exchange-symbol-list
             (senza filtro market_cap, che verrà applicato dopo OHLCV via ADV proxy).

        Returns
        -------
        pd.DataFrame con colonne: ticker, exchange, name, type, market_cap
        """
        # ── Step A: auto-detect soglia ────────────────────────────────────────
        threshold, probe_total = self._detect_cap_threshold()

        # ── Step B: screener paginato ─────────────────────────────────────────
        raw_records = self._fetch_screener_pages(threshold, min_market_cap)
        logger.info(f"Screener: {len(raw_records)} record raw ricevuti")

        if raw_records:
            df = pd.DataFrame(raw_records)

            # Normalizza nomi colonne (EODHD può usare snake_case o camelCase)
            col_map = {
                "code":                 "ticker",
                "Code":                 "ticker",
                "exchange":             "exchange",
                "Exchange":             "exchange",
                "name":                 "name",
                "Name":                 "name",
                "type":                 "type",
                "Type":                 "type",
                "market_capitalization":"market_cap",
                "MarketCapitalization": "market_cap",
                "country":              "country",
                "Country":              "country",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # ── Filtro US ─────────────────────────────────────────────────────
            # Prova prima con country (più affidabile), poi con exchange keyword
            has_country  = "country"  in df.columns
            has_exchange = "exchange" in df.columns

            if has_country:
                mask_us = df["country"].apply(_is_us_country)
                # Integra con exchange per i record senza country
                if has_exchange:
                    mask_us = mask_us | (df["country"].isna() & df["exchange"].apply(_is_us_exchange))
            elif has_exchange:
                mask_us = df["exchange"].apply(_is_us_exchange)
            else:
                mask_us = pd.Series([True] * len(df))  # non possiamo filtrare

            df = df[mask_us].copy()
            logger.info(f"Dopo filtro US: {len(df)} ticker")

            # ── Filtro tipo ───────────────────────────────────────────────────
            if "type" in df.columns:
                df = df[df["type"].isin(VALID_TYPES)].copy()
                logger.info(f"Dopo filtro tipo: {len(df)} ticker")

            # ── Filtro market_cap numerico ────────────────────────────────────
            if "market_cap" in df.columns:
                df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
                # Applica solo se la soglia auto-detected sembra in USD puri
                # (threshold = 2_000_000_000 → min_market_cap in USD)
                # Per soglie più piccole (milioni/migliaia), il confronto
                # con min_market_cap in USD non ha senso → skippa
                if threshold == 2_000_000_000:
                    df = df[df["market_cap"].fillna(0) >= min_market_cap].copy()
                    logger.info(f"Dopo filtro market_cap >= {min_market_cap:,.0f}: {len(df)} ticker")

            # ── Pulizia ───────────────────────────────────────────────────────
            df = df.dropna(subset=["ticker"])
            df = df[df["ticker"].str.strip() != ""]

        else:
            df = pd.DataFrame()

        # ── Step C: fallback se troppo pochi ─────────────────────────────────
        FALLBACK_THRESHOLD = 50
        if len(df) < FALLBACK_THRESHOLD:
            logger.warning(
                f"Screener ha restituito solo {len(df)} ticker dopo i filtri "
                f"(soglia fallback = {FALLBACK_THRESHOLD}). "
                f"Attivo fallback su exchange-symbol-list."
            )
            df_fallback = self._fetch_exchange_symbol_list()

            if not df_fallback.empty:
                if not df.empty:
                    # Merge: usa screener per i ticker già trovati, aggiungi il resto
                    existing = set(df["ticker"].tolist())
                    new_rows = df_fallback[~df_fallback["ticker"].isin(existing)]
                    df = pd.concat([df, new_rows], ignore_index=True)
                else:
                    df = df_fallback

                logger.info(
                    f"Post-fallback universo: {len(df)} ticker "
                    f"(market_cap filtro sarà applicato via ADV proxy su OHLCV)"
                )

        # ── Step D: colonne finali ────────────────────────────────────────────
        for col in ["market_cap", "name", "type", "exchange"]:
            if col not in df.columns:
                df[col] = float("nan") if col == "market_cap" else ""

        keep = ["ticker", "exchange", "name", "type", "market_cap"]
        df   = df[keep].drop_duplicates(subset=["ticker"]).reset_index(drop=True)

        logger.info(f"Universo finale: {len(df)} ticker (Common Stock + ETF, US)")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # OHLCV — singolo ticker
    # ─────────────────────────────────────────────────────────────────────────
    def get_ohlcv(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str]   = None,
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted OHLCV per un ticker US.
        EODHD usa il formato "{TICKER}.US" per tutti i titoli listati su mercati US.

        Returns DataFrame con colonne: date, open, high, low, close,
        adjusted_close, volume — ordinato ascending per date.
        Ritorna DataFrame vuoto in caso di errore.
        """
        symbol = f"{ticker}.US"
        url    = f"{BASE_URL}/eod/{symbol}"
        params = self._p(adjusted_close="true")
        del params["fmt"]
        params["fmt"] = "json"
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.debug(f"OHLCV fetch failed {ticker}: {e}")
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close",
            "adjusted_close": "adjusted_close", "volume": "volume",
        })

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "adjusted_close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # OHLCV — bulk parallelo
    # ─────────────────────────────────────────────────────────────────────────
    def get_bulk_ohlcv(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
        max_workers: int  = 5,
        inter_request_delay: float = 0.1,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV per N ticker in parallelo con ThreadPoolExecutor.

        Returns dict: {ticker: pd.DataFrame}
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
                    results[ticker] = future.result()
                except Exception as e:
                    logger.warning(f"Unexpected error {ticker}: {e}")
                    results[ticker] = pd.DataFrame()

                completed += 1
                if completed % 100 == 0 or completed == total:
                    success = sum(1 for d in results.values() if not d.empty)
                    logger.info(
                        f"OHLCV progress: {completed}/{total} "
                        f"({success} con dati)"
                    )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Earnings calendar
    # ─────────────────────────────────────────────────────────────────────────
    def get_upcoming_earnings(
        self,
        tickers: List[str],
        days_ahead: int = 90,
    ) -> Dict[str, Optional[str]]:
        """
        Fetch prossima data earnings per una lista di ticker US.
        Usa EODHD /api/calendar/earnings con range [oggi, oggi+days_ahead].

        Returns dict: {ticker: "YYYY-MM-DD"} o {ticker: None}.
        """
        if not tickers:
            return {}

        today       = datetime.utcnow().date()
        future_date = today + timedelta(days=days_ahead)
        BATCH_SIZE  = 50
        earnings_map: Dict[str, Optional[str]] = {t: None for t in tickers}
        batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            symbols_str = ",".join(f"{t}.US" for t in batch)
            url         = f"{BASE_URL}/calendar/earnings"
            params      = self._p(
                **{
                    "from":    today.isoformat(),
                    "to":      future_date.isoformat(),
                    "symbols": symbols_str,
                }
            )

            try:
                data = _get_json(url, params)
            except EODHDError as e:
                logger.warning(f"Earnings batch {batch_idx+1}: {e}")
                continue

            if not isinstance(data, dict):
                continue

            for entry in data.get("earnings", []):
                code   = entry.get("code", "")
                ticker = code.split(".")[0].upper()
                report_date = entry.get("report_date") or entry.get("date")
                if ticker in earnings_map and report_date:
                    existing = earnings_map[ticker]
                    if existing is None or str(report_date) < str(existing):
                        earnings_map[ticker] = str(report_date)

            time.sleep(0.3)
            logger.info(f"Earnings calendar: batch {batch_idx+1}/{len(batches)}")

        found = sum(1 for v in earnings_map.values() if v is not None)
        logger.info(f"Earnings date trovate: {found}/{len(tickers)} ticker")
        return earnings_map
