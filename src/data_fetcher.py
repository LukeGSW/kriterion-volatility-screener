"""
data_fetcher.py - EODHD API client for the Kriterion Quant Volatility Screener.

Strategia universo:
  1. /api/exchange-symbol-list/US  -> tutti i ticker US, filtro exchange primari + tipo.
  2. /api/eod-bulk-last-day/US     -> una sola chiamata, volume ultimo giorno.
     Pre-filtro: scarta ticker con volume < 100K shares.
     Se disponibile nel response, usa market_cap per pre-filtro >= 2B.
  3. L'ADV rolling (30d/90d >= 1.5M) e' il filtro definitivo in quant_engine.

Questo riduce l'universo da ~24.000 a ~1.000-3.000 ticker prima del download
triennale, mantenendo i costi API ragionevoli.

No options data. No hard stop logic.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
BASE_URL           = "https://eodhd.com/api"
REQUEST_TIMEOUT    = 60
MAX_RETRIES        = 3
RETRY_BASE_BACKOFF = 2.0

# Exchange primari US (esclude PINK, OTC, OTCQB, OTCGREY, OTCCE)
PRIMARY_US_EXCHANGES = {
    "NYSE", "NASDAQ", "NYSE ARCA", "BATS",
    "NYSE MKT", "AMEX", "NYSE American", "CBOE",
}

VALID_TYPES = {"Common Stock", "ETF"}

# Pre-filtro volume sull'ultimo giorno (conservative - ADV 1.5M fara' il lavoro duro)
MIN_VOLUME_PREFILTER = 100_000


# -- Custom exception ----------------------------------------------------------
class EODHDError(Exception):
    """Raised when an EODHD API call fails after all retries."""


# -- Low-level HTTP ------------------------------------------------------------
def _get_json(url, params, max_retries=MAX_RETRIES):
    """GET con exponential-backoff retry. Raise EODHDError on failure."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                wait = RETRY_BASE_BACKOFF ** (attempt + 1)
                logger.warning("Rate limit 429 - attendo %.0fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.warning("JSON decode error attempt %d: %s", attempt + 1, e)
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                raise EODHDError("HTTP error: {}".format(e)) from e
            time.sleep(RETRY_BASE_BACKOFF)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise EODHDError("Request failed: {}".format(e)) from e
            time.sleep(RETRY_BASE_BACKOFF * (attempt + 1))
    raise EODHDError("Failed GET {} after {} attempts".format(url, max_retries))


# -- EODHD Client --------------------------------------------------------------
class EODHDClient:
    """
    Client EODHD All-In-One.

    Utilizzo:
        client   = EODHDClient(api_token=os.environ["EODHD_API_KEY"])
        universe = client.get_universe()
        ohlcv    = client.get_bulk_ohlcv(universe["ticker"].tolist(), ...)
        earnings = client.get_upcoming_earnings(universe["ticker"].tolist())
    """

    def __init__(self, api_token):
        if not api_token:
            raise ValueError("EODHD API token is required.")
        self.api_token = api_token

    def _p(self, **extra):
        return {"api_token": self.api_token, "fmt": "json", **extra}

    # -- Step 1: Exchange Symbol List ------------------------------------------
    def _fetch_exchange_symbol_list(self):
        """
        GET /api/exchange-symbol-list/US

        Ritorna tutti i ticker listati su mercati US.
        Filtra per exchange primari e tipo (Common Stock, ETF).

        Returns DataFrame: ticker, exchange, name, type
        """
        url    = "{}/exchange-symbol-list/US".format(BASE_URL)
        params = self._p()

        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.error("Exchange-symbol-list fallita: %s", e)
            return pd.DataFrame()

        if not isinstance(data, list) or not data:
            logger.error("Exchange-symbol-list: risposta inattesa (%s)", type(data))
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info("Exchange-symbol-list US raw: %d ticker", len(df))

        df.columns = [c.strip() for c in df.columns]
        col_map = {
            "Code": "ticker", "code": "ticker",
            "Exchange": "exchange", "exchange": "exchange",
            "Name": "name", "name": "name",
            "Type": "type", "type": "type",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "exchange" in df.columns:
            before = len(df)
            df = df[df["exchange"].isin(PRIMARY_US_EXCHANGES)].copy()
            logger.info(
                "Dopo filtro exchange primari: %d ticker (scartati %d OTC/PINK)",
                len(df), before - len(df)
            )

        if "type" in df.columns:
            df = df[df["type"].isin(VALID_TYPES)].copy()
            logger.info("Dopo filtro tipo (Common Stock/ETF): %d ticker", len(df))

        df = df.dropna(subset=["ticker"])
        df = df[df["ticker"].str.strip() != ""]
        keep = [c for c in ["ticker", "exchange", "name", "type"] if c in df.columns]
        return df[keep].reset_index(drop=True)

    # -- Step 2: Bulk Last-Day EOD (pre-filtro volume) -------------------------
    def _fetch_bulk_last_day(self):
        """
        GET /api/eod-bulk-last-day/US

        Una singola chiamata che restituisce volume e close dell'ultimo giorno
        di trading per TUTTI i ticker US. Usato per pre-filtrare prima del
        download OHLCV triennale.

        Returns DataFrame: ticker, last_volume, last_close, [market_cap]
        """
        url    = "{}/eod-bulk-last-day/US".format(BASE_URL)
        params = self._p()
        params["filter"] = "extended"  # <--- AGGIUNGI QUESTA RIGA
        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.warning("Bulk last-day fetch fallita: %s", e)
            return pd.DataFrame()

        if not isinstance(data, list) or not data:
            logger.warning("Bulk last-day: risposta inattesa (%s)", type(data))
            return pd.DataFrame()

        df = pd.DataFrame(data)
        logger.info("Bulk last-day US raw: %d righe", len(df))

        df.columns = [c.strip() for c in df.columns]
        col_map = {
            "code": "ticker",  "Code": "ticker",
            "volume": "last_volume", "Volume": "last_volume",
            "close": "last_close",   "Close": "last_close",
            "adjusted_close": "last_adj_close",
            "market_capitalization": "market_cap",
            "MarketCapitalization":  "market_cap",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["last_volume", "last_close", "last_adj_close", "market_cap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["ticker"])
        df = df[df["ticker"].str.strip() != ""]

        out_cols = ["ticker"] + [
            c for c in ["last_volume", "last_close", "last_adj_close", "market_cap"]
            if c in df.columns
        ]
        return df[out_cols].reset_index(drop=True)

    # -- get_universe ----------------------------------------------------------
    def get_universe(self, min_market_cap=2_000_000_000, min_volume_prefilter=MIN_VOLUME_PREFILTER):
        """
        Recupera l'universo investibile US (Common Stock + ETF).

        Pipeline:
          1. Exchange-symbol-list -> ~11.000 ticker (exchange primari, tipo OK)
          2. Bulk last-day EOD   -> pre-filtro volume > 100K shares
          3. Pre-filtro market_cap >= 2B (se disponibile nel bulk data)

        Il filtro definitivo ADV 1.5M viene applicato in quant_engine
        sui 3 anni di OHLCV.

        Returns DataFrame: ticker, exchange, name, type, market_cap
        """
        # Step 1
        symbol_df = self._fetch_exchange_symbol_list()
        if symbol_df.empty:
            logger.error("Exchange-symbol-list vuota. Nessun ticker da processare.")
            return pd.DataFrame()

        n_after_exchange = len(symbol_df)

        # Step 2
        bulk_df = self._fetch_bulk_last_day()

        if bulk_df.empty:
            logger.warning(
                "Bulk last-day non disponibile. "
                "Procedo con universe completo (nessun pre-filtro volume)."
            )
            symbol_df["market_cap"] = float("nan")
        else:
            symbol_df = symbol_df.merge(bulk_df, on="ticker", how="left")

            # Pre-filtro volume
            if "last_volume" in symbol_df.columns:
                has_volume = symbol_df["last_volume"].notna().sum()
                if has_volume > 0:
                    before = len(symbol_df)
                    symbol_df = symbol_df[
                        symbol_df["last_volume"].fillna(0) >= min_volume_prefilter
                    ].copy()
                    logger.info(
                        "Pre-filtro volume > %s shares: %d ticker (scartati %d)",
                        "{:,}".format(min_volume_prefilter),
                        len(symbol_df), before - len(symbol_df)
                    )
                else:
                    logger.warning("Campo 'last_volume' assente o tutti NaN nel bulk data.")

            # Pre-filtro market_cap (se disponibile)
            if "market_cap" in symbol_df.columns:
                n_with_cap = symbol_df["market_cap"].notna().sum()
                if n_with_cap > 100:
                    before = len(symbol_df)
                    mask_ok = (
                        symbol_df["market_cap"].isna() |
                        (symbol_df["market_cap"] >= min_market_cap)
                    )
                    symbol_df = symbol_df[mask_ok].copy()
                    logger.info(
                        "Pre-filtro market_cap >= $%s (%d con dato): %d ticker (scartati %d)",
                        "{:,.0f}".format(min_market_cap), n_with_cap,
                        len(symbol_df), before - len(symbol_df)
                    )
                else:
                    logger.info(
                        "market_cap disponibile per soli %d ticker - pre-filtro cap saltato.",
                        n_with_cap
                    )
            else:
                symbol_df["market_cap"] = float("nan")

        # Pulizia finale
        drop_cols = [c for c in ["last_volume", "last_close", "last_adj_close"]
                     if c in symbol_df.columns]
        symbol_df = symbol_df.drop(columns=drop_cols, errors="ignore")

        for col in ["market_cap", "name", "type", "exchange"]:
            if col not in symbol_df.columns:
                symbol_df[col] = float("nan") if col == "market_cap" else ""

        keep = ["ticker", "exchange", "name", "type", "market_cap"]
        symbol_df = (
            symbol_df[keep]
            .drop_duplicates(subset=["ticker"])
            .reset_index(drop=True)
        )

        logger.info(
            "Universo finale: %d ticker (da %d post-exchange-filter)",
            len(symbol_df), n_after_exchange
        )
        return symbol_df

    # -- OHLCV: singolo ticker -------------------------------------------------
    def get_ohlcv(self, ticker, from_date=None, to_date=None):
        """
        GET /api/eod/{ticker}.US - daily adjusted OHLCV.

        Returns DataFrame: date, open, high, low, close, adjusted_close, volume
        Ordinato ascending per date. DataFrame vuoto su errore.
        """
        url    = "{}/eod/{}.US".format(BASE_URL, ticker)
        params = {"api_token": self.api_token, "fmt": "json", "adjusted_close": "true"}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            data = _get_json(url, params)
        except EODHDError as e:
            logger.debug("OHLCV %s: %s", ticker, e)
            return pd.DataFrame()

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        df.columns = [c.lower() for c in df.columns]
        rename = {
            "date": "date", "open": "open", "high": "high", "low": "low",
            "close": "close", "adjusted_close": "adjusted_close", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "adjusted_close"])
        return df.sort_values("date").reset_index(drop=True)

    # -- OHLCV: bulk parallelo -------------------------------------------------
    def get_bulk_ohlcv(self, tickers, from_date, to_date,
                       max_workers=5, inter_request_delay=0.1):
        """
        Fetch OHLCV per N ticker in parallelo con ThreadPoolExecutor.
        Returns dict: {ticker: pd.DataFrame}
        """
        results = {}
        total   = len(tickers)

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
                    logger.warning("Error %s: %s", ticker, e)
                    results[ticker] = pd.DataFrame()

                completed += 1
                if completed % 100 == 0 or completed == total:
                    success = sum(1 for d in results.values() if not d.empty)
                    logger.info("OHLCV: %d/%d (%d con dati)", completed, total, success)

        return results

    # -- Market cap da fundamentals -------------------------------------------
    def _get_single_market_cap(self, ticker):
        """
        GET /api/fundamentals/{ticker}.US?filter=General::MarketCapitalization
        Restituisce float o NaN.
        """
        url    = "{}/fundamentals/{}.US".format(BASE_URL, ticker)
        params = {
            "api_token": self.api_token,
            "fmt":       "json",
            "filter":    "General::MarketCapitalization",
        }
        try:
            data = _get_json(url, params, max_retries=2)
            # L'endpoint con filter= può restituire un numero nudo o un dict
            if isinstance(data, (int, float)) and data > 0:
                return float(data)
            if isinstance(data, dict):
                val = data.get("MarketCapitalization") or data.get("market_capitalization")
                if val is not None:
                    return float(val)
            return float("nan")
        except (EODHDError, ValueError, TypeError):
            return float("nan")

    def get_market_caps(self, tickers, max_workers=10, inter_request_delay=0.05):
        """
        Recupera la market capitalization da /api/fundamentals per una lista di ticker.

        Parallelizzato. Chiamata solo sui ticker qualificati (~200-1600),
        non sull'intero universo.

        Returns dict: {ticker: float}  (float("nan") se non disponibile)
        """
        if not tickers:
            return {}

        results = {}
        total   = len(tickers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}
            for ticker in tickers:
                fut = executor.submit(self._get_single_market_cap, ticker)
                future_to_ticker[fut] = ticker
                time.sleep(inter_request_delay)

            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.debug("MarketCap %s: %s", ticker, e)
                    results[ticker] = float("nan")

                completed += 1
                if completed % 200 == 0 or completed == total:
                    found = sum(1 for v in results.values()
                                if isinstance(v, float) and not (v != v))  # not NaN
                    logger.info(
                        "MarketCap: %d/%d (%d con dato)", completed, total, found
                    )

        found = sum(
            1 for v in results.values()
            if isinstance(v, float) and v > 0 and v == v  # > 0 and not NaN
        )
        logger.info("Market cap recuperata per %d/%d ticker", found, total)
        return results

    # -- Earnings calendar -----------------------------------------------------
    def get_upcoming_earnings(self, tickers, days_ahead=90):
        """
        GET /api/calendar/earnings - prossima data earnings per lista ticker US.
        Batch da 50 simboli. Returns {ticker: "YYYY-MM-DD"} o {ticker: None}.
        """
        if not tickers:
            return {}

        today       = datetime.utcnow().date()
        future_date = today + timedelta(days=days_ahead)
        BATCH_SIZE  = 50
        earnings_map = {t: None for t in tickers}
        batches = [tickers[i: i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

        for idx, batch in enumerate(batches):
            symbols_str = ",".join("{}.US".format(t) for t in batch)
            params = {
                "api_token": self.api_token, "fmt": "json",
                "from": today.isoformat(), "to": future_date.isoformat(),
                "symbols": symbols_str,
            }
            try:
                data = _get_json("{}/calendar/earnings".format(BASE_URL), params)
            except EODHDError as e:
                logger.warning("Earnings batch %d: %s", idx + 1, e)
                continue

            if not isinstance(data, dict):
                continue

            for entry in data.get("earnings", []):
                code        = entry.get("code", "")
                ticker      = code.split(".")[0].upper()
                report_date = entry.get("report_date") or entry.get("date")
                if ticker in earnings_map and report_date:
                    existing = earnings_map[ticker]
                    if existing is None or str(report_date) < str(existing):
                        earnings_map[ticker] = str(report_date)

            time.sleep(0.3)
            logger.info("Earnings: batch %d/%d", idx + 1, len(batches))

        found = sum(1 for v in earnings_map.values() if v is not None)
        logger.info("Earnings trovate: %d/%d ticker", found, len(tickers))
        return earnings_map

    # -- News API: blacklist M&A (deal-pending exclusion) ---------------------
    def get_ma_news_tickers(
        self,
        days_lookback: int   = 180,
        min_mentions: int    = 2,
        market_suffix: str   = ".US",
        page_limit: int      = 1000,
        max_pages: int       = 30,
    ) -> Dict[str, int]:
        """
        Recupera la blacklist dei ticker oggetto di M&A negli ultimi N giorni
        usando il News API EODHD con tag 'MERGERS AND ACQUISITIONS'.

        Logica:
          1. Paginazione di /api/news?t=MERGERS+AND+ACQUISITIONS&from=...&to=...
          2. Aggregazione del campo `symbols` di ogni articolo
          3. Filtro per market_suffix (default '.US' per coerenza con l'universo)
          4. Conteggio occorrenze: ogni ticker e' contato 1 volta per articolo distinto
          5. Restituisce solo i ticker con conteggio >= min_mentions

        Parameters
        ----------
        days_lookback : int
            Finestra temporale (giorni indietro da oggi) per cercare news M&A.
        min_mentions : int
            Soglia minima di articoli distinti per includere un ticker.
            min_mentions=2 filtra speculazioni isolate / rumor non confermati.
        market_suffix : str
            Suffisso EODHD per filtrare il market (es. '.US', '.LSE').
            Stringa vuota disabilita il filtro.
        page_limit : int
            Articoli per pagina (max 1000 da spec EODHD).
        max_pages : int
            Safety cap sulla paginazione. Con 1000/pagina e ~10000 articoli
            attesi su 180gg, 30 pagine sono un margine ampio.

        Returns
        -------
        dict {ticker_base: n_mentions} per ticker con n >= min_mentions.
        Il ticker e' restituito SENZA suffisso (es. 'AAPL', non 'AAPL.US')
        per facilitare il matching con l'universo.

        Raises
        ------
        EODHDError se la chiamata API fallisce (no fallback).
        """
        today    = datetime.utcnow().date()
        from_dt  = today - timedelta(days=days_lookback)

        url = "{}/news".format(BASE_URL)
        ticker_counts: Dict[str, int] = {}
        total_articles = 0
        offset         = 0
        page_idx       = 0

        logger.info(
            "News M&A: fetch %s -> %s (lookback %dgg, min_mentions=%d, market='%s')",
            from_dt.isoformat(), today.isoformat(),
            days_lookback, min_mentions, market_suffix,
        )

        while page_idx < max_pages:
            params = {
                "api_token": self.api_token,
                "fmt":       "json",
                "t":         "MERGERS AND ACQUISITIONS",
                "from":      from_dt.isoformat(),
                "to":        today.isoformat(),
                "limit":     page_limit,
                "offset":    offset,
            }
            data = _get_json(url, params)  # raise EODHDError on failure

            if not isinstance(data, list):
                logger.warning(
                    "News API: response non-list a pagina %d (offset %d). "
                    "Tipo: %s. Interrompo paginazione.",
                    page_idx + 1, offset, type(data).__name__,
                )
                break

            n_returned = len(data)
            if n_returned == 0:
                logger.info("News M&A: nessun articolo a offset %d, fine paginazione.", offset)
                break

            total_articles += n_returned

            # Aggrega ticker per articolo. Set per articolo per evitare
            # doppi conteggi quando un ticker appare piu' volte nello stesso articolo.
            for article in data:
                if not isinstance(article, dict):
                    continue
                symbols = article.get("symbols") or []
                if not isinstance(symbols, list):
                    continue
                # Set per articolo: ogni ticker conta 1 volta per articolo
                article_tickers = set()
                for sym in symbols:
                    if not isinstance(sym, str):
                        continue
                    sym = sym.strip()
                    # Filtro market suffix
                    if market_suffix and not sym.endswith(market_suffix):
                        continue
                    # Estrai ticker base (senza suffisso)
                    base = sym[:-len(market_suffix)] if market_suffix else sym
                    base = base.upper()
                    if base:
                        article_tickers.add(base)
                for t in article_tickers:
                    ticker_counts[t] = ticker_counts.get(t, 0) + 1

            logger.info(
                "News M&A: pagina %d (offset %d): %d articoli ricevuti, "
                "totale articoli=%d, ticker unici fino ad ora=%d",
                page_idx + 1, offset, n_returned, total_articles, len(ticker_counts),
            )

            # Se la pagina e' incompleta, abbiamo finito
            if n_returned < page_limit:
                break

            offset   += page_limit
            page_idx += 1
            time.sleep(0.2)  # cortesia verso l'API

        if page_idx >= max_pages:
            logger.warning(
                "News M&A: max_pages=%d raggiunto. "
                "Articoli scansionati=%d (potrebbero esserci ulteriori articoli non visti).",
                max_pages, total_articles,
            )

        # Filtro per soglia minima
        blacklist = {t: n for t, n in ticker_counts.items() if n >= min_mentions}

        logger.info(
            "News M&A: fetch completato. "
            "Articoli totali=%d, ticker unici menzionati=%d, "
            "blacklist finale (n>=%d)=%d ticker",
            total_articles, len(ticker_counts), min_mentions, len(blacklist),
        )

        if blacklist:
            # Log dei top 20 per maggior conteggio (i piu' "caldi")
            top = sorted(blacklist.items(), key=lambda kv: -kv[1])[:20]
            logger.info(
                "Top blacklist (ticker, n_mentions): %s",
                ", ".join("{}={}".format(t, n) for t, n in top),
            )

        return blacklist
