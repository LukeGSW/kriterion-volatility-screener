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

import csv
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


# ── NYLI Merger Arbitrage ETF (MNA) ───────────────────────────────────────────
# Fonte: New York Life Investment Management.
# CSV pubblico con le holdings dell'ETF MNA, che e' LONG sui target di deal
# M&A annunciati globalmente. Lo usiamo come blacklist deal-pending per
# escludere questi titoli dai candidati Long Straddle.
#
# Asset Group da includere (posizioni LONG sui target):
#   - 'Equity Common'  → azioni di target di takeover
#   - 'REIT'           → REIT target di acquisizioni
# Asset Group da ESCLUDERE:
#   - 'TOTAL RETURN SWAPS' → short hedge sugli acquirer in stock-deals
#   - 'CASH', 'MONEY MARKET', 'CURRENCY SECURITY' → liquidita'
#   - 'Exchange Traded Fund' → treasury parking
#   - 'CVR' → Contingent Value Rights, residui di deal chiusi
MNA_ETF_CSV_URL: str = "https://data.nylim.com/MMNA.csv"
MNA_ETF_INCLUDED_ASSET_GROUPS: tuple = ("Equity Common", "REIT")


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

    # -- MNA ETF holdings: blacklist deal-pending -----------------------------
    @staticmethod
    def _clean_excel_quoted(raw: str) -> str:
        """
        Pulisce un campo del formato Excel-quoted usato da NYLI: ='VALUE' o ="VALUE".
        Rimuove anche eventuali spazi.
        """
        if raw is None:
            return ""
        s = raw.strip()
        # Pattern Excel: ="VALUE"  oppure  ='VALUE'
        if s.startswith('="') and s.endswith('"'):
            s = s[2:-1]
        elif s.startswith("='") and s.endswith("'"):
            s = s[2:-1]
        elif s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        return s.strip()

    @staticmethod
    def _normalize_mna_ticker(raw_ticker: str) -> str:
        """
        Normalizza un ticker dell'ETF MNA per matching con universo .US.
        Esempi:
          'TECK/B'  → 'TECK'   (rimuove classe azionaria)
          'AAL SMS' → ''       (swap, scartato — gestito dal filtro Asset Group)
          'EA'      → 'EA'
          '700'     → '700'    (ticker numerico Tencent HK)
        """
        t = (raw_ticker or "").strip().upper()
        if not t:
            return ""
        # Scarta ticker con spazi (di solito sono SMS swap o currency)
        if " " in t:
            return ""
        # Rimuove classe azionaria (TECK/B → TECK, BRK.B → BRK)
        for sep in ("/", ".", "-"):
            if sep in t:
                t = t.split(sep)[0]
        return t

    def get_mna_etf_holdings(
        self,
        url: str = MNA_ETF_CSV_URL,
        included_asset_groups: tuple = MNA_ETF_INCLUDED_ASSET_GROUPS,
        timeout: int = REQUEST_TIMEOUT,
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Scarica le holdings del NYLI Merger Arbitrage ETF (MNA) e ne estrae
        i ticker target di deal-pending con il loro peso percentuale.

        L'ETF e' gestito da NYLI e mantiene posizioni LONG sui target di
        takeover annunciati globalmente. Per la nostra blacklist Long Straddle:
          - INCLUDIAMO le posizioni 'Equity Common' e 'REIT' (target reali)
          - ESCLUDIAMO 'TOTAL RETURN SWAPS' (short hedge sugli acquirer),
            'CASH', 'MONEY MARKET', 'CURRENCY SECURITY', 'CVR', ecc.

        Parameters
        ----------
        url : str
            URL del CSV pubblico delle holdings (default: MNA_ETF_CSV_URL).
        included_asset_groups : tuple
            Asset Group da includere nella blacklist.
        timeout : int
            Timeout HTTP in secondi.

        Returns
        -------
        Tuple di:
            - dict {ticker_normalizzato: pct_net_assets}
            - str | None: data delle holdings (es. '2026-03-27'), o None se
                          non trovata nel CSV header.

        Raises
        ------
        EODHDError se il download fallisce (no fallback, come richiesto).
        """
        logger.info("MNA ETF: download holdings da %s", url)
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise EODHDError("MNA ETF CSV fetch failed: {}".format(e)) from e

        text = resp.text
        holdings_date: Optional[str] = None
        # Parsing CSV
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        # Estrai data holdings dall'header (seconda riga tipicamente)
        for r in rows[:5]:
            if len(r) >= 2 and r[0].strip().rstrip(":").lower() == "holdings":
                holdings_date = r[1].strip()
                break

        # Trova la riga di intestazione delle colonne
        header_idx: Optional[int] = None
        for i, r in enumerate(rows):
            # Header riconosciuto se la prima colonna e' 'Ticker'
            if r and r[0].strip().strip('"').strip('=').strip('"') == "Ticker":
                header_idx = i
                break

        if header_idx is None:
            raise EODHDError("MNA ETF CSV: header 'Ticker' non trovato")

        headers = [h.strip() for h in rows[header_idx]]
        try:
            col_ticker = headers.index("Ticker")
            col_desc   = headers.index("Security Description")
            col_group  = headers.index("Asset Group")
            col_pct    = headers.index("% of Net Assets")
        except ValueError as e:
            raise EODHDError(
                "MNA ETF CSV: colonne attese non trovate ({})".format(e)
            ) from e

        holdings: Dict[str, float] = {}
        n_total       = 0
        n_kept        = 0
        n_skip_group  = 0
        n_skip_ticker = 0
        included_set  = set(g.lower() for g in included_asset_groups)

        for r in rows[header_idx + 1:]:
            if len(r) <= max(col_ticker, col_group, col_pct):
                continue
            n_total += 1

            asset_group = r[col_group].strip()
            if asset_group.lower() not in included_set:
                n_skip_group += 1
                continue

            raw_ticker = self._clean_excel_quoted(r[col_ticker])
            ticker     = self._normalize_mna_ticker(raw_ticker)
            if not ticker:
                n_skip_ticker += 1
                continue

            try:
                pct_str = r[col_pct].strip()
                # Some rows may have already a numeric value, others a string
                pct = float(pct_str) if pct_str else 0.0
            except (ValueError, TypeError):
                pct = 0.0

            # Se duplicato (raro), prendi il peso maggiore
            if ticker in holdings:
                holdings[ticker] = max(holdings[ticker], pct)
            else:
                holdings[ticker] = pct
            n_kept += 1

        logger.info(
            "MNA ETF: holdings_date=%s | righe totali=%d | "
            "kept=%d | skip_asset_group=%d | skip_ticker_vuoto=%d | "
            "ticker unici=%d",
            holdings_date, n_total, n_kept, n_skip_group,
            n_skip_ticker, len(holdings),
        )

        if holdings:
            top10 = sorted(holdings.items(), key=lambda kv: -kv[1])[:10]
            logger.info(
                "MNA ETF top 10 holdings (ticker=peso%%): %s",
                ", ".join("{}={:.2f}%".format(t, p) for t, p in top10),
            )

        return holdings, holdings_date

    # -- Legacy news endpoint (RIMOSSO) ---------------------------------------
    # La funzione get_ma_news_tickers basata sul News API EODHD (tag MERGERS
    # AND ACQUISITIONS) e' stata rimossa perche' produceva falsi positivi
    # massicci (mega-cap acquirer come NVDA, GS, MS, APO, KKR, BAM, BX
    # finivano in blacklist) e falsi negativi sui target veri (WBD, WBS, EA
    # non venivano sempre catturati). Il dato EODHD non distingue tra
    # acquirer e target nel campo `symbols`, ed e' un limite irrisolvibile.
    # Sostituita con get_mna_etf_holdings() sopra.
    def _DELETED_get_ma_news_tickers(self, *args, **kwargs):
        """Deprecato. Sostituito da get_mna_etf_holdings()."""
        raise NotImplementedError(
            "get_ma_news_tickers e' stato rimosso. "
            "Usa get_mna_etf_holdings() che legge il NYLI Merger Arbitrage ETF "
            "(MNA) CSV — fonte molto piu' affidabile per la blacklist deal-pending."
        )
