"""
pipeline.py — Orchestratore principale del Kriterion Quant Volatility Screener.

Flusso di esecuzione:
  1. Carica API key da variabile d'ambiente EODHD_API_KEY.
  2. Recupera l'universo investibile via EODHD Screener (market cap >= $2B).
  3. Scarica la serie storica OHLCV (3 anni) in parallelo per tutti i ticker.
  4. Applica filtri ADV (30d, 90d >= 1.5M) e storia minima (756 gg).
  5. Calcola RV rolling (90gg), percentile rolling (756gg lookback).
  6. Recupera prossime date earnings per i ticker qualificati.
  7. Assembla il dataset finale e lo salva in:
       data/screener_results.parquet
       data/screener_results.csv
       data/run_metadata.json

Invarianti operative rispettate:
  - Nessuna chiamata a dati di opzioni.
  - Nessuna logica di hard stop loss.

Eseguito da GitHub Actions ogni giorno lavorativo dopo la chiusura del mercato USA.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
# Permette di eseguire pipeline.py sia da src/ sia dalla root del repo
_SRC_DIR = Path(__file__).parent
_REPO_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_fetcher import EODHDClient, EODHDError
from quant_engine import (
    COMPRESSION_THRESHOLD,
    MIN_ADV,
    PERCENTILE_LOOKBACK,
    RV_WINDOW,
    run_analysis,
)

# ── Configurazione ─────────────────────────────────────────────────────────────
MIN_MARKET_CAP: float = 2_000_000_000       # $2 miliardi
HISTORY_YEARS: int = 3                       # Anni di storia OHLCV da scaricare
HISTORY_BUFFER_DAYS: int = 90               # Buffer extra per garantire 756 gg lavorativi
EARNINGS_DAYS_AHEAD: int = 90               # Giorni futuri per cercare earnings

DATA_DIR: Path = _REPO_ROOT / "data"
OUTPUT_PARQUET: Path = DATA_DIR / "screener_results.parquet"
OUTPUT_CSV: Path = DATA_DIR / "screener_results.csv"
OUTPUT_METADATA: Path = DATA_DIR / "run_metadata.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _compute_date_range() -> tuple[str, str]:
    """
    Restituisce (from_date, to_date) per il download OHLCV.

    from_date = oggi - (3 anni + buffer)
    to_date   = oggi
    """
    today = datetime.utcnow().date()
    from_date = today - timedelta(days=HISTORY_YEARS * 365 + HISTORY_BUFFER_DAYS)
    return from_date.isoformat(), today.isoformat()


def _days_to_earnings(earnings_date_str: object, reference_date: datetime) -> object:
    """
    Calcola i giorni di calendario tra reference_date e earnings_date_str.

    Returns None se earnings_date_str è None o non parsabile.
    Returns 0 se la data è già passata o è oggi.
    """
    if earnings_date_str is None or pd.isna(earnings_date_str):
        return None
    try:
        ed = pd.to_datetime(str(earnings_date_str)).date()
        delta = (ed - reference_date.date()).days
        return int(max(0, delta))
    except Exception:
        return None


def _save_results(
    results_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Salva il dataset e il file di metadati nella directory data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Parquet (efficiente per Streamlit)
    results_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    logger.info(f"Salvato: {OUTPUT_PARQUET} ({len(results_df)} righe)")

    # CSV (trasparenza / debug)
    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Salvato: {OUTPUT_CSV}")

    # Metadata JSON (letto dall'app Streamlit per header metrics)
    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Salvato: {OUTPUT_METADATA}")


# ── Pipeline principale ───────────────────────────────────────────────────────
def run_pipeline() -> None:
    run_ts = datetime.utcnow()

    logger.info("=" * 65)
    logger.info("  KRITERION QUANT — Volatility Compression Screener")
    logger.info(f"  Run: {run_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 65)

    # ── Step 1: API key ───────────────────────────────────────────────────────
    api_key = os.environ.get("EODHD_API_KEY", "").strip()
    if not api_key:
        logger.critical(
            "EODHD_API_KEY non trovata nelle variabili d'ambiente. "
            "Imposta il secret su GitHub Actions o .env locale."
        )
        sys.exit(1)

    client = EODHDClient(api_token=api_key)

    # ── Step 2: Universo investibile ──────────────────────────────────────────
    logger.info(
        f"[STEP 2/7] Recupero universo "
        f"(market_cap >= ${MIN_MARKET_CAP:,.0f}, US, Common Stock/ETF)..."
    )
    universe_df = client.get_universe(min_market_cap=MIN_MARKET_CAP)

    if universe_df.empty:
        logger.critical("Universo vuoto — nessun ticker recuperato. Abort.")
        sys.exit(1)

    tickers = universe_df["ticker"].tolist()
    n_universe = len(tickers)
    logger.info(f"Universo: {n_universe} ticker dopo filtri screener")

    # ── Step 3: Download OHLCV ────────────────────────────────────────────────
    from_date, to_date = _compute_date_range()
    logger.info(
        f"[STEP 3/7] Download OHLCV [{from_date} → {to_date}] "
        f"per {n_universe} ticker (parallel, max_workers=5)..."
    )

    ohlcv_data = client.get_bulk_ohlcv(
        tickers=tickers,
        from_date=from_date,
        to_date=to_date,
        max_workers=5,
        inter_request_delay=0.1,
    )

    n_with_data = sum(1 for d in ohlcv_data.values() if not d.empty)
    logger.info(f"OHLCV ricevuti: {n_with_data}/{n_universe} ticker con dati")

    # ── Step 4 & 5: Analisi quantitativa (RV, percentile, ADV filter) ─────────
    logger.info(
        f"[STEP 4-5/7] Analisi quantitativa "
        f"(RV window={RV_WINDOW}gg, percentile lookback={PERCENTILE_LOOKBACK}gg)..."
    )

    results_df = run_analysis(
        ohlcv_data=ohlcv_data,
        rv_window=RV_WINDOW,
        percentile_lookback=PERCENTILE_LOOKBACK,
    )

    if results_df.empty:
        logger.warning(
            "Nessun ticker ha superato i filtri quantitativi. "
            "Salvo dataset vuoto."
        )
        empty_cols = [
            "ticker", "rv_current", "rv_percentile", "rv_52w_min", "rv_52w_max",
            "adv_30d", "adv_90d", "close_price", "last_date", "is_compressed",
            "market_cap", "name", "type",
            "next_earnings_date", "days_to_earnings",
        ]
        results_df = pd.DataFrame(columns=empty_cols)
        metadata = {
            "run_timestamp": run_ts.isoformat(),
            "tickers_scanned": n_universe,
            "tickers_with_data": n_with_data,
            "tickers_passed_filters": 0,
            "tickers_compressed": 0,
            "rv_window": RV_WINDOW,
            "percentile_lookback": PERCENTILE_LOOKBACK,
            "compression_threshold": COMPRESSION_THRESHOLD,
            "min_market_cap": MIN_MARKET_CAP,
            "min_adv": MIN_ADV,
        }
        _save_results(results_df, metadata)
        return

    n_qualified = len(results_df)

    # ── Step 6: Arricchimento con metadati universo ───────────────────────────
    logger.info(f"[STEP 6/7] Arricchimento metadati universo per {n_qualified} ticker...")

    meta_cols = [c for c in ["ticker", "market_cap", "name", "type"]
                 if c in universe_df.columns]
    results_df = results_df.merge(
        universe_df[meta_cols],
        on="ticker",
        how="left",
    )

    # ── Step 7: Earnings calendar ─────────────────────────────────────────────
    logger.info(
        f"[STEP 7/7] Recupero earnings calendar "
        f"(prossimi {EARNINGS_DAYS_AHEAD}gg) per {n_qualified} ticker..."
    )

    earnings_map = client.get_upcoming_earnings(
        tickers=results_df["ticker"].tolist(),
        days_ahead=EARNINGS_DAYS_AHEAD,
    )

    results_df["next_earnings_date"] = results_df["ticker"].map(earnings_map)
    results_df["days_to_earnings"] = results_df["next_earnings_date"].apply(
        lambda d: _days_to_earnings(d, run_ts)
    )

    # ── Colonna flag compressione ─────────────────────────────────────────────
    # is_compressed è già calcolata in quant_engine, ma la ricalcoliamo
    # per coerenza nel caso il DataFrame fosse stato riordinato
    results_df["is_compressed"] = results_df["rv_percentile"] <= COMPRESSION_THRESHOLD

    n_compressed = int(results_df["is_compressed"].sum())

    # ── Metadati run ──────────────────────────────────────────────────────────
    metadata = {
        "run_timestamp": run_ts.isoformat(),
        "tickers_scanned": n_universe,
        "tickers_with_data": n_with_data,
        "tickers_passed_filters": n_qualified,
        "tickers_compressed": n_compressed,
        "rv_window": RV_WINDOW,
        "percentile_lookback": PERCENTILE_LOOKBACK,
        "compression_threshold": COMPRESSION_THRESHOLD,
        "min_market_cap": MIN_MARKET_CAP,
        "min_adv": MIN_ADV,
    }

    # ── Salvataggio ───────────────────────────────────────────────────────────
    _save_results(results_df, metadata)

    # ── Summary finale ────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info(
        f"  SUMMARY:"
        f"  {n_universe} scansionati | "
        f"{n_with_data} con dati | "
        f"{n_qualified} qualificati | "
        f"{n_compressed} COMPRESSI (≤{COMPRESSION_THRESHOLD}th pct)"
    )
    logger.info("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()
