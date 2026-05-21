"""
pipeline.py — Orchestratore principale del Kriterion Quant Volatility Screener.

Flusso di esecuzione:
  1. Carica API key da variabile d'ambiente EODHD_API_KEY.
  2. Recupera l'universo investibile via EODHDClient.get_universe()
     (screener con auto-detection unità market_cap + fallback exchange-list).
  3. Scarica la serie storica OHLCV (3 anni + buffer 365gg) in parallelo.
  4. Applica filtri ADV (30d, 90d >= 1.5M) e storia minima (756 RV validi).
  5. Calcola RV rolling (90gg), percentile rolling (756gg lookback).
  6. Recupera prossime date earnings per i ticker qualificati.
  7. Assembla il dataset finale e lo salva in:
       data/screener_results.parquet
       data/screener_results.csv
       data/run_metadata.json

Invarianti operative rispettate:
  - Nessuna chiamata a dati di opzioni.
  - Nessuna logica di hard stop loss.
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_SRC_DIR  = Path(__file__).parent
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
MIN_MARKET_CAP: float  = 2_000_000_000   # $2 miliardi
HISTORY_YEARS: int     = 3               # Anni di storia OHLCV
# Buffer generoso: 365 giorni calendario ≈ 260 trading days extra.
# Garantisce 756+90 = 846 trading days anche con gap e dati mancanti.
HISTORY_BUFFER_DAYS: int = 365
EARNINGS_DAYS_AHEAD: int = 90

DATA_DIR: Path        = _REPO_ROOT / "data"
OUTPUT_PARQUET: Path  = DATA_DIR / "screener_results.parquet"
OUTPUT_CSV: Path      = DATA_DIR / "screener_results.csv"
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
    from_date = oggi - (3 anni + 365 giorni buffer)
    to_date   = oggi

    Il buffer di 365 giorni calendario (~260 trading days) garantisce che
    anche i ticker con giorni festivi, gap o dati parziali abbiano
    abbastanza storia per il calcolo del percentile (756 RV validi richiesti).
    """
    today     = datetime.utcnow().date()
    from_date = today - timedelta(days=HISTORY_YEARS * 365 + HISTORY_BUFFER_DAYS)
    return from_date.isoformat(), today.isoformat()


def _days_to_earnings(earnings_date_str: object, reference_date: datetime) -> object:
    """Giorni calendario da reference_date alla prossima trimestrale."""
    if earnings_date_str is None or pd.isna(earnings_date_str):
        return None
    try:
        ed    = pd.to_datetime(str(earnings_date_str)).date()
        delta = (ed - reference_date.date()).days
        return int(max(0, delta))
    except Exception:
        return None


def _save_results(results_df: pd.DataFrame, metadata: dict) -> None:
    """Salva parquet, CSV e metadata JSON nella directory data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    logger.info(f"Salvato: {OUTPUT_PARQUET} ({len(results_df)} righe)")

    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Salvato: {OUTPUT_CSV}")

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Salvato: {OUTPUT_METADATA}")


def _log_universe_diagnostics(universe_df: pd.DataFrame) -> None:
    """Log utile per diagnosi: distribuzione exchange e tipo."""
    if universe_df.empty:
        return
    if "exchange" in universe_df.columns:
        exc_counts = universe_df["exchange"].value_counts().head(10)
        logger.info(f"Top exchange nell'universo:\n{exc_counts.to_string()}")
    if "type" in universe_df.columns:
        logger.info(f"Tipo strumento:\n{universe_df['type'].value_counts().to_string()}")
    if "market_cap" in universe_df.columns:
        mc = universe_df["market_cap"].dropna()
        if not mc.empty:
            logger.info(
                f"Market cap: min={mc.min():,.0f} "
                f"median={mc.median():,.0f} "
                f"max={mc.max():,.0f}"
            )


# ── Pipeline principale ───────────────────────────────────────────────────────
def run_pipeline() -> None:
    run_ts = datetime.utcnow()

    logger.info("=" * 65)
    logger.info("  KRITERION QUANT — Volatility Compression Screener")
    logger.info(f"  Run: {run_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info(f"  History: {HISTORY_YEARS}y + {HISTORY_BUFFER_DAYS}d buffer")
    logger.info(f"  RV window={RV_WINDOW}d | Pct lookback={PERCENTILE_LOOKBACK}d")
    logger.info("=" * 65)

    # ── Step 1: API key ───────────────────────────────────────────────────────
    api_key = os.environ.get("EODHD_API_KEY", "").strip()
    if not api_key:
        logger.critical(
            "EODHD_API_KEY non trovata. "
            "Imposta il secret su GitHub Actions o esporta la variabile locale."
        )
        sys.exit(1)

    client = EODHDClient(api_token=api_key)

    # ── Step 2: Universo investibile ──────────────────────────────────────────
    logger.info(
        f"[2/7] Recupero universo "
        f"(market_cap >= ${MIN_MARKET_CAP:,.0f}, US, Common Stock/ETF)..."
    )
    universe_df = client.get_universe(min_market_cap=MIN_MARKET_CAP)

    if universe_df.empty:
        logger.critical("Universo vuoto — nessun ticker recuperato. Abort.")
        sys.exit(1)

    _log_universe_diagnostics(universe_df)

    tickers    = universe_df["ticker"].tolist()
    n_universe = len(tickers)
    logger.info(f"Universo: {n_universe} ticker")

    if n_universe < 10:
        logger.warning(
            f"Universo insolitamente piccolo ({n_universe} ticker). "
            "Verifica API key e piano EODHD. "
            "Continuo comunque — consulta il log del debug_pipeline.py per diagnosi."
        )

    # ── Step 3: Download OHLCV ────────────────────────────────────────────────
    from_date, to_date = _compute_date_range()
    logger.info(
        f"[3/7] Download OHLCV [{from_date} → {to_date}] "
        f"per {n_universe} ticker..."
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

    # ── Step 4-5: Analisi quantitativa ───────────────────────────────────────
    logger.info(
        f"[4-5/7] Analisi quantitativa "
        f"(RV {RV_WINDOW}d, percentile {PERCENTILE_LOOKBACK}d lookback)..."
    )

    results_df = run_analysis(
        ohlcv_data=ohlcv_data,
        rv_window=RV_WINDOW,
        percentile_lookback=PERCENTILE_LOOKBACK,
    )

    # ── Step 6a: Metadati universo ────────────────────────────────────────────
    if not results_df.empty:
        logger.info(f"[6/7] Arricchimento metadati per {len(results_df)} ticker...")
        meta_cols = [c for c in ["ticker", "market_cap", "name", "type"]
                     if c in universe_df.columns]
        results_df = results_df.merge(
            universe_df[meta_cols], on="ticker", how="left"
        )

    # ── Step 6b: Earnings calendar ────────────────────────────────────────────
    if not results_df.empty:
        logger.info(
            f"[6/7] Earnings calendar "
            f"(prossimi {EARNINGS_DAYS_AHEAD}gg) per {len(results_df)} ticker..."
        )
        earnings_map = client.get_upcoming_earnings(
            tickers=results_df["ticker"].tolist(),
            days_ahead=EARNINGS_DAYS_AHEAD,
        )
        results_df["next_earnings_date"] = results_df["ticker"].map(earnings_map)
        results_df["days_to_earnings"]   = results_df["next_earnings_date"].apply(
            lambda d: _days_to_earnings(d, run_ts)
        )
        results_df["is_compressed"] = results_df["rv_percentile"] <= COMPRESSION_THRESHOLD

    # Dataset vuoto: scrivi comunque file con schema corretto
    if results_df.empty:
        logger.warning("Nessun ticker qualificato — salvo dataset vuoto con schema.")
        empty_cols = [
            "ticker", "rv_current", "rv_percentile", "rv_52w_min", "rv_52w_max",
            "adv_30d", "adv_90d", "close_price", "last_date", "is_compressed",
            "market_cap", "name", "type",
            "next_earnings_date", "days_to_earnings",
        ]
        results_df = pd.DataFrame(columns=empty_cols)

    n_qualified  = len(results_df)
    n_compressed = int(results_df["is_compressed"].sum()) if "is_compressed" in results_df.columns else 0

    # ── Step 7: Metadati run e salvataggio ────────────────────────────────────
    metadata = {
        "run_timestamp":          run_ts.isoformat(),
        "tickers_scanned":        n_universe,
        "tickers_with_data":      n_with_data,
        "tickers_passed_filters": n_qualified,
        "tickers_compressed":     n_compressed,
        "rv_window":              RV_WINDOW,
        "percentile_lookback":    PERCENTILE_LOOKBACK,
        "compression_threshold":  COMPRESSION_THRESHOLD,
        "min_market_cap":         MIN_MARKET_CAP,
        "min_adv":                MIN_ADV,
        "history_years":          HISTORY_YEARS,
        "history_buffer_days":    HISTORY_BUFFER_DAYS,
        "from_date":              from_date,
        "to_date":                to_date,
    }

    _save_results(results_df, metadata)

    logger.info("=" * 65)
    logger.info(
        f"  SUMMARY: {n_universe} scansionati | "
        f"{n_with_data} con dati | "
        f"{n_qualified} qualificati | "
        f"{n_compressed} COMPRESSI (≤{COMPRESSION_THRESHOLD}° pct)"
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    run_pipeline()
