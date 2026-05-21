"""
quant_engine.py — Motore matematico e quantitativo del Kriterion Quant Screener.

Implementa (completamente vettorizzato con pandas/numpy):
  A. Rendimenti logaritmici giornalieri
  B. Realized Volatility (RV) rolling annualizzata su finestra w (default 90gg)
  C. Rango Percentile rolling su lookback L=756 giorni lavorativi (~3 anni)
  D. Filtro ADV (Average Daily Volume) su finestre 30 e 90 giorni

Nessuna logica di hard stop. Nessuna chiamata a dati di opzioni.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default parameters (allineati al documento di progettazione) ──────────────
RV_WINDOW: int              = 90    # Finestra RV in giorni lavorativi
PERCENTILE_LOOKBACK: int    = 756   # Lookback percentile (~3 anni trading days)
ANNUALIZATION_FACTOR: float = np.sqrt(252)

ADV_WINDOWS: list[int]  = [30, 90]
MIN_ADV: float          = 1_500_000
COMPRESSION_THRESHOLD: float = 5.0  # Percentile soglia flag "Compresso"

# Storia minima in valori RV validi (non righe raw).
# RV_WINDOW = 90 → per PERCENTILE_LOOKBACK valori RV validi servono
# almeno PERCENTILE_LOOKBACK + RV_WINDOW - 1 righe di prezzo.
# Ma controlliamo direttamente i valori RV non-NaN, più robusto.
MIN_VALID_RV_VALUES: int = PERCENTILE_LOOKBACK  # 756 valori RV non-NaN


# ── Phase A: Rendimenti Logaritmici ──────────────────────────────────────────
def compute_log_returns(close_series: pd.Series) -> pd.Series:
    """
    R_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    close_series : pd.Series — adjusted close (index = date)

    Returns
    -------
    pd.Series — primo elemento NaN per costruzione.
    """
    return np.log(close_series / close_series.shift(1))


# ── Phase B: Realized Volatility ─────────────────────────────────────────────
def compute_realized_volatility(
    returns: pd.Series,
    window: int = RV_WINDOW,
) -> pd.Series:
    """
    RV_{t,w} = sqrt(252) * sigma(R_{t-w+1...t})

    Returns
    -------
    pd.Series — RV annualizzata in forma decimale (0.25 = 25%).
    NaN per le prime `window` osservazioni.
    """
    return (
        returns
        .rolling(window=window, min_periods=window)
        .std()
        * ANNUALIZATION_FACTOR
    )


# ── Phase C: Rango Percentile Rolling ────────────────────────────────────────
def compute_rv_percentile(
    rv_series: pd.Series,
    lookback: int = PERCENTILE_LOOKBACK,
) -> pd.Series:
    """
    Percentile rolling del valore RV odierno rispetto ai precedenti `lookback` giorni.

    Interpretazione: 0 = RV ai minimi storici (compressione estrema).

    Implementazione: finestra di (lookback + 1) valori. L'ultimo è "oggi",
    i precedenti `lookback` costituiscono la distribuzione storica.

    Returns
    -------
    pd.Series — percentile [0, 100]. NaN dove storia insufficiente.
    """
    def _pct_of_last(window_arr: np.ndarray) -> float:
        if len(window_arr) < 2:
            return np.nan
        current = window_arr[-1]
        hist    = window_arr[:-1]
        if np.isnan(current):
            return np.nan
        valid_hist = hist[~np.isnan(hist)]
        if len(valid_hist) == 0:
            return np.nan
        return float(np.sum(valid_hist < current) / len(valid_hist) * 100.0)

    return rv_series.rolling(
        window=lookback + 1,
        min_periods=lookback + 1,
    ).apply(_pct_of_last, raw=True)


# ── ADV ───────────────────────────────────────────────────────────────────────
def compute_adv(volume_series: pd.Series, window: int) -> pd.Series:
    """Average Daily Volume rolling su finestra `window` giorni."""
    return volume_series.rolling(window=window, min_periods=window).mean()


# ── Per-ticker full analysis ──────────────────────────────────────────────────
def analyze_ticker(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    rv_window: int           = RV_WINDOW,
    percentile_lookback: int = PERCENTILE_LOOKBACK,
) -> Optional[dict]:
    """
    Analisi quantitativa completa per un singolo ticker.

    Filtri applicati (in ordine):
      1. Colonne minime presenti.
      2. ADV 30d e 90d >= MIN_ADV.
      3. Almeno MIN_VALID_RV_VALUES valori RV non-NaN (= storia sufficiente).
      4. Percentile valido sull'ultimo giorno.

    Returns dict con metriche, o None se il ticker non supera i filtri.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return None

    required = {"date", "adjusted_close", "volume"}
    missing  = required - set(ohlcv_df.columns)
    if missing:
        logger.debug(f"{ticker}: colonne mancanti {missing}")
        return None

    df = ohlcv_df.set_index("date").sort_index()
    df = df.dropna(subset=["adjusted_close"])

    if len(df) < rv_window + 30:
        # Meno dati del necessario anche solo per l'RV
        logger.debug(f"{ticker}: troppo pochi dati ({len(df)} righe)")
        return None

    close  = df["adjusted_close"]
    volume = df["volume"].fillna(0)

    # ── Filtro ADV ────────────────────────────────────────────────────────────
    adv_30 = compute_adv(volume, 30).iloc[-1]
    adv_90 = compute_adv(volume, 90).iloc[-1]

    if pd.isna(adv_30) or pd.isna(adv_90):
        return None
    if adv_30 < MIN_ADV or adv_90 < MIN_ADV:
        logger.debug(
            f"{ticker}: ADV KO — 30d={adv_30:,.0f} 90d={adv_90:,.0f} "
            f"(min={MIN_ADV:,.0f})"
        )
        return None

    # ── Calcoli quantitativi ──────────────────────────────────────────────────
    log_ret   = compute_log_returns(close)
    rv_series = compute_realized_volatility(log_ret, window=rv_window)
    rv_valid  = int(rv_series.notna().sum())

    # Verifica storia tramite valori RV effettivi (più robusto del conteggio righe raw)
    if rv_valid < percentile_lookback:
        logger.debug(
            f"{ticker}: storia RV insufficiente "
            f"({rv_valid} valori validi < {percentile_lookback} richiesti)"
        )
        return None

    pct_series  = compute_rv_percentile(rv_series, lookback=percentile_lookback)
    rv_current  = rv_series.iloc[-1]
    pct_current = pct_series.iloc[-1]

    if pd.isna(rv_current) or pd.isna(pct_current):
        logger.debug(f"{ticker}: RV o percentile NaN sull'ultimo giorno")
        return None

    # RV 52-week per contesto (min/max)
    rv_52w     = rv_series.iloc[-252:] if len(rv_series) >= 252 else rv_series
    rv_52w_min = rv_52w.min()
    rv_52w_max = rv_52w.max()

    return {
        "ticker":        ticker,
        "rv_current":    round(float(rv_current) * 100, 2),
        "rv_percentile": round(float(pct_current), 2),
        "rv_52w_min":    round(float(rv_52w_min) * 100, 2) if pd.notna(rv_52w_min) else None,
        "rv_52w_max":    round(float(rv_52w_max) * 100, 2) if pd.notna(rv_52w_max) else None,
        "adv_30d":       round(float(adv_30)),
        "adv_90d":       round(float(adv_90)),
        "close_price":   round(float(close.iloc[-1]), 2),
        "last_date":     df.index[-1].date().isoformat(),
        "is_compressed": float(pct_current) <= COMPRESSION_THRESHOLD,
    }


# ── Batch analysis ────────────────────────────────────────────────────────────
def run_analysis(
    ohlcv_data: Dict[str, pd.DataFrame],
    rv_window: int           = RV_WINDOW,
    percentile_lookback: int = PERCENTILE_LOOKBACK,
) -> pd.DataFrame:
    """
    Analisi batch su tutti i ticker. Restituisce DataFrame ordinato per
    rv_percentile ascending (più compressi in cima).

    Logga conteggio dettagliato dei fallimenti per diagnosi.
    """
    results        = []
    total          = len(ohlcv_data)
    fail_empty     = 0
    fail_adv       = 0
    fail_history   = 0
    fail_other     = 0

    for i, (ticker, df) in enumerate(ohlcv_data.items()):
        if df is None or df.empty:
            fail_empty += 1
            continue

        result = analyze_ticker(ticker, df, rv_window, percentile_lookback)
        if result is not None:
            results.append(result)
        else:
            # Diagnosi fallimento
            if "adjusted_close" in df.columns:
                n_valid = df["adjusted_close"].notna().sum()
                if n_valid < rv_window + 30:
                    fail_empty += 1
                else:
                    # Prova a capire se è ADV o storia
                    try:
                        vol = df["volume"].fillna(0) if "volume" in df.columns else pd.Series([0])
                        adv30 = compute_adv(vol, 30).iloc[-1]
                        adv90 = compute_adv(vol, 90).iloc[-1]
                        if pd.isna(adv30) or pd.isna(adv90) or adv30 < MIN_ADV or adv90 < MIN_ADV:
                            fail_adv += 1
                        else:
                            fail_history += 1
                    except Exception:
                        fail_other += 1
            else:
                fail_other += 1

        if (i + 1) % 100 == 0:
            logger.info(
                f"Analisi: {i+1}/{total} | qualificati={len(results)} | "
                f"no_data={fail_empty} adv_ko={fail_adv} "
                f"history_ko={fail_history} other={fail_other}"
            )

    logger.info(
        f"Analisi completata: {total} input → "
        f"{len(results)} qualificati | "
        f"{fail_empty} no/poco dato | "
        f"{fail_adv} ADV KO | "
        f"{fail_history} storia insufficiente | "
        f"{fail_other} altro"
    )

    if not results:
        logger.warning("Nessun ticker ha superato tutti i filtri quantitativi.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results).sort_values(
        "rv_percentile", ascending=True
    ).reset_index(drop=True)

    n_comp = int((df_out["rv_percentile"] <= COMPRESSION_THRESHOLD).sum())
    logger.info(f"COMPRESSION ZONE (≤{COMPRESSION_THRESHOLD}° pct): {n_comp} ticker")

    return df_out
