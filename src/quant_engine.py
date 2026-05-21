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
RV_WINDOW: int = 90              # Finestra RV in giorni lavorativi
PERCENTILE_LOOKBACK: int = 756   # Lookback percentile (~3 anni di trading days)
ANNUALIZATION_FACTOR: float = np.sqrt(252)

ADV_WINDOWS: list[int] = [30, 90]  # Finestre per Average Daily Volume
MIN_ADV: float = 1_500_000         # ADV minimo (share)
MIN_HISTORY_DAYS: int = 756        # Storia minima in giorni lavorativi
COMPRESSION_THRESHOLD: float = 5.0  # Percentile soglia per flag "Compresso"


# ── Phase A: Rendimenti Logaritmici ──────────────────────────────────────────
def compute_log_returns(close_series: pd.Series) -> pd.Series:
    """
    Calcola i rendimenti giornalieri continui (logaritmici).

    Formula: R_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    close_series : pd.Series
        Serie temporale dei prezzi di chiusura adjusted (index = date).

    Returns
    -------
    pd.Series
        Rendimenti logaritmici. Il primo elemento è NaN per costruzione.
    """
    return np.log(close_series / close_series.shift(1))


# ── Phase B: Realized Volatility ─────────────────────────────────────────────
def compute_realized_volatility(
    returns: pd.Series,
    window: int = RV_WINDOW,
) -> pd.Series:
    """
    Calcola la Realized Volatility annualizzata su finestra rolling.

    Formula: RV_{t,w} = sqrt(252) * sigma(R_{t-w+1...t})

    Parameters
    ----------
    returns : pd.Series
        Serie dei rendimenti logaritmici.
    window : int
        Finestra rolling in giorni lavorativi (default 90).

    Returns
    -------
    pd.Series
        RV annualizzata in forma decimale (es. 0.25 = 25%).
        Valori NaN per le prime `window` osservazioni.
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
    Calcola il rango percentile rolling del valore odierno di RV rispetto
    alla sua distribuzione storica nei precedenti `lookback` giorni.

    Interpretazione: un valore prossimo a 0 indica che la RV corrente è
    ai minimi storici (zona di compressione estrema).

    Implementazione: per ogni giorno t, considera la finestra di dimensione
    (lookback + 1). L'ultimo elemento della finestra è il valore "oggi";
    il percentile viene calcolato rispetto ai precedenti `lookback` elementi
    (= distribuzione storica).

    Parameters
    ----------
    rv_series : pd.Series
        Serie della Realized Volatility annualizzata.
    lookback : int
        Numero di osservazioni storiche (default 756 = ~3 anni).

    Returns
    -------
    pd.Series
        Percentile [0, 100] per ogni giorno. NaN dove non ci sono dati
        sufficienti.
    """
    def _percentile_of_last(window_arr: np.ndarray) -> float:
        """
        Dato un array di dimensione (lookback + 1), restituisce il percentile
        dell'ultimo elemento rispetto ai precedenti (lookback).
        """
        if len(window_arr) < 2:
            return np.nan
        current = window_arr[-1]
        historical = window_arr[:-1]

        if np.isnan(current):
            return np.nan

        valid_hist = historical[~np.isnan(historical)]
        if len(valid_hist) == 0:
            return np.nan

        # Percentile = fraction of historical values strictly below current
        pct = np.sum(valid_hist < current) / len(valid_hist) * 100.0
        return float(pct)

    # La finestra deve essere (lookback + 1) per includere il giorno corrente
    # più i `lookback` giorni di storia
    return rv_series.rolling(
        window=lookback + 1,
        min_periods=lookback + 1,
    ).apply(_percentile_of_last, raw=True)


# ── ADV filter ────────────────────────────────────────────────────────────────
def compute_adv(volume_series: pd.Series, window: int) -> pd.Series:
    """
    Calcola l'Average Daily Volume su finestra rolling.

    Parameters
    ----------
    volume_series : pd.Series
        Serie del volume giornaliero (shares traded).
    window : int
        Finestra in giorni lavorativi.

    Returns
    -------
    pd.Series
        ADV rolling. NaN dove i dati sono insufficienti.
    """
    return volume_series.rolling(window=window, min_periods=window).mean()


# ── Per-ticker full analysis ──────────────────────────────────────────────────
def analyze_ticker(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    rv_window: int = RV_WINDOW,
    percentile_lookback: int = PERCENTILE_LOOKBACK,
) -> Optional[dict]:
    """
    Esegue l'analisi quantitativa completa per un singolo ticker.

    Applica i filtri:
      - Storia minima: MIN_HISTORY_DAYS giorni lavorativi.
      - ADV 30d e ADV 90d entrambi >= MIN_ADV.

    Returns
    -------
    dict con le metriche chiave, oppure None se il ticker non supera i filtri.
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return None

    required = {"date", "adjusted_close", "volume"}
    missing = required - set(ohlcv_df.columns)
    if missing:
        logger.debug(f"{ticker}: colonne mancanti {missing}")
        return None

    df = ohlcv_df.set_index("date").sort_index()

    # Rimuovi righe con adjusted_close NaN
    df = df.dropna(subset=["adjusted_close"])

    # ── Filtro storia minima ──────────────────────────────────────────────────
    # Necessitiamo di almeno (lookback + rv_window) osservazioni per poter
    # calcolare il percentile dell'ultimo giorno
    min_required = percentile_lookback + rv_window
    if len(df) < min_required:
        logger.debug(
            f"{ticker}: storia insufficiente "
            f"({len(df)} gg < {min_required} richiesti)"
        )
        return None

    close = df["adjusted_close"]
    volume = df["volume"].fillna(0)

    # ── Filtro ADV ────────────────────────────────────────────────────────────
    adv_30 = compute_adv(volume, 30).iloc[-1]
    adv_90 = compute_adv(volume, 90).iloc[-1]

    if pd.isna(adv_30) or pd.isna(adv_90):
        return None
    if adv_30 < MIN_ADV or adv_90 < MIN_ADV:
        logger.debug(
            f"{ticker}: ADV filter KO "
            f"(30d={adv_30:,.0f}, 90d={adv_90:,.0f}, min={MIN_ADV:,.0f})"
        )
        return None

    # ── Calcoli quantitativi ──────────────────────────────────────────────────
    log_ret = compute_log_returns(close)
    rv_series = compute_realized_volatility(log_ret, window=rv_window)
    pct_series = compute_rv_percentile(rv_series, lookback=percentile_lookback)

    rv_current = rv_series.iloc[-1]
    pct_current = pct_series.iloc[-1]

    if pd.isna(rv_current) or pd.isna(pct_current):
        logger.debug(f"{ticker}: RV o percentile NaN all'ultimo giorno")
        return None

    # RV a 52 settimane (min/max) per contesto
    rv_52w = rv_series.iloc[-252:] if len(rv_series) >= 252 else rv_series
    rv_52w_min = rv_52w.min()
    rv_52w_max = rv_52w.max()

    return {
        "ticker": ticker,
        "rv_current": round(float(rv_current) * 100, 2),       # in % (es. 18.5)
        "rv_percentile": round(float(pct_current), 2),          # [0, 100]
        "rv_52w_min": round(float(rv_52w_min) * 100, 2) if pd.notna(rv_52w_min) else None,
        "rv_52w_max": round(float(rv_52w_max) * 100, 2) if pd.notna(rv_52w_max) else None,
        "adv_30d": round(float(adv_30)),
        "adv_90d": round(float(adv_90)),
        "close_price": round(float(close.iloc[-1]), 2),
        "last_date": df.index[-1].date().isoformat(),
        "is_compressed": float(pct_current) <= COMPRESSION_THRESHOLD,
    }


# ── Batch analysis ────────────────────────────────────────────────────────────
def run_analysis(
    ohlcv_data: Dict[str, pd.DataFrame],
    rv_window: int = RV_WINDOW,
    percentile_lookback: int = PERCENTILE_LOOKBACK,
) -> pd.DataFrame:
    """
    Esegue l'analisi su tutti i ticker e restituisce un DataFrame ordinato.

    Parameters
    ----------
    ohlcv_data : dict
        {ticker: pd.DataFrame con colonne date/adjusted_close/volume}
    rv_window : int
        Finestra per la Realized Volatility (default 90).
    percentile_lookback : int
        Lookback per il percentile rolling (default 756).

    Returns
    -------
    pd.DataFrame
        Tutti i ticker che superano i filtri ADV + storia, ordinati per
        rv_percentile ascending (più compressi in cima).
        DataFrame vuoto se nessun ticker supera i filtri.
    """
    results = []
    total = len(ohlcv_data)
    passed_adv = 0
    failed_history = 0
    failed_adv = 0

    for i, (ticker, df) in enumerate(ohlcv_data.items()):
        result = analyze_ticker(ticker, df, rv_window, percentile_lookback)

        if result is not None:
            results.append(result)
            passed_adv += 1
        else:
            # Distinguish failure reason for logging
            if df is not None and not df.empty:
                n_rows = len(df.dropna(subset=["adjusted_close"]) if "adjusted_close" in df.columns else df)
                if n_rows < percentile_lookback + rv_window:
                    failed_history += 1
                else:
                    failed_adv += 1

        if (i + 1) % 100 == 0:
            logger.info(
                f"Analisi: {i + 1}/{total} ticker processati, "
                f"{len(results)} qualificati finora"
            )

    logger.info(
        f"Analisi completata: {total} input → "
        f"{passed_adv} qualificati | "
        f"{failed_history} storia insufficiente | "
        f"{failed_adv} ADV sotto soglia"
    )

    if not results:
        logger.warning("Nessun ticker ha superato i filtri quantitativi.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Ordina per rv_percentile ascending (più compressi in cima)
    results_df = results_df.sort_values(
        "rv_percentile", ascending=True
    ).reset_index(drop=True)

    n_compressed = int((results_df["rv_percentile"] <= COMPRESSION_THRESHOLD).sum())
    logger.info(
        f"COMPRESSION ZONE (≤{COMPRESSION_THRESHOLD}th pct): "
        f"{n_compressed} ticker"
    )

    return results_df
