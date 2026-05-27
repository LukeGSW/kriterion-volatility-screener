"""
quant_engine.py - Motore matematico e quantitativo del Kriterion Quant Screener.

Implementa (completamente vettorizzato con pandas/numpy):
  A. Rendimenti logaritmici giornalieri
  B. Realized Volatility (RV) rolling annualizzata su finestre multiple (20/60/90)
  C. Rango Percentile rolling su lookback L=756 giorni lavorativi (~3 anni)
  D. Filtro ADV (Average Daily Volume) su finestre 30 e 90 giorni
  E. ATR(14) normalizzato (ATR/Close) e percentile rolling 252gg
  F. Term Structure RV (rv_20 / rv_60) come proxy di squeeze attivo
  G. Expansion Ratio (rv_52w_max / rv_current) con classificazione tier
  H. Borda ranking aggregato per selezione candidati Long Straddle
  I. Deal-Pending detector: identifica titoli oggetto di acquisizione

Nessuna logica di hard stop. Nessuna chiamata a dati di opzioni.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default parameters (allineati al documento di progettazione) ──────────────
RV_WINDOW: int              = 90    # Finestra RV "lunga" — regime storico
RV_SHORT_WINDOW: int        = 20    # Finestra RV "corta" — squeeze recente
RV_INTERMEDIATE_WINDOW: int = 60    # Finestra RV "intermedia" — regime recente
PERCENTILE_LOOKBACK: int    = 756   # Lookback percentile (~3 anni trading days)
ANNUALIZATION_FACTOR: float = np.sqrt(252)

# ATR
ATR_WINDOW: int                  = 14   # Finestra ATR (standard Wilder)
ATR_PERCENTILE_LOOKBACK: int     = 252  # Lookback percentile ATR (~1 anno trading)

ADV_WINDOWS: list[int]  = [30, 90]
MIN_ADV: float          = 1_500_000

# Soglie operative
COMPRESSION_THRESHOLD: float = 5.0     # Percentile soglia flag "Compresso" (legacy)
STRADDLE_GATE_PCT: float     = 20.0    # Gate RV percentile per candidati straddle

# Expansion ratio tier — basati sulla matematica del P/L straddle ATM
# (vedi documentazione di progetto):
#   - ratio < 2.0   → INSUFFICIENT  (espansione non sufficiente per target operativi)
#   - 2.0 ≤ r < 3.0 → LOW           (target realistico: +50% premio)
#   - 3.0 ≤ r < 4.5 → MEDIUM        (target realistico: +100% premio)
#   - r ≥ 4.5       → HIGH          (target realistico: +200% premio)
EXPANSION_TIER_LOW: float    = 2.0
EXPANSION_TIER_MEDIUM: float = 3.0
EXPANSION_TIER_HIGH: float   = 4.5

# ── Deal-Pending detector ─────────────────────────────────────────────────────
# Identifica titoli oggetto di acquisizione con deal price definito.
# Questi titoli mostrano una firma statistica inequivocabile:
#   - Price pinning estremo (close clustered intorno al deal price)
#   - Range giornaliero collassato (arbitraggio meccanico)
#   - Volume spike storico (giorno dell'annuncio)
# Sono i PEGGIORI candidati per Long Straddle: vol-dead per natura.
DEAL_CV_WINDOW: int                = 30      # giorni per CV close
DEAL_RANGE_WINDOW: int             = 20      # giorni per range relativo
DEAL_VOLUME_WINDOW: int            = 180     # giorni per volume spike (~6m)
DEAL_CV_THRESHOLD: float           = 0.005   # CV close < 0.5%  → price pinning
DEAL_RANGE_THRESHOLD: float        = 0.008   # range medio < 0.8% → range collassato
DEAL_VOLUME_SPIKE_THRESHOLD: float = 5.0     # max/median volume > 5x → spike annuncio

# Storia minima in valori RV validi (non righe raw).
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


# ── ATR ───────────────────────────────────────────────────────────────────────
def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ATR_WINDOW,
) -> pd.Series:
    """
    Average True Range (Wilder).
    True Range_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    ATR_t = SMA(TR, window)   (qui usiamo SMA per semplicità e replicabilità;
                                la formula Wilder con EMA produce risultati
                                molto correlati per window=14 sui ranking).

    Returns
    -------
    pd.Series — ATR in unità di prezzo. NaN prime `window` osservazioni.
    """
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def compute_atr_normalized(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ATR_WINDOW,
) -> pd.Series:
    """
    ATR normalizzato: ATR(window) / Close.
    Esprime il range giornaliero medio come frazione del prezzo,
    rendendo l'indicatore confrontabile cross-section.

    Returns
    -------
    pd.Series — ATR%/100 (es. 0.025 = 2.5% del prezzo).
    """
    atr = compute_atr(high, low, close, window=window)
    return atr / close


def compute_atr_pct_percentile(
    atr_normalized: pd.Series,
    lookback: int = ATR_PERCENTILE_LOOKBACK,
) -> pd.Series:
    """
    Percentile rolling dell'ATR normalizzato corrente vs lookback recente.
    Logica identica a compute_rv_percentile ma con lookback più corto
    (252gg vs 756gg) per maggiore reattività agli squeeze in formazione.
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

    return atr_normalized.rolling(
        window=lookback + 1,
        min_periods=lookback + 1,
    ).apply(_pct_of_last, raw=True)


# ── ADV ───────────────────────────────────────────────────────────────────────
def compute_adv(volume_series: pd.Series, window: int) -> pd.Series:
    """Average Daily Volume rolling su finestra `window` giorni."""
    return volume_series.rolling(window=window, min_periods=window).mean()


# ── Expansion tier classification ─────────────────────────────────────────────
def classify_expansion_tier(ratio: float) -> str:
    """
    Classifica il potenziale di espansione in base al rapporto rv_52w_max / rv_current.

    INSUFFICIENT : ratio < 2.0   — espansione strutturalmente insufficiente
    LOW          : 2.0–3.0       — target operativo realistico ≈ +50% premio
    MEDIUM       : 3.0–4.5       — target operativo realistico ≈ +100% premio
    HIGH         : ≥ 4.5         — target operativo realistico ≈ +200% premio
    """
    if pd.isna(ratio):
        return "N/A"
    r = float(ratio)
    if r < EXPANSION_TIER_LOW:
        return "INSUFFICIENT"
    if r < EXPANSION_TIER_MEDIUM:
        return "LOW"
    if r < EXPANSION_TIER_HIGH:
        return "MEDIUM"
    return "HIGH"


# ── Deal-Pending detector ─────────────────────────────────────────────────────
def compute_deal_pending_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    cv_window: int           = DEAL_CV_WINDOW,
    range_window: int        = DEAL_RANGE_WINDOW,
    volume_window: int       = DEAL_VOLUME_WINDOW,
    cv_threshold: float      = DEAL_CV_THRESHOLD,
    range_threshold: float   = DEAL_RANGE_THRESHOLD,
    vol_spike_threshold: float = DEAL_VOLUME_SPIKE_THRESHOLD,
) -> dict:
    """
    Rileva titoli "deal-pending" (oggetto di acquisizione con prezzo definito).

    Calcola tre signature statistiche:
      1. CV close (Coefficient of Variation) sugli ultimi cv_window giorni:
         std(close) / mean(close). Valori molto bassi indicano price pinning.
      2. Range relativo medio sugli ultimi range_window giorni:
         mean((H - L) / Close). Valori bassi indicano range giornaliero collassato.
      3. Volume spike sugli ultimi volume_window giorni:
         max(volume) / median(volume). Valori alti indicano un evento
         (tipicamente l'annuncio del deal).

    Il flag is_deal_pending e' True quando TUTTE e tre le condizioni sono soddisfatte
    simultaneamente — combinazione che è praticamente esclusiva dei deal-pending.

    Returns
    -------
    dict con keys: cv_30, range_rel_20, volume_spike_180, is_deal_pending
    """
    # CV close
    recent_close = close.tail(cv_window)
    if len(recent_close) < cv_window:
        cv = None
    else:
        mu = recent_close.mean()
        if mu is None or pd.isna(mu) or mu == 0:
            cv = None
        else:
            sigma = recent_close.std()
            cv = float(sigma / mu) if pd.notna(sigma) else None

    # Range relativo medio
    recent_high  = high.tail(range_window)
    recent_low   = low.tail(range_window)
    recent_cls   = close.tail(range_window)
    if (
        len(recent_high) < range_window
        or len(recent_low) < range_window
        or len(recent_cls) < range_window
    ):
        rr = None
    else:
        # Mantieni solo righe con close > 0 e high/low validi
        valid_mask = (recent_cls > 0) & recent_high.notna() & recent_low.notna()
        if valid_mask.sum() < range_window // 2:
            rr = None
        else:
            rng_rel = (
                (recent_high[valid_mask] - recent_low[valid_mask])
                / recent_cls[valid_mask]
            )
            rr = float(rng_rel.mean()) if not rng_rel.empty else None

    # Volume spike
    recent_vol = volume.tail(volume_window)
    valid_vol  = recent_vol[recent_vol > 0]
    if len(valid_vol) < volume_window // 2:
        vs = None
    else:
        med = valid_vol.median()
        if med is None or pd.isna(med) or med == 0:
            vs = None
        else:
            vs = float(valid_vol.max() / med)

    # Flag combinato: serve TUTTE e tre le condizioni
    if cv is None or rr is None or vs is None:
        is_deal_pending = False
    else:
        is_deal_pending = bool(
            cv < cv_threshold
            and rr < range_threshold
            and vs > vol_spike_threshold
        )

    return {
        "cv_30":            round(cv, 5) if cv is not None else None,
        "range_rel_20":     round(rr, 5) if rr is not None else None,
        "volume_spike_180": round(vs, 2) if vs is not None else None,
        "is_deal_pending":  is_deal_pending,
    }


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

    # Per ATR servono anche high/low; se mancano, fallback a sole metriche RV.
    has_hl = {"high", "low"}.issubset(set(ohlcv_df.columns))

    required = {"date", "adjusted_close", "volume"}
    missing  = required - set(ohlcv_df.columns)
    if missing:
        logger.debug(f"{ticker}: colonne mancanti {missing}")
        return None

    df = ohlcv_df.set_index("date").sort_index()
    df = df.dropna(subset=["adjusted_close"])

    if len(df) < rv_window + 30:
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

    # ── Calcoli quantitativi RV ───────────────────────────────────────────────
    log_ret   = compute_log_returns(close)

    rv_long   = compute_realized_volatility(log_ret, window=rv_window)              # default 90
    rv_short  = compute_realized_volatility(log_ret, window=RV_SHORT_WINDOW)        # 20
    rv_mid    = compute_realized_volatility(log_ret, window=RV_INTERMEDIATE_WINDOW) # 60

    rv_valid  = int(rv_long.notna().sum())

    if rv_valid < percentile_lookback:
        logger.debug(
            f"{ticker}: storia RV insufficiente "
            f"({rv_valid} valori validi < {percentile_lookback} richiesti)"
        )
        return None

    pct_series  = compute_rv_percentile(rv_long, lookback=percentile_lookback)

    rv_current      = rv_long.iloc[-1]
    rv_short_curr   = rv_short.iloc[-1]
    rv_mid_curr     = rv_mid.iloc[-1]
    pct_current     = pct_series.iloc[-1]

    if pd.isna(rv_current) or pd.isna(pct_current):
        logger.debug(f"{ticker}: RV o percentile NaN sull'ultimo giorno")
        return None

    # ── Term structure (squeeze attivo) ───────────────────────────────────────
    if pd.notna(rv_short_curr) and pd.notna(rv_mid_curr) and rv_mid_curr > 0:
        rv_term_structure = float(rv_short_curr / rv_mid_curr)
    else:
        rv_term_structure = None

    # ── RV 52-week per contesto e expansion ratio ─────────────────────────────
    rv_52w     = rv_long.iloc[-252:] if len(rv_long) >= 252 else rv_long
    rv_52w_min = rv_52w.min()
    rv_52w_max = rv_52w.max()

    if pd.notna(rv_52w_max) and rv_current > 0:
        expansion_ratio = float(rv_52w_max / rv_current)
    else:
        expansion_ratio = None

    expansion_tier = classify_expansion_tier(
        expansion_ratio if expansion_ratio is not None else float("nan")
    )

    # ── ATR(14)/Close e percentile 252gg ──────────────────────────────────────
    atr_pct_current   = None
    atr_pct_pctile    = None
    deal_signals: dict = {
        "cv_30": None,
        "range_rel_20": None,
        "volume_spike_180": None,
        "is_deal_pending": False,
    }
    if has_hl:
        high = df["high"]
        low  = df["low"]
        atr_norm = compute_atr_normalized(high, low, close, window=ATR_WINDOW)
        atr_pct_series = compute_atr_pct_percentile(
            atr_norm, lookback=ATR_PERCENTILE_LOOKBACK
        )
        atr_last       = atr_norm.iloc[-1]
        atr_pctile_last = atr_pct_series.iloc[-1]
        if pd.notna(atr_last):
            atr_pct_current = round(float(atr_last) * 100, 3)  # in %
        if pd.notna(atr_pctile_last):
            atr_pct_pctile = round(float(atr_pctile_last), 2)

        # Detector deal-pending (richiede high/low/volume)
        deal_signals = compute_deal_pending_signals(close, high, low, volume)

    # ── Flag operativi ────────────────────────────────────────────────────────
    is_compressed   = float(pct_current) <= COMPRESSION_THRESHOLD
    is_deal_pending = bool(deal_signals["is_deal_pending"])

    # Gate Long Straddle:
    #   - rv_percentile_90 ≤ STRADDLE_GATE_PCT  (regime compresso)
    #   - rv_20 < rv_60                         (term structure inversa, squeeze attivo)
    #   - NOT is_deal_pending                   (esclude titoli oggetto di acquisizione)
    gate_pct  = float(pct_current) <= STRADDLE_GATE_PCT
    gate_term = (
        rv_term_structure is not None and rv_term_structure < 1.0
    )
    is_straddle_candidate = bool(gate_pct and gate_term and not is_deal_pending)

    return {
        "ticker":           ticker,
        "rv_current":       round(float(rv_current) * 100, 2),
        "rv_percentile":    round(float(pct_current), 2),
        "rv_52w_min":       round(float(rv_52w_min) * 100, 2) if pd.notna(rv_52w_min) else None,
        "rv_52w_max":       round(float(rv_52w_max) * 100, 2) if pd.notna(rv_52w_max) else None,
        # Nuove metriche multi-window
        "rv_20":            round(float(rv_short_curr) * 100, 2) if pd.notna(rv_short_curr) else None,
        "rv_60":            round(float(rv_mid_curr)   * 100, 2) if pd.notna(rv_mid_curr)   else None,
        "rv_term_structure": round(rv_term_structure, 3) if rv_term_structure is not None else None,
        # ATR
        "atr_pct":          atr_pct_current,    # ATR(14)/Close in %
        "atr_pct_percentile": atr_pct_pctile,   # percentile 252gg
        # Expansion
        "expansion_ratio":  round(expansion_ratio, 2) if expansion_ratio is not None else None,
        "expansion_tier":   expansion_tier,
        # Volume / prezzo
        "adv_30d":          round(float(adv_30)),
        "adv_90d":          round(float(adv_90)),
        "close_price":      round(float(close.iloc[-1]), 2),
        "last_date":        df.index[-1].date().isoformat(),
        # Deal-pending detector
        "cv_30":            deal_signals["cv_30"],
        "range_rel_20":     deal_signals["range_rel_20"],
        "volume_spike_180": deal_signals["volume_spike_180"],
        # Flags
        "is_compressed":         is_compressed,
        "is_deal_pending":       is_deal_pending,
        "is_straddle_candidate": is_straddle_candidate,
    }


# ── Borda ranking aggregato ───────────────────────────────────────────────────
def compute_borda_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il ranking Borda aggregato sui ticker candidati Long Straddle.

    Tre metriche di rank (tutte ascending, rank 1 = "piu' estremo"):
        R1 = rank(rv_percentile)         — regime compresso (lookback 3y)
        R2 = rank(atr_pct_percentile)    — ATR(14)/Close percentile 1y
        R3 = rank(rv_term_structure)     — intensita' squeeze (rv_20/rv_60)

    Borda Rank totale = R1 + R2 + R3.
    Piu' basso = candidato migliore.

    Tie-break su rv_percentile (metrica piu' strutturalmente robusta).

    Il ranking e' calcolato SOLO sui ticker che passano il gate
    (is_straddle_candidate == True). Per gli altri le colonne ranking
    sono NaN.

    Parameters
    ----------
    df : DataFrame con almeno le colonne
        ['rv_percentile', 'atr_pct_percentile', 'rv_term_structure',
         'is_straddle_candidate']

    Returns
    -------
    DataFrame con colonne aggiunte:
        rank_rv_pct, rank_atr_pct, rank_term_structure,
        borda_score, borda_rank
    """
    out = df.copy()

    # Inizializza colonne a NaN
    out["rank_rv_pct"]         = np.nan
    out["rank_atr_pct"]        = np.nan
    out["rank_term_structure"] = np.nan
    out["borda_score"]         = np.nan
    out["borda_rank"]          = np.nan

    if "is_straddle_candidate" not in out.columns:
        return out

    mask = out["is_straddle_candidate"] == True
    cand = out.loc[mask].copy()

    if cand.empty:
        return out

    # Rank ascending: valore piu' basso → rank 1
    # method="min" gestisce i pareggi in modo deterministico
    cand["rank_rv_pct"] = cand["rv_percentile"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    cand["rank_atr_pct"] = cand["atr_pct_percentile"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    cand["rank_term_structure"] = cand["rv_term_structure"].rank(
        method="min", ascending=True, na_option="bottom"
    )

    # Borda score = somma dei rank (piu' basso = meglio)
    cand["borda_score"] = (
        cand["rank_rv_pct"].fillna(len(cand))
        + cand["rank_atr_pct"].fillna(len(cand))
        + cand["rank_term_structure"].fillna(len(cand))
    )

    # Rank finale: ordina per borda_score, tie-break su rv_percentile
    cand_sorted = cand.sort_values(
        by=["borda_score", "rv_percentile"],
        ascending=[True, True],
        kind="mergesort",
    )
    cand_sorted["borda_rank"] = np.arange(1, len(cand_sorted) + 1)

    # Riporta indietro nelle posizioni originali
    for col in ["rank_rv_pct", "rank_atr_pct", "rank_term_structure",
                "borda_score", "borda_rank"]:
        out.loc[cand_sorted.index, col] = cand_sorted[col]

    return out


# ── Batch analysis ────────────────────────────────────────────────────────────
def run_analysis(
    ohlcv_data: Dict[str, pd.DataFrame],
    rv_window: int           = RV_WINDOW,
    percentile_lookback: int = PERCENTILE_LOOKBACK,
) -> pd.DataFrame:
    """
    Analisi batch su tutti i ticker. Restituisce DataFrame ordinato per
    rv_percentile ascending (piu' compressi in cima).

    Aggiunge ranking Borda per candidati Long Straddle.
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

    # ── Deal-pending diagnostics ──────────────────────────────────────────────
    if "is_deal_pending" in df_out.columns:
        n_deal = int((df_out["is_deal_pending"] == True).sum())
        if n_deal > 0:
            deal_tickers = df_out.loc[
                df_out["is_deal_pending"] == True, "ticker"
            ].tolist()
            logger.info(
                f"DEAL-PENDING rilevati: {n_deal} ticker → "
                f"{', '.join(deal_tickers[:20])}"
                + (" ..." if n_deal > 20 else "")
            )

    # ── Borda ranking sui candidati straddle ──────────────────────────────────
    df_out = compute_borda_ranking(df_out)

    n_cand = int((df_out["is_straddle_candidate"] == True).sum())
    logger.info(
        f"STRADDLE CANDIDATES "
        f"(gate: pct≤{STRADDLE_GATE_PCT} & rv_20<rv_60 & NOT deal_pending): "
        f"{n_cand} ticker"
    )

    return df_out
