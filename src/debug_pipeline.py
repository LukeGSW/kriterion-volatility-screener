"""
debug_pipeline.py — Script di diagnostica per il Kriterion Quant Screener.

Esegui con:
    EODHD_API_KEY=xxx python src/debug_pipeline.py

Testa ogni stadio del pipeline in isolamento e identifica il collo di bottiglia.
Output: report testuale dettagliato su console.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Aggiungi src/ al path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from quant_engine import (
        MIN_ADV,
        PERCENTILE_LOOKBACK,
        RV_WINDOW,
        compute_adv,
        compute_log_returns,
        compute_realized_volatility,
        compute_rv_percentile,
    )
    QUANT_ENGINE_OK = True
except ImportError as e:
    QUANT_ENGINE_OK = False
    QUANT_ENGINE_ERR = str(e)

BASE_URL = "https://eodhd.com/api"
SEP      = "=" * 68
SEP_THIN = "-" * 68


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str)   -> None: print(f"  ✓  {msg}")
def fail(msg: str) -> None: print(f"  ✗  {msg}")
def info(msg: str) -> None: print(f"  →  {msg}")
def warn(msg: str) -> None: print(f"  ⚠  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Screener grezzo senza filtri (verifica struttura risposta)
# ─────────────────────────────────────────────────────────────────────────────
def test_screener_raw(api_key: str) -> dict | None:
    section("TEST 1 — Screener API: risposta grezza senza filtri (limit=5)")

    url = f"{BASE_URL}/screener"
    params = {"api_token": api_key, "limit": 5, "fmt": "json"}

    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"  HTTP status: {r.status_code}")

        if r.status_code != 200:
            fail(f"Errore HTTP {r.status_code}: {r.text[:200]}")
            return None

        data = r.json()
    except Exception as e:
        fail(f"Eccezione: {e}")
        return None

    print(f"  Tipo risposta: {type(data).__name__}")

    if isinstance(data, dict):
        keys = list(data.keys())
        info(f"Chiavi dict: {keys}")
        total = data.get("total", "ASSENTE")
        info(f"Campo 'total': {total}")
        records = data.get("data", data.get("results", []))
        info(f"Records nella pagina: {len(records)}")

        if records:
            first = records[0]
            info(f"Campi del primo record: {list(first.keys())}")
            print(f"\n  --- Primo record completo ---")
            for k, v in first.items():
                print(f"      {k:35s}: {v}")
            print()

            # Controlla valore di market_capitalization per i primi 3
            print(f"  market_capitalization nei primi record:")
            for rec in records[:5]:
                mc  = rec.get("market_capitalization", rec.get("market_cap", "N/A"))
                exc = rec.get("exchange", "?")
                typ = rec.get("type", "?")
                print(f"      {rec.get('code','?'):10s} | {exc:8s} | {typ:15s} | market_cap={mc}")

        return data

    elif isinstance(data, list):
        warn("La risposta è una LISTA (non un dict) — struttura diversa dall'atteso!")
        info(f"Lunghezza lista: {len(data)}")
        if data:
            info(f"Campi primo elemento: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
            print(f"  Primo elemento: {json.dumps(data[0], default=str)[:300]}")
        return {"data": data, "total": len(data)}

    else:
        fail(f"Risposta di tipo inatteso: {type(data)} — {str(data)[:300]}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Indagine unità market_capitalization nel filtro screener
# ─────────────────────────────────────────────────────────────────────────────
def test_screener_cap_units(api_key: str) -> int | None:
    """
    Prova tre soglie diverse per capire in quali unità EODHD interpreta
    il campo market_capitalization nel parametro filters.

    Scenario atteso:
      Se unità = USD      → soglia 2_000_000_000  restituisce ~600-800 ticker US
      Se unità = migliaia → soglia 2_000_000      restituisce ~600-800 ticker US
      Se unità = milioni  → soglia 2_000          restituisce ~600-800 ticker US

    Restituisce la soglia corretta da usare nel codice.
    """
    section("TEST 2 — Unità market_cap nel filtro screener (auto-detection)")

    candidates = [
        (2_000_000_000, "USD puri       (2,000,000,000)"),
        (2_000_000,     "Migliaia USD   (2,000,000)    "),
        (2_000,         "Milioni USD    (2,000)        "),
    ]

    correct_threshold = None
    url = f"{BASE_URL}/screener"

    for threshold, label in candidates:
        filters = json.dumps([["market_capitalization", ">=", threshold]])
        params  = {"api_token": api_key, "filters": filters, "limit": 1, "fmt": "json"}
        try:
            r    = requests.get(url, params=params, timeout=30)
            data = r.json()
            total = int(data.get("total", 0)) if isinstance(data, dict) else len(data)
            status = "✓ PLAUSIBILE" if 300 <= total <= 3000 else ("⚠ TROPPO BASSO" if total < 300 else "⚠ TROPPO ALTO")
            print(f"  threshold={label} → total={total:6d}  {status}")
            if 300 <= total <= 3000 and correct_threshold is None:
                correct_threshold = threshold
        except Exception as e:
            print(f"  threshold={label} → ERRORE: {e}")
        time.sleep(0.4)

    if correct_threshold:
        ok(f"Soglia corretta rilevata: {correct_threshold:,}")
        info(f"Usa questo valore in data_fetcher.py e pipeline.py")
    else:
        warn("Nessuna soglia produce un risultato plausibile (300-3000 ticker US).")
        info("Potrebbe esserci un problema con il piano API o con i parametri del filtro.")

    return correct_threshold


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Screener con filtri attivi (struttura e numero risultati)
# ─────────────────────────────────────────────────────────────────────────────
def test_screener_filtered(api_key: str, threshold: int | None) -> int:
    section("TEST 3 — Screener con filtro market_cap attivo")

    if threshold is None:
        threshold = 2_000_000_000
        warn(f"Soglia non rilevata automaticamente, uso default {threshold:,}")

    url     = f"{BASE_URL}/screener"
    filters = json.dumps([["market_capitalization", ">=", threshold]])
    params  = {
        "api_token": api_key,
        "filters":   filters,
        "limit":     10,
        "fmt":       "json",
    }

    try:
        r    = requests.get(url, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        fail(f"Eccezione: {e}")
        return 0

    total   = int(data.get("total", 0)) if isinstance(data, dict) else 0
    records = data.get("data", []) if isinstance(data, dict) else data[:10]

    info(f"Total (API): {total}")
    info(f"Records nella prima pagina: {len(records)}")

    exchange_counts: dict = {}
    type_counts: dict     = {}
    for rec in records:
        exc = rec.get("exchange", "?")
        typ = rec.get("type", "?")
        exchange_counts[exc] = exchange_counts.get(exc, 0) + 1
        type_counts[typ]     = type_counts.get(typ,     0) + 1
        mc  = rec.get("market_capitalization", rec.get("market_cap", "N/A"))
        print(f"    {rec.get('code','?'):10s} | {exc:8s} | {typ:15s} | mc={mc}")

    if total > 0:
        ok(f"Screener funziona: {total} ticker totali trovati con filtro >= {threshold:,}")
    else:
        fail("Screener restituisce 0 risultati con questo filtro.")

    return total


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Exchange Symbol List (alternativa al Screener)
# ─────────────────────────────────────────────────────────────────────────────
def test_exchange_list(api_key: str) -> list:
    section("TEST 4 — Exchange Symbol List /api/exchange-symbol-list/US")

    url    = f"{BASE_URL}/exchange-symbol-list/US"
    params = {"api_token": api_key, "fmt": "json"}

    try:
        r    = requests.get(url, params=params, timeout=60)
        data = r.json()
    except Exception as e:
        fail(f"Eccezione: {e}")
        return []

    if isinstance(data, list):
        ok(f"Lista exchange US: {len(data)} ticker totali")
        if data:
            info(f"Campi disponibili: {list(data[0].keys())}")
            for item in data[:3]:
                print(f"    {json.dumps(item)}")
        return data
    else:
        warn(f"Risposta inattesa: {type(data)} — {str(data)[:300]}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — OHLCV per un ticker noto (AAPL)
# ─────────────────────────────────────────────────────────────────────────────
def test_ohlcv(api_key: str, ticker: str = "AAPL") -> list:
    section(f"TEST 5 — OHLCV per {ticker} (4 anni di storia)")

    today     = datetime.utcnow().date()
    from_date = (today - timedelta(days=4 * 365)).isoformat()

    url    = f"{BASE_URL}/eod/{ticker}.US"
    params = {
        "api_token":     api_key,
        "from":          from_date,
        "to":            today.isoformat(),
        "adjusted_close":"true",
        "fmt":           "json",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"  HTTP status: {r.status_code}")
        data = r.json()
    except Exception as e:
        fail(f"Eccezione: {e}")
        return []

    if isinstance(data, list) and data:
        ok(f"Righe ricevute: {len(data)}")
        info(f"Colonne: {list(data[0].keys())}")
        info(f"Prima riga: {data[0]}")
        info(f"Ultima riga: {data[-1]}")

        # Verifica campo adjusted_close
        has_adj = "adjusted_close" in data[0]
        if has_adj:
            ok("Campo 'adjusted_close' presente")
        else:
            warn("Campo 'adjusted_close' ASSENTE — presente solo 'close'")
            info(f"Campi disponibili: {list(data[0].keys())}")

        return data
    else:
        fail(f"Risposta inattesa o vuota: {type(data)} — {str(data)[:300]}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Motore quantitativo su AAPL (simula l'analisi reale)
# ─────────────────────────────────────────────────────────────────────────────
def test_quant_engine(ohlcv_data: list, ticker: str = "AAPL") -> None:
    section(f"TEST 6 — Motore quantitativo su {ticker}")

    if not QUANT_ENGINE_OK:
        fail(f"quant_engine.py non importabile: {QUANT_ENGINE_ERR}")
        return

    if not ohlcv_data:
        warn("Nessun dato OHLCV da analizzare — salta.")
        return

    # Build DataFrame
    df = pd.DataFrame(ohlcv_data)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # adjusted_close o fallback a close
    if "adjusted_close" in df.columns:
        df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    elif "close" in df.columns:
        warn("adjusted_close assente, uso 'close' come fallback")
        df["adjusted_close"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        fail("Nessuna colonna di prezzo trovata.")
        return

    df["volume"] = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df.dropna(subset=["adjusted_close"])

    n_rows = len(df)
    min_req = PERCENTILE_LOOKBACK + RV_WINDOW  # 756 + 90 = 846
    info(f"Righe dopo pulizia: {n_rows}")
    info(f"Periodo: {df['date'].min().date()} → {df['date'].max().date()}")
    print()

    # ── Storia minima ──────────────────────────────────────────────────────
    print(f"  {'STORIA MINIMA':35s}: richiesti={min_req} gg, disponibili={n_rows} gg", end="  ")
    if n_rows >= min_req:
        ok("OK")
    else:
        fail(f"MANCANO {min_req - n_rows} giorni lavorativi")
        info("Fix: aumenta HISTORY_BUFFER_DAYS in pipeline.py (da 90 a 365+)")

    # ── ADV ────────────────────────────────────────────────────────────────
    adv_30 = compute_adv(df["volume"], 30).iloc[-1]
    adv_90 = compute_adv(df["volume"], 90).iloc[-1]
    print(f"  {'ADV 30d':35s}: {adv_30:>12,.0f}  ", end="")
    ok("OK") if adv_30 >= MIN_ADV else fail(f"sotto soglia {MIN_ADV:,.0f}")
    print(f"  {'ADV 90d':35s}: {adv_90:>12,.0f}  ", end="")
    ok("OK") if adv_90 >= MIN_ADV else fail(f"sotto soglia {MIN_ADV:,.0f}")

    # ── RV rolling ────────────────────────────────────────────────────────
    log_ret  = compute_log_returns(df["adjusted_close"])
    rv_series = compute_realized_volatility(log_ret, window=RV_WINDOW)
    rv_valid  = rv_series.notna().sum()
    rv_now    = rv_series.iloc[-1]
    print(f"  {'RV valori non-NaN':35s}: {rv_valid}  ", end="")
    ok("OK") if rv_valid > 0 else fail("nessun valore RV calcolato")
    if pd.notna(rv_now):
        info(f"RV corrente ({RV_WINDOW}d): {rv_now * 100:.2f}%")

    # ── Percentile rolling ────────────────────────────────────────────────
    pct_series = compute_rv_percentile(rv_series, lookback=PERCENTILE_LOOKBACK)
    pct_valid  = pct_series.notna().sum()
    pct_now    = pct_series.iloc[-1]
    print(f"  {'Percentile valori non-NaN':35s}: {pct_valid}  ", end="")
    ok("OK") if pct_valid > 0 else fail("nessun percentile calcolato")

    if pd.notna(pct_now):
        label = "⚡ COMPRESSO!" if pct_now <= 5 else ("vicino" if pct_now <= 15 else "nella norma")
        info(f"RV Percentile attuale: {pct_now:.2f}° — {label}")
    else:
        warn("RV Percentile è NaN sull'ultimo giorno")
        needed = PERCENTILE_LOOKBACK + RV_WINDOW + 1
        info(f"Per percentile valido servono {needed} gg lavorativi di storia ({needed - n_rows:+d} rispetto al disponibile)")

    # ── Distribuzione RV ──────────────────────────────────────────────────
    if pct_valid > 0:
        print()
        print(f"  Distribuzione percentile RV (su {pct_valid} valori calcolati):")
        for q in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
            v = np.nanpercentile(pct_series.dropna(), q)
            print(f"      p{q:3d}: {v:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Simulazione pipeline completo su 3 ticker noti
# ─────────────────────────────────────────────────────────────────────────────
def test_pipeline_sample(api_key: str) -> None:
    section("TEST 7 — Pipeline completo su 3 ticker campione (AAPL, MSFT, NVDA)")

    tickers   = ["AAPL", "MSFT", "NVDA"]
    today     = datetime.utcnow().date()
    from_date = (today - timedelta(days=4 * 365)).isoformat()

    for ticker in tickers:
        print(f"\n  {SEP_THIN}")
        print(f"  {ticker}")
        print(f"  {SEP_THIN}")
        url    = f"{BASE_URL}/eod/{ticker}.US"
        params = {
            "api_token":      api_key,
            "from":           from_date,
            "to":             today.isoformat(),
            "adjusted_close": "true",
            "fmt":            "json",
        }
        try:
            r    = requests.get(url, params=params, timeout=30)
            data = r.json()
        except Exception as e:
            fail(f"OHLCV fetch fallito: {e}")
            continue

        if not isinstance(data, list) or not data:
            fail(f"Risposta vuota o non lista: {type(data)}")
            continue

        test_quant_engine(data, ticker)
        time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Fundamentals endpoint (market cap per AAPL)
# ─────────────────────────────────────────────────────────────────────────────
def test_fundamentals_marketcap(api_key: str) -> None:
    section("TEST 8 — Fundamentals endpoint: market_cap di AAPL (verifica unità)")

    url    = f"{BASE_URL}/fundamentals/AAPL.US"
    params = {"api_token": api_key, "filter": "General", "fmt": "json"}

    try:
        r    = requests.get(url, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        fail(f"Eccezione: {e}")
        return

    if not isinstance(data, dict):
        warn(f"Risposta inattesa: {type(data)}")
        return

    general = data.get("General", data)
    mc_raw  = general.get("MarketCapitalization", general.get("market_capitalization", "N/A"))
    info(f"MarketCapitalization AAPL (raw): {mc_raw}")

    if mc_raw != "N/A":
        try:
            mc_float = float(mc_raw)
            if mc_float > 1e12:
                ok(f"Unità = USD puri (AAPL = ${mc_float/1e12:.2f}T)")
                info("→ usa threshold=2_000_000_000 nel filtro screener")
            elif mc_float > 1e9:
                ok(f"Unità = migliaia USD (AAPL = ${mc_float/1e9:.2f}B in migliaia)")
                info("→ usa threshold=2_000_000 nel filtro screener")
            elif mc_float > 1e6:
                ok(f"Unità = milioni USD (AAPL = ${mc_float/1e6:.2f}T in milioni)")
                info("→ usa threshold=2_000 nel filtro screener")
            else:
                warn(f"Valore inatteso: {mc_float}")
        except (TypeError, ValueError):
            warn(f"Non numerico: {mc_raw}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("EODHD_API_KEY", "").strip()
    if not api_key:
        print("ERRORE: variabile d'ambiente EODHD_API_KEY non impostata.")
        print("  Esegui: EODHD_API_KEY=<tuo_token> python src/debug_pipeline.py")
        sys.exit(1)

    print(f"\n{'⚡ KRITERION QUANT — DIAGNOSTICA PIPELINE':^68}")
    print(f"{'Run: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'):^68}")

    # Stadio 1: struttura risposta screener
    test_screener_raw(api_key)

    # Stadio 2: rileva unità market_cap
    correct_threshold = test_screener_cap_units(api_key)

    # Stadio 3: screener con filtro
    test_screener_filtered(api_key, correct_threshold)

    # Stadio 4: exchange list (alternativa)
    test_exchange_list(api_key)

    # Stadio 5: OHLCV per AAPL
    ohlcv = test_ohlcv(api_key, "AAPL")

    # Stadio 6: motore quantitativo su AAPL
    test_quant_engine(ohlcv, "AAPL")

    # Stadio 7: pipeline completo su 3 ticker
    test_pipeline_sample(api_key)

    # Stadio 8: fundamentals per verifica unità
    test_fundamentals_marketcap(api_key)

    print(f"\n{SEP}")
    print("  Diagnostica completata.")
    print("  Incolla l'output completo per analisi approfondita.")
    print(SEP + "\n")
