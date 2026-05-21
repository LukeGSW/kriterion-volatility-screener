"""
debug_pipeline.py - Diagnostica per il Kriterion Quant Volatility Screener.

Esegui con:
    EODHD_API_KEY=xxx python src/debug_pipeline.py

Testa ogni stadio in isolamento e identifica il collo di bottiglia.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))

try:
    from quant_engine import (
        MIN_ADV, PERCENTILE_LOOKBACK, RV_WINDOW,
        compute_adv, compute_log_returns,
        compute_realized_volatility, compute_rv_percentile,
    )
    QUANT_OK = True
except ImportError as e:
    QUANT_OK  = False
    QUANT_ERR = str(e)

BASE_URL = "https://eodhd.com/api"
SEP      = "=" * 68

def section(t):
    print("\n" + SEP)
    print("  " + t)
    print(SEP)

def ok(m):   print("  OK  " + m)
def fail(m): print("  KO  " + m)
def info(m): print("  ->  " + m)
def warn(m): print("  !!  " + m)


# -- TEST 1: Exchange Symbol List ---------------------------------------------
def test_exchange_list(api_key):
    section("TEST 1 - Exchange Symbol List /api/exchange-symbol-list/US")
    url    = "{}/exchange-symbol-list/US".format(BASE_URL)
    params = {"api_token": api_key, "fmt": "json"}
    try:
        r    = requests.get(url, params=params, timeout=60)
        data = r.json()
    except Exception as e:
        fail("Eccezione: {}".format(e))
        return []

    print("  HTTP status: {}".format(r.status_code))
    if not isinstance(data, list):
        fail("Risposta non e' una lista: {}".format(type(data)))
        return []

    ok("Ticker totali: {:,}".format(len(data)))
    if data:
        info("Campi: {}".format(list(data[0].keys())))
        # Distribuzione exchange
        exchanges = {}
        types     = {}
        for item in data:
            exc = item.get("Exchange", item.get("exchange", "?"))
            typ = item.get("Type",     item.get("type",     "?"))
            exchanges[exc] = exchanges.get(exc, 0) + 1
            types[typ]     = types.get(typ,     0) + 1
        top_exc = sorted(exchanges.items(), key=lambda x: -x[1])[:8]
        print("  Top exchange:")
        for exc, cnt in top_exc:
            print("      {:20s}: {:>6,}".format(exc, cnt))
        print("  Tipi strumento:")
        for typ, cnt in sorted(types.items(), key=lambda x: -x[1])[:5]:
            print("      {:20s}: {:>6,}".format(typ, cnt))

    primary = {"NYSE","NASDAQ","NYSE ARCA","BATS","NYSE MKT","AMEX","NYSE American","CBOE"}
    valid_t = {"Common Stock","ETF"}
    filtered = [
        x for x in data
        if x.get("Exchange", x.get("exchange","")) in primary
        and x.get("Type", x.get("type","")) in valid_t
    ]
    ok("Dopo filtro exchange primari + tipo: {:,} ticker".format(len(filtered)))
    return filtered


# -- TEST 2: Bulk Last-Day EOD ------------------------------------------------
def test_bulk_last_day(api_key):
    section("TEST 2 - Bulk Last-Day /api/eod-bulk-last-day/US")
    url    = "{}/eod-bulk-last-day/US".format(BASE_URL)
    params = {"api_token": api_key, "fmt": "json"}
    try:
        r    = requests.get(url, params=params, timeout=120)
        data = r.json()
    except Exception as e:
        fail("Eccezione: {}".format(e))
        return pd.DataFrame()

    print("  HTTP status: {}".format(r.status_code))
    if not isinstance(data, list):
        warn("Risposta non e' una lista: {}".format(type(data)))
        return pd.DataFrame()

    ok("Record ricevuti: {:,}".format(len(data)))
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    info("Colonne: {}".format(list(df.columns)))
    info("Esempio primo record: {}".format(data[0]))

    # Rinomina standard
    df.columns = [c.strip() for c in df.columns]
    col_map = {
        "code":"ticker","Code":"ticker",
        "volume":"last_volume","Volume":"last_volume",
        "close":"last_close","Close":"last_close",
        "market_capitalization":"market_cap","MarketCapitalization":"market_cap",
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})

    for col in ["last_volume","last_close","market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "last_volume" in df.columns:
        above_100k = (df["last_volume"] >= 100_000).sum()
        above_500k = (df["last_volume"] >= 500_000).sum()
        above_1m   = (df["last_volume"] >= 1_500_000).sum()
        ok("Volume >= 100K shares: {:,} ticker".format(above_100k))
        info("Volume >= 500K shares: {:,} ticker".format(above_500k))
        info("Volume >= 1.5M shares: {:,} ticker".format(above_1m))
    else:
        warn("Campo 'last_volume' non presente nel response!")

    if "market_cap" in df.columns:
        n_cap = df["market_cap"].notna().sum()
        info("market_cap disponibile per {:,} ticker".format(n_cap))
        if n_cap > 0:
            above_2b = (df["market_cap"] >= 2_000_000_000).sum()
            info("market_cap >= $2B: {:,} ticker".format(above_2b))
    else:
        info("market_cap NON presente nel response bulk last-day")

    return df


# -- TEST 3: Combinato (exchange-list x bulk-last-day) ------------------------
def test_combined_universe(sym_list, bulk_df):
    section("TEST 3 - Universo combinato (exchange-list + bulk last-day pre-filtro)")

    if not sym_list:
        warn("Exchange-list vuota - salta.")
        return

    sym_df = pd.DataFrame(sym_list)
    col_map = {
        "Code":"ticker","code":"ticker",
        "Exchange":"exchange","exchange":"exchange",
        "Type":"type","type":"type",
    }
    sym_df = sym_df.rename(columns={k:v for k,v in col_map.items() if k in sym_df.columns})
    info("Ticker da exchange-list (primari+tipo): {:,}".format(len(sym_df)))

    if bulk_df.empty or "last_volume" not in bulk_df.columns:
        warn("Bulk data non disponibile - pre-filtro volume saltato.")
        return

    merged = sym_df.merge(bulk_df[["ticker","last_volume"]], on="ticker", how="left")
    info("Dopo join con bulk data: {:,} ticker".format(len(merged)))

    after_vol = merged[merged["last_volume"].fillna(0) >= 100_000]
    ok("Dopo pre-filtro volume >= 100K: {:,} ticker".format(len(after_vol)))
    info("Questi ticker verranno scaricati per 3 anni di OHLCV")
    info("Tempo stimato download OHLCV (5 workers, 0.1s delay): ~{:.0f} minuti".format(
        len(after_vol) * 0.1 / 5 / 60
    ))


# -- TEST 4: OHLCV singolo ticker (AAPL) --------------------------------------
def test_ohlcv(api_key, ticker="AAPL"):
    section("TEST 4 - OHLCV per {} (4 anni)".format(ticker))
    today     = datetime.utcnow().date()
    from_date = (today - timedelta(days=4 * 365)).isoformat()

    url    = "{}/eod/{}.US".format(BASE_URL, ticker)
    params = {
        "api_token": api_key, "fmt": "json",
        "from": from_date, "to": today.isoformat(),
        "adjusted_close": "true",
    }
    try:
        r    = requests.get(url, params=params, timeout=30)
        data = r.json()
    except Exception as e:
        fail("Eccezione: {}".format(e))
        return []

    print("  HTTP status: {}".format(r.status_code))
    if not isinstance(data, list) or not data:
        fail("Risposta inattesa: {}".format(type(data)))
        return []

    ok("Righe ricevute: {:,}".format(len(data)))
    info("Colonne: {}".format(list(data[0].keys())))
    info("Prima riga: {}".format(data[0]))
    info("Ultima riga: {}".format(data[-1]))

    has_adj = "adjusted_close" in data[0]
    if has_adj:
        ok("Campo 'adjusted_close' presente")
    else:
        warn("Campo 'adjusted_close' ASSENTE - solo 'close' disponibile")

    return data


# -- TEST 5: Motore quantitativo su AAPL --------------------------------------
def test_quant_engine(ohlcv_data, ticker="AAPL"):
    section("TEST 5 - Motore quantitativo su {}".format(ticker))

    if not QUANT_OK:
        fail("quant_engine non importabile: {}".format(QUANT_ERR))
        return
    if not ohlcv_data:
        warn("Nessun dato OHLCV - salta.")
        return

    df = pd.DataFrame(ohlcv_data)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    ac = "adjusted_close" if "adjusted_close" in df.columns else "close"
    df["adjusted_close"] = pd.to_numeric(df[ac], errors="coerce")
    df["volume"]         = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df = df.dropna(subset=["adjusted_close"])

    n   = len(df)
    req = PERCENTILE_LOOKBACK + RV_WINDOW
    info("Righe dopo pulizia: {:,}".format(n))
    info("Periodo: {} -> {}".format(df["date"].min().date(), df["date"].max().date()))

    print()
    print("  {:35s}: req={} have={}  {}".format(
        "Storia minima", req, n, "OK" if n >= req else "KO - MANCANO {} gg".format(req - n)
    ))

    adv30 = compute_adv(df["volume"], 30).iloc[-1]
    adv90 = compute_adv(df["volume"], 90).iloc[-1]
    print("  {:35s}: {:>15,.0f}  {}".format("ADV 30d", adv30, "OK" if adv30 >= MIN_ADV else "KO"))
    print("  {:35s}: {:>15,.0f}  {}".format("ADV 90d", adv90, "OK" if adv90 >= MIN_ADV else "KO"))

    log_ret  = compute_log_returns(df["adjusted_close"])
    rv_s     = compute_realized_volatility(log_ret, window=RV_WINDOW)
    rv_valid = rv_s.notna().sum()
    rv_now   = rv_s.iloc[-1]
    print("  {:35s}: {}  {}".format("Valori RV non-NaN", rv_valid, "OK" if rv_valid > 0 else "KO"))

    pct_s     = compute_rv_percentile(rv_s, lookback=PERCENTILE_LOOKBACK)
    pct_valid = pct_s.notna().sum()
    pct_now   = pct_s.iloc[-1]
    print("  {:35s}: {}  {}".format("Percentile non-NaN", pct_valid, "OK" if pct_valid > 0 else "KO"))

    if pd.notna(rv_now):
        info("RV corrente ({}d): {:.2f}%".format(RV_WINDOW, rv_now * 100))
    if pd.notna(pct_now):
        tag = "COMPRESSO!" if pct_now <= 5 else ("attenzione" if pct_now <= 15 else "normale")
        info("RV Percentile: {:.1f} - {}".format(pct_now, tag))
    else:
        warn("RV Percentile NaN - storia insufficiente (need {} RV validi, have {})".format(
            PERCENTILE_LOOKBACK, rv_valid
        ))


# -- MAIN ---------------------------------------------------------------------
if __name__ == "__main__":
    api_key = os.environ.get("EODHD_API_KEY", "").strip()
    if not api_key:
        print("ERRORE: EODHD_API_KEY non impostata.")
        print("  Esegui: EODHD_API_KEY=<token> python src/debug_pipeline.py")
        sys.exit(1)

    print("\n{:^68}".format("KRITERION QUANT - DIAGNOSTICA PIPELINE"))
    print("{:^68}".format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")))

    sym_list = test_exchange_list(api_key)
    bulk_df  = test_bulk_last_day(api_key)
    test_combined_universe(sym_list, bulk_df)
    ohlcv    = test_ohlcv(api_key, "AAPL")
    test_quant_engine(ohlcv, "AAPL")

    print("\n" + SEP)
    print("  Diagnostica completata.")
    print(SEP + "\n")
