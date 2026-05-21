"""
streamlit_app.py — Kriterion Quant | Volatility Compression Screener Dashboard.

Layer di visualizzazione pura: legge il dataset pre-calcolato da GitHub Actions
(data/screener_results.parquet) senza ricalcoli on-the-fly.

Tema: Kriterion Quant Dark — sfondo #0d1117, accenti gold #f0a500.
Deploy target: Streamlit Community Cloud (legge i file direttamente dalla repo).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config — DEVE essere il primo comando Streamlit ─────────────────────
st.set_page_config(
    page_title="Kriterion Quant | Volatility Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://kriterionquant.com",
        "About": "Kriterion Quant — Volatility Compression Screener v1.0",
    },
)

# ── Percorsi dati ─────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).parent
_REPO_ROOT = _APP_DIR.parent
_DATA_DIR = _REPO_ROOT / "data"
_PARQUET_PATH = _DATA_DIR / "screener_results.parquet"
_METADATA_PATH = _DATA_DIR / "run_metadata.json"

# ── CSS Kriterion Quant Dark Theme ────────────────────────────────────────────
_CSS = """
<style>
/* ── Tipografia e base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif;
}
.stApp { background-color: #0d1117; color: #e6edf3; }

/* ── Header brand ──────────────────────────────────────────────────────── */
.kq-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid rgba(240,165,0,0.3);
    padding: 1.2rem 2rem 1rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-radius: 0 0 10px 10px;
}
.kq-brand { line-height: 1.2; }
.kq-logo  { font-size: 1.5rem; font-weight: 800; color: #f0a500;
            letter-spacing: -0.5px; }
.kq-sub   { font-size: 0.8rem; color: #8b949e; margin-top: 0.1rem; }
.kq-badge {
    background: rgba(240,165,0,0.12);
    color: #f0a500;
    border: 1px solid rgba(240,165,0,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Metric cards ──────────────────────────────────────────────────────── */
.kq-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
    height: 100%;
}
.kq-card:hover { border-color: rgba(240,165,0,0.5); }
.kq-card-value {
    font-size: 2rem; font-weight: 700; color: #f0a500; line-height: 1.1;
}
.kq-card-value.accent-green { color: #3fb950; }
.kq-card-value.accent-white { color: #e6edf3; font-size: 0.9rem;
                               padding-top: 0.5rem; }
.kq-card-label {
    font-size: 0.7rem; color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.07em; margin-top: 0.3rem;
}

/* ── Compression zone ticker cards ─────────────────────────────────────── */
.kq-ticker-card {
    background: #161b22;
    border: 1px solid rgba(240,165,0,0.4);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
}
.kq-ticker-card:hover { border-color: #f0a500; transform: translateY(-2px); }
.kq-ticker-name  { font-size: 1.4rem; font-weight: 700; color: #e6edf3; }
.kq-ticker-rv    { font-size: 0.9rem; color: #f0a500; margin: 0.3rem 0; }
.kq-ticker-pct   { font-size: 0.8rem; color: #f0c040; font-weight: 600; }
.kq-ticker-earn  { font-size: 0.7rem; margin-top: 0.3rem; }
.kq-ticker-links { font-size: 0.7rem; color: #8b949e; margin-top: 0.4rem; }
.kq-ticker-links a { color: #58a6ff; text-decoration: none; margin: 0 0.3rem; }
.kq-ticker-links a:hover { text-decoration: underline; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3; }

/* ── Slider accent ──────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { color: #f0a500; }

/* ── Section titles ─────────────────────────────────────────────────────── */
.kq-section {
    font-size: 0.85rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem; margin-bottom: 0.8rem;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.kq-footer {
    text-align: center; color: #8b949e; font-size: 0.72rem;
    border-top: 1px solid #21262d;
    padding-top: 1rem; margin-top: 2rem;
}
.kq-footer strong { color: #f0a500; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #21262d !important; }

/* ── Hide Streamlit default elements ────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""


# ── Utility functions ─────────────────────────────────────────────────────────
def _fmt_cap(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    v = float(val)
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"


def _fmt_vol(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    v = float(val)
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:,.0f}"


def _fmt_pct(val: object, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.{decimals}f}%"


def _fmt_num(val: object, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.{decimals}f}"


def _tv_url(ticker: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={ticker}"


def _yf_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker}"


def _pct_color(pct: float) -> str:
    """Colore hex per un valore di percentile."""
    if pct <= 5:
        return "#f0a500"   # amber — zona compressione
    if pct <= 15:
        return "#f0c040"   # giallo
    if pct <= 30:
        return "#a8c880"   # verde chiaro
    return "#3fb950"       # verde


def _earn_color(dte: Optional[float]) -> str:
    if dte is None or np.isnan(float(dte if dte is not None else float("nan"))):
        return "#8b949e"
    if float(dte) < 15:
        return "#f85149"   # rosso — imminenti
    if float(dte) < 30:
        return "#f0c040"   # giallo — attenzione
    return "#8b949e"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_data() -> tuple[pd.DataFrame, dict]:
    """Carica parquet e metadata. Cache 5 minuti."""
    metadata: dict = {}
    if _METADATA_PATH.exists():
        with open(_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

    if not _PARQUET_PATH.exists():
        return pd.DataFrame(), metadata

    df = pd.read_parquet(_PARQUET_PATH)

    # Forza tipi corretti
    for col in ["rv_current", "rv_percentile", "rv_52w_min", "rv_52w_max",
                "close_price", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["adv_30d", "adv_90d"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "days_to_earnings" in df.columns:
        df["days_to_earnings"] = pd.to_numeric(df["days_to_earnings"], errors="coerce")

    if "is_compressed" in df.columns:
        df["is_compressed"] = df["is_compressed"].astype(bool)

    return df, metadata


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="kq-header">
        <div class="kq-brand">
            <div class="kq-logo">⚡ KRITERION QUANT</div>
            <div class="kq-sub">Volatility Compression Screener — Long Straddle Engine</div>
        </div>
        <div class="kq-badge">Live · US Markets</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df, meta = _load_data()

    if df.empty and not meta:
        st.error(
            "⚠️ Nessun dato disponibile. "
            "Il pipeline GitHub Actions potrebbe non aver ancora girato."
        )
        st.info(
            "Lo screener viene aggiornato automaticamente ogni giorno lavorativo "
            "dopo la chiusura del mercato USA (≈22:00 UTC). "
            "Per un run manuale: **Actions → Daily Volatility Screener → Run workflow**."
        )
        return

    # ── Global metric cards ───────────────────────────────────────────────────
    ts_raw = meta.get("run_timestamp", "")
    try:
        ts_display = datetime.fromisoformat(ts_raw).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_display = ts_raw or "—"

    n_scanned    = meta.get("tickers_scanned", len(df) if not df.empty else 0)
    n_qualified  = meta.get("tickers_passed_filters", len(df) if not df.empty else 0)
    n_compressed = meta.get("tickers_compressed", 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kq-card">
            <div class="kq-card-value">{n_scanned:,}</div>
            <div class="kq-card-label">Ticker Scansionati</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kq-card">
            <div class="kq-card-value accent-green">{n_qualified:,}</div>
            <div class="kq-card-label">Qualificati (ADV + Storia)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kq-card">
            <div class="kq-card-value">{n_compressed}</div>
            <div class="kq-card-label">⚡ Compressi ≤5° Pct</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kq-card">
            <div class="kq-card-value accent-white">{ts_display}</div>
            <div class="kq-card-label">Ultimo Aggiornamento</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty:
        st.warning("Il dataset è vuoto. Nessun ticker ha superato i filtri in quest'ultimo run.")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Filtri")
        st.markdown("---")

        # Percentile soglia
        max_pct = st.slider(
            "RV Percentile massimo",
            min_value=1, max_value=50, value=10, step=1,
            help="Mostra solo ticker con RV Percentile ≤ questo valore",
        )

        st.markdown("---")

        # Market cap
        st.markdown("**Market Cap minima**")
        cap_map = {
            "Tutte (≥$2B)": 2e9,
            "Large Cap (≥$10B)": 10e9,
            "Mega Cap (≥$100B)": 100e9,
        }
        cap_label = st.radio(
            "Market Cap",
            list(cap_map.keys()),
            index=0,
            label_visibility="collapsed",
        )
        min_cap = cap_map[cap_label]

        st.markdown("---")

        # Tipo strumento
        type_options = ["Tutti"]
        if "type" in df.columns:
            type_options += sorted(df["type"].dropna().unique().tolist())
        type_sel = st.selectbox("Tipo strumento", type_options, index=0)

        st.markdown("---")

        # Earnings filter
        exclude_earnings = st.toggle(
            "Escludi Earnings imminenti",
            value=False,
            help="Nasconde ticker con earnings nei prossimi N giorni (Event Risk Rule)",
        )
        earn_threshold = 15
        if exclude_earnings:
            earn_threshold = st.slider(
                "Soglia giorni a Earnings",
                min_value=5, max_value=30, value=15, step=5,
                help="Nascondi se Days_To_Earnings < soglia",
            )

        st.markdown("---")

        # Parametri run (sola lettura)
        rv_w  = meta.get("rv_window", 90)
        pct_l = meta.get("percentile_lookback", 756)
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#8b949e;line-height:1.8">
            <b style="color:#e6edf3">Parametri run:</b><br>
            RV Window: <b style="color:#f0a500">{rv_w} gg</b><br>
            Percentile Lookback: <b style="color:#f0a500">{pct_l} gg (~3y)</b><br>
            Soglia compressione: <b style="color:#f0a500">≤5° pct</b><br>
            ADV minimo: <b style="color:#f0a500">1.5M share</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Apply filters ─────────────────────────────────────────────────────────
    filt = df.copy()

    filt = filt[filt["rv_percentile"] <= max_pct]

    if "market_cap" in filt.columns:
        has_cap = filt["market_cap"].notna()
        filt = filt[~has_cap | (filt["market_cap"] >= min_cap)]

    if type_sel != "Tutti" and "type" in filt.columns:
        filt = filt[filt["type"] == type_sel]

    if exclude_earnings and "days_to_earnings" in filt.columns:
        mask_no_earn = filt["days_to_earnings"].isna()
        mask_safe    = filt["days_to_earnings"] >= earn_threshold
        filt = filt[mask_no_earn | mask_safe]

    # ── Main results table ────────────────────────────────────────────────────
    st.markdown(
        f'<div class="kq-section">📋 Risultati — {len(filt)} ticker '
        f'(≤{max_pct}° percentile)</div>',
        unsafe_allow_html=True,
    )

    if filt.empty:
        st.info("Nessun ticker corrisponde ai filtri attivi. Rilassa i parametri nella sidebar.")
    else:
        # Prepara DataFrame display
        disp = pd.DataFrame()
        disp["Ticker"] = filt["ticker"]

        if "name" in filt.columns:
            disp["Nome"] = filt["name"].fillna("—")

        if "type" in filt.columns:
            disp["Tipo"] = filt["type"].fillna("—")

        if "market_cap" in filt.columns:
            disp["Market Cap"] = filt["market_cap"].apply(_fmt_cap)

        if "close_price" in filt.columns:
            disp["Prezzo"] = filt["close_price"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "—"
            )

        if "rv_current" in filt.columns:
            disp["RV 90d (%)"] = filt["rv_current"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
            )

        # Percentile — colonna chiave, mantenuta numerica per column_config
        if "rv_percentile" in filt.columns:
            disp["RV Pct (3y)"] = filt["rv_percentile"]

        if "rv_52w_min" in filt.columns:
            disp["RV 52w Min"] = filt["rv_52w_min"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
            )

        if "rv_52w_max" in filt.columns:
            disp["RV 52w Max"] = filt["rv_52w_max"].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
            )

        if "adv_30d" in filt.columns:
            disp["ADV 30d"] = filt["adv_30d"].apply(_fmt_vol)

        if "adv_90d" in filt.columns:
            disp["ADV 90d"] = filt["adv_90d"].apply(_fmt_vol)

        # Days to earnings — numerico per column_config
        if "days_to_earnings" in filt.columns:
            disp["Days to Earn."] = filt["days_to_earnings"]

        if "next_earnings_date" in filt.columns:
            disp["Earnings Date"] = filt["next_earnings_date"].fillna("—")

        # Link columns (URL come stringa)
        disp["TradingView"] = filt["ticker"].apply(_tv_url)
        disp["Yahoo Finance"] = filt["ticker"].apply(_yf_url)

        disp = disp.reset_index(drop=True)

        col_config: dict = {
            "TradingView": st.column_config.LinkColumn(
                "TradingView",
                display_text="📈 Chart",
                help="Apri grafico TradingView",
            ),
            "Yahoo Finance": st.column_config.LinkColumn(
                "Yahoo Finance",
                display_text="🔗 YF",
                help="Apri Yahoo Finance",
            ),
        }

        if "RV Pct (3y)" in disp.columns:
            col_config["RV Pct (3y)"] = st.column_config.NumberColumn(
                "RV Pct (3y)",
                format="%.1f",
                help="Percentile rolling 3 anni. Più basso = più compresso.",
            )

        if "Days to Earn." in disp.columns:
            col_config["Days to Earn."] = st.column_config.NumberColumn(
                "Days to Earn.",
                format="%d",
                help="Giorni calendario alla prossima trimestrale (Event Risk Rule: chiudi prima degli earnings)",
            )

        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=min(620, 60 + len(disp) * 36),
            column_config=col_config,
        )

    # ── Compression Zone cards ────────────────────────────────────────────────
    compressed = filt[filt["rv_percentile"] <= 5.0] if "rv_percentile" in filt.columns else pd.DataFrame()

    if not compressed.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="kq-section">⚡ Compression Zone — '
            f'{len(compressed)} ticker al ≤5° Percentile</div>',
            unsafe_allow_html=True,
        )

        # Cards in griglia: max 4 per riga
        N_COLS = 4
        rows = [
            compressed.iloc[i: i + N_COLS]
            for i in range(0, len(compressed), N_COLS)
        ]

        for row_df in rows:
            cols = st.columns(N_COLS)
            for col_idx, (_, row) in enumerate(row_df.iterrows()):
                ticker  = row.get("ticker", "")
                rv_val  = row.get("rv_current")
                pct_val = row.get("rv_percentile")
                dte     = row.get("days_to_earnings")
                earn_dt = row.get("next_earnings_date")

                rv_str  = f"{rv_val:.1f}% RV" if pd.notna(rv_val) else "RV N/A"
                pct_str = f"{pct_val:.1f}° pct" if pd.notna(pct_val) else "—"
                dte_str = (
                    f"Earnings: {int(dte)}gg"
                    if pd.notna(dte) else
                    f"Earnings: {earn_dt}" if (earn_dt and str(earn_dt) != "—") else
                    "Earnings: N/A"
                )
                ec = _earn_color(dte if pd.notna(dte) else None)

                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="kq-ticker-card">
                        <div class="kq-ticker-name">{ticker}</div>
                        <div class="kq-ticker-rv">{rv_str}</div>
                        <div class="kq-ticker-pct">{pct_str}</div>
                        <div class="kq-ticker-earn" style="color:{ec}">
                            {dte_str}
                        </div>
                        <div class="kq-ticker-links">
                            <a href="{_tv_url(ticker)}" target="_blank">📈 TV</a>
                            <a href="{_yf_url(ticker)}" target="_blank">🔗 YF</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Distribuzione RV Percentile ───────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="kq-section">📊 Distribuzione RV Percentile — Universo qualificato</div>',
        unsafe_allow_html=True,
    )

    if "rv_percentile" in df.columns and df["rv_percentile"].notna().any():
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df["rv_percentile"].dropna(),
            nbinsx=50,
            marker_color="#f0a500",
            opacity=0.7,
            name="Tutti i ticker qualificati",
            hovertemplate="Pct: %{x:.0f} | Count: %{y}<extra></extra>",
        ))

        # Evidenzia distribuzione filtrata
        if not filt.empty and "rv_percentile" in filt.columns:
            fig.add_trace(go.Histogram(
                x=filt["rv_percentile"].dropna(),
                nbinsx=50,
                marker_color="#58a6ff",
                opacity=0.5,
                name=f"Filtrati (≤{max_pct}° pct)",
                hovertemplate="Pct: %{x:.0f} | Count: %{y}<extra></extra>",
            ))

        fig.add_vline(
            x=5, line_dash="dash", line_color="#f85149", line_width=1.5,
            annotation_text="5° Pct — Compression Zone",
            annotation_font_color="#f85149",
            annotation_font_size=11,
        )
        fig.add_vline(
            x=10, line_dash="dot", line_color="#f0c040", line_width=1.2,
            annotation_text="10° Pct",
            annotation_font_color="#f0c040",
            annotation_font_size=10,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", size=12),
            xaxis=dict(
                title="RV Percentile Rolling (3 anni)",
                gridcolor="#21262d",
                range=[0, 100],
            ),
            yaxis=dict(
                title="N° Ticker",
                gridcolor="#21262d",
            ),
            barmode="overlay",
            legend=dict(
                bgcolor="rgba(22,27,34,0.8)",
                bordercolor="#30363d",
                borderwidth=1,
            ),
            height=320,
            margin=dict(l=10, r=10, t=20, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

    # ── RV Scatter: RV Attuale vs Percentile ──────────────────────────────────
    if (
        "rv_percentile" in df.columns
        and "rv_current" in df.columns
        and df[["rv_percentile", "rv_current"]].notna().all(axis=1).any()
    ):
        st.markdown(
            '<div class="kq-section">🔵 Scatter: RV Attuale vs Percentile</div>',
            unsafe_allow_html=True,
        )

        scatter_df = df[["ticker", "rv_current", "rv_percentile", "market_cap"]].dropna(
            subset=["rv_current", "rv_percentile"]
        )

        colors = scatter_df["rv_percentile"].apply(_pct_color).tolist()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=scatter_df["rv_percentile"],
            y=scatter_df["rv_current"],
            mode="markers",
            marker=dict(
                color=scatter_df["rv_percentile"],
                colorscale=[[0, "#f0a500"], [0.1, "#f0c040"],
                            [0.3, "#a8c880"], [1, "#3fb950"]],
                size=6,
                opacity=0.75,
                colorbar=dict(title="Percentile", tickfont=dict(size=10)),
            ),
            text=scatter_df["ticker"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Percentile: %{x:.1f}<br>"
                "RV 90d: %{y:.1f}%<extra></extra>"
            ),
        ))

        # Linea verticale zona compressione
        fig2.add_vline(
            x=5, line_dash="dash", line_color="#f85149", line_width=1.5,
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", size=12),
            xaxis=dict(
                title="RV Percentile (3y)",
                gridcolor="#21262d",
                range=[0, 100],
            ),
            yaxis=dict(
                title="RV 90d (%)",
                gridcolor="#21262d",
            ),
            height=350,
            margin=dict(l=10, r=10, t=20, b=40),
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ── Strategy reminder box ─────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                padding:1.2rem 1.5rem;font-size:0.8rem;color:#8b949e;line-height:1.9">
        <b style="color:#f0a500;font-size:0.85rem">⚡ Regole operative Long Straddle (reference)</b><br>
        <b style="color:#e6edf3">Entry:</b>
            Acquisto Call + Put ATM, stessa scadenza, DTE ≈ 90 giorni.<br>
        <b style="color:#e6edf3">Time Stop:</b>
            Chiusura tassativa a DTE residui = 30 (max permanenza 60 gg).<br>
        <b style="color:#f0c040">Event Risk Rule:</b>
            Chiusura il giorno precedente agli Earnings — incassa Vega run-up, evita IV Crush.<br>
        <b style="color:#3fb950">Profit Target:</b>
            +100% sul premio netto pagato, o in prossimità di livelli volumetrici rilevanti.<br>
        <b style="color:#f85149">No hard stop loss</b> basato su percentuale di perdita o livello di prezzo.
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kq-footer">
        <strong>KRITERION QUANT</strong> — Volatility Compression Screener<br>
        Data source: EODHD API (All-In-One) ·
        Aggiornamento automatico via GitHub Actions · Mon–Fri ~22:00 UTC<br>
        <em>Solo a scopo educativo e di ricerca. Non costituisce consulenza finanziaria.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
