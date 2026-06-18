#!/usr/bin/env python3
"""
fetch_data.py — Market Health Dashboard data pipeline
=====================================================

Pulls every metric defined in the spec, computes the derived scores, and writes
a single self-describing JSON file to data/latest.json.

DESIGN PRINCIPLES
-----------------
1. Nothing is fatal. Every external source is wrapped so one failure (a dead
   Yahoo endpoint, an FRED hiccup, a Google Trends rate-limit) degrades that one
   metric to status="unavailable" instead of killing the whole run.
2. Every metric is self-describing: {value, status, source, asof, notes, ...}.
   The dashboard and the alert script read `status` before trusting `value`.
3. Proxies are labeled. Several spec metrics have no free first-party source
   (NYSE breadth internals, ISM PMI, Conference Board LEI, Forward P/E). Where we
   substitute, status is "proxy" or "manual" and `notes` says exactly what it is.
4. Composite scores are computed only from inputs that actually came back, and
   the output records how many inputs were available so confidence is visible.

SECRETS (read from environment — set as GitHub Actions secrets)
---------------------------------------------------------------
  FRED_API_KEY    required for all FRED series
  ALPACA_KEY      required for breadth internals (S&P 500 constituent pull)
  ALPACA_SECRET   required for breadth internals

USAGE
-----
  python fetch_data.py                  # full run, writes data/latest.json
  python fetch_data.py --no-breadth     # skip the heavy 500-symbol Alpaca pull
  python fetch_data.py --selftest       # offline: exercise pure math, no network
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

# Third-party (installed via requirements.txt in CI)
try:
    import numpy as np
    import pandas as pd
    import requests
except Exception as e:  # pragma: no cover
    print(f"FATAL: core dependency missing: {e}", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "docs/data/latest.json")

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Manual inputs (update these monthly, or override via env)
FORWARD_PE = float(os.environ.get("FORWARD_PE", "22.0"))          # S&P 500 fwd P/E
FORWARD_PE_HIST_MEAN = float(os.environ.get("FORWARD_PE_MEAN", "16.0"))

# 11 GICS sector SPDR ETFs
SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLP": "Consumer Staples", "XLRE": "Real Estate",
    "XLY": "Consumer Discretionary", "XLC": "Communication Services",
}

# FRED series. Note where these are proxies for proprietary series.
FRED_SERIES = {
    # Cycle position
    "lei":               ("USALOLITOAASTSAM",   "OECD Composite Leading Indicator, US (amplitude-adjusted) — current; replaces discontinued USSLIND"),
    "yield_curve_10y3m": ("T10Y3M",             "10yr minus 3mo Treasury spread"),
    "nfci_leverage":     ("NFCINONFINLEVERAGE", "Chicago Fed NFCI nonfinancial leverage subindex"),
    "lending_standards": ("DRTSCILM",           "SLOOS: net % of banks tightening C&I loan standards — leads recessions 2-4 quarters"),
    "unemployment":      ("UNRATE",             "Civilian unemployment rate"),
    "cc_delinquency":    ("DRCCLACBS",          "Credit card delinquency rate, all commercial banks"),
    "auto_delinquency":  ("DRALACBN",           "Auto/other consumer loan delinquency rate"),
    "savings_rate":      ("PSAVERT",            "Personal saving rate"),
    "debt_service":      ("TDSP",               "Household debt service ratio"),
    "corp_profits":      ("A053RC1Q027SBEA",    "Corporate profits (with IVA/CCAdj)"),
    "gdp":               ("GDP",                "Nominal GDP"),
    "core_cpi":          ("CPILFESL",           "Core CPI (ex food & energy), index"),
    "core_pce":          ("PCEPILFE",           "Core PCE price index (Fed's preferred inflation gauge), index"),
    "ppi":               ("PPIFIS",             "Producer Price Index, Final Demand (SA) — leads CPI by 1-3 months"),
    "fed_funds":         ("DFF",                "Effective federal funds rate"),
    "umich_sentiment":   ("UMCSENT",            "U. Michigan Consumer Sentiment"),
    "avg_hourly_earnings":("CES0500000003",     "Average hourly earnings, total private (for real wage growth)"),
    "hy_oas":            ("BAMLH0A0HYM2",       "ICE BofA US High-Yield option-adjusted credit spread"),
    "ig_oas":            ("BAMLC0A0CM",         "ICE BofA US Investment-Grade option-adjusted credit spread"),
    # Valuation building blocks
    "equity_mktcap":     ("NCBEILQ027S",        "Fed B.103 market value of equities (Buffett numerator; Wilshire removed from FRED Jun 2024)"),
    "real_10y":          ("DFII10",             "10yr TIPS yield (real) — for Excess CAPE Yield"),
}

# How far back to look when computing a metric's trend, by series cadence.
TREND_LOOKBACK = {
    "lei": 3, "yield_curve_10y3m": 21, "nfci_leverage": 4, "unemployment": 3,
    "lending_standards": 1,
    "cc_delinquency": 1, "auto_delinquency": 1, "savings_rate": 3, "debt_service": 1,
    "corp_profits": 1, "gdp": 1, "core_cpi": 3, "core_pce": 3, "ppi": 3, "fed_funds": 21,
    "umich_sentiment": 3, "avg_hourly_earnings": 3, "hy_oas": 21, "ig_oas": 21,
    "real_10y": 21, "equity_mktcap": 1,
}

# Shiller CAPE candidate download URLs (tried in order)
SHILLER_URLS = [
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
    "https://shillerdata.com/wp-content/uploads/2024/ie_data.xls",
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls",
]

# Google Trends themes
TRENDS_THEMES = {
    "ai_tech":   ["artificial intelligence stocks", "nvidia stock"],
    "energy":    ["oil stocks", "uranium stocks"],
    "defense":   ["defense stocks"],
    "financials":["bank stocks", "interest rates"],
    "fear_greed":["stock market crash", "recession 2026"],
}

# Static catalyst calendar (update the year-ahead schedule as needed).
# These are the recurring macro events; earnings come from the watchlist.
CATALYST_CALENDAR = [
    # (ISO date, label)
    ("2026-06-11", "CPI Release"),
    ("2026-06-17", "FOMC Meeting + Decision"),
    ("2026-06-26", "PCE Release"),
    ("2026-07-02", "Nonfarm Payrolls (NFP)"),
    ("2026-07-15", "CPI Release"),
    ("2026-07-29", "FOMC Meeting + Decision"),
    ("2026-07-31", "PCE Release"),
    ("2026-08-01", "Nonfarm Payrolls (NFP)"),
    ("2026-08-13", "CPI Release"),
    ("2026-08-29", "PCE Release"),
    ("2026-09-04", "Nonfarm Payrolls (NFP)"),
    ("2026-09-16", "FOMC Meeting + Decision"),
]


def now_stamps():
    utc = datetime.now(timezone.utc)
    pt = utc - timedelta(hours=8)  # PST; close enough for a display stamp
    return utc.isoformat(), pt.strftime("%b %d %Y, %-I:%M %p PT") if os.name != "nt" \
        else pt.strftime("%b %d %Y, %I:%M %p PT")


def metric(value, status="ok", source="", asof=None, notes="", **extra):
    """Uniform metric envelope."""
    d = {"value": value, "status": status, "source": source}
    if asof:
        d["asof"] = asof
    if notes:
        d["notes"] = notes
    d.update(extra)
    return d


def unavailable(source="", error="", notes=""):
    return {"value": None, "status": "unavailable", "source": source,
            "error": str(error)[:300], "notes": notes}


# ---------------------------------------------------------------------------
# Pure calculations (no network — covered by --selftest)
# ---------------------------------------------------------------------------

def ema(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    close = pd.Series(close).astype(float)
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def rsi(close, period=14):
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return float(out.iloc[-1])


def slope_label(series, lookback=21, flat_band=0.001):
    """Classify the slope of a moving-average tail as rising/flat/declining.
    flat_band is the fractional change below which we call it flat."""
    s = pd.Series(series).dropna()
    if len(s) < lookback + 1:
        return "unknown", 0.0
    change = (s.iloc[-1] - s.iloc[-lookback]) / abs(s.iloc[-lookback] or 1)
    if change > flat_band:
        return "rising", float(change)
    if change < -flat_band:
        return "declining", float(change)
    return "flat", float(change)


def percentile_rank(value, history):
    """Percentile (0-100) of `value` within `history`. Used for Goldman-style scoring."""
    h = pd.Series(history).dropna().astype(float)
    if len(h) == 0 or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float((h < value).mean() * 100.0)


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def fred_series(series_id, api_key, observations=400):
    """Return a pandas Series indexed by date (most recent last), or raise."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "sort_order": "asc", "limit": 100000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = [(o["date"], o["value"]) for o in obs if o.get("value") not in (".", "", None)]
    if not rows:
        raise ValueError(f"FRED {series_id} returned no usable observations")
    idx = pd.to_datetime([d for d, _ in rows])
    vals = pd.to_numeric([v for _, v in rows], errors="coerce")
    s = pd.Series(vals, index=idx).dropna()
    return s.iloc[-observations:]


def series_trend(s, periods):
    """Direction + magnitude of a series over the last `periods` observations."""
    try:
        s = pd.Series(s).dropna()
        if len(s) < periods + 1:
            return None
        cur = float(s.iloc[-1])
        prev = float(s.iloc[-periods - 1])
        chg = cur - prev
        pct = (chg / abs(prev) * 100) if prev else 0.0
        # 'flat' band scaled to the series' own typical move
        band = (s.diff().abs().median() or 0) * 0.5
        direction = "rising" if chg > band else "falling" if chg < -band else "flat"
        return {"direction": direction, "change": round(chg, 3),
                "pct": round(pct, 2), "periods": periods}
    except Exception:
        return None


def yoy_pct(s):
    """Year-over-year percent change of a monthly index series (e.g. CPI/PCE)."""
    try:
        s = pd.Series(s).dropna()
        if len(s) < 13:
            return None
        return round((float(s.iloc[-1]) / float(s.iloc[-13]) - 1) * 100, 2)
    except Exception:
        return None


def pull_fred_all(api_key):
    out = {}
    raw = {}
    if not api_key:
        for key, (sid, desc) in FRED_SERIES.items():
            out[key] = unavailable(source=f"FRED:{sid}", error="FRED_API_KEY not set", notes=desc)
        return out, raw
    for key, (sid, desc) in FRED_SERIES.items():
        try:
            s = fred_series(sid, api_key)
            raw[key] = s
            m = metric(
                value=round(float(s.iloc[-1]), 4),
                source=f"FRED:{sid}",
                asof=s.index[-1].strftime("%Y-%m-%d"),
                notes=desc,
            )
            tr = series_trend(s, TREND_LOOKBACK.get(key, 3))
            if tr:
                m["trend"] = tr
            # For inflation indices, surface YoY (the headline people read).
            if key in ("core_cpi", "core_pce", "ppi"):
                yoy = yoy_pct(s)
                if yoy is not None:
                    m["yoy_pct"] = yoy
                    # is inflation accelerating or decelerating?
                    yoy_prev = yoy_pct(s.iloc[:-3]) if len(s) > 16 else None
                    if yoy_prev is not None:
                        m["yoy_direction"] = "accelerating" if yoy > yoy_prev + 0.1 \
                            else "decelerating" if yoy < yoy_prev - 0.1 else "steady"
            out[key] = m
        except Exception as e:
            out[key] = unavailable(source=f"FRED:{sid}", error=e, notes=desc)
        time.sleep(0.15)  # be polite to FRED
    return out, raw


# ---------------------------------------------------------------------------
# yfinance (price / sectors / VIX)
# ---------------------------------------------------------------------------

def _yf_history(ticker, period="2y", interval="1d"):
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty for {ticker}")
    return df


def pull_spx_technicals():
    """S&P 500 price vs 200-DMA, 200-DMA slope, weekly MACD."""
    try:
        df = _yf_history("^GSPC", period="3y", interval="1d")
        close = df["Close"].dropna()
        dma200 = close.rolling(200).mean()
        dma50 = close.rolling(50).mean()
        last = float(close.iloc[-1])
        last_200 = float(dma200.iloc[-1])
        last_50 = float(dma50.iloc[-1])
        slope, change = slope_label(dma200, lookback=21)

        # Weekly MACD
        wk = close.resample("W-FRI").last().dropna()
        macd_line, signal_line, hist = macd(wk.values)

        return {
            "spx_price": metric(round(last, 2), source="yfinance:^GSPC",
                                asof=close.index[-1].strftime("%Y-%m-%d")),
            "spx_vs_200dma": metric(
                round((last / last_200 - 1) * 100, 2), source="yfinance:^GSPC",
                notes="percent above/below 200-day moving average",
                price=round(last, 2), dma200=round(last_200, 2)),
            "dma200_slope": metric(slope, source="yfinance:^GSPC",
                                   notes="slope of 200-DMA over ~1 month",
                                   change_pct=round(change * 100, 3)),
            "dma50": metric(round(last_50, 2), source="yfinance:^GSPC"),
            "weekly_macd": metric(
                round(hist, 3), source="yfinance:^GSPC",
                notes="weekly MACD histogram; <0 = bearish momentum",
                macd_line=round(macd_line, 3), signal_line=round(signal_line, 3),
                bearish=bool(hist < 0)),
        }
    except Exception as e:
        return {"spx_technicals": unavailable(source="yfinance:^GSPC", error=e)}


def pull_vix():
    try:
        df = _yf_history("^VIX", period="6mo", interval="1d")
        close = df["Close"].dropna()
        last = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        return metric(round(last, 2), source="yfinance:^VIX",
                      asof=close.index[-1].strftime("%Y-%m-%d"),
                      notes="CBOE Volatility Index",
                      ma20=round(ma20, 2),
                      above_30=bool(last > 30), above_40=bool(last > 40))
    except Exception as e:
        return unavailable(source="yfinance:^VIX", error=e)


def pull_ad_proxy():
    """RSP/SPY ratio as an A/D-line / breadth-divergence proxy.
    Falling ratio while SPY rises = narrow leadership = bearish breadth divergence."""
    try:
        import yfinance as yf
        data = yf.download(["RSP", "SPY"], period="1y", interval="1d",
                           auto_adjust=False, progress=False)
        close = data["Close"].dropna()
        ratio = (close["RSP"] / close["SPY"])
        last = float(ratio.iloc[-1])
        ratio_ma50 = float(ratio.rolling(50).mean().iloc[-1])
        spy_chg = float(close["SPY"].iloc[-1] / close["SPY"].iloc[-21] - 1)
        ratio_chg = float(ratio.iloc[-1] / ratio.iloc[-21] - 1)
        divergence = bool(spy_chg > 0 and ratio_chg < 0)
        return metric(round(last, 5), source="yfinance:RSP/SPY",
                      notes="equal-weight/cap-weight ratio; proxy for NYSE A/D divergence",
                      ratio_ma50=round(ratio_ma50, 5),
                      spy_21d_pct=round(spy_chg * 100, 2),
                      ratio_21d_pct=round(ratio_chg * 100, 2),
                      bearish_divergence=divergence)
    except Exception as e:
        return unavailable(source="yfinance:RSP/SPY", error=e)


def pull_sectors():
    """Sector relative strength vs SPY and sector RSI for the 11 SPDRs."""
    import yfinance as yf
    out = {}
    try:
        tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
        data = yf.download(tickers, period="1y", interval="1d",
                           auto_adjust=False, progress=False)["Close"].dropna()
        spy_3m = float(data["SPY"].iloc[-1] / data["SPY"].iloc[-63] - 1)
        for etf, name in SECTOR_ETFS.items():
            if etf not in data:
                out[etf] = unavailable(source=f"yfinance:{etf}", notes=name)
                continue
            s = data[etf].dropna()
            sec_3m = float(s.iloc[-1] / s.iloc[-63] - 1)
            rel = sec_3m - spy_3m
            out[etf] = metric(round(rel * 100, 2), source=f"yfinance:{etf}",
                              notes=f"{name}: 3-month relative strength vs SPY (pct pts)",
                              name=name, rsi=round(rsi(s.values), 1),
                              sector_3m_pct=round(sec_3m * 100, 2))
        return out
    except Exception as e:
        return {etf: unavailable(source=f"yfinance:{etf}", error=e, notes=name)
                for etf, name in SECTOR_ETFS.items()}


# ---------------------------------------------------------------------------
# Shiller CAPE
# ---------------------------------------------------------------------------

def pull_cape():
    """Current CAPE. Primary: multpl.com monthly table (current, Shiller-sourced).
    Fallback: Shiller ie_data.xls. Returns (cape_metric, cape_value, ecy_value)."""
    ua = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
    # Primary: multpl.com by-month table (most current machine-readable CAPE)
    try:
        import re
        from io import StringIO
        r = requests.get("https://www.multpl.com/shiller-pe/table/by-month",
                         headers=ua, timeout=30)
        r.raise_for_status()
        t = pd.read_html(StringIO(r.text))[0]
        asof = str(t.iloc[0, 0]).strip()
        m = re.search(r"[-+]?\d+\.?\d*", str(t.iloc[0, 1]))
        cape = float(m.group()) if m else None
        if cape is not None and 3 < cape < 100:
            return (metric(round(cape, 2), source="multpl.com (Shiller PE)", asof=asof,
                           notes="Cyclically adjusted P/E (P/E10)"), cape, None)
    except Exception:
        pass
    # Fallback: Shiller workbook (may lag if the hosted file is cached)
    return pull_shiller_cape()


def pull_shiller_cape():
    """Download Shiller's ie_data.xls and extract latest CAPE (P/E10) and, if present,
    the pre-computed Excess CAPE Yield column. Returns (cape_metric, cape_value, ecy_value)."""
    ua = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
    last_err = None
    for url in SHILLER_URLS:
        try:
            r = requests.get(url, headers=ua, timeout=45)
            r.raise_for_status()
            from io import BytesIO
            xls = pd.ExcelFile(BytesIO(r.content))
            sheet = "Data" if "Data" in xls.sheet_names else xls.sheet_names[0]
            df = xls.parse(sheet, header=None)

            # Find the header row (first row mentioning CAPE / P/E10)
            header_row = None
            for ri in range(min(20, len(df))):
                rowtext = " ".join(str(df.iat[ri, ci]).upper() for ci in range(df.shape[1]))
                if "CAPE" in rowtext or "P/E10" in rowtext:
                    header_row = ri
                    break
            if header_row is None:
                raise ValueError("no CAPE header row found")
            start = header_row + 1

            def last_in_range(ci, lo, hi):
                col = pd.to_numeric(df.iloc[start:, ci], errors="coerce").dropna()
                col = col[(col > lo) & (col < hi)]
                return float(col.iloc[-1]) if len(col) else None

            # Candidate CAPE columns + the ECY column, from header text.
            cape_candidates = []
            ecy_col = None
            for ci in range(df.shape[1]):
                h = str(df.iat[header_row, ci]).upper().strip()
                if "EXCESS CAPE YIELD" in h:
                    ecy_col = ci
                if ("CAPE" in h or "P/E10" in h or "PE10" in h) \
                        and "EXCESS" not in h and "TR CAPE" not in h and "TR_CAPE" not in h:
                    cape_candidates.append(ci)

            # Pick the candidate whose latest value is a plausible CAPE (3..70).
            # This rejects mis-matches like the ECY column (~0.02) or TR CAPE.
            cape = None
            for ci in cape_candidates:
                v = last_in_range(ci, 3, 70)
                if v is not None:
                    cape = v
                    break
            # Last resort: scan every column for a CAPE-shaped series (mostly 8..55).
            if cape is None:
                for ci in range(df.shape[1]):
                    col = pd.to_numeric(df.iloc[start:, ci], errors="coerce").dropna()
                    if len(col) < 50:
                        continue
                    if float(((col > 8) & (col < 55)).mean()) > 0.6:
                        v = last_in_range(ci, 3, 70)
                        if v is not None:
                            cape = v
                            break
            if cape is None:
                raise ValueError("could not locate a plausible CAPE column")
            if not (3 < cape < 100):
                raise ValueError(f"CAPE sanity check failed: {cape}")

            # Excess CAPE Yield straight from the workbook, if present.
            # Shiller stores it as a fraction (0.0183); normalize to percent (1.83).
            ecy_val = None
            if ecy_col is not None:
                ecy_val = last_in_range(ecy_col, -5, 15)
                if ecy_val is not None and abs(ecy_val) < 0.5:
                    ecy_val *= 100.0

            return (metric(round(cape, 2), source="Shiller ie_data.xls",
                           notes="Cyclically adjusted P/E (P/E10)"), cape, ecy_val)
        except Exception as e:
            last_err = e
            continue
    return (unavailable(source="Shiller ie_data.xls", error=last_err), None, None)


# ---------------------------------------------------------------------------
# Alpaca — S&P 500 constituent breadth internals
# (powers % > 200-DMA, McClellan, Hindenburg/Titanic proxies)
# ---------------------------------------------------------------------------

def sp500_constituents():
    """Best-effort S&P 500 ticker list in Alpaca-native format (dots for class shares).
    Order: maintained GitHub CSV -> Wikipedia (with UA) -> baked-in static list."""
    ua = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
    # 1) Maintained dataset CSV (stable, ~500 names, already dot-formatted)
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(url, headers=ua, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        syms = df["Symbol"].astype(str).str.strip().tolist()
        if len(syms) > 400:
            return syms, "github_csv"
    except Exception:
        pass
    # 2) Wikipedia (needs a UA or it 403s)
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=ua, timeout=30)
        r.raise_for_status()
        from io import StringIO
        tbls = pd.read_html(StringIO(r.text))
        syms = tbls[0]["Symbol"].astype(str).str.strip().tolist()
        if len(syms) > 400:
            return syms, "wikipedia"
    except Exception:
        pass
    # 3) Static safety net
    return list(FALLBACK_SP500), "fallback_static"


def alpaca_daily_bars(symbols, api_key, api_secret, days=320):
    """Pull daily bars for a list of symbols from Alpaca. Returns {sym: DataFrame}.

    Resilient to bad/unsupported symbols: if Alpaca 400s a chunk with
    'invalid symbol: X', X is dropped and the chunk is retried. One bad ticker
    (e.g. a class share or a delisted name) never kills the whole pull.
    """
    import re
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.6))).strftime("%Y-%m-%d")
    out = {}
    dropped = []
    base = f"{ALPACA_DATA_URL}/v2/stocks/bars"
    CHUNK = 100
    for i in range(0, len(symbols), CHUNK):
        chunk = list(symbols[i:i + CHUNK])
        attempts = 0
        while chunk and attempts < 15:
            attempts += 1
            page_token = None
            chunk_done = True
            while True:
                params = {
                    "symbols": ",".join(chunk), "timeframe": "1Day",
                    "start": start, "limit": 10000, "adjustment": "split",
                    "feed": "iex",
                }
                if page_token:
                    params["page_token"] = page_token
                r = requests.get(base, headers=headers, params=params, timeout=60)
                if r.status_code == 400 and "invalid symbol" in r.text.lower():
                    m = re.search(r"invalid symbol:\s*([A-Za-z0-9.\-/]+)", r.text)
                    if m:
                        bad = m.group(1).strip().rstrip('"').rstrip("\\")
                        dropped.append(bad)
                        chunk = [s for s in chunk if s != bad]
                    else:
                        chunk = []  # unparseable; abandon this chunk
                    chunk_done = False
                    break  # retry the (now smaller) chunk from the top
                if r.status_code != 200:
                    raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:200]}")
                j = r.json()
                bars = j.get("bars", {})
                for sym, blist in bars.items():
                    if not blist:
                        continue
                    df = pd.DataFrame(blist)
                    df["t"] = pd.to_datetime(df["t"])
                    df = df.set_index("t").sort_index()
                    out[sym] = pd.concat([out.get(sym), df]) if sym in out else df
                page_token = j.get("next_page_token")
                if not page_token:
                    break
            if chunk_done:
                break  # chunk fully processed; move to next chunk
        time.sleep(0.2)
    if dropped:
        print(f"  [alpaca] dropped {len(dropped)} unsupported symbols: {dropped[:10]}"
              f"{'...' if len(dropped) > 10 else ''}", file=sys.stderr)
    return out


def compute_breadth(bars):
    """From {sym: df with 'c','h','l'} compute breadth internals for the universe."""
    pct_above_200 = []
    new_highs = new_lows = advancers = decliners = 0
    counted = 0
    for sym, df in bars.items():
        c = df["c"].dropna()
        if len(c) < 200:
            continue
        counted += 1
        ma200 = c.rolling(200).mean().iloc[-1]
        if c.iloc[-1] > ma200:
            pct_above_200.append(1)
        else:
            pct_above_200.append(0)
        # 52-week (252d) new highs/lows
        window = c.iloc[-252:]
        if c.iloc[-1] >= window.max():
            new_highs += 1
        if c.iloc[-1] <= window.min():
            new_lows += 1
        # advancers/decliners vs prior close
        if len(c) >= 2:
            if c.iloc[-1] > c.iloc[-2]:
                advancers += 1
            elif c.iloc[-1] < c.iloc[-2]:
                decliners += 1
    pct = (sum(pct_above_200) / len(pct_above_200) * 100) if pct_above_200 else None
    return {
        "universe_counted": counted,
        "pct_above_200dma": round(pct, 1) if pct is not None else None,
        "new_highs": new_highs, "new_lows": new_lows,
        "advancers": advancers, "decliners": decliners,
    }


def breadth_signals(breadth, mcclellan_val=None, spx_50d_ret=None):
    """Hindenburg/Titanic/Zweig proxies computed on the S&P 500 universe.
    Clearly labeled as proxies for the true NYSE-wide indicators."""
    n = breadth.get("universe_counted") or 0
    nh, nl = breadth.get("new_highs", 0), breadth.get("new_lows", 0)
    adv, dec = breadth.get("advancers", 0), breadth.get("decliners", 0)
    signals = {}

    if n > 0:
        nh_pct = nh / n * 100
        nl_pct = nl / n * 100
        # ---- Hindenburg Omen: ALL FOUR classic same-day conditions ----
        # 1) new highs AND new lows each >= 2.2% of issues
        # 2) neither side overwhelms the other (ratio guard)
        # 3) index in an uptrend: above its level ~50 sessions ago (spx_50d_ret > 0)
        # 4) McClellan Oscillator negative (deteriorating internals)
        cond_counts    = nh_pct >= 2.2 and nl_pct >= 2.2 and min(nh, nl) > 0
        cond_ratio     = max(nh_pct, nl_pct) < 2.8 * min(nh_pct, nl_pct) + 5
        cond_uptrend   = (spx_50d_ret is None) or (spx_50d_ret > 0)
        cond_mcclellan = (mcclellan_val is None) or (mcclellan_val < 0)
        hindenburg_raw = bool(cond_counts and cond_ratio and cond_uptrend and cond_mcclellan)
        signals["hindenburg_omen_today"] = metric(
            hindenburg_raw, status="proxy", source="Alpaca S&P500 internals",
            notes="PROXY on S&P500 (true indicator is NYSE-wide). RAW same-day flag — "
                  "not actionable alone. Confirmed signal = 2+ raw flags within 30 days "
                  "(see hindenburg_omen_confirmed).",
            new_high_pct=round(nh_pct, 2), new_low_pct=round(nl_pct, 2),
            cond_counts=bool(cond_counts), cond_ratio=bool(cond_ratio),
            cond_uptrend=bool(cond_uptrend), cond_mcclellan=bool(cond_mcclellan))
        # ---- Titanic Syndrome: new lows exceed new highs (raw same-day) ----
        signals["titanic_syndrome_today"] = metric(
            bool(nl > nh and nh > 0), status="proxy",
            source="Alpaca S&P500 internals",
            notes="PROXY on S&P500. RAW same-day flag. Confirmed signal requires a 52wk "
                  "index high within ~7 sessions AND the flag (see titanic_syndrome_confirmed).",
            new_highs=nh, new_lows=nl)
    if (adv + dec) > 0:
        zweig_ratio = adv / (adv + dec)
        signals["zweig_adv_ratio"] = metric(
            round(zweig_ratio, 3), status="proxy", source="Alpaca S&P500 internals",
            notes="Advancers/(Adv+Dec). Zweig Breadth Thrust = 10-day avg moving "
                  "<0.40 to >0.615 within 10 days; needs multi-day history (alert script).")
    return signals


def mcclellan_oscillator(prev_state, breadth):
    """McClellan Oscillator needs a running EMA of net advances. On a single run we
    can only seed it; the alert/state file carries the EMAs forward day to day.
    Here we emit today's net-advance ratio and let state accumulate over runs."""
    adv, dec = breadth.get("advancers", 0), breadth.get("decliners", 0)
    if (adv + dec) == 0:
        return unavailable(source="Alpaca S&P500 internals",
                           notes="no advance/decline data this run")
    rana = (adv - dec) / (adv + dec) * 1000  # ratio-adjusted net advances
    prev = (prev_state or {}).get("mcclellan", {})
    ema19 = prev.get("ema19")
    ema39 = prev.get("ema39")
    a19, a39 = 2 / (19 + 1), 2 / (39 + 1)
    ema19 = rana if ema19 is None else (rana - ema19) * a19 + ema19
    ema39 = rana if ema39 is None else (rana - ema39) * a39 + ema39
    osc = ema19 - ema39
    return metric(round(osc, 2), status="proxy", source="Alpaca S&P500 internals",
                  notes="Ratio-adjusted McClellan Oscillator (S&P500 proxy). Seeds/accumulates "
                        "via state file across daily runs; trust after ~40 sessions.",
                  ema19=round(ema19, 3), ema39=round(ema39, 3),
                  negative=bool(osc < 0), _state={"ema19": ema19, "ema39": ema39})


def _spx_50d_return():
    """SPX % change vs ~50 trading sessions ago. Powers the Hindenburg uptrend filter."""
    try:
        close = _yf_history("^GSPC", period="6mo", interval="1d")["Close"].dropna()
        if len(close) < 51:
            return None
        return float(close.iloc[-1] / close.iloc[-51] - 1) * 100
    except Exception:
        return None


def confirm_clusters(history, window_days=30, min_count=2):
    """Promote RAW daily breadth flags to CONFIRMED signals using rolling history.
    `history` must already include today's snapshot (with hindenburg_raw/titanic_raw).
    Returns dict of confirmed metrics to merge into result['breadth']. These are
    marked derived=True so they are NOT double-counted in source-health/confidence."""
    today = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=window_days)

    def _cluster(flag_key):
        dates = []
        for h in history:
            if not h.get(flag_key):
                continue
            try:
                d = datetime.fromisoformat(h["date"]).date()
            except Exception:
                continue
            if cutoff <= d <= today:
                dates.append(h["date"])
        return sorted(set(dates))

    hind_dates = _cluster("hindenburg_raw")
    tit_dates = _cluster("titanic_raw")
    return {
        "hindenburg_omen_confirmed": metric(
            bool(len(hind_dates) >= min_count), status="proxy",
            source="Alpaca S&P500 internals (cluster)", derived=True,
            notes=f"CONFIRMED when {min_count}+ raw flags occur within {window_days} days. "
                  "This is the actionable signal; the daily flag is not.",
            count=len(hind_dates), window_days=window_days, dates=hind_dates),
        "titanic_syndrome_confirmed": metric(
            bool(len(tit_dates) >= min_count), status="proxy",
            source="Alpaca S&P500 internals (cluster)", derived=True,
            notes=f"CONFIRMED when {min_count}+ raw flags occur within {window_days} days.",
            count=len(tit_dates), window_days=window_days, dates=tit_dates),
    }


def pull_breadth(api_key, api_secret, prev_state, enabled=True):
    if not enabled:
        return {"breadth": unavailable(source="Alpaca", notes="breadth pull disabled (--no-breadth)")}, None
    if not (api_key and api_secret):
        return {"breadth": unavailable(source="Alpaca",
                error="ALPACA_KEY/SECRET not set")}, None
    try:
        syms, src = sp500_constituents()
        bars = alpaca_daily_bars(syms, api_key, api_secret)
        if len(bars) < 100:
            # Alpaca IEX feed can be sparse; try Finviz fallback for the headline % > 200DMA
            raise RuntimeError(f"Only {len(bars)} symbols returned from Alpaca")
        breadth = compute_breadth(bars)
        out = {
            "constituent_source": metric(src, source=src),
            "pct_above_200dma": metric(
                breadth["pct_above_200dma"], source="Alpaca S&P500 internals",
                notes="percent of S&P 500 members above their 200-DMA",
                universe=breadth["universe_counted"],
                below_40=bool((breadth["pct_above_200dma"] or 100) < 40)),
        }
        # McClellan first, so its value can feed the Hindenburg same-day filter
        out["mcclellan"] = mcclellan_oscillator(prev_state, breadth)
        mc_state = out["mcclellan"].pop("_state", None) if isinstance(out["mcclellan"], dict) else None
        mc_val = out["mcclellan"].get("value") if isinstance(out["mcclellan"], dict) else None
        spx_50d_ret = _spx_50d_return()
        out.update(breadth_signals(breadth, mcclellan_val=mc_val, spx_50d_ret=spx_50d_ret))
        return out, {"mcclellan": mc_state} if mc_state else None
    except Exception as e:
        # Fallback: Finviz for % above 200-DMA only
        fv = finviz_pct_above_200()
        return {"breadth_error": unavailable(source="Alpaca", error=e),
                "pct_above_200dma": fv}, None


def finviz_pct_above_200():
    """Fallback: scrape Finviz group page for % of S&P stocks above 200-DMA.
    Fragile by design; used only when Alpaca fails."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
        # Finviz exposes SMA200 breadth on the index/group pages; this is best-effort.
        r = requests.get("https://finviz.com/api/counts.ashx?t=sma200",
                         headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        # shape varies; attempt common keys
        val = None
        if isinstance(j, dict):
            for k in ("above", "pct_above", "value"):
                if k in j:
                    val = float(j[k]); break
        if val is None:
            raise ValueError("Finviz response shape unrecognized")
        return metric(round(val, 1), status="proxy", source="Finviz (fallback)",
                      notes="fallback breadth source; Alpaca preferred")
    except Exception as e:
        return unavailable(source="Finviz (fallback)", error=e)


# ---------------------------------------------------------------------------
# Put/Call ratio (CBOE)
# ---------------------------------------------------------------------------

def _parse_cboe_csv(text):
    """Parse CBOE put/call CSV. Returns (date_str, ratio_float) or raises."""
    import io, csv as _csv
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("For")]
    # Skip header rows until we find the data header
    data_rows = []
    in_data = False
    for line in lines:
        low = line.lower()
        if "trade_date" in low or ("date" in low and "call" in low):
            in_data = True
            continue
        if in_data and line.strip():
            data_rows.append(line)
    if not data_rows:
        raise ValueError("No data rows found in CSV")
    # Last row = most recent trading day
    last = list(_csv.reader([data_rows[-1]]))[0]
    # Columns: Date, Call, Put, Total, P/C Ratio  (sometimes 4-column, ratio last)
    date_str = last[0].strip()
    ratio = float(last[-1].strip())   # P/C Ratio is always last column
    if not (0.1 <= ratio <= 5.0):
        raise ValueError(f"Implausible ratio: {ratio}")
    return date_str, ratio


def pull_put_call():
    """
    Fetch CBOE put/call ratio. Tries three endpoints in order:
    1. totalpc.csv   — total (equity + index) — preferred; most complete
    2. equitypc.csv  — equity-only (less index hedging distortion)
    3. Legacy JSON   — old endpoint; often 403 but kept as final fallback
    CSV format: Date, Call, Put, Total, P/C Ratio
    """
    HEADERS = {"User-Agent": "Mozilla/5.0 (market-health-dashboard/1.0)"}
    candidates = [
        ("https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",  "total"),
        ("https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv", "equity"),
        ("https://cdn.cboe.com/api/global/us_indices/daily_prices/_data/put_call_ratio.json", "json"),
    ]
    last_err = None
    for url, flavor in candidates:
        try:
            r = requests.get(url, timeout=20, headers=HEADERS)
            r.raise_for_status()
            if flavor == "json":
                j = r.json()
                val = None
                if isinstance(j, dict) and "data" in j and j["data"]:
                    last = j["data"][-1]
                    for k in ("ratio", "total", "pc_ratio", "value"):
                        if k in last:
                            val = float(last[k]); break
                if val is None:
                    raise ValueError("JSON shape unrecognized")
                date_str = "unknown"
            else:
                date_str, val = _parse_cboe_csv(r.text)
            print(f"  put/call: {val:.3f} ({flavor}, {date_str})", flush=True)
            return metric(round(val, 3), source=f"CBOE ({flavor})",
                          notes="Put/call ratio. <0.5=euphoria (contrarian warning); "
                                ">1.2=fear (contrarian opportunity). "
                                "Equity-only cleaner than total for sentiment; total more complete.",
                          asof=date_str,
                          euphoria=bool(val < 0.5),
                          fear=bool(val > 1.2))
        except Exception as e:
            last_err = e
            print(f"  put/call {flavor} failed: {e}", flush=True)
            continue
    return unavailable(source="CBOE", error=last_err,
                       notes="All three CBOE put/call endpoints failed. "
                             "If totalpc.csv 403s on GitHub Actions, add cdn.cboe.com "
                             "to network egress allowlist.")


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------

def compute_fear_greed(result):
    """
    Fetch CNN's Fear & Greed Index directly from their API.
    Falls back to z-score approximation if CNN is unreachable.
    Endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata
    Response: {fear_and_greed: {score, rating, timestamp},
               fear_and_greed_historical: {data: [{x: unix_ms, y: score, rating}]}}
    """
    CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
    }
    try:
        r = requests.get(CNN_URL, headers=HEADERS, timeout=15)
        r.raise_for_status()
        j = r.json()
        fg = j.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "")
        ts = fg.get("timestamp", "")
        if score is None:
            raise ValueError("score missing from CNN response")
        score = round(float(score), 1)
        # Normalize label to title case
        label = rating.replace("_", " ").title() if rating else (
            "Extreme Fear" if score < 25 else
            "Fear"         if score < 45 else
            "Neutral"      if score < 55 else
            "Greed"        if score < 75 else
            "Extreme Greed")
        print(f"  CNN Fear & Greed: {score} ({label})", flush=True)
        return metric(score, source="CNN (production.dataviz.cnn.io)",
                      notes="CNN's official Fear & Greed Index. "
                            "0=Extreme Fear, 100=Extreme Greed. "
                            "Updated by CNN throughout the trading day.",
                      label=label, timestamp=str(ts))
    except Exception as e:
        print(f"  CNN Fear & Greed fetch failed ({e}); falling back to z-score", flush=True)

    # ---- Fallback: z-score approximation from our own inputs ----
    def _zs(val, mean, std, invert=False):
        if not std: return 50.0
        z = (val - mean) / std
        if invert: z = -z
        return max(0.0, min(100.0, 50.0 + (z / 3.0) * 50.0))

    scores = {}
    try:
        spx_dma = (result.get("trend") or {}).get("spx_vs_200dma") or {}
        m1 = spx_dma.get("value")
        if m1 is not None:
            scores["momentum"] = _zs(m1, 2.0, 8.0)
        ho = (result.get("breadth") or {}).get("hindenburg_omen_today") or {}
        scores["strength"] = _zs((ho.get("new_high_pct") or 0) - (ho.get("new_low_pct") or 0), 0.0, 5.0)
        mc = (result.get("breadth") or {}).get("mcclellan") or {}
        if mc.get("value") is not None:
            scores["breadth"] = _zs(mc["value"], 0.0, 60.0)
        vx = (result.get("sentiment") or {}).get("vix") or {}
        if vx.get("value") is not None:
            scores["volatility"] = _zs(vx["value"], 18.0, 8.0, invert=True)
        hy = (result.get("macro") or {}).get("hy_oas") or {}
        if hy.get("value") is not None:
            scores["junk_demand"] = _zs(hy["value"], 4.5, 2.5, invert=True)
        sl = ((result.get("trend") or {}).get("dma200_slope") or {}).get("value")
        if sl:
            scores["safe_haven"] = _zs({"rising":1.5,"flat":0.0,"falling":-1.5}.get(str(sl),0.0), 0.0, 1.0)
        pc = (result.get("sentiment") or {}).get("put_call") or {}
        if pc.get("value") and pc.get("status") == "ok":
            scores["put_call"] = _zs(pc["value"], 0.85, 0.18, invert=True)
    except Exception as e2:
        print(f"WARN: fear_greed fallback failed: {e2}", file=sys.stderr)

    if not scores:
        return metric(None, status="unavailable", source="CNN / computed",
                      notes="CNN unreachable and insufficient local inputs.")
    composite = round(sum(scores.values()) / len(scores), 1)
    label = ("Extreme Fear" if composite < 25 else "Fear" if composite < 45 else
             "Neutral" if composite < 55 else "Greed" if composite < 75 else "Extreme Greed")
    return metric(composite, source="computed fallback (CNN unreachable)",
                  notes="CNN API unavailable. Z-score approximation from local inputs. "
                        "Expect ~10-15pt divergence from official CNN score.",
                  label=label, components=scores, n_inputs=len(scores))


def buffett_indicator(raw):
    try:
        w = raw.get("equity_mktcap")  # NCBEILQ027S, Millions of $, quarterly
        if w is None and FRED_API_KEY:
            try:
                w = fred_series("NCBEILQ027S", FRED_API_KEY)
            except Exception:
                w = None
        g = raw.get("gdp")  # Billions of $, quarterly
        if w is None or g is None:
            return unavailable(source="FRED:NCBEILQ027S/GDP", error="inputs missing")
        # Align quarterly series; scale equities (millions) to billions to match GDP.
        g_aligned = g.reindex(w.index, method="ffill")
        ratio_series = ((w / 1000.0) / g_aligned).dropna()
        if ratio_series.empty:
            return unavailable(source="FRED:NCBEILQ027S/GDP", error="no overlapping dates")
        latest = float(ratio_series.iloc[-1])
        x = np.arange(len(ratio_series))
        coeffs = np.polyfit(x, ratio_series.values, 1)
        trend = np.polyval(coeffs, x)
        resid = ratio_series.values - trend
        z = (resid[-1] - resid.mean()) / (resid.std() or 1)
        return metric(round(latest, 3), source="FRED:NCBEILQ027S/GDP",
                      asof=ratio_series.index[-1].strftime("%Y-%m-%d"),
                      notes="Market value of US equities / GDP (Buffett indicator). "
                            "Watch deviation from trend, not the absolute level.",
                      deviation_z=round(float(z), 2),
                      elevated=bool(z > 1.0))
    except Exception as e:
        return unavailable(source="FRED:NCBEILQ027S/GDP", error=e)


def excess_cape_yield(cape_value, raw, prefilled=None):
    try:
        # Prefer the value Shiller already computes in the workbook.
        if prefilled is not None:
            return metric(round(float(prefilled), 2), source="Shiller ie_data.xls",
                          notes="Excess CAPE Yield (from Shiller workbook): CAPE earnings yield "
                                "minus real bond yield. Lower = stocks less attractive vs bonds.",
                          low=bool(prefilled < 1.0))
        if cape_value is None:
            return unavailable(source="Shiller+FRED:DFII10", error="CAPE unavailable")
        real10 = raw.get("real_10y")
        if real10 is None:
            return unavailable(source="FRED:DFII10", error="real yield unavailable")
        ecy = (1.0 / cape_value) * 100 - float(real10.iloc[-1])
        return metric(round(ecy, 2), source="Shiller + FRED:DFII10 (computed)",
                      notes="Excess CAPE Yield = CAPE earnings yield minus real 10yr. "
                            "Lower = stocks less attractive vs bonds.",
                      low=bool(ecy < 1.0))
    except Exception as e:
        return unavailable(source="Shiller+FRED:DFII10", error=e)


def goldman_composite(raw, cape_value):
    """Percentile-rank five inputs and average -> 0-100 bear-risk-ish score.
    Higher percentile = more stretched/bear-prone, per Goldman framing."""
    parts = {}
    try:
        # Valuation (CAPE percentile, high = risky)
        if cape_value is not None:
            # Use a long-run CAPE distribution proxy: rank vs a fixed historical spread.
            parts["valuation"] = clamp(percentile_rank(cape_value,
                                       list(np.linspace(5, 44, 200))) or 50)
        # Yield curve (low/inverted = risky -> invert percentile)
        yc = raw.get("yield_curve_10y3m")
        if yc is not None:
            p = percentile_rank(float(yc.iloc[-1]), yc.values)
            if p is not None:
                parts["yield_curve"] = clamp(100 - p)
        # Unemployment (very low = late cycle = risky -> invert)
        un = raw.get("unemployment")
        if un is not None:
            p = percentile_rank(float(un.iloc[-1]), un.values)
            if p is not None:
                parts["unemployment"] = clamp(100 - p)
        # Core inflation YoY (high = risky)
        cpi = raw.get("core_cpi")
        if cpi is not None and len(cpi) > 13:
            yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100
            parts["core_inflation"] = clamp(percentile_rank(
                yoy, ((cpi.pct_change(12) * 100).dropna().values)) or 50)
        # LEI trend (falling = risky -> invert percentile of level)
        lei = raw.get("lei")
        if lei is not None:
            p = percentile_rank(float(lei.iloc[-1]), lei.values)
            if p is not None:
                parts["lei"] = clamp(100 - p)
        if not parts:
            return unavailable(source="FRED composite", error="no inputs available")
        score = sum(parts.values()) / len(parts)
        return metric(round(score, 1), source="Goldman-style composite (self-calc)",
                      notes="percentile-ranked composite; >70 historically high-risk, <40 favorable",
                      inputs=parts, n_inputs=len(parts),
                      above_70=bool(score > 70), above_50=bool(score > 50))
    except Exception as e:
        return unavailable(source="FRED composite", error=e)


def cycle_phase(raw):
    """Crude business-cycle phase from LEI direction + yield-curve level."""
    try:
        lei = raw.get("lei")
        yc = raw.get("yield_curve_10y3m")
        lei_dir = None
        if lei is not None and len(lei) > 6:
            lei_dir = "rising" if lei.iloc[-1] > lei.iloc[-6] else "falling"
        yc_level = float(yc.iloc[-1]) if yc is not None else None

        phase = "Indeterminate"
        if lei_dir == "rising" and (yc_level is None or yc_level > 1.0):
            phase = "Early Cycle"
        elif lei_dir == "rising" and yc_level is not None and 0 <= yc_level <= 1.0:
            phase = "Mid Cycle"
        elif lei_dir == "falling" and yc_level is not None and yc_level < 0:
            phase = "Late Cycle"
        elif lei_dir == "falling":
            phase = "Late / Contraction Risk"
        favored = {
            "Early Cycle": ["XLF", "XLY", "XLI"],
            "Mid Cycle": ["XLK", "XLI", "XLC"],
            "Late Cycle": ["XLE", "XLB", "XLU"],
            "Late / Contraction Risk": ["XLU", "XLV", "XLP"],
            "Indeterminate": [],
        }[phase]
        return metric(phase, source="derived: LEI + yield curve",
                      notes="business-cycle phase heuristic",
                      lei_direction=lei_dir, yield_curve=yc_level,
                      favored_sectors=favored)
    except Exception as e:
        return unavailable(source="derived", error=e)


def composite_risk(structural_inputs, cycle_inputs):
    """Average available 0-100 inputs into structural & cycle sub-scores, then combine."""
    def avg(d):
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    s = avg(structural_inputs)
    c = avg(cycle_inputs)
    comp = None
    if s is not None and c is not None:
        comp = (s + c) / 2
    elif s is not None:
        comp = s
    elif c is not None:
        comp = c

    def label(x):
        if x is None:
            return "Unknown"
        if x <= 25:
            return "Low Risk"
        if x <= 50:
            return "Moderate Risk"
        if x <= 75:
            return "Elevated Risk"
        return "High Risk"

    return {
        "composite": metric(round(comp, 1) if comp is not None else None,
                            status="ok" if comp is not None else "unavailable",
                            source="derived",
                            notes="average of structural & cycle sub-scores (available inputs only)",
                            label=label(comp),
                            structural_inputs=structural_inputs,
                            cycle_inputs=cycle_inputs),
        "structural_score": metric(round(s, 1) if s is not None else None,
                                   status="ok" if s is not None else "unavailable",
                                   source="derived", label=label(s)),
        "cycle_score": metric(round(c, 1) if c is not None else None,
                              status="ok" if c is not None else "unavailable",
                              source="derived", label=label(c)),
    }


# ---------------------------------------------------------------------------
# Catalyst calendar
# ---------------------------------------------------------------------------

def upcoming_catalysts(days=30):
    today = datetime.now(timezone.utc).date()
    out = []
    for iso, label in CATALYST_CALENDAR:
        try:
            d = datetime.strptime(iso, "%Y-%m-%d").date()
            if today <= d <= today + timedelta(days=days):
                out.append({"date": iso, "label": label,
                            "days_away": (d - today).days})
        except Exception:
            continue
    return sorted(out, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# State (carried run-to-run via data/state.json, committed by CI)
# ---------------------------------------------------------------------------

def load_state(path="data/state.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state, path="data/state.json"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"WARN: could not save state: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ordinal(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def pull_portfolio(positions, risk_score=None):
    """Calculate portfolio-level beta, Sharpe, concentration, and sector overlap."""
    if not positions:
        return {"status": "no_positions", "notes": "No positions configured in watchlist.py"}
    try:
        import yfinance as yf
        tickers = [t for t, _ in positions]
        shares_map = {t: s for t, s in positions}

        # Fetch SPY first as benchmark
        spy_df = yf.download("SPY", period="2y", interval="1d",
                             auto_adjust=True, progress=False)
        if spy_df is None or spy_df.empty:
            return {"status": "unavailable", "error": "SPY benchmark data unavailable"}
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.droplevel(1)
        spy = spy_df["Close"].dropna() if "Close" in spy_df.columns else spy_df.iloc[:, 0].dropna()

        # Fetch each ticker individually — avoids all multi-ticker column structure issues
        holdings = []
        skipped = []
        for ticker, shares in positions:
            try:
                df = yf.download(ticker, period="2y", interval="1d",
                                 auto_adjust=True, progress=False)
                if df is None or df.empty or len(df) < 30:
                    skipped.append(ticker)
                    continue
                # yfinance sometimes returns MultiIndex columns even for single tickers
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                px = df["Close"].dropna() if "Close" in df.columns else df.iloc[:, 0].dropna()
                if len(px) < 30:
                    skipped.append(ticker)
                    continue
                price = float(px.iloc[-1])
                value = price * shares
                # Beta vs SPY
                ret = px.pct_change().dropna()
                spy_ret = spy.pct_change().dropna()
                r, s = ret.align(spy_ret, join="inner")
                if len(r) > 60:
                    cov = float(r.cov(s))
                    var = float(s.var())
                    beta = round(cov / var, 2) if var > 0 else 1.0
                else:
                    beta = 1.0
                # Sharpe (annualised, ~4.33% risk-free)
                rf_daily = 0.0433 / 252
                excess = ret - rf_daily
                sharpe = round(float(excess.mean() / excess.std() * (252 ** 0.5)), 2) if excess.std() > 0 else 0.0
                # 1-yr return
                ret_1y = round(float(px.iloc[-1] / px.iloc[max(-252, -len(px))] - 1) * 100, 1) if len(px) >= 21 else None
                holdings.append({
                    "ticker": ticker, "shares": shares, "price": round(price, 2),
                    "value": round(value, 2), "beta": beta,
                    "sharpe": sharpe, "ret_1y": ret_1y,
                })
                time.sleep(0.1)  # be gentle with Yahoo
            except Exception as e:
                skipped.append(f"{ticker}")
                print(f"  [portfolio] {ticker} skipped: {str(e)[:60]}", file=sys.stderr)

        if not holdings:
            return {"status": "unavailable", "error": "no price data returned", "skipped": skipped}

        total_value = sum(h["value"] for h in holdings)
        for h in holdings:
            h["weight"] = round(h["value"] / total_value * 100, 1) if total_value else 0

        port_beta = round(sum(h["weight"] / 100 * h["beta"] for h in holdings), 2)
        port_sharpe = round(sum(h["weight"] / 100 * h["sharpe"] for h in holdings), 2)
        top5_weight = round(sum(h["weight"] for h in sorted(holdings, key=lambda x: -x["weight"])[:5]), 1)

        beta_context = ""
        if risk_score is not None:
            if risk_score >= 75 and port_beta > 1.2:
                beta_context = "High beta in high-risk environment — consider reducing exposure"
            elif risk_score >= 50 and port_beta > 1.4:
                beta_context = "Elevated beta vs elevated macro risk — worth monitoring"
            elif risk_score >= 50 and port_beta > 1.1:
                beta_context = "Moderately elevated beta in an elevated-risk tape — stay selective"
            elif risk_score <= 30 and port_beta < 0.8:
                beta_context = "Low beta in low-risk environment — room to add exposure"
            else:
                beta_context = "Beta and risk score are reasonably aligned"

        holdings.sort(key=lambda x: -x["value"])

        return {
            "status": "ok",
            "total_value": round(total_value, 2),
            "position_count": len(holdings),
            "portfolio_beta": port_beta,
            "portfolio_sharpe": port_sharpe,
            "top5_concentration_pct": top5_weight,
            "risk_score_context": beta_context,
            "macro_risk_score": risk_score,
            "holdings": holdings,
            "skipped": skipped,
            "asof": spy.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)[:300]}


def interpret(result, fred_raw):
    """Generate dynamic per-metric readings + a top-level summary and a
    slightly-prescriptive Watch/Consider list, all from the assembled result."""
    R = {"readings": {}, "watch": [], "summary": ""}

    def mval(section, key, field="value"):
        m = result.get(section, {}).get(key)
        if isinstance(m, dict):
            return m.get(field)
        return None

    def trend_word(section, key):
        m = result.get(section, {}).get(key)
        if isinstance(m, dict) and isinstance(m.get("trend"), dict):
            return m["trend"].get("direction")
        return None

    # ---------- Per-metric readings ----------

    # CAPE
    cape = mval("structural", "cape")
    cape_pct = None
    if cape is not None:
        cape_pct = percentile_rank(cape, list(np.linspace(5, 44, 200))) or 50
        if cape_pct >= 90:
            tail = "among the most expensive readings ever recorded — deeply stretched"
        elif cape_pct >= 75:
            tail = "well above the long-run average of ~17 — elevated"
        elif cape_pct >= 40:
            tail = "near its historical middle"
        else:
            tail = "below average — relatively cheap"
        R["readings"]["cape"] = (f"At the {_ordinal(round(cape_pct))} percentile of all history; {tail}. "
                                 "Says little about this month, a lot about the next decade.")

    # Buffett
    bz = mval("structural", "buffett_indicator", "deviation_z")
    if bz is not None:
        if bz >= 2:
            tail = "at a record extreme versus its own trend"
        elif bz >= 1:
            tail = "stretched above trend"
        elif bz >= -1:
            tail = "near its long-run trend"
        else:
            tail = "below trend"
        R["readings"]["buffett_indicator"] = (f"Total market value vs GDP is {tail} "
                                              f"(z = {bz:+.1f}). Watch the deviation, not the level.")

    # Excess CAPE Yield
    ecy = mval("structural", "excess_cape_yield")
    if ecy is not None:
        if ecy < 0:
            tail = "negative — by this gauge stocks are priced to underperform bonds"
        elif ecy < 1:
            tail = "low — thin compensation for holding stocks over bonds"
        else:
            tail = "positive — stocks still offer a yield cushion over bonds"
        R["readings"]["excess_cape_yield"] = f"{ecy:.2f}% — {tail}."

    # Forward P/E
    fpe = mval("structural", "forward_pe")
    fpe_mean = mval("structural", "forward_pe", "vs_mean")
    if fpe is not None:
        state = "elevated" if fpe >= 20 else "moderate" if fpe >= 16 else "below average"
        R["readings"]["forward_pe"] = (f"{fpe:.1f}x next-12-month earnings — {state}"
                                       + (f", {fpe_mean:+.1f} vs the ~16x long-run mean." if fpe_mean is not None else "."))

    # Yield curve
    yc = mval("macro", "yield_curve_10y3m")
    yct = trend_word("macro", "yield_curve_10y3m")
    if yc is not None:
        if yc < 0:
            base = "Inverted — the classic recession warning; every U.S. recession since the 1950s followed an inversion"
        elif yc < 0.5:
            base = "Positive but flat — the recession cushion has thinned"
        else:
            base = "Comfortably positive — no recession signal here"
        if yct:
            base += f"; {yct} lately"
        R["readings"]["yield_curve_10y3m"] = base + "."

    # Credit spreads (HY OAS)
    hy = mval("macro", "hy_oas")
    hyt = trend_word("macro", "hy_oas")
    if hy is not None:
        if hy < 3:
            base = "Very tight — credit markets are complacent, little stress priced in"
        elif hy < 5:
            base = "Normal — no credit stress"
        elif hy < 7:
            base = "Widening into caution territory — watch closely"
        else:
            base = "Stressed — credit markets are flashing real risk"
        if hyt == "rising":
            base += "; spreads are widening (bond market getting nervous)"
        elif hyt == "falling":
            base += "; spreads are tightening (risk appetite firm)"
        R["readings"]["hy_oas"] = base + "."

    # LEI
    lei = mval("macro", "lei")
    leit = trend_word("macro", "lei")
    if lei is not None:
        if leit == "rising":
            base = "Rising — leading indicators point to continued expansion"
        elif leit == "falling":
            base = "Falling — leading indicators are softening, an early caution"
        else:
            base = "Flat — leading indicators are stalling"
        R["readings"]["lei"] = base + "."

    # Unemployment
    un = mval("macro", "unemployment")
    unt = trend_word("macro", "unemployment")
    if un is not None:
        if unt == "rising":
            base = f"{un:.1f}% and rising — a turn up from lows is a late-cycle warning (Sahm-rule territory if it accelerates)"
        elif unt == "falling":
            base = f"{un:.1f}% and falling — labor market still firm"
        else:
            base = f"{un:.1f}%, holding steady"
        R["readings"]["unemployment"] = base + "."

    # Inflation (Core PCE preferred, CPI fallback)
    for key, label in (("core_pce", "Core PCE"), ("core_cpi", "Core CPI")):
        yoy = mval("macro", key, "yoy_pct")
        ydir = mval("macro", key, "yoy_direction")
        if yoy is not None:
            vs = "above" if yoy > 2.2 else "near" if yoy > 1.8 else "below"
            base = f"{label} running {yoy:.1f}% YoY, {vs} the Fed's 2% target"
            if ydir:
                base += f" and {ydir}"
            R["readings"][key] = base + "."

    # Fed funds stance
    ff = mval("macro", "fed_funds")
    fft = trend_word("macro", "fed_funds")
    if ff is not None:
        stance = "cutting" if fft == "falling" else "hiking" if fft == "rising" else "on hold"
        R["readings"]["fed_funds"] = (f"Policy rate {ff:.2f}%, currently {stance}. "
                                      "Rate-cut cycles have preceded recent recessions — easing is not always bullish.")

    # Consumer sentiment
    um = mval("macro", "umich_sentiment")
    umt = trend_word("macro", "umich_sentiment")
    if um is not None:
        lvl = "weak" if um < 70 else "moderate" if um < 90 else "strong"
        base = f"{um:.0f} — {lvl} consumer mood"
        if umt:
            base += f", {umt}"
        R["readings"]["umich_sentiment"] = base + "."

    # NFCI leverage
    nf = mval("macro", "nfci_leverage")
    if nf is not None:
        base = ("Positive — financial leverage/conditions tighter than average"
                if nf > 0 else "Negative — leverage/conditions looser than average (easy money)")
        R["readings"]["nfci_leverage"] = base + "."

    # Delinquencies
    for key, label in (("cc_delinquency", "Credit-card"), ("auto_delinquency", "Auto-loan")):
        v = mval("macro", key)
        t = trend_word("macro", key)
        if v is not None:
            base = f"{label} delinquencies at {v:.2f}%"
            if t == "rising":
                base += " and rising — early sign of consumer stress (leads the economy by months)"
            elif t == "falling":
                base += " and easing"
            R["readings"][key] = base + "."

    # VIX
    vix = mval("sentiment", "vix")
    if vix is not None:
        if vix < 15:
            tail = "calm — possibly complacent"
        elif vix < 20:
            tail = "normal"
        elif vix < 30:
            tail = "elevated — markets are nervous"
        else:
            tail = "high — stress/fear regime (often a contrarian bottom signal above 40)"
        R["readings"]["vix"] = f"{vix:.1f} — {tail}."

    # Breadth
    pa = mval("breadth", "pct_above_200dma")
    if pa is not None:
        if pa >= 60:
            tail = "broad participation — healthy"
        elif pa >= 40:
            tail = "narrowing — fewer names holding up the index"
        else:
            tail = "weak — distribution underway beneath the surface"
        R["readings"]["pct_above_200dma"] = f"{pa:.0f}% of the S&P 500 above its 200-DMA — {tail}."

    ad_div = mval("structural", "ad_line_proxy", "bearish_divergence")
    if ad_div is not None:
        R["readings"]["ad_line_proxy"] = ("Equal-weight is lagging cap-weight — a few megacaps are masking weaker breadth."
                                          if ad_div else "Breadth confirms the index — no divergence.")

    # ---------- Summary + Watch/Consider ----------
    score = mval("scores", "composite")
    label = mval("scores", "composite", "label") or "Unknown"
    s_parts = []
    if score is not None:
        s_parts.append(f"{label} ({score:.0f}/100).")

    # Valuation clause
    val_flags = []
    if cape_pct is not None and cape_pct >= 85:
        val_flags.append("CAPE near historic highs")
    if bz is not None and bz >= 1.5:
        val_flags.append("the Buffett indicator at a record")
    if ecy is not None and ecy < 1:
        val_flags.append("a thin equity-risk premium")
    if val_flags:
        s_parts.append("Valuations are stretched — " + ", ".join(val_flags) + ".")

    # Cycle clause
    cyc_bits = []
    if leit == "rising":
        cyc_bits.append("leading indicators still rising")
    elif leit == "falling":
        cyc_bits.append("leading indicators softening")
    if yc is not None:
        cyc_bits.append("the yield curve " + ("inverted" if yc < 0 else "still positive"))
    if hy is not None:
        cyc_bits.append("credit spreads " + ("calm" if hy < 5 else "widening"))
    if unt == "rising":
        cyc_bits.append("unemployment ticking up")
    if cyc_bits:
        joined = ", ".join(cyc_bits)
        if val_flags and ("positive" in joined or "rising" in joined or "calm" in joined):
            s_parts.append("Still, the cycle holds — " + joined + ".")
        else:
            s_parts.append("On the cycle: " + joined + ".")

    # Breadth/trigger clause
    if pa is not None:
        if pa < 40:
            s_parts.append("Breadth has broken down — distribution beneath the surface.")
        elif ad_div:
            s_parts.append("Breadth is narrowing, a yellow flag.")
        else:
            s_parts.append("Breadth is healthy, so no regime break yet.")

    R["summary"] = " ".join(s_parts)

    # Watch / Consider (slightly prescriptive)
    watch = []
    if hy is not None and (hy >= 5 or hyt == "rising"):
        watch.append("Credit spreads are the key tell right now — widening here would confirm rising risk.")
    if yc is not None and 0 <= yc < 0.5:
        watch.append("Yield curve is flat; an inversion would be a fresh recession warning.")
    if val_flags:
        watch.append("With valuations this stretched, favor quality and proven earnings over speculative names.")
    if pa is not None and pa < 50:
        watch.append("Thin breadth argues for tighter stops and smaller new positions.")
    elif pa is not None and pa >= 60:
        watch.append("Breadth is broad — the tape supports staying invested.")
    if unt == "rising":
        watch.append("Rising unemployment is the cleanest recession trigger to monitor from here.")
    for key in ("core_pce", "core_cpi"):
        ydir = mval("macro", key, "yoy_direction")
        if ydir == "accelerating":
            watch.append("Inflation is re-accelerating — watch the next CPI/PCE print and Fed tone.")
            break
    cats = result.get("catalysts", {}).get("upcoming", [])
    near = [c for c in cats if isinstance(c, dict) and c.get("days_away", 99) <= 7]
    if near:
        nm = near[0]
        watch.append(f"Next catalyst: {nm['label']} on {nm['date']} ({nm['days_away']}d).")
    if not watch:
        watch.append("No pressing risks firing; conditions are within normal ranges.")
    R["watch"] = watch[:6]

    return R


def run(no_breadth=False):
    utc_iso, pt_disp = now_stamps()
    prev_state = load_state()
    new_state = dict(prev_state)

    result = {
        "meta": {
            "generated_utc": utc_iso,
            "generated_display": pt_disp,
            "schema_version": 2,
            "notes": "Each metric carries status/source. status 'proxy' = computed "
                     "stand-in for a proprietary/unavailable series (see notes).",
        },
        "macro": {}, "structural": {}, "trend": {}, "breadth": {},
        "sentiment": {}, "sectors": {}, "catalysts": {}, "scores": {},
        "source_health": {},
    }

    # ---- FRED ----
    fred_out, fred_raw = pull_fred_all(FRED_API_KEY)
    result["macro"].update(fred_out)

    # ---- Shiller CAPE ----
    cape_metric, cape_value, cape_ecy = pull_cape()
    result["structural"]["cape"] = cape_metric

    # ---- Derived valuation ----
    result["structural"]["buffett_indicator"] = buffett_indicator(fred_raw)
    result["structural"]["excess_cape_yield"] = excess_cape_yield(cape_value, fred_raw, cape_ecy)
    result["structural"]["forward_pe"] = metric(
        FORWARD_PE, status="manual", source="manual (Yardeni)",
        notes=f"manual monthly input; historical mean ~{FORWARD_PE_HIST_MEAN}",
        vs_mean=round(FORWARD_PE - FORWARD_PE_HIST_MEAN, 2))

    # ---- Trend / VIX / A-D proxy ----
    result["trend"].update(pull_spx_technicals())
    result["sentiment"]["vix"] = pull_vix()
    result["structural"]["ad_line_proxy"] = pull_ad_proxy()

    # ---- Breadth internals (heavy) ----
    breadth_out, breadth_state = pull_breadth(
        ALPACA_KEY, ALPACA_SECRET, prev_state, enabled=not no_breadth)
    result["breadth"].update(breadth_out)
    if breadth_state:
        new_state.update(breadth_state)

    # ---- Sentiment: put/call + Fear & Greed ----
    result["sentiment"]["put_call"] = pull_put_call()
    # Fear & Greed computed AFTER breadth/trend/macro are assembled
    result["sentiment"]["fear_greed"] = compute_fear_greed(result)

    # ---- Sectors ----
    result["sectors"]["relative_strength"] = pull_sectors()
    result["sectors"]["cycle_phase"] = cycle_phase(fred_raw)

    # ---- Goldman composite ----
    result["scores"]["goldman_composite"] = goldman_composite(fred_raw, cape_value)

    # ---- Composite risk score ----
    structural_inputs = {}
    cyc_inputs = {}
    # Structural 0-100 contributions (higher = riskier)
    cv = result["structural"]["cape"]
    if cv["status"] == "ok" and cv["value"]:
        structural_inputs["cape"] = clamp(percentile_rank(cv["value"],
                                          list(np.linspace(5, 44, 200))) or 50)
    bi = result["structural"]["buffett_indicator"]
    if bi["status"] == "ok":
        structural_inputs["buffett"] = clamp(50 + (bi.get("deviation_z", 0) or 0) * 20)
    ecy = result["structural"]["excess_cape_yield"]
    if ecy["status"] == "ok" and ecy["value"] is not None:
        # lower ECY = riskier; map ~[-1,5] to [100,0]
        structural_inputs["ecy"] = clamp(100 - (ecy["value"] + 1) / 6 * 100)
    pct200 = result["breadth"].get("pct_above_200dma")
    if pct200 and pct200.get("status") in ("ok", "proxy") and pct200.get("value") is not None:
        # lower % above 200dma = riskier
        structural_inputs["breadth"] = clamp(100 - pct200["value"])
    # Cycle 0-100 contributions
    gc = result["scores"]["goldman_composite"]
    if gc["status"] == "ok":
        cyc_inputs["goldman"] = gc["value"]
    nfci = fred_raw.get("nfci_leverage")
    if nfci is not None:
        p = percentile_rank(float(nfci.iloc[-1]), nfci.values)
        if p is not None:
            cyc_inputs["nfci_leverage"] = clamp(p)  # higher leverage = riskier
    ds = fred_raw.get("debt_service")
    if ds is not None:
        p = percentile_rank(float(ds.iloc[-1]), ds.values)
        if p is not None:
            cyc_inputs["debt_service"] = clamp(p)

    result["scores"].update(composite_risk(structural_inputs, cyc_inputs))

    # ---- Catalysts ----
    result["catalysts"]["upcoming"] = upcoming_catalysts(30)

    # ---- Interpretation: per-metric readings + summary + watch list ----
    try:
        result["interpretation"] = interpret(result, fred_raw)
    except Exception as e:
        result["interpretation"] = {"summary": "", "watch": [], "readings": {},
                                    "error": str(e)[:200]}

    # ---- History log (compounds over time; stored in committed state.json) ----
    try:
        hist = new_state.get("history", [])
        today = utc_iso[:10]
        snap = {
            "date": today,
            "composite": mval_path(result, "scores", "composite", "value"),
            "structural": mval_path(result, "scores", "structural_score", "value"),
            "cycle": mval_path(result, "scores", "cycle_score", "value"),
            "cape": mval_path(result, "structural", "cape", "value"),
            "buffett_z": mval_path(result, "structural", "buffett_indicator", "deviation_z"),
            "yield_curve": mval_path(result, "macro", "yield_curve_10y3m", "value"),
            "hy_oas": mval_path(result, "macro", "hy_oas", "value"),
            "vix": mval_path(result, "sentiment", "vix", "value"),
            "pct_above_200": mval_path(result, "breadth", "pct_above_200dma", "value"),
            "goldman": mval_path(result, "scores", "goldman_composite", "value"),
            "hindenburg_raw": bool(mval_path(result, "breadth", "hindenburg_omen_today", "value")),
            "titanic_raw": bool(mval_path(result, "breadth", "titanic_syndrome_today", "value")),
            "new_high_pct": mval_path(result, "breadth", "hindenburg_omen_today", "new_high_pct"),
            "new_low_pct": mval_path(result, "breadth", "hindenburg_omen_today", "new_low_pct"),
        }
        # one entry per day (replace today's if re-run)
        hist = [h for h in hist if h.get("date") != today]
        hist.append(snap)
        new_state["history"] = hist[-460:]  # ~15 months of daily points
        # Expose a compact recent window in latest.json so the dashboard can draw sparklines
        result["history"] = new_state["history"][-90:]
        # Promote raw flags -> confirmed clusters using the freshly-updated history
        try:
            result["breadth"].update(confirm_clusters(new_state["history"]))
        except Exception as e:
            print(f"WARN: cluster confirmation failed: {e}", file=sys.stderr)
    except Exception as e:
        print(f"WARN: history log failed: {e}", file=sys.stderr)

    # ---- Source health summary ----
    def walk_status(node, acc):
        if isinstance(node, dict):
            if "status" in node and "value" in node:
                if not node.get("derived"):   # derived signals don't count vs confidence
                    acc[node["status"]] = acc.get(node["status"], 0) + 1
            else:
                for v in node.values():
                    walk_status(v, acc)
        return acc
    health = {}
    walk_status(result, health)
    result["source_health"] = health

    # ---- Persist state ----
    new_state["last_run_utc"] = utc_iso
    save_state(new_state)

    return result


def mval_path(result, section, key, field="value"):
    m = result.get(section, {}).get(key)
    return m.get(field) if isinstance(m, dict) else None


def selftest():
    """Offline validation of the pure-math functions (no network)."""
    print("Running offline self-test...")
    # MACD / RSI on a synthetic uptrend
    series = list(100 + np.cumsum(np.random.RandomState(0).randn(300) + 0.05))
    m, s, h = macd(series)
    assert isinstance(m, float) and isinstance(h, float), "MACD failed"
    r = rsi(series)
    assert 0 <= r <= 100, f"RSI out of range: {r}"
    lbl, chg = slope_label(pd.Series(series).rolling(50).mean())
    assert lbl in ("rising", "flat", "declining", "unknown"), "slope label failed"
    p = percentile_rank(50, list(range(100)))
    assert abs(p - 50) < 2, f"percentile failed: {p}"
    # composite
    cr = composite_risk({"a": 80, "b": 60}, {"c": 40})
    assert cr["composite"]["value"] == round((70 + 40) / 2, 1), "composite math failed"
    # breadth signal proxy
    bs = breadth_signals({"universe_counted": 500, "new_highs": 20, "new_lows": 18,
                          "advancers": 250, "decliners": 240})
    assert "hindenburg_omen_today" in bs, "breadth signals failed"
    # mcclellan seeding
    mc = mcclellan_oscillator({}, {"advancers": 300, "decliners": 200})
    assert mc["status"] == "proxy", "mcclellan failed"
    # cycle phase + catalysts smoke
    assert upcoming_catalysts(3650), "catalyst calendar empty"
    print("  MACD:", round(m, 3), "RSI:", round(r, 1), "slope:", lbl)
    print("  composite:", cr["composite"]["value"], "label:", cr["composite"]["label"])
    print("  mcclellan osc:", mc["value"])
    print("ALL SELF-TESTS PASSED ✅")


# Minimal static S&P500 fallback (partial — enough to function if Wikipedia is down).
# In production Wikipedia provides the full list; this is the safety net.
FALLBACK_SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","BRK.B","LLY","AVGO",
    "TSLA","JPM","V","XOM","UNH","MA","JNJ","PG","HD","COST","ABBV","MRK",
    "CVX","ADBE","CRM","WMT","PEP","KO","BAC","NFLX","AMD","TMO","ACN","LIN",
    "MCD","CSCO","ABT","DHR","WFC","TXN","QCOM","INTU","DIS","CAT","VZ","INTC",
    "AMGN","CMCSA","PFE","NOW","IBM","UNP","GE","HON","SPGI","LOW","COP","ISRG",
    "PM","RTX","GS","NEE","UBER","T","ELV","BKNG","MS","PLD","BLK","SCHW","C",
    "MDT","SBUX","DE","LMT","ADP","GILD","MDLZ","CB","REGN","ADI","BMY","VRTX",
    "MMC","TJX","SO","ETN","PGR","CI","BSX","FI","ZTS","DUK","SLB","BDX","AON",
    "ITW","CL","WM","MO","EOG","APD","NOC","CME","ICE","MCK","SHW","PYPL","CDNS",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-breadth", action="store_true",
                    help="skip the heavy 500-symbol Alpaca breadth pull")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: validate pure-math functions, no network")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        sys.exit(0)

    try:
        data = run(no_breadth=args.no_breadth)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {OUTPUT_PATH}")
        print("Source health:", data["source_health"])
    except Exception as e:
        print("FATAL during run:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
