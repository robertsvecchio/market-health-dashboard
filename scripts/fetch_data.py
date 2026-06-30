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
    "philly_coincident": ("USPHCI",             "Philly Fed Coincident Index — 4-factor current economic activity"),
    "sahm_rule":         ("SAHMREALTIME",       "Sahm Rule: 3mo avg unemployment rise from 12mo low. >=0.5 = recession onset"),
    "recession_prob":    ("RECPROUSM156N",       "Chauvet-Piger smoothed recession probability (0-100%)"),
    "fed_funds":         ("DFF",                "Effective federal funds rate"),
    "umich_sentiment":   ("UMCSENT",            "U. Michigan Consumer Sentiment"),
    "avg_hourly_earnings":("CES0500000003",     "Average hourly earnings, total private (for real wage growth)"),
    "hy_oas":            ("BAMLH0A0HYM2",       "ICE BofA US High-Yield option-adjusted credit spread"),
    "ig_oas":            ("BAMLC0A0CM",         "ICE BofA US Investment-Grade option-adjusted credit spread"),
    # Valuation building blocks
    "equity_mktcap":     ("NCBEILQ027S",        "Fed B.103 market value of equities (Buffett numerator; Wilshire removed from FRED Jun 2024)"),
    "real_10y":          ("DFII10",             "10yr TIPS yield (real) — for Excess CAPE Yield"),
    # NEW: Margin debt via Fed Flow of Funds (quarterly)
    "margin_debt":       ("BOGZ1FL663067003Q",  "FINRA margin debt (Fed Flow of Funds, quarterly)"),
}

# How far back to look when computing a metric's trend, by series cadence.
TREND_LOOKBACK = {
    "lei": 3, "yield_curve_10y3m": 21, "nfci_leverage": 4, "unemployment": 3,
    "philly_coincident": 3, "sahm_rule": 1, "recession_prob": 3,
    "lending_standards": 1,
    "cc_delinquency": 1, "auto_delinquency": 1, "savings_rate": 3, "debt_service": 1,
    "corp_profits": 1, "gdp": 1, "core_cpi": 3, "core_pce": 3, "ppi": 3, "fed_funds": 21,
    "umich_sentiment": 3, "avg_hourly_earnings": 3, "hy_oas": 21, "ig_oas": 21,
    "real_10y": 21, "equity_mktcap": 1,
    "margin_debt": 1,
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
# Empirically-grounded threshold maps
# ---------------------------------------------------------------------------

METRIC_THRESHOLDS = {
    # CAPE: Shiller empirical research; breakpoints validated against 1881-2026 data
    # Source: Shiller (2000), Campbell & Shiller (1998), Wikipedia CAPE article
    "cape": {
        "low":      (None, 15),    # <15: 10yr returns historically >10% real
        "moderate": (15,   25),    # 15-25: 7-10% expected real returns
        "elevated": (25,   35),    # 25-35: <5% expected; Shiller called >25 "thin ice"
        "high":     (35,   None),  # >35: near-zero or negative real 10yr returns
        "source": "Shiller (2000); Campbell & Shiller (1998)",
    },
    # HY OAS level: BofA/FRED historical series 1996-2026
    # Source: ICE BofA HY OAS (BAMLH0A0HYM2); Eco3min framework
    "hy_oas": {
        "complacent": (None, 3.5), # <350bps: risk premium compressed, complacency
        "low":        (3.5,  4.5), # 350-450bps: normal range (LT avg ~4.5-5%)
        "moderate":   (4.5,  5.0), # 450-500bps: approaching stress
        "elevated":   (5.0,  6.0), # 500-600bps: stress
        "high":       (6.0,  None),# >600bps: 85% recession within 12-18mo historically
        "source": "ICE BofA OAS historical 1996-2024; Eco3min research",
    },
    # HY momentum: 90-day bps change
    "hy_momentum_bps": {
        "low":      (None, -50),   # tightening >50bps: bullish
        "moderate": (-50,   80),   # neutral zone
        "elevated": (80,   150),   # widening fast: watch
        "high":     (150,  None),  # >150bps widening: credit stress signal
        "source": "Eco3min HY OAS framework; empirical 3-9mo recession lead",
    },
    # SLOOS: net % banks tightening C&I standards
    # Source: Federal Reserve SLOOS research; Fed FEDS Notes (2024)
    "sloos": {
        "low":      (None,   0),   # easing: banks loosening credit = bullish
        "moderate": (0,     20),   # mild tightening: normal
        "elevated": (20,    40),   # >20%: preceded every post-1990 recession
        "high":     (40,   None),  # >40%: active credit crunch territory
        "source": "Federal Reserve SLOOS research; FEDS Notes May 2024",
    },
    # Yield curve 10y-3m
    # Source: Estrella & Mishkin (1998); NY Fed recession model
    "yield_curve_10y3m": {
        "low":      (1.5,  None),  # >150bps: healthy, expansionary
        "moderate": (0.5,  1.5),   # flattening: watch zone
        "elevated": (-0.5, 0.5),   # flat/near-zero: includes re-steepening danger
        "high":     (None, -0.5),  # inverted: 7/8 historical crashes preceded
        "source": "Estrella & Mishkin (1998); NY Fed yield curve model",
    },
    # Sahm Rule: exact empirical threshold from Sahm (2019)
    "sahm_rule": {
        "low":      (None, 0.3),   # no signal
        "moderate": (0.3,  0.5),   # approaching threshold, watch
        "elevated": (0.5,  0.8),   # TRIGGERED: recession onset historically confirmed
        "high":     (0.8,  None),  # deep in recession signal
        "source": "Sahm (2019); Federal Reserve real-time series SAHMREALTIME",
    },
    # Margin debt YoY %
    # Source: FINRA historical data; Goepfert (SentimenTrader) research
    "margin_debt_yoy": {
        "low":      (None,  10),   # <10% YoY: normal leverage growth
        "moderate": (10,    30),   # moderate accumulation
        "elevated": (30,    50),   # elevated speculation
        "high":     (50,   None),  # >50% YoY: preceded 2000 and 2007 peaks
        "source": "FINRA margin debt history; SentimenTrader research",
    },
}


def threshold_label(metric_key, value):
    """Return Low/Moderate/Elevated/High/Complacent based on METRIC_THRESHOLDS."""
    if metric_key not in METRIC_THRESHOLDS or value is None:
        return None
    t = METRIC_THRESHOLDS[metric_key]
    # Special case: complacent (below-normal is its own risk category)
    if "complacent" in t:
        lo, hi = t["complacent"]
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return "Complacent"
    for lbl in ("low", "moderate", "elevated", "high"):
        if lbl not in t:
            continue
        lo, hi = t[lbl]
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return lbl.capitalize()
    return None


# ---------------------------------------------------------------------------
# NEW: HY OAS 90-day momentum score
# ---------------------------------------------------------------------------

def hy_momentum_score(fred_raw):
    """Compute HY OAS 90-day momentum score (0-100) and raw bps change.
    Tightening = low risk, widening = high risk.
    >+150bps = 100, 0 change = 50, <-100bps = 0.
    Returns (score, change_bps) or (None, None) if insufficient data."""
    hy = fred_raw.get("hy_oas")
    if hy is None or len(hy) < 65:
        return None, None
    try:
        current = float(hy.iloc[-1])
        prior_90d = float(hy.iloc[-65])  # ~90 trading days
        change_bps = (current - prior_90d) * 100  # OAS is in %, convert to bps
        # Score: tightening = low risk, widening = high risk
        # >+150bps change = 100, 0 = 50, <-100bps = 0
        score = clamp(50 + change_bps / 3)
        return round(score, 1), round(change_bps, 1)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Five-bucket weighted composite_risk (spec v2 + Spec Addition 1)
# ---------------------------------------------------------------------------

def _risk_label(x):
    if x is None:
        return "Unknown"
    if x <= 25:
        return "Low Risk"
    if x <= 50:
        return "Moderate Risk"
    if x <= 65:
        return "Elevated Risk"
    return "High Risk"


def composite_risk(fred_raw, breadth_result, scores_result, cape_value):
    """
    Five-bucket weighted composite risk score.
    All inputs normalized to 0-100 (0=no risk, 100=max risk).
    Missing inputs: dropped and weights renormalized proportionally.

    Bucket weights (empirically grounded on historical signal quality):
      Credit     30%  — highest backtested hit rate (SLOOS, HY momentum, NFCI)
      Cycle      25%  — LEI + yield curve + Goldman + debt service (Spec Addition 1)
      Valuation  20%  — CAPE + Buffett + ECY (structural, not timing); capped at 70
      Breadth    15%  — % above 200DMA + McClellan
      Labor      10%  — Sahm + unemployment trend + CC delinquency

    Spec Addition 1 changes:
      - Yield curve added directly to Cycle bucket
      - Cycle bucket weights: lei 0.30, yc 0.30, goldman 0.30, debt_service 0.10
      - Co-firing bonus: LEI declining + YC in danger zone → cycle_score +15
      - Valuation capped at 70 before composite (prevents structural CAPE crowding
        out time-sensitive signals)
    """

    def wavg(inputs_weights):
        """Weighted average of (value, weight) pairs; skip None; renormalize."""
        available = [(v, w) for v, w in inputs_weights if v is not None]
        if not available:
            return None
        total_w = sum(w for _, w in available)
        return sum(v * w for v, w in available) / total_w

    def prank(series_key, invert=False):
        """Percentile rank of latest value in fred_raw series."""
        s = fred_raw.get(series_key)
        if s is None or len(s) < 10:
            return None
        p = percentile_rank(float(s.iloc[-1]), s.values)
        if p is None:
            return None
        return clamp(100 - p if invert else p)

    # ── CREDIT BUCKET (weight 0.30) ──────────────────────────────────────────
    sloos_raw = fred_raw.get("lending_standards")
    sloos_score = None
    if sloos_raw is not None:
        v = float(sloos_raw.iloc[-1])
        # -30 (easing) -> 0, 0 -> 40, +20 -> 60, +40 -> 80, +60 -> 100
        sloos_score = clamp(40 + v * 1.0)

    hy_mom_score = None
    hm = scores_result.get("hy_momentum_score")
    if hm is not None:
        hy_mom_score = hm

    nfci_score = prank("nfci_leverage")

    credit_score = wavg([
        (sloos_score,  0.40),
        (hy_mom_score, 0.35),
        (nfci_score,   0.25),
    ])

    # ── CYCLE BUCKET (weight 0.25) — Spec Addition 1: add yield curve ────────
    # LEI direction: falling=100, flat=50, rising=0 (6-month lookback)
    lei_score = None
    lei_raw = fred_raw.get("lei")
    if lei_raw is not None and len(lei_raw) >= 7:
        delta = float(lei_raw.iloc[-1]) - float(lei_raw.iloc[-7])
        if delta > 0.2:
            lei_score = 20.0
        elif delta < -0.2:
            lei_score = 80.0
        else:
            lei_score = 50.0

    # Yield curve → 0-100 risk score (Spec Addition 1)
    yc_raw = fred_raw.get("yield_curve_10y3m")
    yc_val = float(yc_raw.iloc[-1]) if yc_raw is not None else None

    # Detect post-inversion: was YC negative at any point in last 390 trading days?
    post_inv = False
    if yc_raw is not None and len(yc_raw) >= 390:
        post_inv = bool((yc_raw.iloc[-390:] < 0).any())

    def yc_to_risk(yc_pct, post_inversion=False):
        if yc_pct is None:
            return None
        if yc_pct > 1.5:
            s = 10
        elif yc_pct > 0.5:
            s = 30
        elif yc_pct > -0.5:
            s = 65
        else:
            s = 85
        # Post-inversion re-steepening bonus: +15 pts if YC is positive but was
        # inverted within 18 months — historically the danger zone
        if post_inversion and 0 <= yc_pct < 1.0:
            s = min(100, s + 15)
        return s

    yc_score_val = yc_to_risk(yc_val, post_inversion=post_inv)

    goldman_score = None
    gc = scores_result.get("goldman_composite")
    if gc and gc.get("status") == "ok":
        goldman_score = gc["value"]

    ds_score = prank("debt_service")

    # Revised cycle weights per Spec Addition 1
    cycle_score = wavg([
        (lei_score,     0.30),
        (yc_score_val,  0.30),
        (goldman_score, 0.30),
        (ds_score,      0.10),
    ])

    # Co-firing bonus: LEI declining + YC in danger zone → +15 (7/8 historical hit rate)
    co_fire = bool(
        lei_score is not None and lei_score > 70 and
        yc_score_val is not None and yc_score_val > 55
    )
    if co_fire and cycle_score is not None:
        cycle_score = min(100.0, cycle_score + 15)

    # ── VALUATION BUCKET (weight 0.20) ────────────────────────────────────────
    # CAPE: map to 0-100 using empirical thresholds
    cape_score = None
    if cape_value is not None:
        if cape_value < 15:
            cape_score = max(0.0, cape_value / 15 * 20)
        elif cape_value < 25:
            cape_score = 20 + (cape_value - 15) / 10 * 30
        elif cape_value < 35:
            cape_score = 50 + (cape_value - 25) / 10 * 30
        else:
            cape_score = clamp(80 + (cape_value - 35) / 10 * 20)

    # Buffett z-score: deviation from trend -> risk
    buffett_score = None
    bi = breadth_result.get("buffett_indicator") or {}
    bz = bi.get("deviation_z")
    if bz is not None:
        buffett_score = clamp(50 + bz * 20)

    # ECY: lower ECY = riskier; map [-1, 5] -> [100, 0]
    ecy_score = None
    ecy = breadth_result.get("excess_cape_yield") or {}
    ev = ecy.get("value")
    if ev is not None:
        ecy_score = clamp(100 - (ev + 1) / 6 * 100)

    valuation_score = wavg([
        (cape_score,    0.40),
        (buffett_score, 0.35),
        (ecy_score,     0.25),
    ])

    # Valuation ceiling: above 70, additional stretching adds no new information.
    # Prevents permanently elevated CAPE from crowding out time-sensitive signals.
    valuation_capped = min(valuation_score, 70.0) if valuation_score is not None else None

    # ── BREADTH BUCKET (weight 0.15) ──────────────────────────────────────────
    breadth_200_score = None
    p200 = breadth_result.get("pct_above_200dma") or {}
    p200v = p200.get("value")
    if p200v is not None:
        breadth_200_score = clamp(100 - p200v)

    mcc_score = None
    mcc = breadth_result.get("mcclellan_oscillator") or {}
    mv = mcc.get("value")
    if mv is None:
        # also check key used in this codebase
        mcc2 = breadth_result.get("mcclellan") or {}
        mv = mcc2.get("value")
    if mv is not None:
        # Range roughly -150 to +150; center at 0 = 50
        mcc_score = clamp(50 - mv / 3)

    breadth_score = wavg([
        (breadth_200_score, 0.60),
        (mcc_score,         0.40),
    ])

    # ── LABOR BUCKET (weight 0.10) ────────────────────────────────────────────
    sahm_score = None
    sahm_raw = fred_raw.get("sahm_rule")
    if sahm_raw is not None:
        sv = float(sahm_raw.iloc[-1])
        sahm_score = clamp(sv / 0.5 * 100)

    unemp_score = None
    un = fred_raw.get("unemployment")
    if un is not None and len(un) > 12:
        current_u = float(un.iloc[-1])
        prior_u = float(un.iloc[-13])
        rising = current_u > prior_u + 0.3
        low_base = current_u < 4.5
        if rising and low_base:
            unemp_score = 70.0
        elif rising:
            unemp_score = 50.0
        else:
            unemp_score = 20.0

    cc_score = prank("cc_delinquency")

    labor_score = wavg([
        (sahm_score,  0.50),
        (unemp_score, 0.30),
        (cc_score,    0.20),
    ])

    # ── COMPOSITE ─────────────────────────────────────────────────────────────
    # Use valuation_capped (not raw) to prevent structural CAPE distorting timing
    composite = wavg([
        (credit_score,    0.30),
        (cycle_score,     0.25),
        (valuation_capped, 0.20),
        (breadth_score,   0.15),
        (labor_score,     0.10),
    ])

    def mk(val, **extra):
        return metric(
            round(val, 1) if val is not None else None,
            status="ok" if val is not None else "unavailable",
            source="derived", label=_risk_label(val), derived=True,
            **extra
        )

    return {
        "composite":       mk(composite,
                              notes="5-bucket weighted composite: credit(30%) cycle(25%) "
                                    "valuation(20%,capped70) breadth(15%) labor(10%)"),
        "credit_score":    mk(credit_score,
                              notes="Credit bucket: SLOOS(40%) HY momentum(35%) NFCI leverage(25%)"),
        "cycle_score":     mk(cycle_score,
                              notes="Cycle bucket (v2): LEI direction(30%) YC risk(30%) "
                                    "Goldman composite(30%) Debt service(10%)",
                              post_inversion=post_inv,
                              co_fire=co_fire,
                              yc_risk_score=round(yc_score_val, 1) if yc_score_val is not None else None),
        "valuation_score": mk(valuation_score,
                              notes="Valuation bucket: CAPE(40%) Buffett(35%) ECY(25%). "
                                    "Capped at 70 in composite calculation."),
        "breadth_score":   mk(breadth_score,
                              notes="Breadth bucket: % above 200DMA(60%) McClellan percentile(40%)"),
        "labor_score":     mk(labor_score,
                              notes="Labor bucket: Sahm Rule(50%) Unemployment trend(30%) "
                                    "CC delinquency percentile(20%)"),
        # Legacy alias for send_alerts.py backward compat
        "structural_score": mk(valuation_score,
                               notes="alias for valuation_score (backward compat)"),
        # Expose bucket weights for frontend contributor bar
        "bucket_weights": {
            "credit": 0.30, "cycle": 0.25, "valuation": 0.20,
            "breadth": 0.15, "labor": 0.10,
        },
    }


# ---------------------------------------------------------------------------
# Regime Condition Counter
# ---------------------------------------------------------------------------

def regime_conditions(fred_raw, result, history):
    """
    Count how many of the 4 historically-validated crash precursor conditions
    are currently active. 3+ conditions have preceded every endogenous 20%+
    drawdown in the post-1950 US market (excluding pure exogenous shocks).

    Returns dict with individual condition booleans, count, label, and context.
    """
    conditions = {}

    # CONDITION 1 — Credit stress
    # HY OAS widened >80bps in 90 days OR SLOOS net tightening >20%
    try:
        hy_mom = result.get("macro", {}).get("hy_momentum", {})
        hy_chg = hy_mom.get("change_90d_bps") or 0
        sloos_raw = fred_raw.get("lending_standards")
        sloos_v = float(sloos_raw.iloc[-1]) if sloos_raw is not None else 0
        conditions["credit_stress"] = bool(hy_chg > 80 or sloos_v > 20)
    except Exception:
        conditions["credit_stress"] = False

    # CONDITION 2 — Cycle breakdown
    # LEI declining for 3+ consecutive months
    try:
        lei = fred_raw.get("lei")
        if lei is not None and len(lei) >= 4:
            recent = [float(lei.iloc[i]) for i in [-4, -3, -2, -1]]
            conditions["cycle_breakdown"] = bool(
                recent[1] < recent[0] and recent[2] < recent[1] and recent[3] < recent[2]
            )
        else:
            conditions["cycle_breakdown"] = False
    except Exception:
        conditions["cycle_breakdown"] = False

    # CONDITION 3 — Yield curve (inverted OR re-steepening after inversion)
    try:
        yc = fred_raw.get("yield_curve_10y3m")
        yc_active = False
        if yc is not None and len(yc) >= 400:
            current_yc = float(yc.iloc[-1])
            if current_yc < 0:
                yc_active = True
            else:
                lookback = yc.iloc[-390:]
                was_inverted = bool((lookback < 0).any())
                # Re-steepening after inversion: positive but < 1.0 — the historically
                # dangerous normalization zone
                if was_inverted and 0 <= current_yc < 1.0:
                    yc_active = True
        elif yc is not None and len(yc) >= 2:
            # Shorter history: just check current
            current_yc = float(yc.iloc[-1])
            if current_yc < 0:
                yc_active = True
        conditions["yield_curve"] = yc_active
    except Exception:
        conditions["yield_curve"] = False

    # CONDITION 4 — Leverage stress
    # Margin debt YoY >40% AND trend now rolling over (falling)
    try:
        md = result.get("macro", {}).get("margin_debt", {})
        md_yoy = md.get("yoy_pct")
        md_trend = (md.get("trend") or {}).get("direction")
        conditions["leverage_stress"] = bool(
            md_yoy is not None and md_yoy > 40 and md_trend == "falling"
        )
    except Exception:
        conditions["leverage_stress"] = False

    # CONDITION 5 — Labor market deterioration
    # Sahm Rule >= 0.5 (exact empirical threshold from Sahm 2019)
    try:
        sahm = fred_raw.get("sahm_rule")
        if sahm is not None:
            conditions["labor"] = bool(float(sahm.iloc[-1]) >= 0.5)
        else:
            conditions["labor"] = False
    except Exception:
        conditions["labor"] = False

    # n counts only the four primary conditions surfaced in the UI
    # (yield_curve, credit_stress, cycle_breakdown, leverage_stress).
    # "labor" (Sahm Rule) remains computed and available in `conditions` for
    # other consumers, but is excluded from the headline count so
    # conditions_active never exceeds 4 — keeping it consistent with the
    # "N of 4 conditions active" framing used throughout the dashboard.
    n = sum(v for k, v in conditions.items() if k != "labor")

    if n == 0:
        regime_label = "No Signal"
        context = ("0 of 4 conditions active. Historically, endogenous bear markets "
                   "require 3+ conditions simultaneously. Invest normally.")
    elif n == 1:
        regime_label = "No Signal"
        context = ("1 of 4 conditions active. Below historical watch threshold. "
                   "Monitor — not yet actionable.")
    elif n == 2:
        regime_label = "Watch"
        context = ("2 of 4 conditions active. Risk is building. "
                   "Monitor remaining conditions closely.")
    elif n == 3:
        regime_label = "Caution"
        context = ("3 of 4 conditions active. This combination preceded the 2000 "
                   "dot-com crash, the 2008 GFC, and the 1973 bear market. "
                   "Consider reducing risk exposure.")
    else:
        regime_label = "High Alert"
        context = ("4 of 4 conditions active. Every endogenous US bear market of 20%+ "
                   "since 1950 occurred with this combination. Reduce risk exposure.")

    return {
        "conditions_active": n,
        "conditions": conditions,
        "label": regime_label,
        "context": context,
        "threshold_note": "3+ conditions historically precede 20%+ drawdowns "
                          "(excludes exogenous shocks: COVID, geopolitical events)",
        "derived": True,
    }


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


def yoy_pct_quarterly(s):
    """Year-over-year percent change for quarterly series (4 observations back)."""
    try:
        s = pd.Series(s).dropna()
        if len(s) < 5:
            return None
        return round((float(s.iloc[-1]) / float(s.iloc[-5]) - 1) * 100, 2)
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
            # Embed a small recent history for tiles that get an Overview sparkline
            if key == "umich_sentiment":
                try:
                    m["spark"] = [round(float(x), 1) for x in s.iloc[-24:].tolist()]
                except Exception:
                    pass
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
            # Margin debt: compute YoY% (quarterly series — 4 obs back = ~1 year)
            if key == "margin_debt":
                yoy_q = yoy_pct_quarterly(s)
                if yoy_q is not None:
                    m["yoy_pct"] = yoy_q
                    m["threshold_label"] = threshold_label("margin_debt_yoy", yoy_q)
            # Margin debt: 8-quarter YoY spark + rolling_over flag (additive to yoy_pct block above)
            if key == "margin_debt":
                try:
                    spark_yoy = []
                    md_list = s.dropna().tolist()
                    for i in range(4, len(md_list)):
                        try:
                            chg = round((md_list[i] / md_list[i - 4] - 1) * 100, 1)
                            spark_yoy.append(chg)
                        except Exception:
                            pass
                    m["spark_yoy"] = spark_yoy[-8:] if spark_yoy else []
                    yoy_val = m.get("yoy_pct")
                    md_trend = (m.get("trend") or {}).get("direction")
                    m["rolling_over"] = bool(
                        yoy_val is not None and yoy_val > 40 and md_trend == "falling"
                    )
                    m["above_threshold"] = bool(yoy_val is not None and yoy_val > 40)
                except Exception:
                    pass
            # Yield curve: 24-month monthly-resampled spark + post-inversion flag
            if key == "yield_curve_10y3m":
                try:
                    spark_monthly = s.resample("MS").last().iloc[-24:]
                    m["spark"] = [round(float(x), 3) for x in spark_monthly.tolist()]
                    m["any_negative_in_spark"] = bool((spark_monthly < 0).any())
                    current_yc = float(s.iloc[-1])
                    was_inverted = bool(len(s) >= 390 and (s.iloc[-390:] < 0).any())
                    m["post_inversion"] = bool(was_inverted and current_yc >= 0 and current_yc < 1.0)
                except Exception:
                    pass
            # SLOOS: 8-quarter spark + threshold metadata
            if key == "lending_standards":
                try:
                    m["spark"] = [round(float(x), 1) for x in s.iloc[-8:].tolist()]
                    m["above_threshold"] = bool(float(s.iloc[-1]) > 20)
                    m["loosening"] = bool(float(s.iloc[-1]) < 0)
                except Exception:
                    pass
            # LEI: 18-month spark + consecutive decline count on the metric envelope
            if key == "lei":
                try:
                    m["spark"] = [round(float(x), 2) for x in s.iloc[-18:].tolist()]
                    lei_list = s.dropna().tolist()
                    consecutive = 0
                    for i in range(len(lei_list) - 1, 0, -1):
                        if lei_list[i] < lei_list[i - 1]:
                            consecutive += 1
                        else:
                            break
                    m["consecutive_declines"] = consecutive
                    m["decline_alert"] = bool(consecutive >= 3)
                    m["decline_note"] = (
                        f"LEI declining {consecutive} consecutive months — recession precursor active"
                        if consecutive >= 3 else
                        f"LEI declining {consecutive} consecutive month(s) — threshold is 3"
                    )
                except Exception:
                    pass
            # NY Fed recession probability: 36-month spark + threshold metadata
            if key == "recession_prob":
                try:
                    m["spark"] = [round(float(x), 1) for x in s.iloc[-36:].tolist()]
                    m["threshold"] = 30.0
                    m["above_threshold"] = bool(float(s.iloc[-1]) >= 30.0)
                    m["threshold_note"] = (
                        "Above 30% has preceded every US recession since 1967 — zero false positives."
                    )
                except Exception:
                    pass
            # Apply threshold labels to key series
            if key in ("lending_standards", "sahm_rule", "yield_curve_10y3m"):
                thr_key = "sloos" if key == "lending_standards" else key
                m["threshold_label"] = threshold_label(thr_key, float(s.iloc[-1]))
            # HY OAS: threshold label + complacency flag
            if key == "hy_oas":
                hy_val = float(s.iloc[-1])
                m["threshold_label"] = threshold_label("hy_oas", hy_val)
                m["complacent"] = bool(hy_val < 3.5)
                m["complacency_note"] = (
                    "Spreads below 350bps indicate compressed risk premiums. "
                    "This is not 'safe' — it means risk is being mispriced. "
                    "The GFC peak (Oct 2007) saw HY OAS at only 270bps before spreads "
                    "exploded to 1,900bps within 14 months."
                    if hy_val < 3.5 else ""
                )
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
                asof=close.index[-1].strftime("%Y-%m-%d"),
                notes="% above/below 200-DMA. Negative = below."),
            "dma200_slope": metric(slope, source="yfinance:^GSPC",
                                   asof=close.index[-1].strftime("%Y-%m-%d"),
                                   notes="200-DMA direction over last 21 sessions"),
            "weekly_macd": metric(round(macd_line, 4), source="yfinance:^GSPC",
                                  asof=close.index[-1].strftime("%Y-%m-%d"),
                                  notes="Weekly MACD line vs signal",
                                  signal=round(signal_line, 4),
                                  histogram=round(hist, 4),
                                  bearish=bool(macd_line < signal_line)),
        }
    except Exception as e:
        return {"spx_price": unavailable(source="yfinance:^GSPC", error=e),
                "spx_vs_200dma": unavailable(source="yfinance:^GSPC", error=e),
                "dma200_slope": unavailable(source="yfinance:^GSPC", error=e),
                "weekly_macd": unavailable(source="yfinance:^GSPC", error=e)}


def pull_vix():
    """VIX from yfinance."""
    try:
        df = _yf_history("^VIX", period="1y", interval="1d")
        close = df["Close"].dropna()
        v = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        return metric(round(v, 2), source="yfinance:^VIX",
                      asof=close.index[-1].strftime("%Y-%m-%d"),
                      notes="CBOE Volatility Index",
                      ma20=round(ma20, 2),
                      above_40=bool(v > 40),
                      above_30=bool(v > 30))
    except Exception as e:
        return unavailable(source="yfinance:^VIX", error=e)


def pull_ad_proxy():
    """RSP/SPY ratio as an A-D line proxy."""
    try:
        import yfinance as yf
        data = yf.download(["RSP", "SPY"], period="2y", interval="1d",
                           auto_adjust=False, progress=False)["Close"].dropna()
        ratio = (data["RSP"] / data["SPY"]).dropna()
        latest = float(ratio.iloc[-1])
        ratio_21d = float(ratio.iloc[-22]) if len(ratio) >= 22 else float(ratio.iloc[0])
        ratio_chg = (latest - ratio_21d) / ratio_21d
        divergence = bool(ratio_chg < -0.02)
        return metric(round(latest, 4), source="yfinance:RSP/SPY",
                      asof=ratio.index[-1].strftime("%Y-%m-%d"),
                      notes="RSP/SPY ratio as breadth divergence proxy. Falling = megacap masking.",
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

            cape = None
            for ci in cape_candidates:
                v = last_in_range(ci, 3, 70)
                if v is not None:
                    cape = v
                    break
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
# ---------------------------------------------------------------------------

def sp500_constituents():
    """Best-effort S&P 500 ticker list."""
    ua = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
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
    return list(FALLBACK_SP500), "fallback_static"


def alpaca_daily_bars(symbols, api_key, api_secret, days=320):
    """Pull daily bars for a list of symbols from Alpaca."""
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
                        chunk = []
                    chunk_done = False
                    break
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
                break
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
        window = c.iloc[-252:]
        if c.iloc[-1] >= window.max():
            new_highs += 1
        if c.iloc[-1] <= window.min():
            new_lows += 1
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
    """Hindenburg/Titanic proxies computed on the S&P 500 universe.
    Raw flags only — not surfaced in alerts per spec (high false positive rate).
    Retained as breadth data for the Technical tab."""
    n = breadth.get("universe_counted") or 0
    nh, nl = breadth.get("new_highs", 0), breadth.get("new_lows", 0)
    adv, dec = breadth.get("advancers", 0), breadth.get("decliners", 0)
    signals = {}

    if n > 0:
        nh_pct = nh / n * 100
        nl_pct = nl / n * 100
        cond_counts    = nh_pct >= 2.2 and nl_pct >= 2.2 and min(nh, nl) > 0
        cond_ratio     = max(nh_pct, nl_pct) < 2.8 * min(nh_pct, nl_pct) + 5
        cond_uptrend   = (spx_50d_ret is None) or (spx_50d_ret > 0)
        cond_mcclellan = (mcclellan_val is None) or (mcclellan_val < 0)
        hindenburg_raw = bool(cond_counts and cond_ratio and cond_uptrend and cond_mcclellan)
        signals["hindenburg_omen_today"] = metric(
            hindenburg_raw, status="proxy", source="Alpaca S&P500 internals",
            notes="PROXY on S&P500 (true indicator is NYSE-wide). RAW same-day flag — "
                  "retained as breadth data only. High false positive rate; removed from alerts.",
            new_high_pct=round(nh_pct, 2), new_low_pct=round(nl_pct, 2),
            cond_counts=bool(cond_counts), cond_ratio=bool(cond_ratio),
            cond_uptrend=bool(cond_uptrend), cond_mcclellan=bool(cond_mcclellan))
        signals["titanic_syndrome_today"] = metric(
            bool(nl > nh and nh > 0), status="proxy",
            source="Alpaca S&P500 internals",
            notes="PROXY on S&P500. RAW same-day flag. "
                  "Retained as breadth data only. High false positive rate; removed from alerts.",
            new_highs=nh, new_lows=nl)
    if (adv + dec) > 0:
        zweig_ratio = adv / (adv + dec)
        signals["zweig_adv_ratio"] = metric(
            round(zweig_ratio, 3), status="proxy", source="Alpaca S&P500 internals",
            notes="Advancers/(Adv+Dec). Zweig Breadth Thrust = 10-day avg moving "
                  "<0.40 to >0.615 within 10 days; needs multi-day history (alert script).")
    return signals


def mcclellan_oscillator(prev_state, breadth):
    """McClellan Oscillator — seeds from state, accumulates across daily runs."""
    adv, dec = breadth.get("advancers", 0), breadth.get("decliners", 0)
    if (adv + dec) == 0:
        return unavailable(source="Alpaca S&P500 internals",
                           notes="no advance/decline data this run")
    rana = (adv - dec) / (adv + dec) * 1000
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
    """SPX % change vs ~50 trading sessions ago."""
    try:
        close = _yf_history("^GSPC", period="6mo", interval="1d")["Close"].dropna()
        if len(close) < 51:
            return None
        return float(close.iloc[-1] / close.iloc[-51] - 1) * 100
    except Exception:
        return None


def confirm_clusters(history, window_days=30, min_count=2):
    """Promote RAW daily breadth flags to CONFIRMED signals using rolling history.
    Returns dict of confirmed metrics to merge into result['breadth']. These are
    marked derived=True so they are NOT double-counted in source-health/confidence.
    NOTE: confirmed flags are NOT surfaced in Active Alerts per spec (high FPR).
    They remain available in the Technical tab as breadth data."""
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
                  "Retained as breadth data only — NOT an active alert (high false positive rate).",
            count=len(hind_dates), window_days=window_days, dates=hind_dates),
        "titanic_syndrome_confirmed": metric(
            bool(len(tit_dates) >= min_count), status="proxy",
            source="Alpaca S&P500 internals (cluster)", derived=True,
            notes=f"CONFIRMED when {min_count}+ raw flags occur within {window_days} days. "
                  "Retained as breadth data only — NOT an active alert (high false positive rate).",
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
        out["mcclellan"] = mcclellan_oscillator(prev_state, breadth)
        mc_state = out["mcclellan"].pop("_state", None) if isinstance(out["mcclellan"], dict) else None
        mc_val = out["mcclellan"].get("value") if isinstance(out["mcclellan"], dict) else None
        spx_50d_ret = _spx_50d_return()
        out.update(breadth_signals(breadth, mcclellan_val=mc_val, spx_50d_ret=spx_50d_ret))
        return out, {"mcclellan": mc_state} if mc_state else None
    except Exception as e:
        fv = finviz_pct_above_200()
        return {"breadth_error": unavailable(source="Alpaca", error=e),
                "pct_above_200dma": fv}, None


def finviz_pct_above_200():
    """Fallback: scrape Finviz group page for % of S&P stocks above 200-DMA."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (market-health-dashboard)"}
        r = requests.get("https://finviz.com/api/counts.ashx?t=sma200",
                         headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
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
    last = list(_csv.reader([data_rows[-1]]))[0]
    date_str = last[0].strip()
    ratio = float(last[-1].strip())
    if not (0.1 <= ratio <= 5.0):
        raise ValueError(f"Implausible ratio: {ratio}")
    return date_str, ratio


def pull_put_call():
    """Fetch CBOE put/call ratio. Tries three endpoints in order."""
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
                                ">1.2=fear (contrarian opportunity).",
                          asof=date_str,
                          euphoria=bool(val < 0.5),
                          fear=bool(val > 1.2))
        except Exception as e:
            last_err = e
            print(f"  put/call {flavor} failed: {e}", flush=True)
            continue
    return unavailable(source="CBOE", error=last_err,
                       notes="All three CBOE put/call endpoints failed.")


# ---------------------------------------------------------------------------
# CNN Fear & Greed
# ---------------------------------------------------------------------------

def compute_fear_greed(result):
    """Fetch CNN's Fear & Greed Index. Falls back to z-score approximation."""
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
        label = rating.replace("_", " ").title() if rating else (
            "Extreme Fear" if score < 25 else
            "Fear"         if score < 45 else
            "Neutral"      if score < 55 else
            "Greed"        if score < 75 else
            "Extreme Greed")
        spark = []
        try:
            hist_data = (j.get("fear_and_greed_historical") or {}).get("data") or []
            for pt in hist_data[-40:]:
                yv = pt.get("y")
                if yv is not None:
                    spark.append(round(float(yv), 1))
        except Exception:
            spark = []
        print(f"  CNN Fear & Greed: {score} ({label}), {len(spark)} history pts", flush=True)
        return metric(score, source="CNN (production.dataviz.cnn.io)",
                      notes="CNN's official Fear & Greed Index. "
                            "0=Extreme Fear, 100=Extreme Greed.",
                      label=label, timestamp=str(ts), spark=spark)
    except Exception as e:
        print(f"  CNN Fear & Greed fetch failed ({e}); falling back to z-score", flush=True)

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
                  notes="CNN API unavailable. Z-score approximation from local inputs.",
                  label=label, components=scores, n_inputs=len(scores))


# ---------------------------------------------------------------------------
# Valuation derived metrics
# ---------------------------------------------------------------------------

def buffett_indicator(raw):
    try:
        w = raw.get("equity_mktcap")
        if w is None and FRED_API_KEY:
            try:
                w = fred_series("NCBEILQ027S", FRED_API_KEY)
            except Exception:
                w = None
        g = raw.get("gdp")
        if w is None or g is None:
            return unavailable(source="FRED:NCBEILQ027S/GDP", error="inputs missing")
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
                      notes="Market value of US equities / GDP (Buffett indicator).",
                      deviation_z=round(float(z), 2),
                      elevated=bool(z > 1.0))
    except Exception as e:
        return unavailable(source="FRED:NCBEILQ027S/GDP", error=e)


def excess_cape_yield(cape_value, raw, prefilled=None):
    try:
        if prefilled is not None:
            return metric(round(float(prefilled), 2), source="Shiller ie_data.xls",
                          notes="Excess CAPE Yield: CAPE earnings yield minus real bond yield.",
                          low=bool(prefilled < 1.0))
        if cape_value is None:
            return unavailable(source="Shiller+FRED:DFII10", error="CAPE unavailable")
        real10 = raw.get("real_10y")
        if real10 is None:
            return unavailable(source="FRED:DFII10", error="real yield unavailable")
        ecy = (1.0 / cape_value) * 100 - float(real10.iloc[-1])
        return metric(round(ecy, 2), source="Shiller + FRED:DFII10 (computed)",
                      notes="Excess CAPE Yield = CAPE earnings yield minus real 10yr.",
                      low=bool(ecy < 1.0))
    except Exception as e:
        return unavailable(source="Shiller+FRED:DFII10", error=e)


def goldman_composite(raw, cape_value):
    """Percentile-rank five inputs and average -> 0-100 bear-risk-ish score."""
    parts = {}
    try:
        if cape_value is not None:
            parts["valuation"] = clamp(percentile_rank(cape_value,
                                       list(np.linspace(5, 44, 200))) or 50)
        yc = raw.get("yield_curve_10y3m")
        if yc is not None:
            p = percentile_rank(float(yc.iloc[-1]), yc.values)
            if p is not None:
                parts["yield_curve"] = clamp(100 - p)
        un = raw.get("unemployment")
        if un is not None:
            p = percentile_rank(float(un.iloc[-1]), un.values)
            if p is not None:
                parts["unemployment"] = clamp(100 - p)
        cpi = raw.get("core_cpi")
        if cpi is not None and len(cpi) > 13:
            yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100
            parts["core_inflation"] = clamp(percentile_rank(
                yoy, ((cpi.pct_change(12) * 100).dropna().values)) or 50)
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
        return unavailable(source="derived: LEI + yield curve", error=e)


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

        spy_df = yf.download("SPY", period="2y", interval="1d",
                             auto_adjust=True, progress=False)
        if spy_df is None or spy_df.empty:
            return {"status": "unavailable", "error": "SPY benchmark data unavailable"}
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.droplevel(1)
        spy = spy_df["Close"].dropna() if "Close" in spy_df.columns else spy_df.iloc[:, 0].dropna()

        holdings = []
        skipped = []
        for ticker, shares in positions:
            try:
                df = yf.download(ticker, period="2y", interval="1d",
                                 auto_adjust=True, progress=False)
                if df is None or df.empty or len(df) < 30:
                    skipped.append(ticker)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                px = df["Close"].dropna() if "Close" in df.columns else df.iloc[:, 0].dropna()
                if len(px) < 30:
                    skipped.append(ticker)
                    continue
                price = float(px.iloc[-1])
                value = price * shares
                ret = px.pct_change().dropna()
                spy_ret = spy.pct_change().dropna()
                r, s = ret.align(spy_ret, join="inner")
                if len(r) > 60:
                    cov = float(r.cov(s))
                    var = float(s.var())
                    beta = round(cov / var, 2) if var > 0 else 1.0
                else:
                    beta = 1.0
                rf_daily = 0.0433 / 252
                excess = ret - rf_daily
                sharpe = round(float(excess.mean() / excess.std() * (252 ** 0.5)), 2) if excess.std() > 0 else 0.0
                ret_1y = round(float(px.iloc[-1] / px.iloc[max(-252, -len(px))] - 1) * 100, 1) if len(px) >= 21 else None
                holdings.append({
                    "ticker": ticker, "shares": shares, "price": round(price, 2),
                    "value": round(value, 2), "beta": beta,
                    "sharpe": sharpe, "ret_1y": ret_1y,
                })
                time.sleep(0.1)
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
    """Generate dynamic per-metric readings + a top-level summary."""
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

    # ECY
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

    # HY OAS
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
            base = f"{un:.1f}% and rising — a turn up from lows is a late-cycle warning"
        elif unt == "falling":
            base = f"{un:.1f}% and falling — labor market still firm"
        else:
            base = f"{un:.1f}%, holding steady"
        R["readings"]["unemployment"] = base + "."

    # Inflation
    for key, label in (("core_pce", "Core PCE"), ("core_cpi", "Core CPI")):
        yoy = mval("macro", key, "yoy_pct")
        ydir = mval("macro", key, "yoy_direction")
        if yoy is not None:
            vs = "above" if yoy > 2.2 else "near" if yoy > 1.8 else "below"
            base = f"{label} running {yoy:.1f}% YoY, {vs} the Fed's 2% target"
            if ydir:
                base += f" and {ydir}"
            R["readings"][key] = base + "."

    # Fed funds
    ff = mval("macro", "fed_funds")
    fft = trend_word("macro", "fed_funds")
    if ff is not None:
        stance = "cutting" if fft == "falling" else "hiking" if fft == "rising" else "on hold"
        R["readings"]["fed_funds"] = (f"Policy rate {ff:.2f}%, currently {stance}. "
                                      "Rate-cut cycles have preceded recent recessions.")

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
                base += " and rising — early sign of consumer stress"
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
            tail = "high — stress/fear regime"
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

    # Summary
    score = mval("scores", "composite")
    label = mval("scores", "composite", "label") or "Unknown"
    s_parts = []
    if score is not None:
        s_parts.append(f"{label} ({score:.0f}/100).")

    val_flags = []
    if cape_pct is not None and cape_pct >= 85:
        val_flags.append("CAPE near historic highs")
    if bz is not None and bz >= 1.5:
        val_flags.append("the Buffett indicator at a record")
    if ecy is not None and ecy < 1:
        val_flags.append("a thin equity-risk premium")
    if val_flags:
        s_parts.append("Valuations are stretched — " + ", ".join(val_flags) + ".")

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

    if pa is not None:
        if pa < 40:
            s_parts.append("Breadth has broken down — distribution beneath the surface.")
        elif ad_div:
            s_parts.append("Breadth is narrowing, a yellow flag.")
        else:
            s_parts.append("Breadth is healthy, so no regime break yet.")

    R["summary"] = " ".join(s_parts)

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
            "schema_version": 3,
            "notes": "Each metric carries status/source. status 'proxy' = computed "
                     "stand-in for a proprietary/unavailable series (see notes).",
        },
        "macro": {}, "structural": {}, "trend": {}, "breadth": {},
        "sentiment": {}, "sectors": {}, "catalysts": {}, "scores": {},
        "source_health": {}, "regime": {},
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
    result["sentiment"]["fear_greed"] = compute_fear_greed(result)

    # ---- Sectors ----
    result["sectors"]["relative_strength"] = pull_sectors()
    result["sectors"]["cycle_phase"] = cycle_phase(fred_raw)

    # ---- HY OAS 90d momentum derived metric ----
    hy_mom_score, hy_mom_bps = hy_momentum_score(fred_raw)
    result["macro"]["hy_momentum"] = metric(
        hy_mom_score,
        status="ok" if hy_mom_score is not None else "unavailable",
        source="derived:FRED:BAMLH0A0HYM2",
        notes="HY OAS 90-day rate-of-change converted to 0-100 risk score. "
              ">75 = spreads widening fast = credit stress signal.",
        change_90d_bps=hy_mom_bps,
        derived=True,
    )
    # Apply threshold label to hy_momentum bps
    if hy_mom_bps is not None:
        result["macro"]["hy_momentum"]["threshold_label"] = threshold_label(
            "hy_momentum_bps", hy_mom_bps)

    # ---- Goldman composite (needed before composite_risk) ----
    result["scores"]["goldman_composite"] = goldman_composite(fred_raw, cape_value)

    # ---- Build scores_result stub for composite_risk hy_momentum_score input ----
    # composite_risk() reads scores_result.get("hy_momentum_score") and
    # scores_result.get("goldman_composite")
    _scores_for_composite = {
        "hy_momentum_score": hy_mom_score,
        "goldman_composite": result["scores"]["goldman_composite"],
    }

    # ---- Composite risk score (new signature: fred_raw, structural, scores, cape) ----
    result["scores"].update(
        composite_risk(fred_raw, result["structural"], _scores_for_composite, cape_value)
    )

    # ---- Regime Condition Counter ----
    try:
        result["regime"] = regime_conditions(
            fred_raw, result, new_state.get("history", []))
    except Exception as e:
        result["regime"] = {
            "conditions_active": 0,
            "conditions": {},
            "label": "Unknown",
            "context": "regime counter error",
            "threshold_note": "",
            "error": str(e)[:200],
            "derived": True,
        }

    # ---- LEI Sustained Decline Alert (Spec Addition 2) ----
    try:
        lei_raw_sa = fred_raw.get("lei")
        if lei_raw_sa is not None and len(lei_raw_sa) >= 7:
            recent_lei = [float(lei_raw_sa.iloc[i]) for i in [-7, -6, -5, -4, -3, -2, -1]]
            consecutive_declines = 0
            for i in range(len(recent_lei) - 1, 0, -1):
                if recent_lei[i] < recent_lei[i - 1]:
                    consecutive_declines += 1
                else:
                    break
            result["macro"]["lei_decline_streak"] = {
                "value": consecutive_declines,
                "status": "ok",
                "source": "derived:FRED:USALOLITOAASTSAM",
                "alert": consecutive_declines >= 3,
                "notes": f"LEI has declined for {consecutive_declines} consecutive months. "
                         "3+ months preceded every post-1950 US recession.",
                "derived": True,
            }
        else:
            result["macro"]["lei_decline_streak"] = {
                "value": None, "status": "unavailable",
                "source": "derived:FRED:USALOLITOAASTSAM",
                "alert": False, "notes": "Insufficient LEI history", "derived": True,
            }
    except Exception as e:
        result["macro"]["lei_decline_streak"] = {
            "value": None, "status": "unavailable",
            "source": "derived:FRED:USALOLITOAASTSAM",
            "alert": False, "notes": str(e)[:200], "derived": True,
        }

    # ---- Catalysts ----
    result["catalysts"]["upcoming"] = upcoming_catalysts(30)

    # ---- Interpretation ----
    try:
        result["interpretation"] = interpret(result, fred_raw)
    except Exception as e:
        result["interpretation"] = {"summary": "", "watch": [], "readings": {},
                                    "error": str(e)[:200]}

    # ---- History log ----
    try:
        hist = new_state.get("history", [])
        today = utc_iso[:10]
        snap = {
            "date": today,
            "composite": mval_path(result, "scores", "composite", "value"),
            "structural": mval_path(result, "scores", "valuation_score", "value"),
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
            "fear_greed": mval_path(result, "sentiment", "fear_greed", "value"),
            "umich": mval_path(result, "macro", "umich_sentiment", "value"),
            # Spec history additions
            "sloos":            mval_path(result, "macro", "lending_standards", "value"),
            "margin_debt_yoy":  (result.get("macro", {}).get("margin_debt") or {}).get("yoy_pct"),
            "hy_momentum_bps":  (result.get("macro", {}).get("hy_momentum") or {}).get("change_90d_bps"),
            "hy_mom_score":     mval_path(result, "macro", "hy_momentum", "value"),
            "regime_count":     (result.get("regime") or {}).get("conditions_active"),
            "credit_score":     mval_path(result, "scores", "credit_score", "value"),
            "cycle_score":      mval_path(result, "scores", "cycle_score", "value"),
            "valuation_score":  mval_path(result, "scores", "valuation_score", "value"),
            "breadth_score":    mval_path(result, "scores", "breadth_score", "value"),
            "labor_score":      mval_path(result, "scores", "labor_score", "value"),
            "lei_streak":       (result.get("macro", {}).get("lei_decline_streak") or {}).get("value"),
            "nyfed_prob":       mval_path(result, "macro", "recession_prob", "value"),
        }
        hist = [h for h in hist if h.get("date") != today]
        hist.append(snap)
        new_state["history"] = hist[-460:]
        result["history"] = new_state["history"][-90:]
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
                if not node.get("derived"):
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

    # ---- Threshold label ----
    assert threshold_label("cape", 40) == "High", f"threshold cape 40: {threshold_label('cape', 40)}"
    assert threshold_label("cape", 20) == "Moderate", f"threshold cape 20: {threshold_label('cape', 20)}"
    assert threshold_label("hy_oas", 2.6) == "Complacent", f"threshold hy_oas 2.6: {threshold_label('hy_oas', 2.6)}"
    assert threshold_label("sahm_rule", 0.55) == "Elevated", f"threshold sahm 0.55: {threshold_label('sahm_rule', 0.55)}"
    print("  threshold_label checks ✓")

    # ---- HY momentum ----
    score_none, bps_none = hy_momentum_score({})  # empty fred_raw
    assert score_none is None and bps_none is None, "hy_momentum_score empty dict should return None"
    short_series = pd.Series(list(range(10)), index=pd.date_range("2024-01-01", periods=10))
    s_none, b_none = hy_momentum_score({"hy_oas": short_series})
    assert s_none is None and b_none is None, "hy_momentum_score short series should return None"
    long_series = pd.Series(
        [3.5] * 65 + [4.5],
        index=pd.date_range("2023-01-01", periods=66, freq="B")
    )
    s_ok, b_ok = hy_momentum_score({"hy_oas": long_series})
    assert s_ok is not None, "hy_momentum_score long series returned None"
    assert abs(b_ok - 100.0) < 0.5, f"hy_momentum_score bps wrong: {b_ok}"
    print(f"  hy_momentum_score ✓ score={s_ok} bps={b_ok}")

    # ---- Regime counter (all-None input -> 0 conditions) ----
    rc = regime_conditions({}, {}, [])
    assert rc["conditions_active"] == 0, f"empty regime should be 0, got {rc['conditions_active']}"
    assert rc["label"] == "No Signal", f"empty regime label wrong: {rc['label']}"
    print(f"  regime_conditions empty ✓ label={rc['label']}")

    # ---- Composite (all-None input -> unavailable) ----
    cr = composite_risk({}, {}, {}, cape_value=None)
    assert cr["composite"]["status"] == "unavailable", \
        f"all-None composite should be unavailable, got {cr['composite']['status']}"
    print("  composite_risk all-None ✓")

    # Composite with some inputs
    _mock_fred = {
        "lending_standards": pd.Series([15.0], index=pd.date_range("2024-01-01", periods=1)),
        "nfci_leverage": pd.Series([0.1, 0.2, 0.3, 0.2, 0.1], index=pd.date_range("2020-01-01", periods=5, freq="MS")),
        "debt_service": pd.Series([10.0, 10.5, 11.0, 11.2, 11.5], index=pd.date_range("2020-01-01", periods=5, freq="MS")),
        "cc_delinquency": pd.Series([2.0, 2.1, 2.2, 2.3, 2.4], index=pd.date_range("2020-01-01", periods=5, freq="MS")),
        "unemployment": pd.Series([4.0]*14, index=pd.date_range("2023-01-01", periods=14, freq="MS")),
        "sahm_rule": pd.Series([0.2], index=pd.date_range("2024-01-01", periods=1)),
    }
    _mock_structural = {
        "buffett_indicator": {"value": 1.5, "status": "ok", "deviation_z": 1.5},
        "excess_cape_yield": {"value": 0.5, "status": "ok"},
        "pct_above_200dma": {"value": 55.0, "status": "ok"},
    }
    _mock_scores = {
        "hy_momentum_score": 55.0,
        "goldman_composite": {"value": 60.0, "status": "ok"},
    }
    cr2 = composite_risk(_mock_fred, _mock_structural, _mock_scores, cape_value=32.0)
    assert cr2["composite"]["value"] is not None, "composite with inputs returned None"
    assert 0 <= cr2["composite"]["value"] <= 100, f"composite out of range: {cr2['composite']['value']}"
    assert "co_fire" in cr2["cycle_score"], "co_fire missing from cycle_score"
    assert "post_inversion" in cr2["cycle_score"], "post_inversion missing from cycle_score"
    print(f"  composite_risk with inputs ✓ {cr2['composite']['value']} label={cr2['composite']['label']}")

    # ---- breadth signals ----
    bs = breadth_signals({"universe_counted": 500, "new_highs": 20, "new_lows": 18,
                          "advancers": 250, "decliners": 240})
    assert "hindenburg_omen_today" in bs, "breadth signals failed"

    # ---- LEI consecutive declines ----
    _lei_mock = pd.Series(
        [100.5, 100.3, 100.1, 99.9, 99.7, 99.5, 99.3],
        index=pd.date_range("2024-01-01", periods=7, freq="MS")
    )
    consecutive = 0
    lei_list = _lei_mock.tolist()
    for i in range(len(lei_list) - 1, 0, -1):
        if lei_list[i] < lei_list[i - 1]:
            consecutive += 1
        else:
            break
    assert consecutive == 6, f"LEI consecutive declines wrong: {consecutive}"
    print(f"  LEI consecutive declines ✓ streak={consecutive}")

    # ---- mcclellan seeding ----
    mc = mcclellan_oscillator({}, {"advancers": 300, "decliners": 200})
    assert mc["status"] == "proxy", "mcclellan failed"

    # ---- catalysts smoke ----
    assert upcoming_catalysts(3650), "catalyst calendar empty"

    print(f"  MACD: {round(m, 3)}  RSI: {round(r, 1)}  slope: {lbl}")
    print("ALL SELF-TESTS PASSED ✅")


# Minimal static S&P500 fallback
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
