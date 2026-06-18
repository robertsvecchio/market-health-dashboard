#!/usr/bin/env python3
"""
backfill_breadth_history.py — one-off backfill for the cluster-confirmation fix.
============================================================================

The Hindenburg/Titanic cluster logic (in fetch_data.py) needs ~30 days of daily
RAW breadth flags in data/state.json["history"] before it can confirm anything.
A fresh dashboard only has a few days. This script reconstructs the missing daily
flags by pulling one large constituent panel from Alpaca and recomputing breadth
AS OF each of the last N trading days (look-ahead-free), then merges the synthesized
hindenburg_raw/titanic_raw into the existing history.

Run ONCE, locally or as a manual GitHub Actions step, with the same secrets the
daily workflow uses:

    ALPACA_KEY=... ALPACA_SECRET=... python scripts/backfill_breadth_history.py
    # add --dry-run to print without writing state.json

It is idempotent: re-running recomputes the same days and replaces them in place.
It does NOT fabricate the McClellan/uptrend same-day conditions for past days where
that state is unavailable — it reconstructs the breadth-count conditions (the core
of the signal) and is explicit in the notes that backfilled flags are count-based.
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone

# Reuse the production functions so behavior matches the live pipeline exactly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import (  # noqa: E402
    sp500_constituents, alpaca_daily_bars, load_state, save_state,
    _yf_history,
)

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
HIST_PATH = os.environ.get("STATE_PATH", "data/state.json")


def breadth_asof(panel, asof_ts, hi_lo_win=252, ma_win=200):
    """Recompute breadth internals using ONLY bars up to asof_ts (no look-ahead)."""
    nh = nl = above = counted = 0
    for sym, df in panel.items():
        s = df["c"].dropna()
        s = s[s.index <= asof_ts]
        if len(s) < ma_win:
            continue
        counted += 1
        ma = s.iloc[-ma_win:].mean()
        if s.iloc[-1] > ma:
            above += 1
        win = s.iloc[-hi_lo_win:]
        if s.iloc[-1] >= win.max():
            nh += 1
        if s.iloc[-1] <= win.min():
            nl += 1
    if not counted:
        return None
    nh_pct = nh / counted * 100
    nl_pct = nl / counted * 100
    # Same count-based conditions as the live raw flag (conditions 1 & 2).
    hind_raw = bool(nh_pct >= 2.2 and nl_pct >= 2.2 and min(nh, nl) > 0
                    and max(nh_pct, nl_pct) < 2.8 * min(nh_pct, nl_pct) + 5)
    tit_raw = bool(nl > nh and nh > 0)
    return {
        "date": asof_ts.date().isoformat(),
        "pct_above_200": round(above / counted * 100, 1),
        "new_high_pct": round(nh_pct, 2),
        "new_low_pct": round(nl_pct, 2),
        "hindenburg_raw": hind_raw,
        "titanic_raw": tit_raw,
        "backfilled": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45,
                    help="trading days to reconstruct (default 45 → covers the 30d window)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (ALPACA_KEY and ALPACA_SECRET):
        print("ERROR: ALPACA_KEY/ALPACA_SECRET not set", file=sys.stderr)
        sys.exit(1)

    syms, src = sp500_constituents()
    print(f"Constituents: {len(syms)} ({src})")
    # 320 trading days gives a full 252-day window even for the oldest backfilled day.
    bars = alpaca_daily_bars(syms, ALPACA_KEY, ALPACA_SECRET, days=320)
    print(f"Panel pulled: {len(bars)} symbols")
    if len(bars) < 100:
        print("ERROR: too few symbols returned; aborting", file=sys.stderr)
        sys.exit(1)

    # Trading calendar = SPX sessions (reliable, avoids per-symbol gaps).
    spx = _yf_history("^GSPC", period="6mo", interval="1d")["Close"].dropna()
    sessions = list(spx.index[-args.days:])

    rebuilt = []
    for ts in sessions:
        row = breadth_asof(bars, ts)
        if row:
            rebuilt.append(row)
    print(f"Reconstructed {len(rebuilt)} sessions "
          f"({rebuilt[0]['date']} → {rebuilt[-1]['date']})")
    hind = sum(r["hindenburg_raw"] for r in rebuilt)
    tit = sum(r["titanic_raw"] for r in rebuilt)
    print(f"  Hindenburg raw flags in window: {hind}")
    print(f"  Titanic raw flags in window:    {tit}")

    # Merge into existing history: update matching dates, insert missing ones.
    state = load_state(HIST_PATH)
    hist = state.get("history", [])
    by_date = {h["date"]: h for h in hist}
    for r in rebuilt:
        cur = by_date.get(r["date"], {"date": r["date"]})
        # Only fill the breadth fields; never overwrite live score fields.
        for k in ("pct_above_200", "new_high_pct", "new_low_pct",
                  "hindenburg_raw", "titanic_raw", "backfilled"):
            cur.setdefault(k, r[k]) if k in cur else cur.update({k: r[k]})
        by_date[r["date"]] = cur
    merged = sorted(by_date.values(), key=lambda h: h["date"])[-460:]
    state["history"] = merged

    if args.dry_run:
        print("\n--dry-run: not writing. Last 5 reconstructed rows:")
        for r in rebuilt[-5:]:
            print(" ", r)
        return
    save_state(state, HIST_PATH)
    print(f"\nWrote {HIST_PATH} — history now {len(merged)} entries. "
          "Cluster confirmation will evaluate immediately on next dashboard run.")


if __name__ == "__main__":
    main()
