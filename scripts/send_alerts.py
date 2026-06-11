#!/usr/bin/env python3
"""
send_alerts.py — Market Health Dashboard email notifications
============================================================

Runs after fetch_data.py in the GitHub Actions workflow. Reads data/latest.json,
compares against the prior run's state (stored in data/state.json, which is
committed by CI so it persists day to day), then sends ONE consolidated email:

  - A daily digest: composite score, sub-scores, top sector signals, upcoming
    catalysts, and all currently-active alerts.
  - Threshold crossings since the last run are called out at the top (e.g. the
    yield curve just inverted, Goldman just crossed 70, VIX just crossed 30,
    breadth just dropped below 40%, the composite risk band changed).

Design:
  - Never fatal. Missing keys/secrets -> print and exit 0 so the workflow stays green.
  - State is merged into data/state.json so it coexists with fetch_data's state
    (McClellan EMAs etc.).

Secrets (env, set as GitHub Actions secrets):
  RESEND_API_KEY   Resend API key
  ALERT_EMAIL      recipient (your Gmail)

Usage:
  python send_alerts.py                # normal: build + send
  python send_alerts.py --dry-run      # build + print, do not send
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

LATEST_PATH = os.environ.get("LATEST_PATH", "data/latest.json")
STATE_PATH = os.environ.get("STATE_PATH", "data/state.json")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "")
FROM_ADDR = os.environ.get("ALERT_FROM", "Market Health <onboarding@resend.dev>")

# Risk band colors (inline styles for email clients)
BAND_COLOR = {"Low Risk": "#3FB950", "Moderate Risk": "#D8A657",
              "Elevated Risk": "#E8833A", "High Risk": "#F85149",
              "Unknown": "#8B949E"}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"WARN: could not save state: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Pure logic (testable offline)
# ---------------------------------------------------------------------------

def _g(d, *path, default=None):
    """Safe nested getter."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def snapshot(data):
    """Extract the small set of values we track for threshold crossings."""
    return {
        "composite": _g(data, "scores", "composite", "value"),
        "label": _g(data, "scores", "composite", "label"),
        "goldman": _g(data, "scores", "goldman_composite", "value"),
        "yield_curve": _g(data, "macro", "yield_curve_10y3m", "value"),
        "vix": _g(data, "sentiment", "vix", "value"),
        "pct_above_200": _g(data, "breadth", "pct_above_200dma", "value"),
        "hindenburg": _g(data, "breadth", "hindenburg_omen_today", "value"),
        "titanic": _g(data, "breadth", "titanic_syndrome_today", "value"),
    }


def crossed(prev, now, level, rising=True):
    """True if value crossed `level` between prev and now in the given direction."""
    if prev is None or now is None:
        return False
    return (prev <= level < now) if rising else (prev >= level > now)


def detect_crossings(prev, cur):
    """Compare prior snapshot to current; return list of {txt, pri} crossings."""
    out = []
    if not prev:  # first run — no crossings, just a baseline digest
        return out

    # Composite risk band change
    if prev.get("label") and cur.get("label") and prev["label"] != cur["label"]:
        worse = ["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]
        try:
            up = worse.index(cur["label"]) > worse.index(prev["label"])
        except ValueError:
            up = True
        out.append({"txt": f"Composite risk moved {prev['label']} → {cur['label']}",
                    "pri": "HIGH" if up else "MED"})

    # Goldman composite
    if crossed(prev.get("goldman"), cur.get("goldman"), 70, True):
        out.append({"txt": f"Goldman bear-risk composite crossed above 70 ({cur['goldman']:.0f})", "pri": "HIGH"})
    elif crossed(prev.get("goldman"), cur.get("goldman"), 50, True):
        out.append({"txt": f"Goldman composite crossed above 50 ({cur['goldman']:.0f})", "pri": "MED"})
    elif crossed(prev.get("goldman"), cur.get("goldman"), 50, False):
        out.append({"txt": f"Goldman composite fell back below 50 ({cur['goldman']:.0f})", "pri": "MED"})

    # Yield curve inversion / normalization
    if crossed(prev.get("yield_curve"), cur.get("yield_curve"), 0, False):
        out.append({"txt": f"Yield curve INVERTED (10y–3m now {cur['yield_curve']:.2f})", "pri": "HIGH"})
    elif crossed(prev.get("yield_curve"), cur.get("yield_curve"), 0, True):
        out.append({"txt": f"Yield curve normalized (10y–3m now {cur['yield_curve']:.2f})", "pri": "MED"})

    # VIX
    if crossed(prev.get("vix"), cur.get("vix"), 40, True):
        out.append({"txt": f"VIX crossed above 40 ({cur['vix']:.1f}) — stress", "pri": "HIGH"})
    elif crossed(prev.get("vix"), cur.get("vix"), 30, True):
        out.append({"txt": f"VIX crossed above 30 ({cur['vix']:.1f})", "pri": "MED"})

    # Breadth
    if crossed(prev.get("pct_above_200"), cur.get("pct_above_200"), 40, False):
        out.append({"txt": f"% of S&P 500 above 200-DMA dropped below 40% ({cur['pct_above_200']:.0f}%)", "pri": "HIGH"})

    # Breadth alert flags newly tripped
    if cur.get("hindenburg") and not prev.get("hindenburg"):
        out.append({"txt": "Hindenburg Omen flag tripped (S&P 500 proxy; watch for a cluster)", "pri": "MED"})
    if cur.get("titanic") and not prev.get("titanic"):
        out.append({"txt": "Titanic Syndrome flag tripped (S&P 500 proxy)", "pri": "MED"})

    order = {"HIGH": 0, "MED": 1}
    out.sort(key=lambda x: order.get(x["pri"], 9))
    return out


def active_alerts(data):
    """Currently-firing conditions (point-in-time), independent of crossings."""
    out = []
    yc = _g(data, "macro", "yield_curve_10y3m", "value")
    if yc is not None:
        if yc < 0:
            out.append(("HIGH", "Yield curve inverted"))
        elif yc < 0.5:
            out.append(("MED", f"Yield curve flat ({yc:.2f})"))
    g = _g(data, "scores", "goldman_composite")
    if g and g.get("value") is not None:
        if g.get("above_70"):
            out.append(("HIGH", f"Goldman composite > 70 ({g['value']:.0f})"))
        elif g.get("above_50"):
            out.append(("MED", f"Goldman composite > 50 ({g['value']:.0f})"))
    vix = _g(data, "sentiment", "vix")
    if vix and vix.get("value") is not None:
        if vix.get("above_40"):
            out.append(("HIGH", f"VIX > 40 ({vix['value']:.1f})"))
        elif vix.get("above_30"):
            out.append(("MED", f"VIX > 30 ({vix['value']:.1f})"))
    pa = _g(data, "breadth", "pct_above_200dma")
    if pa and pa.get("below_40"):
        out.append(("HIGH", f"Only {pa['value']:.0f}% of S&P 500 above 200-DMA"))
    if _g(data, "breadth", "hindenburg_omen_today", "value"):
        out.append(("MED", "Hindenburg Omen flag today"))
    if _g(data, "breadth", "titanic_syndrome_today", "value"):
        out.append(("MED", "Titanic Syndrome flag today"))
    if _g(data, "structural", "ad_line_proxy", "bearish_divergence"):
        out.append(("MED", "Breadth divergence (equal-weight lagging)"))
    if _g(data, "structural", "excess_cape_yield", "low"):
        out.append(("MED", "Excess CAPE Yield low — stocks expensive vs bonds"))
    order = {"HIGH": 0, "MED": 1}
    out.sort(key=lambda x: order.get(x[0], 9))
    return out


def top_sectors(data, n=3):
    rs = _g(data, "sectors", "relative_strength", default={}) or {}
    rows = [(k, v.get("value"), v.get("name")) for k, v in rs.items()
            if isinstance(v, dict) and v.get("value") is not None]
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n]


def upcoming(data, days=7):
    cats = _g(data, "catalysts", "upcoming", default=[]) or []
    return [c for c in cats if isinstance(c, dict) and c.get("days_away", 99) <= days]


def build_email(data, prev_snap):
    cur = snapshot(data)
    crossings = detect_crossings(prev_snap, cur)
    alerts = active_alerts(data)
    sectors = top_sectors(data)
    cats = upcoming(data, 7)

    score = cur.get("composite")
    label = cur.get("label") or "Unknown"
    color = BAND_COLOR.get(label, "#8B949E")
    disp = _g(data, "meta", "generated_display", default="")

    # Subject line
    hi = any(c["pri"] == "HIGH" for c in crossings)
    score_txt = f"{score:.0f}" if isinstance(score, (int, float)) else "—"
    prefix = "⚠️ " if hi else ""
    subject = f"{prefix}Market Health: {label} {score_txt}"
    if crossings:
        subject += f" · {len(crossings)} change{'s' if len(crossings) != 1 else ''}"

    # HTML body (inline styles; email-client friendly, dark)
    def chip(pri):
        c = "#F85149" if pri == "HIGH" else "#D8A657"
        return (f'<span style="font:600 11px monospace;color:{c};'
                f'background:rgba(216,166,87,.12);padding:1px 6px;border-radius:4px">{pri}</span>')

    parts = []
    parts.append(f'''
      <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:560px;margin:0 auto;
                  background:#0E1116;color:#E6EDF3;padding:20px;border-radius:14px">
        <div style="font:600 11px sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#8B949E">Market Health</div>
        <div style="font-size:13px;color:#6E7681;margin-top:2px">{disp}</div>
        <div style="margin:16px 0 6px">
          <span style="font:600 46px monospace;color:{color}">{score_txt}</span>
          <span style="font:18px monospace;color:#6E7681"> / 100</span>
          <div style="font:700 15px sans-serif;color:{color};margin-top:2px">{label}</div>
        </div>
        <div style="font-size:13px;color:#8B949E">
          Structural {_fmt(_g(data,"scores","structural_score","value"))} ·
          Cycle {_fmt(_g(data,"scores","cycle_score","value"))}
        </div>
    ''')

    if crossings:
        rows = "".join(
            f'<div style="padding:9px 0;border-top:1px solid #232A33;font-size:14px">'
            f'{chip(c["pri"])} &nbsp;{c["txt"]}</div>' for c in crossings)
        parts.append(f'''
          <div style="margin-top:18px">
            <div style="font:600 11px sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#8B949E;margin-bottom:4px">Changed since last run</div>
            {rows}
          </div>''')

    if alerts:
        rows = "".join(
            f'<div style="padding:7px 0;border-top:1px solid #232A33;font-size:13.5px">'
            f'{chip(p)} &nbsp;{t}</div>' for p, t in alerts)
        parts.append(f'''
          <div style="margin-top:18px">
            <div style="font:600 11px sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#8B949E;margin-bottom:4px">Active alerts</div>
            {rows}
          </div>''')
    else:
        parts.append('<div style="margin-top:18px;font-size:13.5px;color:#8B949E">No alerts firing — conditions within normal ranges.</div>')

    if sectors:
        cells = "".join(
            f'<span style="font:600 12px monospace;color:{"#3FB950" if v>=0 else "#F85149"};'
            f'margin-right:14px">{tk} {"+" if v>=0 else ""}{v:.1f}</span>'
            for tk, v, nm in sectors)
        phase = _g(data, "sectors", "cycle_phase", "value", default="—")
        parts.append(f'''
          <div style="margin-top:18px">
            <div style="font:600 11px sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#8B949E;margin-bottom:6px">Sector rotation — {phase}</div>
            <div>{cells}</div>
          </div>''')

    if cats:
        rows = "".join(
            f'<div style="padding:6px 0;border-top:1px solid #232A33;font-size:13.5px">'
            f'<span style="font:12px monospace;color:#8B949E;display:inline-block;width:90px">{c["date"]}</span>'
            f'{c["label"]} <span style="color:#6E7681;font:12px monospace">({c["days_away"]}d)</span></div>'
            for c in cats)
        parts.append(f'''
          <div style="margin-top:18px">
            <div style="font:600 11px sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#8B949E;margin-bottom:4px">Next 7 days</div>
            {rows}
          </div>''')

    parts.append('''
        <div style="margin-top:20px;font-size:11px;color:#6E7681;line-height:1.5">
          Breadth signals are S&amp;P 500 proxies for NYSE-wide indicators. Not investment advice.
        </div>
      </div>''')

    return subject, "".join(parts), cur, crossings


def _fmt(v):
    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_email(subject, html):
    r = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                 "Content-Type": "application/json"},
        json={"from": FROM_ADDR, "to": [ALERT_EMAIL], "subject": subject, "html": html},
        timeout=30,
    )
    if r.status_code in (200, 201):
        print(f"Sent: {subject}")
        return True
    print(f"Resend error {r.status_code}: {r.text[:300]}", file=sys.stderr)
    return False


def main(dry_run=False):
    data = load_json(LATEST_PATH)
    if not data:
        print("No latest.json — nothing to send.")
        return
    state = load_json(STATE_PATH, default={})
    prev_snap = state.get("alerts", {}).get("last_snapshot")

    subject, html, cur, crossings = build_email(data, prev_snap)

    if dry_run:
        print("SUBJECT:", subject)
        print("CROSSINGS:", crossings)
        print("HTML length:", len(html))
        return

    if not (RESEND_API_KEY and ALERT_EMAIL):
        print("RESEND_API_KEY or ALERT_EMAIL not set — skipping send (non-fatal).")
        return

    ok = send_email(subject, html)

    # Persist snapshot for next run's crossing detection (merge into shared state).
    state.setdefault("alerts", {})
    state["alerts"]["last_snapshot"] = cur
    state["alerts"]["last_sent_utc"] = datetime.now(timezone.utc).isoformat()
    state["alerts"]["last_send_ok"] = bool(ok)
    save_state(state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build but do not send")
    args = ap.parse_args()
    try:
        main(dry_run=args.dry_run)
    except Exception as e:
        # Never break the workflow over an alert failure.
        print(f"send_alerts non-fatal error: {e}", file=sys.stderr)
        sys.exit(0)
