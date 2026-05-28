"""
analysis/wyckoff_detector.py
=============================
Detects Wyckoff events based ONLY on rules from the original SMI course.
Rules source: WYCKOFF_BEHAVIOR_RULES.txt + WYCKOFF_COMPLETE_COURSE_FULLTEXT.txt
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def avg_vol(df, window=20):
    return df["Volume"].rolling(window).mean()

def avg_spread(df, window=20):
    return (df["High"] - df["Low"]).rolling(window).mean()

def close_pos(row):
    rng = row["High"] - row["Low"]
    if rng == 0: return 0.5
    return (row["Close"] - row["Low"]) / rng

def vol_rank(df, i, lookback=30):
    start = max(0, i - lookback)
    return (df["Volume"].iloc[start:i+1] <= df["Volume"].iloc[i]).mean()


# ─────────────────────────────────────────────────────────────────
#  SWING POINTS — find local highs & lows
# ─────────────────────────────────────────────────────────────────

def find_swings(df, window=10):
    """Find all swing highs and lows."""
    highs, lows = [], []
    prices = df["Close"].values
    hi = df["High"].values
    lo = df["Low"].values

    for i in range(window, len(df) - window):
        if hi[i] == max(hi[i-window:i+window+1]):
            highs.append(i)
        if lo[i] == min(lo[i-window:i+window+1]):
            lows.append(i)
    return highs, lows


# ─────────────────────────────────────────────────────────────────
#  SELLING CLIMAX (SC)
#  Rules from course:
#  - After prolonged downtrend
#  - Climactic volume (highest or near highest)
#  - Wide spread bar
#  - Close in upper half of bar
#  - Price bounces after
# ─────────────────────────────────────────────────────────────────

def find_SC(df):
    _, lows = find_swings(df, window=10)
    avv = avg_vol(df, 20)
    avs = avg_spread(df, 20)
    candidates = []

    for i in lows:
        if i < 20 or i > len(df) - 5:
            continue
        row = df.iloc[i]

        # Must be preceded by downtrend (price 15 bars ago > current)
        prev_close = df["Close"].iloc[max(0, i-15)]
        if prev_close <= row["Close"] * 1.05:
            continue

        # Climactic volume
        vr = vol_rank(df, i, lookback=50)
        if vr < 0.75:
            continue

        # Wide spread
        spread = row["High"] - row["Low"]
        if avs.iloc[i] > 0 and spread < avs.iloc[i] * 1.0:
            continue

        # Close in upper half
        if close_pos(row) < 0.45:
            continue

        # Price bounces after (at least 2 of next 5 bars close higher)
        next5 = df["Close"].iloc[i+1:i+6]
        if (next5 > row["Close"]).sum() < 2:
            continue

        candidates.append({
            "event": "SC",
            "date": str(df.index[i].date()),
            "price": round(float(row["Low"]), 2),
            "volume": int(row["Volume"]),
            "bar_index": i
        })

    if not candidates:
        return None
    # Return the one with highest volume rank
    return max(candidates, key=lambda x: x["volume"])


# ─────────────────────────────────────────────────────────────────
#  AUTOMATIC RALLY (AR)
#  Rules: First sharp rally after SC, lower volume than SC,
#         establishes top of trading range
# ─────────────────────────────────────────────────────────────────

def find_AR(df, sc):
    if not sc:
        return None
    sc_i = sc["bar_index"]
    sc_vol = sc["volume"]

    # Look for highest high in next 20 bars
    search = df.iloc[sc_i+1: sc_i+25]
    if search.empty:
        return None

    ar_loc = search["High"].idxmax()
    ar_i = df.index.get_loc(ar_loc)
    row = df.iloc[ar_i]

    # Volume should be less than SC
    if row["Volume"] > sc_vol * 1.2:
        return None

    # Must rally at least 3%
    if row["High"] < df.iloc[sc_i]["Low"] * 1.03:
        return None

    return {
        "event": "AR",
        "date": str(df.index[ar_i].date()),
        "price": round(float(row["High"]), 2),
        "volume": int(row["Volume"]),
        "bar_index": ar_i
    }


# ─────────────────────────────────────────────────────────────────
#  SECONDARY TEST (ST)
#  Rules: Returns to SC area, LOWER volume than SC, holds above SC
# ─────────────────────────────────────────────────────────────────

def find_STs(df, sc, ar):
    if not sc or not ar:
        return []
    sc_i = sc["bar_index"]
    ar_i = ar["bar_index"]
    sc_low = sc["price"]
    sc_vol = sc["volume"]
    sts = []

    search = df.iloc[ar_i+1: ar_i+60]
    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        # Near SC low (within 5%)
        if abs(row["Low"] - sc_low) / sc_low < 0.05:
            if row["Volume"] < sc_vol * 0.85:
                if row["Low"] >= sc_low * 0.97:
                    sts.append({
                        "event": "ST",
                        "date": str(date.date()),
                        "price": round(float(row["Low"]), 2),
                        "volume": int(row["Volume"]),
                        "bar_index": i
                    })
                    if len(sts) >= 3:
                        break
    return sts


# ─────────────────────────────────────────────────────────────────
#  SPRING
#  Rules: Breaks below SC low briefly, low volume, closes back above
# ─────────────────────────────────────────────────────────────────

def find_Spring(df, sc, ar):
    if not sc or not ar:
        return None
    ar_i = ar["bar_index"]
    sc_low = sc["price"]
    sc_vol = sc["volume"]

    search = df.iloc[ar_i+1: ar_i+80]
    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        if row["Low"] < sc_low:
            if row["Volume"] < sc_vol:
                if row["Close"] > sc_low:
                    return {
                        "event": "Spring",
                        "date": str(date.date()),
                        "price": round(float(row["Low"]), 2),
                        "volume": int(row["Volume"]),
                        "bar_index": i
                    }
    return None


# ─────────────────────────────────────────────────────────────────
#  SIGN OF STRENGTH (SOS)
#  Rules: Strong advance on wide spread + high volume, breaks AR high
# ─────────────────────────────────────────────────────────────────

def find_SOS(df, sc, ar):
    if not sc or not ar:
        return None
    ar_i = ar["bar_index"]
    ar_high = ar["price"]
    avv = avg_vol(df, 20)
    avs = avg_spread(df, 20)

    search = df.iloc[ar_i+1: ar_i+100]
    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        if row["High"] > ar_high:
            spread = row["High"] - row["Low"]
            if avs.iloc[i] > 0 and spread > avs.iloc[i]:
                if avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i] * 1.2:
                    if close_pos(row) > 0.55:
                        return {
                            "event": "SOS",
                            "date": str(date.date()),
                            "price": round(float(row["High"]), 2),
                            "volume": int(row["Volume"]),
                            "bar_index": i
                        }
    return None


# ─────────────────────────────────────────────────────────────────
#  LPS — Last Point of Support
#  Rules: Higher low after SOS, low volume pullback
# ─────────────────────────────────────────────────────────────────

def find_LPS(df, sc, sos):
    if not sc or not sos:
        return None
    sos_i = sos["bar_index"]
    avv = avg_vol(df, 20)
    _, lows = find_swings(df, window=5)

    for i in lows:
        if i <= sos_i or i >= len(df) - 3:
            continue
        row = df.iloc[i]
        # Must be higher than SC
        if row["Low"] < sc["price"]:
            continue
        # Low volume
        if avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i] * 0.9:
            continue
        return {
            "event": "LPS",
            "date": str(df.index[i].date()),
            "price": round(float(row["Low"]), 2),
            "volume": int(row["Volume"]),
            "bar_index": i
        }
    return None


# ─────────────────────────────────────────────────────────────────
#  BUYING CLIMAX (BC) — Distribution
#  Rules: After uptrend, climactic volume, wide spread, close near LOW
# ─────────────────────────────────────────────────────────────────

def find_BC(df):
    highs, _ = find_swings(df, window=10)
    avv = avg_vol(df, 20)
    avs = avg_spread(df, 20)
    candidates = []

    for i in highs:
        if i < 20 or i > len(df) - 5:
            continue
        row = df.iloc[i]

        # Must be preceded by uptrend
        prev_close = df["Close"].iloc[max(0, i-15)]
        if prev_close >= row["Close"] * 0.95:
            continue

        # Climactic volume
        vr = vol_rank(df, i, lookback=50)
        if vr < 0.75:
            continue

        # Wide spread
        spread = row["High"] - row["Low"]
        if avs.iloc[i] > 0 and spread < avs.iloc[i] * 1.0:
            continue

        # Close in LOWER half of bar
        if close_pos(row) > 0.55:
            continue

        candidates.append({
            "event": "BC",
            "date": str(df.index[i].date()),
            "price": round(float(row["High"]), 2),
            "volume": int(row["Volume"]),
            "bar_index": i
        })

    if not candidates:
        return None
    return max(candidates, key=lambda x: x["volume"])


# ─────────────────────────────────────────────────────────────────
#  PSY — Preliminary Supply
#  Rules: First selling after prolonged advance, volume increases
# ─────────────────────────────────────────────────────────────────

def find_PSY(df, bc):
    if not bc:
        return None
    bc_i = bc["bar_index"]
    avv = avg_vol(df, 20)
    highs, _ = find_swings(df, window=8)

    # Look for swing high before BC
    before_bc = [i for i in highs if i < bc_i - 5]
    if not before_bc:
        return None

    i = before_bc[-1]
    row = df.iloc[i]

    # Volume increasing
    if avv.iloc[i] > 0 and row["Volume"] < avv.iloc[i]:
        return None

    return {
        "event": "PSY",
        "date": str(df.index[i].date()),
        "price": round(float(row["High"]), 2),
        "volume": int(row["Volume"]),
        "bar_index": i
    }


# ─────────────────────────────────────────────────────────────────
#  SOW — Sign of Weakness
#  Rules: Strong decline on wide spread + high volume, breaks AR low
# ─────────────────────────────────────────────────────────────────

def find_SOW(df, bc, ar_dist):
    if not bc or not ar_dist:
        return None
    ar_i = ar_dist["bar_index"]
    ar_low = ar_dist["price"]
    avv = avg_vol(df, 20)
    avs = avg_spread(df, 20)

    search = df.iloc[ar_i+1: ar_i+80]
    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        if row["Low"] < ar_low:
            spread = row["High"] - row["Low"]
            if avs.iloc[i] > 0 and spread > avs.iloc[i]:
                if avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i] * 1.2:
                    if close_pos(row) < 0.45:
                        return {
                            "event": "SOW",
                            "date": str(date.date()),
                            "price": round(float(row["Low"]), 2),
                            "volume": int(row["Volume"]),
                            "bar_index": i
                        }
    return None


# ─────────────────────────────────────────────────────────────────
#  RE-ACCUMULATION — Stepping Stone TR during Markup
# ─────────────────────────────────────────────────────────────────

def find_reaccumulation(df, sos):
    if not sos:
        return []
    sos_i = sos["bar_index"]
    results = []
    avv = avg_vol(df, 20)
    window = 20
    step = 15

    i = sos_i + 20
    while i < len(df) - window:
        chunk = df.iloc[i:i+window]
        price_range = (chunk["High"].max() - chunk["Low"].min()) / chunk["Close"].mean()
        avg_v = avv.iloc[i+window//2] if avv.iloc[i+window//2] > 0 else 1
        avg_vol_chunk = chunk["Volume"].mean()

        # Tight price range (< 8%) with decreasing volume = re-accumulation
        if price_range < 0.08 and avg_vol_chunk < avg_v * 0.85:
            mid_date = df.index[i + window//2]
            mid_price = chunk["Close"].mean()
            results.append({
                "event": "ReAcc",
                "date": str(mid_date.date()),
                "price": round(float(mid_price), 2),
                "volume": int(avg_vol_chunk),
                "bar_index": i + window//2
            })
            i += window
        else:
            i += step

    return results[:3]  # max 3 re-accumulation zones


# ─────────────────────────────────────────────────────────────────
#  WAVE ANALYSIS (Section 5M)
# ─────────────────────────────────────────────────────────────────

def analyze_waves(df):
    closes = df["Close"].values
    volumes = df["Volume"].values
    window = 5

    highs_idx, lows_idx = [], []
    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[i-window:i+window+1]):
            highs_idx.append((i, closes[i]))
        if closes[i] == min(closes[i-window:i+window+1]):
            lows_idx.append((i, closes[i]))

    trend_strength = "Unknown"
    if len(highs_idx) >= 2 and len(lows_idx) >= 2:
        lh = [h[1] for h in highs_idx[-3:]]
        ll = [l[1] for l in lows_idx[-3:]]
        if all(lh[i] > lh[i-1] for i in range(1, len(lh))):
            if all(ll[i] > ll[i-1] for i in range(1, len(ll))):
                trend_strength = "Strong Uptrend — Rising Highs & Lows"
            else:
                trend_strength = "Weakening Uptrend — Lows not rising"
        elif all(lh[i] < lh[i-1] for i in range(1, len(lh))):
            if all(ll[i] < ll[i-1] for i in range(1, len(ll))):
                trend_strength = "Strong Downtrend — Falling Highs & Lows"
        else:
            trend_strength = "Sideways / Transition"

    recent_vol = volumes[-20:]
    vol_trend = "Increasing" if recent_vol[-1] > np.mean(recent_vol) else "Decreasing"

    return {
        "trend_strength": trend_strength,
        "recent_volume_trend": vol_trend,
    }


# ─────────────────────────────────────────────────────────────────
#  PHASE DETERMINATION
# ─────────────────────────────────────────────────────────────────

def determine_phase(events):
    names = [e["event"] for e in events]

    if "SOS" in names or "LPS" in names:
        return "D-E", "BULLISH"
    elif "Spring" in names:
        return "C", "BULLISH"
    elif "ST" in names and "SC" in names:
        return "B", "NEUTRAL"
    elif "AR" in names and "SC" in names:
        return "A", "NEUTRAL"
    elif "SOW" in names:
        return "D", "BEARISH"
    elif "BC" in names:
        return "A", "BEARISH"
    return "?", "NEUTRAL"


# ─────────────────────────────────────────────────────────────────
#  BUILD ZONES FROM EVENTS
# ─────────────────────────────────────────────────────────────────

def build_zones(events, df):
    zones = []
    sc_e = next((e for e in events if e["event"] == "SC"), None)
    ar_e = next((e for e in events if e["event"] == "AR"), None)
    sos_e = next((e for e in events if e["event"] == "SOS"), None)
    bc_e = next((e for e in events if e["event"] == "BC"), None)

    last_date = df.index[-1].strftime("%Y-%m-%d")

    # Accumulation zone between SC and AR
    if sc_e and ar_e:
        end_date = sos_e["date"] if sos_e else last_date
        zones.append({
            "x0": sc_e["date"],
            "x1": end_date,
            "y0": sc_e["price"] * 0.98,
            "y1": ar_e["price"] * 1.01,
            "name": "Accumulation TR",
            "color": "green"
        })

    # Distribution zone between AR and BC
    if bc_e:
        zones.append({
            "x0": bc_e["date"],
            "x1": last_date,
            "y0": bc_e["price"] * 0.95,
            "y1": bc_e["price"] * 1.02,
            "name": "Distribution TR",
            "color": "red"
        })

    return zones


# ─────────────────────────────────────────────────────────────────
#  BUILD SUPPORT/RESISTANCE LINES
# ─────────────────────────────────────────────────────────────────

def build_support_lines(events):
    lines = []
    for e in events:
        if e["event"] == "SC":
            lines.append({"y": e["price"], "label": f"SC Low ${e['price']:.1f}", "color": "#e03c3c"})
        elif e["event"] == "AR":
            lines.append({"y": e["price"], "label": f"AR High ${e['price']:.1f}", "color": "#4fc3f7"})
        elif e["event"] == "Spring":
            lines.append({"y": e["price"], "label": f"Spring ${e['price']:.1f}", "color": "#a5d6a7"})
        elif e["event"] == "SOS":
            lines.append({"y": e["price"], "label": f"SOS ${e['price']:.1f}", "color": "#43a047"})
        elif e["event"] == "LPS":
            lines.append({"y": e["price"], "label": f"LPS ${e['price']:.1f}", "color": "#66bb6a"})
        elif e["event"] == "BC":
            lines.append({"y": e["price"], "label": f"BC High ${e['price']:.1f}", "color": "#ff7043"})
        elif e["event"] == "SOW":
            lines.append({"y": e["price"], "label": f"SOW ${e['price']:.1f}", "color": "#ef5350"})
    return lines


# ─────────────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────

def detect_wyckoff_events(df):
    events = []
    print("🔍 Running Wyckoff event detection...")

    # ── Try Accumulation first ──
    sc = find_SC(df)
    if sc:
        events.append(sc)
        print(f"  ✅ SC  : {sc['date']} @ ${sc['price']}")

        ar = find_AR(df, sc)
        if ar:
            events.append(ar)
            print(f"  ✅ AR  : {ar['date']} @ ${ar['price']}")

            for st in find_STs(df, sc, ar):
                events.append(st)
                print(f"  ✅ ST  : {st['date']} @ ${st['price']}")

            spring = find_Spring(df, sc, ar)
            if spring:
                events.append(spring)
                print(f"  ✅ Spring: {spring['date']} @ ${spring['price']}")

            sos = find_SOS(df, sc, ar)
            if sos:
                events.append(sos)
                print(f"  ✅ SOS : {sos['date']} @ ${sos['price']}")

                lps = find_LPS(df, sc, sos)
                if lps:
                    events.append(lps)
                    print(f"  ✅ LPS : {lps['date']} @ ${lps['price']}")

                # Re-accumulation zones
                for ra in find_reaccumulation(df, sos):
                    events.append(ra)
                    print(f"  ✅ ReAcc: {ra['date']} @ ${ra['price']}")

    # ── Try Distribution if no Accumulation ──
    if not sc:
        bc = find_BC(df)
        if bc:
            events.append(bc)
            print(f"  ✅ BC  : {bc['date']} @ ${bc['price']}")

            psy = find_PSY(df, bc)
            if psy:
                events.append(psy)
                print(f"  ✅ PSY : {psy['date']} @ ${psy['price']}")

    phase, bias = determine_phase(events)
    waves = analyze_waves(df)
    zones = build_zones(events, df)
    support_lines = build_support_lines(events)

    names = [e["event"] for e in events]
    summary = f"Events: {', '.join(names)}" if names else "No clear Wyckoff events detected"

    print(f"  📊 Phase: {phase} | Bias: {bias}")
    print(f"  🌊 {waves['trend_strength']}")

    return {
        "events": events,
        "support_lines": support_lines,
        "trend_lines": [],
        "zones": zones,
        "trading_ranges": [],
        "markup_lines": [],
        "phase": phase,
        "bias": bias,
        "wave_analysis": waves,
        "summary_ar": summary,
        "events_found": len(events) > 0
    }
