"""
wyckoff_detector.py
====================
Methodology (Wyckoff SMI Course):
1. Scan for TREND CHANGES: Markup → Sideways, Markdown → Sideways
2. For each Sideways zone → classify: Accumulation / Distribution / ReAcc / ReDist
3. Detect Wyckoff events INSIDE each zone
"""
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def avg_vol(df, w=20):   return df["Volume"].rolling(w).mean()
def avg_spd(df, w=20):   return (df["High"]-df["Low"]).rolling(w).mean()

def close_pos(row):
    r = row["High"]-row["Low"]
    return (row["Close"]-row["Low"])/r if r else 0.5

def vol_rank(df, i, lb=60):
    s = max(0, i-lb)
    return (df["Volume"].iloc[s:i+1] <= df["Volume"].iloc[i]).mean()

def find_swings(df, w=10):
    hi, lo = df["High"].values, df["Low"].values
    peaks, troughs = [], []
    for i in range(w, len(df)-w):
        if hi[i] == max(hi[i-w:i+w+1]): peaks.append(i)
        if lo[i] == min(lo[i-w:i+w+1]): troughs.append(i)
    return peaks, troughs


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FIND ALL SIDEWAYS ZONES
#  Rule: A sideways zone is a period where price oscillates within a range
#  without making significant new highs or lows.
#  Detection: Rolling window where (High-Low)/Close < threshold
# ══════════════════════════════════════════════════════════════════════════════

def find_sideways_zones(df, min_bars=30, range_pct=0.12):
    """
    Scans the chart for all consolidation/sideways zones.
    Returns list of zones with start, end, high, low.
    """
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n = len(df)
    zones = []
    i = min_bars

    while i < n - min_bars:
        # Look for a window where price is contained
        found = False
        for width in range(min_bars, min(120, n-i), 5):
            chunk_hi = max(highs[i:i+width])
            chunk_lo = min(lows[i:i+width])
            mid = (chunk_hi + chunk_lo) / 2
            rng = (chunk_hi - chunk_lo) / mid if mid else 1

            if rng < range_pct:
                # Extend the zone as far as price stays within ±5%
                end = i + width
                while end < n:
                    new_hi = max(highs[i:end+1])
                    new_lo = min(lows[i:end+1])
                    new_rng = (new_hi-new_lo)/((new_hi+new_lo)/2)
                    if new_rng > range_pct * 1.4:
                        break
                    end += 1

                if end - i >= min_bars:
                    zones.append({
                        "start": i,
                        "end":   end,
                        "x0":    str(df.index[i].date()),
                        "x1":    str(df.index[min(end, n-1)].date()),
                        "high":  float(max(highs[i:end])),
                        "low":   float(min(lows[i:end])),
                        "bars":  end - i,
                    })
                    i = end + 5
                    found = True
                    break
        if not found:
            i += 5

    # Merge overlapping zones
    merged = []
    for z in zones:
        if merged and z["start"] < merged[-1]["end"] + 10:
            merged[-1]["end"]  = max(merged[-1]["end"],  z["end"])
            merged[-1]["x1"]   = str(df.index[min(merged[-1]["end"], n-1)].date())
            merged[-1]["high"] = max(merged[-1]["high"], z["high"])
            merged[-1]["low"]  = min(merged[-1]["low"],  z["low"])
            merged[-1]["bars"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(z)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — CLASSIFY EACH ZONE
#  Rules from course:
#  - After Markdown  + climactic volume at BOTTOM  → Accumulation
#  - After Markup    + climactic volume at TOP     → Distribution
#  - During Markup   + tight range + low volume   → Re-Accumulation
#  - During Markdown + tight range + low volume   → Re-Distribution
# ══════════════════════════════════════════════════════════════════════════════

def classify_zone(df, zone, lookback=30):
    len_df = len(df)
    """
    Classify a sideways zone based on:
    1. What came before it (trend direction)
    2. Volume at the extremes (climactic at top or bottom)
    3. Price behavior inside
    """
    s, e = zone["start"], zone["end"]
    n = len(df)

    # ── What was the trend BEFORE this zone? ──
    before_start = max(0, s - lookback)
    before_chunk = df.iloc[before_start:s]
    if len(before_chunk) < 5:
        prior_trend = "unknown"
    else:
        price_change = (before_chunk["Close"].iloc[-1] -
                        before_chunk["Close"].iloc[0]) / before_chunk["Close"].iloc[0]
        if price_change > 0.08:
            prior_trend = "markup"
        elif price_change < -0.08:
            prior_trend = "markdown"
        else:
            prior_trend = "sideways"

    # ── Zone size relative to price ──
    zone_range = (zone["high"] - zone["low"]) / ((zone["high"]+zone["low"])/2)

    # ── Volume behavior inside zone ──
    chunk = df.iloc[s:e]
    avv_val = float(avg_vol(df, 20).iloc[min(s+10, n-1)])
    avg_chunk_vol = float(chunk["Volume"].mean())
    vol_ratio = avg_chunk_vol / avv_val if avv_val > 0 else 1.0

    # ── Climactic bars at extremes ──
    # Find bar with highest volume in zone
    max_vol_idx = int(chunk["Volume"].idxmax().name if hasattr(chunk["Volume"].idxmax(),'name') else 0)
    try:
        max_vol_loc = chunk["Volume"].idxmax()
        max_vol_bar = df.index.get_loc(max_vol_loc)
        max_vol_row = df.iloc[max_vol_bar]
        climax_at_bottom = max_vol_row["Low"] < (zone["low"] + (zone["high"]-zone["low"])*0.35)
        climax_at_top    = max_vol_row["High"] > (zone["low"] + (zone["high"]-zone["low"])*0.65)
    except:
        climax_at_bottom = False
        climax_at_top    = False

    # ── Classification logic ──
    # Primary classification based on prior trend direction
    if prior_trend == "markdown":
        # After a downtrend = Accumulation (always)
        zone_type = "Accumulation"
    elif prior_trend == "markup":
        # After an uptrend = Distribution (always)
        zone_type = "Distribution"
    elif prior_trend == "sideways":
        # During sideways — use volume and range to determine
        if vol_ratio < 0.85 and zone_range < 0.09:
            zone_type = "Re-Accumulation"
        else:
            zone_type = "Re-Accumulation"
    else:
        # Unknown prior — use volume hint
        if vol_ratio > 1.2:
            zone_type = "Accumulation"
        else:
            zone_type = "Re-Accumulation"

    # Override: if zone is SMALL and tight during a clear uptrend = Re-Accumulation
    # Check if price after zone is higher than before = markup context
    after_end = min(zone["end"]+30, len_df-1)
    before_start_price = df["Close"].iloc[max(0,zone["start"]-30)] if zone["start"] > 0 else df["Close"].iloc[0]
    zone_mid_price = (zone["high"]+zone["low"])/2
    if (zone_type == "Accumulation" and
        zone_range < 0.10 and
        zone_mid_price > before_start_price * 1.10):
        zone_type = "Re-Accumulation"
    if (zone_type == "Distribution" and
        zone_range < 0.10 and
        zone_mid_price < before_start_price * 0.90):
        zone_type = "Re-Distribution"

    zone["type"]        = zone_type
    zone["prior_trend"] = prior_trend
    zone["vol_ratio"]   = round(vol_ratio, 2)
    return zone


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — DETECT EVENTS INSIDE EACH ZONE
# ══════════════════════════════════════════════════════════════════════════════

def detect_SC_in_zone(df, zone):
    s, e = zone["start"], zone["end"]
    chunk = df.iloc[s:e]
    avs = avg_spd(df, 20)
    for date, row in chunk.iterrows():
        i = df.index.get_loc(date)
        if row["Low"] > zone["low"] * 1.02: continue
        if vol_rank(df, i, 60) < 0.65: continue
        if close_pos(row) < 0.38: continue
        return {"event":"SC","date":str(date.date()),
                "price":round(float(row["Low"]),2),
                "volume":int(row["Volume"]),"bar_index":i}
    return None

def detect_BC_in_zone(df, zone):
    s, e = zone["start"], zone["end"]
    chunk = df.iloc[s:e]
    for date, row in chunk.iterrows():
        i = df.index.get_loc(date)
        if row["High"] < zone["high"] * 0.98: continue
        if vol_rank(df, i, 60) < 0.65: continue
        if close_pos(row) > 0.55: continue
        return {"event":"BC","date":str(date.date()),
                "price":round(float(row["High"]),2),
                "volume":int(row["Volume"]),"bar_index":i}
    return None

def detect_AR_after_SC(df, sc, zone_end):
    if not sc: return None
    si = sc["bar_index"]
    seg = df.iloc[si+1:min(si+30, zone_end)]
    if seg.empty: return None
    loc = seg["High"].idxmax(); ai = df.index.get_loc(loc)
    row = df.iloc[ai]
    return {"event":"AR","date":str(df.index[ai].date()),
            "price":round(float(row["High"]),2),
            "volume":int(row["Volume"]),"bar_index":ai}

def detect_AR_after_BC(df, bc, zone_end):
    if not bc: return None
    bi = bc["bar_index"]
    seg = df.iloc[bi+1:min(bi+30, zone_end)]
    if seg.empty: return None
    loc = seg["Low"].idxmin(); ai = df.index.get_loc(loc)
    row = df.iloc[ai]
    return {"event":"AR","date":str(df.index[ai].date()),
            "price":round(float(row["Low"]),2),
            "volume":int(row["Volume"]),"bar_index":ai}

def detect_Spring(df, sc, ar, zone_end):
    if not sc or not ar: return None
    ai, sl, sv = ar["bar_index"], sc["price"], sc["volume"]
    for date, row in df.iloc[ai+5:zone_end].iterrows():
        i = df.index.get_loc(date)
        if row["Low"] < sl and row["Volume"] < sv*1.1 and row["Close"] > sl:
            return {"event":"Spring","date":str(date.date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None

def detect_SOS(df, ar, zone_end):
    if not ar: return None
    ai, ah = ar["bar_index"], ar["price"]
    avv, avs = avg_vol(df,20), avg_spd(df,20)
    # Search from zone end onwards (SOS breaks OUT of the zone)
    for date, row in df.iloc[zone_end:zone_end+200].iterrows():
        i = df.index.get_loc(date)
        sp = row["High"]-row["Low"]
        if (row["High"] > ah*1.01 and
            avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i]*1.0 and
            close_pos(row) > 0.50):
            return {"event":"SOS","date":str(date.date()),
                    "price":round(float(row["High"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None

def detect_SOW(df, ar_dist, zone_end):
    if not ar_dist: return None
    ai, al = ar_dist["bar_index"], ar_dist["price"]
    avv = avg_vol(df, 20)
    for date, row in df.iloc[zone_end:zone_end+200].iterrows():
        i = df.index.get_loc(date)
        if (row["Low"] < al*0.99 and
            avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i]*1.0 and
            close_pos(row) < 0.50):
            return {"event":"SOW","date":str(date.date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — BUILD ZONE BOXES FOR CHART
# ══════════════════════════════════════════════════════════════════════════════

ZONE_COLORS = {
    "Accumulation":    "#4fc3f7",
    "Distribution":    "#e03c3c",
    "Re-Accumulation": "#26c6da",
    "Re-Distribution": "#ff7043",
}

def build_zone_boxes(classified_zones, df):
    boxes = []
    for z in classified_zones:
        col = ZONE_COLORS.get(z["type"], "#90caf9")
        boxes.append({
            "x0":    z["x0"],
            "x1":    z["x1"],
            "y0":    round(z["low"]  * 0.995, 2),
            "y1":    round(z["high"] * 1.005, 2),
            "label": z["type"],
            "color": col,
            "vol_ratio": z.get("vol_ratio", 1.0),
        })
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
#  WAVE ANALYSIS (Section 5M)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_waves(df):
    c = df["Close"].values; w = 5
    hi_idx = [i for i in range(w,len(c)-w) if c[i]==max(c[i-w:i+w+1])]
    lo_idx = [i for i in range(w,len(c)-w) if c[i]==min(c[i-w:i+w+1])]
    trend = "Unknown"
    if len(hi_idx)>=2 and len(lo_idx)>=2:
        lh=[c[i] for i in hi_idx[-3:]]; ll=[c[i] for i in lo_idx[-3:]]
        rh=all(lh[i]>lh[i-1] for i in range(1,len(lh)))
        rl=all(ll[i]>ll[i-1] for i in range(1,len(ll)))
        fh=all(lh[i]<lh[i-1] for i in range(1,len(lh)))
        fl=all(ll[i]<ll[i-1] for i in range(1,len(ll)))
        if rh and rl:   trend="Strong Uptrend — Rising Highs & Lows"
        elif fh and fl: trend="Strong Downtrend — Falling Highs & Lows"
        else:           trend="Sideways / Transition"
    rv = df["Volume"].values[-20:]
    return {"trend_strength":trend,
            "recent_volume_trend":"Increasing" if rv[-1]>np.mean(rv) else "Decreasing"}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def detect_wyckoff_events(df):
    events = []
    print("🔍 Wyckoff event detection — full chart scan...")

    # ── STEP 1: Find all sideways zones ──────────────────────────────────────
    zones = find_sideways_zones(df, min_bars=25, range_pct=0.11)
    print(f"  📦 Sideways zones found: {len(zones)}")

    # ── STEP 2: Classify each zone ───────────────────────────────────────────
    classified = []
    for z in zones:
        z = classify_zone(df, z)
        classified.append(z)
        print(f"  🏷️  [{z['type']:18s}] {z['x0']} → {z['x1']}  "
              f"${z['low']:.0f}–${z['high']:.0f}  "
              f"prior={z['prior_trend']}  vol={z['vol_ratio']:.2f}x")

    # ── STEP 3: Detect events inside each zone ───────────────────────────────
    for z in classified:
        s, e = z["start"], z["end"]
        ztype = z["type"]

        if ztype == "Accumulation":
            sc = detect_SC_in_zone(df, z)
            if sc:
                events.append(sc)
                ar = detect_AR_after_SC(df, sc, e)
                if ar:
                    events.append(ar)
                    sp = detect_Spring(df, sc, ar, e)
                    if sp: events.append(sp)
                    sos = detect_SOS(df, ar, e)
                    if sos: events.append(sos)

        elif ztype == "Distribution":
            bc = detect_BC_in_zone(df, z)
            if bc:
                events.append(bc)
                ar_d = detect_AR_after_BC(df, bc, e)
                if ar_d:
                    events.append(ar_d)
                    sow = detect_SOW(df, ar_d, e)
                    if sow: events.append(sow)

        elif ztype == "Re-Accumulation":
            # Mark the zone center as a ReAcc event
            mid_i = (s + e) // 2
            events.append({
                "event":     "ReAcc",
                "date":      str(df.index[mid_i].date()),
                "price":     round(float(df["Close"].iloc[mid_i]), 2),
                "volume":    int(df["Volume"].iloc[mid_i]),
                "bar_index": mid_i,
                "x0": z["x0"], "x1": z["x1"],
                "y0": round(z["low"]*0.995,2),
                "y1": round(z["high"]*1.005,2),
            })

        elif ztype == "Re-Distribution":
            mid_i = (s + e) // 2
            events.append({
                "event":     "ReDist",
                "date":      str(df.index[mid_i].date()),
                "price":     round(float(df["Close"].iloc[mid_i]), 2),
                "volume":    int(df["Volume"].iloc[mid_i]),
                "bar_index": mid_i,
                "x0": z["x0"], "x1": z["x1"],
                "y0": round(z["low"]*0.995,2),
                "y1": round(z["high"]*1.005,2),
            })

    # ── Determine bias from events ────────────────────────────────────────────
    nm = [e["event"] for e in events]
    has_acc  = any(z["type"]=="Accumulation" for z in classified)
    has_dist = any(z["type"]=="Distribution" for z in classified)
    last_z   = classified[-1]["type"] if classified else ""

    if "SOS" in nm:
        phase, bias = "D-E", "BULLISH"
    elif "Spring" in nm or has_acc:
        phase, bias = "C",   "BULLISH"
    elif "SOW" in nm or has_dist:
        phase, bias = "D",   "BEARISH"
    elif "ReAcc" in nm:
        phase, bias = "E",   "BULLISH"
    else:
        phase, bias = "?",   "NEUTRAL"

    waves = analyze_waves(df)
    zone_boxes = build_zone_boxes(classified, df)

    print(f"\n  📊 Phase: {phase} | Bias: {bias}")
    print(f"  🌊 {waves['trend_strength']}")
    print(f"  📋 Events: {', '.join(nm) if nm else 'None'}")

    return {
        "events":        events,
        "zone_boxes":    zone_boxes,
        "classified_zones": classified,
        "support_lines": [],
        "trend_lines":   [],
        "zones":         [],
        "trading_ranges":[],
        "markup_lines":  [],
        "phase_labels":  [],
        "phase":  phase,
        "bias":   bias,
        "wave_analysis": waves,
        "summary_ar": f"Events: {', '.join(nm)}" if nm else "No events",
        "events_found": len(events) > 0
    }
