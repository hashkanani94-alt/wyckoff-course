"""
wyckoff_detector.py — Full chart scan based on Wyckoff SMI Course
"""
import pandas as pd
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────
def avg_vol(df, w=20):  return df["Volume"].rolling(w).mean()
def avg_spd(df, w=20):  return (df["High"]-df["Low"]).rolling(w).mean()
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


# ─── SC ───────────────────────────────────────────────────────────────────────
def find_SC(df):
    _, troughs = find_swings(df, 10)
    avs = avg_spd(df, 20)
    best = None
    for i in troughs:
        if i < 20 or i > len(df)-5: continue
        row = df.iloc[i]
        if df["Close"].iloc[max(0,i-15)] <= row["Close"]*1.04: continue
        if vol_rank(df, i, 60) < 0.70: continue
        spread = row["High"]-row["Low"]
        if avs.iloc[i] > 0 and spread < avs.iloc[i]*0.85: continue
        if close_pos(row) < 0.38: continue
        nxt = df["Close"].iloc[i+1:i+6]
        if (nxt > row["Close"]).sum() < 2: continue
        if best is None or row["Volume"] > best["volume"]:
            best = {"event":"SC","date":str(df.index[i].date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return best


# ─── AR ───────────────────────────────────────────────────────────────────────
def find_AR(df, sc):
    if not sc: return None
    si = sc["bar_index"]
    seg = df.iloc[si+1:si+30]
    if seg.empty: return None
    loc = seg["High"].idxmax(); ai = df.index.get_loc(loc)
    row = df.iloc[ai]
    if row["High"] < df.iloc[si]["Low"]*1.03: return None
    return {"event":"AR","date":str(df.index[ai].date()),
            "price":round(float(row["High"]),2),
            "volume":int(row["Volume"]),"bar_index":ai}


# ─── STs ──────────────────────────────────────────────────────────────────────
def find_STs(df, sc, ar):
    if not sc or not ar: return []
    ai, sl, sv = ar["bar_index"], sc["price"], sc["volume"]
    sts = []
    for date, row in df.iloc[ai+1:ai+120].iterrows():
        i = df.index.get_loc(date)
        if abs(row["Low"]-sl)/sl < 0.06 and row["Volume"] < sv*0.9 and row["Low"] >= sl*0.96:
            sts.append({"event":"ST","date":str(date.date()),
                        "price":round(float(row["Low"]),2),
                        "volume":int(row["Volume"]),"bar_index":i})
            if len(sts) >= 3: break
    return sts


# ─── Spring ───────────────────────────────────────────────────────────────────
def find_Spring(df, sc, ar):
    if not sc or not ar: return None
    ai, sl, sv = ar["bar_index"], sc["price"], sc["volume"]
    # Must be at least 10 bars after AR to avoid overlap with ST
    for date, row in df.iloc[ai+10:ai+150].iterrows():
        i = df.index.get_loc(date)
        if row["Low"] < sl and row["Volume"] < sv and row["Close"] > sl:
            return {"event":"Spring","date":str(date.date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None


# ─── SOS — search the ENTIRE chart after AR ──────────────────────────────────
def find_SOS(df, sc, ar):
    if not sc or not ar: return None
    ai, ah = ar["bar_index"], ar["price"]
    avv, avs = avg_vol(df,20), avg_spd(df,20)
    # Search ALL remaining bars after AR (not just 120!)
    for date, row in df.iloc[ai+20:].iterrows():
        i = df.index.get_loc(date)
        sp = row["High"]-row["Low"]
        if (row["High"] > ah*1.02 and
            avs.iloc[i] > 0 and sp > avs.iloc[i]*0.85 and
            avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i]*1.05 and
            close_pos(row) > 0.55):
            return {"event":"SOS","date":str(date.date()),
                    "price":round(float(row["High"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None


# ─── LPS ──────────────────────────────────────────────────────────────────────
def find_LPS(df, sc, sos):
    if not sc or not sos: return None
    _, troughs = find_swings(df, 5)
    avv = avg_vol(df, 20)
    for i in troughs:
        if i <= sos["bar_index"] or i >= len(df)-3: continue
        row = df.iloc[i]
        if row["Low"] < sc["price"]: continue
        if avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i]*0.95: continue
        return {"event":"LPS","date":str(df.index[i].date()),
                "price":round(float(row["Low"]),2),
                "volume":int(row["Volume"]),"bar_index":i}
    return None


# ─── Re-Accumulation zones (scan markup phase) ────────────────────────────────
def find_reaccumulation(df, sos):
    """
    Finds consolidation zones during markup.
    Rules (Section on Re-Accumulation):
    - Tight price range during an uptrend
    - Decreasing volume = absorption
    - Similar structure to accumulation but shorter
    """
    if not sos: return []
    results = []
    avv = avg_vol(df, 20)
    w = 30   # window size per zone
    step = 20
    i = sos["bar_index"] + 30

    while i < len(df) - w:
        chunk = df.iloc[i:i+w]
        hi = float(chunk["High"].max())
        lo = float(chunk["Low"].min())
        mid = float(chunk["Close"].mean())
        rng = (hi - lo) / mid if mid else 1

        # Get avg volume for this period
        av = float(avv.iloc[i+w//2]) if avv.iloc[i+w//2] > 0 else 1
        cv = float(chunk["Volume"].mean())

        # Re-Acc: tight range (<10%) + volume < average
        if rng < 0.10 and cv < av * 0.90:
            x0_date = str(df.index[i].date())
            x1_date = str(df.index[min(i+w, len(df)-1)].date())
            results.append({
                "event": "ReAcc",
                "date":  str(df.index[i+w//2].date()),
                "price": round(mid, 2),
                "volume": int(cv),
                "bar_index": i+w//2,
                "x0": x0_date,
                "x1": x1_date,
                "y0": round(lo * 0.995, 2),
                "y1": round(hi * 1.005, 2),
            })
            i += w + 10   # skip ahead after finding a zone
            if len(results) >= 4: break
        else:
            i += step

    return results


# ─── BC / PSY ─────────────────────────────────────────────────────────────────
def find_BC(df):
    peaks, _ = find_swings(df, 10)
    avs = avg_spd(df, 20)
    best = None
    for i in peaks:
        if i < 20 or i > len(df)-5: continue
        row = df.iloc[i]
        if df["Close"].iloc[max(0,i-15)] >= row["Close"]*0.96: continue
        if vol_rank(df, i, 60) < 0.70: continue
        sp = row["High"]-row["Low"]
        if avs.iloc[i] > 0 and sp < avs.iloc[i]*0.85: continue
        if close_pos(row) > 0.55: continue
        if best is None or row["Volume"] > best["volume"]:
            best = {"event":"BC","date":str(df.index[i].date()),
                    "price":round(float(row["High"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return best


# ─── Phase determination ──────────────────────────────────────────────────────
def determine_phase(events):
    nm = [e["event"] for e in events]
    if "SOS" in nm or "LPS" in nm: return "D-E","BULLISH"
    if "Spring" in nm:              return "C",  "BULLISH"
    if "ST" in nm and "SC" in nm:   return "B",  "NEUTRAL"
    if "AR" in nm and "SC" in nm:   return "A",  "NEUTRAL"
    if "BC" in nm:                  return "A",  "BEARISH"
    return "?","NEUTRAL"


# ─── Phase labels for chart ───────────────────────────────────────────────────
def build_phase_labels(sc, ar, spring, sos, bc, df):
    labels = []
    last = str(df.index[-1].date())
    if sc and ar:
        labels.append({"text":"Phase A","x0":sc["date"],"x1":ar["date"],"color":"#e53935"})
    if ar and spring:
        labels.append({"text":"Phase B","x0":ar["date"],"x1":spring["date"],"color":"#fb8c00"})
    if spring and sos:
        labels.append({"text":"Phase C","x0":spring["date"],"x1":sos["date"],"color":"#fdd835"})
    elif spring:
        labels.append({"text":"Phase C","x0":spring["date"],"x1":last,"color":"#fdd835"})
    if sos:
        labels.append({"text":"Phase D→E  Markup","x0":sos["date"],"x1":last,"color":"#43a047"})
    return labels


# ─── Wave analysis ────────────────────────────────────────────────────────────
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
        elif rh:        trend="Possible Accumulation — Rising Highs"
        else:           trend="Sideways / Transition"
    rv = df["Volume"].values[-20:]
    return {"trend_strength":trend,
            "recent_volume_trend":"Increasing" if rv[-1]>np.mean(rv) else "Decreasing"}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def detect_wyckoff_events(df):
    events = []
    print("🔍 Running Wyckoff event detection...")

    sc = find_SC(df)
    ar = spring = sos = lps = bc = None
    reaccs = []

    if sc:
        events.append(sc)
        print(f"  ✅ SC      : {sc['date']} @ ${sc['price']}")

        ar = find_AR(df, sc)
        if ar:
            events.append(ar)
            print(f"  ✅ AR      : {ar['date']} @ ${ar['price']}")

            for st in find_STs(df, sc, ar):
                events.append(st)
                print(f"  ✅ ST      : {st['date']} @ ${st['price']}")

            spring = find_Spring(df, sc, ar)
            if spring:
                events.append(spring)
                print(f"  ✅ Spring  : {spring['date']} @ ${spring['price']}")

            # Search ENTIRE remaining chart for SOS
            sos = find_SOS(df, sc, ar)
            if sos:
                events.append(sos)
                print(f"  ✅ SOS     : {sos['date']} @ ${sos['price']}")

                lps = find_LPS(df, sc, sos)
                if lps:
                    events.append(lps)
                    print(f"  ✅ LPS     : {lps['date']} @ ${lps['price']}")

                # Find all Re-Accumulation zones in markup
                reaccs = find_reaccumulation(df, sos)
                for ra in reaccs:
                    events.append(ra)
                    print(f"  ✅ ReAcc   : {ra['date']} @ ${ra['price']:.0f}  [{ra['x0']} → {ra['x1']}]")
            else:
                print(f"  ⚠️  SOS not found — accumulation may still be in progress")
    else:
        bc = find_BC(df)
        if bc:
            events.append(bc)
            print(f"  ✅ BC      : {bc['date']} @ ${bc['price']}")

    phase, bias = determine_phase(events)
    waves = analyze_waves(df)
    phase_labels = build_phase_labels(sc, ar, spring, sos, bc, df)
    nm = [e["event"] for e in events]

    print(f"  📊 Phase: {phase} | Bias: {bias}")
    print(f"  🌊 {waves['trend_strength']}")
    print(f"  📦 Re-Acc zones found: {len(reaccs)}")

    return {
        "events":       events,
        "support_lines":[],
        "trend_lines":  [],
        "zones":        [],
        "trading_ranges":[],
        "markup_lines": [],
        "phase_labels": phase_labels,
        "phase":  phase,
        "bias":   bias,
        "wave_analysis": waves,
        "summary_ar": f"Events: {', '.join(nm)}" if nm else "No clear Wyckoff events",
        "events_found": len(events) > 0
    }
