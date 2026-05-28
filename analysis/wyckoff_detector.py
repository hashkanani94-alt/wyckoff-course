"""
analysis/wyckoff_detector.py — Full Wyckoff SMI Course Detector
"""
import pandas as pd
import numpy as np


def avg_vol(df, w=20):  return df["Volume"].rolling(w).mean()
def avg_spd(df, w=20):  return (df["High"] - df["Low"]).rolling(w).mean()
def close_pos(row):
    r = row["High"] - row["Low"]
    return (row["Close"] - row["Low"]) / r if r else 0.5
def vol_rank(df, i, lb=50):
    s = max(0, i-lb)
    return (df["Volume"].iloc[s:i+1] <= df["Volume"].iloc[i]).mean()

def find_swings(df, w=10):
    hi, lo = df["High"].values, df["Low"].values
    peaks, troughs = [], []
    for i in range(w, len(df)-w):
        if hi[i] == max(hi[i-w:i+w+1]): peaks.append(i)
        if lo[i] == min(lo[i-w:i+w+1]): troughs.append(i)
    return peaks, troughs

# ── SC ────────────────────────────────────────────────────────────
def find_SC(df):
    _, troughs = find_swings(df, 10)
    avs = avg_spd(df, 20)
    best = None
    for i in troughs:
        if i < 20 or i > len(df)-5: continue
        row = df.iloc[i]
        if df["Close"].iloc[max(0,i-15)] <= row["Close"]*1.04: continue
        if vol_rank(df, i, 60) < 0.72: continue
        spread = row["High"]-row["Low"]
        if avs.iloc[i] > 0 and spread < avs.iloc[i]*0.9: continue
        if close_pos(row) < 0.40: continue
        nxt = df["Close"].iloc[i+1:i+6]
        if (nxt > row["Close"]).sum() < 2: continue
        if best is None or row["Volume"] > best["volume"]:
            best = {"event":"SC","date":str(df.index[i].date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return best

# ── AR ────────────────────────────────────────────────────────────
def find_AR(df, sc):
    if not sc: return None
    si, sv = sc["bar_index"], sc["volume"]
    seg = df.iloc[si+1:si+25]
    if seg.empty: return None
    loc = seg["High"].idxmax(); ai = df.index.get_loc(loc)
    row = df.iloc[ai]
    if row["Volume"] > sv*1.3: return None
    if row["High"] < df.iloc[si]["Low"]*1.03: return None
    return {"event":"AR","date":str(df.index[ai].date()),
            "price":round(float(row["High"]),2),
            "volume":int(row["Volume"]),"bar_index":ai}

# ── STs ───────────────────────────────────────────────────────────
def find_STs(df, sc, ar):
    if not sc or not ar: return []
    ai, sl, sv = ar["bar_index"], sc["price"], sc["volume"]
    sts = []
    for date, row in df.iloc[ai+1:ai+80].iterrows():
        i = df.index.get_loc(date)
        if abs(row["Low"]-sl)/sl < 0.06 and row["Volume"] < sv*0.88 and row["Low"] >= sl*0.96:
            sts.append({"event":"ST","date":str(date.date()),
                        "price":round(float(row["Low"]),2),
                        "volume":int(row["Volume"]),"bar_index":i})
            if len(sts) >= 3: break
    return sts

# ── Spring ────────────────────────────────────────────────────────
def find_Spring(df, sc, ar):
    if not sc or not ar: return None
    ai, sl, sv = ar["bar_index"], sc["price"], sc["volume"]
    for date, row in df.iloc[ai+1:ai+100].iterrows():
        i = df.index.get_loc(date)
        if row["Low"] < sl and row["Volume"] < sv and row["Close"] > sl:
            return {"event":"Spring","date":str(date.date()),
                    "price":round(float(row["Low"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None

# ── SOS ───────────────────────────────────────────────────────────
def find_SOS(df, sc, ar):
    if not sc or not ar: return None
    ai, ah = ar["bar_index"], ar["price"]
    avv, avs = avg_vol(df,20), avg_spd(df,20)
    for date, row in df.iloc[ai+1:ai+120].iterrows():
        i = df.index.get_loc(date)
        sp = row["High"]-row["Low"]
        if (row["High"] > ah and
            avs.iloc[i] > 0 and sp > avs.iloc[i]*0.9 and
            avv.iloc[i] > 0 and row["Volume"] > avv.iloc[i]*1.1 and
            close_pos(row) > 0.55):
            return {"event":"SOS","date":str(date.date()),
                    "price":round(float(row["High"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return None

# ── LPS ───────────────────────────────────────────────────────────
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

# ── BC ────────────────────────────────────────────────────────────
def find_BC(df):
    peaks, _ = find_swings(df, 10)
    avs = avg_spd(df, 20)
    best = None
    for i in peaks:
        if i < 20 or i > len(df)-5: continue
        row = df.iloc[i]
        if df["Close"].iloc[max(0,i-15)] >= row["Close"]*0.96: continue
        if vol_rank(df, i, 60) < 0.72: continue
        sp = row["High"]-row["Low"]
        if avs.iloc[i] > 0 and sp < avs.iloc[i]*0.9: continue
        if close_pos(row) > 0.55: continue
        if best is None or row["Volume"] > best["volume"]:
            best = {"event":"BC","date":str(df.index[i].date()),
                    "price":round(float(row["High"]),2),
                    "volume":int(row["Volume"]),"bar_index":i}
    return best

# ── PSY ───────────────────────────────────────────────────────────
def find_PSY(df, bc):
    if not bc: return None
    peaks, _ = find_swings(df, 8)
    before = [i for i in peaks if i < bc["bar_index"]-5]
    if not before: return None
    i = before[-1]; row = df.iloc[i]
    avv = avg_vol(df, 20)
    if avv.iloc[i] > 0 and row["Volume"] < avv.iloc[i]*0.8: return None
    return {"event":"PSY","date":str(df.index[i].date()),
            "price":round(float(row["High"]),2),
            "volume":int(row["Volume"]),"bar_index":i}

# ── Re-Accumulation zones ─────────────────────────────────────────
def find_reaccumulation(df, sos):
    if not sos: return []
    results, avv, w = [], avg_vol(df, 20), 25
    i = sos["bar_index"] + 25
    while i < len(df) - w:
        chunk = df.iloc[i:i+w]
        rng = (chunk["High"].max()-chunk["Low"].min())/chunk["Close"].mean()
        av = avv.iloc[i+w//2] if avv.iloc[i+w//2] > 0 else 1
        if rng < 0.07 and chunk["Volume"].mean() < av*0.85:
            results.append({
                "event":"ReAcc",
                "date":str(df.index[i+w//2].date()),
                "price":round(float(chunk["Close"].mean()),2),
                "volume":int(chunk["Volume"].mean()),
                "bar_index":i+w//2,
                "x0": str(df.index[i].date()),
                "x1": str(df.index[min(i+w, len(df)-1)].date()),
                "y0": round(float(chunk["Low"].min()*0.995),2),
                "y1": round(float(chunk["High"].max()*1.005),2),
            })
            i += w
            if len(results) >= 3: break
        else:
            i += 10
    return results

# ── Phase labels for chart ────────────────────────────────────────
def build_phase_labels(events, sc, ar, spring, sos, bc, df):
    """Returns list of phase band annotations for the chart."""
    labels = []
    last = str(df.index[-1].date())

    if sc and ar:
        labels.append({"text":"Phase A","x0":sc["date"],"x1":ar["date"],"color":"#e53935"})
    if ar and spring:
        labels.append({"text":"Phase B","x0":ar["date"],"x1":spring["date"],"color":"#fb8c00"})
    if spring and sos:
        labels.append({"text":"Phase C","x0":spring["date"],"x1":sos["date"],"color":"#fdd835"})
    if sos:
        labels.append({"text":"Phase D→E (Markup)","x0":sos["date"],"x1":last,"color":"#43a047"})
    return labels

# ── Trend lines ───────────────────────────────────────────────────
def build_trend_lines(events, sc, ar, spring, sos, lps):
    tls = []
    # Demand line: SC → Spring or SC → LPS
    p2 = spring or lps
    if sc and p2:
        tls.append({"x0":sc["date"],"y0":sc["price"],
                    "x1":p2["date"],"y1":p2["price"],
                    "label":"Demand Line","style":"demand"})
    # Supply line: AR → SOS area (resistance)
    if ar and sos:
        tls.append({"x0":ar["date"],"y0":ar["price"],
                    "x1":sos["date"],"y1":sos["price"],
                    "label":"Supply Line","style":"supply"})
    return tls

# ── Support lines ─────────────────────────────────────────────────
def build_support_lines(events):
    color_map = {"SC":"#e03c3c","AR":"#4fc3f7","Spring":"#a5d6a7",
                 "SOS":"#43a047","LPS":"#66bb6a","BC":"#ff7043","SOW":"#ef5350"}
    label_map = {"SC":"SC Support","AR":"AR Resistance","Spring":"Spring Low",
                 "SOS":"SOS Breakout","LPS":"LPS Support","BC":"BC High","SOW":"SOW"}
    lines = []
    for e in events:
        ev = e["event"]
        if ev in color_map:
            lines.append({"y":e["price"],"label":f'{label_map[ev]} ${e["price"]:.1f}',
                          "color":color_map[ev]})
    return lines

# ── Zones ─────────────────────────────────────────────────────────
def build_zones(sc, ar, spring, sos, reaccs, df):
    zones = []
    last = str(df.index[-1].date())
    if sc and ar:
        end = sos["date"] if sos else last
        zones.append({"x0":sc["date"],"x1":end,
                      "y0":min(sc["price"],getattr(spring,"__class__",type("",(),{"price":sc["price"]})).__dict__.get("price",sc["price"]) if spring else sc["price"])*0.995,
                      "y1":ar["price"]*1.005,"name":"Accumulation Zone","color":"green"})
    for ra in reaccs:
        zones.append({"x0":ra["x0"],"x1":ra["x1"],
                      "y0":ra["y0"],"y1":ra["y1"],
                      "name":"Re-Accumulation","color":"green"})
    return zones

# ── Wave analysis ─────────────────────────────────────────────────
def analyze_waves(df):
    c = df["Close"].values; w = 5
    hi_idx = [i for i in range(w,len(c)-w) if c[i]==max(c[i-w:i+w+1])]
    lo_idx = [i for i in range(w,len(c)-w) if c[i]==min(c[i-w:i+w+1])]
    trend = "Unknown"
    if len(hi_idx)>=2 and len(lo_idx)>=2:
        lh=[c[i] for i in hi_idx[-3:]]; ll=[c[i] for i in lo_idx[-3:]]
        rh = all(lh[i]>lh[i-1] for i in range(1,len(lh)))
        rl = all(ll[i]>ll[i-1] for i in range(1,len(ll)))
        fh = all(lh[i]<lh[i-1] for i in range(1,len(lh)))
        fl = all(ll[i]<ll[i-1] for i in range(1,len(ll)))
        if rh and rl:   trend = "Strong Uptrend — Rising Highs & Lows"
        elif fh and fl: trend = "Strong Downtrend — Falling Highs & Lows"
        elif rh:        trend = "Possible Accumulation — Rising Highs"
        else:           trend = "Sideways / Transition"
    rv = df["Volume"].values[-20:]
    return {"trend_strength":trend,
            "recent_volume_trend":"Increasing" if rv[-1]>np.mean(rv) else "Decreasing"}

# ── Phase determination ───────────────────────────────────────────
def determine_phase(events):
    nm = [e["event"] for e in events]
    if "SOS" in nm or "LPS" in nm: return "D-E","BULLISH"
    if "Spring" in nm: return "C","BULLISH"
    if "ST" in nm and "SC" in nm: return "B","NEUTRAL"
    if "AR" in nm and "SC" in nm: return "A","NEUTRAL"
    if "BC" in nm: return "A","BEARISH"
    return "?","NEUTRAL"

# ── MAIN ──────────────────────────────────────────────────────────
def detect_wyckoff_events(df):
    events = []
    print("🔍 Running Wyckoff event detection...")

    sc = find_SC(df)
    ar = spring = sos = lps = bc = psy = None
    reaccs = []

    if sc:
        events.append(sc); print(f"  ✅ SC     : {sc['date']} @ ${sc['price']}")
        ar = find_AR(df, sc)
        if ar:
            events.append(ar); print(f"  ✅ AR     : {ar['date']} @ ${ar['price']}")
            for st in find_STs(df, sc, ar):
                events.append(st); print(f"  ✅ ST     : {st['date']} @ ${st['price']}")
            spring = find_Spring(df, sc, ar)
            if spring:
                events.append(spring); print(f"  ✅ Spring : {spring['date']} @ ${spring['price']}")
            sos = find_SOS(df, sc, ar)
            if sos:
                events.append(sos); print(f"  ✅ SOS    : {sos['date']} @ ${sos['price']}")
                lps = find_LPS(df, sc, sos)
                if lps:
                    events.append(lps); print(f"  ✅ LPS    : {lps['date']} @ ${lps['price']}")
                reaccs = find_reaccumulation(df, sos)
                for ra in reaccs:
                    events.append(ra); print(f"  ✅ ReAcc  : {ra['date']} @ ${ra['price']}")
    else:
        bc = find_BC(df)
        if bc:
            events.append(bc); print(f"  ✅ BC     : {bc['date']} @ ${bc['price']}")
            psy = find_PSY(df, bc)
            if psy:
                events.append(psy); print(f"  ✅ PSY    : {psy['date']} @ ${psy['price']}")

    phase, bias = determine_phase(events)
    waves       = analyze_waves(df)
    zones       = build_zones(sc, ar, spring, sos, reaccs, df)
    support_lines = build_support_lines(events)
    trend_lines   = build_trend_lines(events, sc, ar, spring, sos, lps)
    phase_labels  = build_phase_labels(events, sc, ar, spring, sos, bc, df)
    nm = [e["event"] for e in events]

    print(f"  📊 Phase: {phase} | Bias: {bias}")
    print(f"  🌊 {waves['trend_strength']}")

    return {
        "events":        events,
        "support_lines": support_lines,
        "trend_lines":   trend_lines,
        "zones":         zones,
        "trading_ranges":[],
        "markup_lines":  [],
        "phase_labels":  phase_labels,
        "phase":  phase,
        "bias":   bias,
        "wave_analysis": waves,
        "summary_ar": f"Events: {', '.join(nm)}" if nm else "No clear Wyckoff events",
        "events_found": len(events) > 0
    }
