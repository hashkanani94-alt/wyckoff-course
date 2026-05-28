"""
analysis/wyckoff_detector.py
=============================
Detects Wyckoff events based ONLY on rules from the original SMI course.
Rules source: WYCKOFF_BEHAVIOR_RULES.txt
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def avg_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return df["Volume"].rolling(window).mean()


def avg_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    return (df["High"] - df["Low"]).rolling(window).mean()


def close_position(row) -> float:
    """Returns where close is within the bar. 1.0 = top, 0.0 = bottom."""
    rng = row["High"] - row["Low"]
    if rng == 0:
        return 0.5
    return (row["Close"] - row["Low"]) / rng


# ─────────────────────────────────────────────────────────────────
#  TREND DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_trend_before(df: pd.DataFrame, idx: int, lookback: int = 15) -> str:
    """Returns 'up', 'down', or 'sideways' for bars before idx."""
    start = max(0, idx - lookback)
    window = df.iloc[start:idx]
    if len(window) < 5:
        return "sideways"
    price_change = window["Close"].iloc[-1] - window["Close"].iloc[0]
    threshold = window["Close"].mean() * 0.05
    if price_change < -threshold:
        return "down"
    elif price_change > threshold:
        return "up"
    return "sideways"


# ─────────────────────────────────────────────────────────────────
#  SELLING CLIMAX (SC)
# ─────────────────────────────────────────────────────────────────

def find_SC(df: pd.DataFrame) -> dict | None:
    """
    SC Rules (from course):
    - After prolonged downtrend (10-15 bars declining)
    - Price makes new low
    - Volume is climactic (highest or near highest)
    - Bar spread is wide
    - Close in upper half of the bar
    - Price bounces upward after
    - Only ONE SC — the absolute lowest point
    """
    avg_vol = avg_volume(df, 20)
    avg_sprd = avg_spread(df, 20)

    # Find the absolute lowest close in the dataset
    low_idx = df["Low"].idxmin()
    i = df.index.get_loc(low_idx)

    if i < 15 or i > len(df) - 5:
        return None

    row = df.iloc[i]

    # Check downtrend before
    trend = detect_trend_before(df, i, lookback=15)
    if trend != "down":
        return None

    # Check climactic volume
    vol_rank = (df["Volume"].iloc[max(0, i-30):i+1] <= row["Volume"]).mean()
    if vol_rank < 0.80:
        return None

    # Check wide spread
    spread = row["High"] - row["Low"]
    if avg_sprd.iloc[i] > 0 and spread < avg_sprd.iloc[i] * 1.2:
        return None

    # Close in upper half
    if close_position(row) < 0.5:
        return None

    # Price must bounce after
    if i + 3 >= len(df):
        return None
    next_closes = df["Close"].iloc[i+1:i+4]
    if next_closes.mean() <= row["Close"]:
        return None

    return {
        "event": "SC",
        "date": str(df.index[i].date()),
        "price": round(float(row["Low"]), 2),
        "volume": int(row["Volume"]),
        "bar_index": i
    }


# ─────────────────────────────────────────────────────────────────
#  AUTOMATIC RALLY (AR) after SC
# ─────────────────────────────────────────────────────────────────

def find_AR(df: pd.DataFrame, sc: dict) -> dict | None:
    """
    AR Rules (from course):
    - First sharp rally after SC
    - Volume LESS than SC
    - Establishes TOP boundary of Trading Range
    - Duration: 2-5 bars
    """
    if sc is None:
        return None

    sc_i = sc["bar_index"]
    sc_vol = sc["volume"]
    search = df.iloc[sc_i+1: sc_i+15]

    if search.empty:
        return None

    # Find highest high after SC in next 15 bars
    ar_loc = search["High"].idxmax()
    ar_i = df.index.get_loc(ar_loc)
    row = df.iloc[ar_i]

    # Volume should be less than SC
    if row["Volume"] >= sc_vol * 1.1:
        return None

    # Must be higher than SC close
    if row["Close"] <= df.iloc[sc_i]["Close"]:
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
# ─────────────────────────────────────────────────────────────────

def find_ST(df: pd.DataFrame, sc: dict, ar: dict) -> list[dict]:
    """
    ST Rules (from course):
    - Returns to SC area
    - Volume MUST BE LESS than SC
    - Price holds ABOVE SC low
    """
    if sc is None or ar is None:
        return []

    sc_i = sc["bar_index"]
    ar_i = ar["bar_index"]
    sc_low = sc["price"]
    sc_vol = sc["volume"]

    sts = []
    search = df.iloc[ar_i+1: ar_i+40]

    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        # Price returns near SC low (within 3%)
        if abs(row["Low"] - sc_low) / sc_low < 0.03:
            # Volume less than SC
            if row["Volume"] < sc_vol * 0.9:
                # Holds above SC low
                if row["Low"] >= sc_low * 0.98:
                    sts.append({
                        "event": f"ST",
                        "date": str(date.date()),
                        "price": round(float(row["Low"]), 2),
                        "volume": int(row["Volume"]),
                        "bar_index": i
                    })
                    if len(sts) >= 2:
                        break

    return sts


# ─────────────────────────────────────────────────────────────────
#  SPRING
# ─────────────────────────────────────────────────────────────────

def find_Spring(df: pd.DataFrame, sc: dict, ar: dict) -> dict | None:
    """
    Spring Rules (from course):
    - Breaks BELOW SC low briefly
    - Volume less than SC (Type 3 = most bullish)
    - Price immediately recovers above SC low
    - Shakeout of weak holders
    """
    if sc is None or ar is None:
        return None

    ar_i = ar["bar_index"]
    sc_low = sc["price"]
    sc_vol = sc["volume"]

    search = df.iloc[ar_i+1: ar_i+60]

    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        # Breaks below SC low
        if row["Low"] < sc_low:
            # Volume less than SC (bullish spring)
            if row["Volume"] < sc_vol:
                # Closes back above SC low
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
# ─────────────────────────────────────────────────────────────────

def find_SOS(df: pd.DataFrame, sc: dict, ar: dict) -> dict | None:
    """
    SOS Rules (from course):
    - Strong advance on widening spread and increasing volume
    - Breaks above AR high
    - Closes near the HIGH of the bar
    """
    if sc is None or ar is None:
        return None

    ar_i = ar["bar_index"]
    ar_high = ar["price"]
    avg_vol = avg_volume(df, 20)
    avg_sprd = avg_spread(df, 20)

    search = df.iloc[ar_i+1: ar_i+80]

    for date, row in search.iterrows():
        i = df.index.get_loc(date)
        # Breaks above AR high
        if row["High"] > ar_high:
            spread = row["High"] - row["Low"]
            # Wide spread
            if avg_sprd.iloc[i] > 0 and spread > avg_sprd.iloc[i]:
                # Strong volume
                if avg_vol.iloc[i] > 0 and row["Volume"] > avg_vol.iloc[i] * 1.2:
                    # Closes near high
                    if close_position(row) > 0.6:
                        return {
                            "event": "SOS",
                            "date": str(date.date()),
                            "price": round(float(row["High"]), 2),
                            "volume": int(row["Volume"]),
                            "bar_index": i
                        }
    return None


# ─────────────────────────────────────────────────────────────────
#  BUYING CLIMAX (BC) — for Distribution
# ─────────────────────────────────────────────────────────────────

def find_BC(df: pd.DataFrame) -> dict | None:
    """
    BC Rules (from course):
    - After prolonged uptrend
    - Price makes new high
    - Climactic volume
    - Wide spread
    - Closes in LOWER half of bar
    """
    avg_vol = avg_volume(df, 20)
    avg_sprd = avg_spread(df, 20)

    high_idx = df["High"].idxmax()
    i = df.index.get_loc(high_idx)

    if i < 15 or i > len(df) - 5:
        return None

    row = df.iloc[i]
    trend = detect_trend_before(df, i, lookback=15)
    if trend != "up":
        return None

    vol_rank = (df["Volume"].iloc[max(0, i-30):i+1] <= row["Volume"]).mean()
    if vol_rank < 0.80:
        return None

    spread = row["High"] - row["Low"]
    if avg_sprd.iloc[i] > 0 and spread < avg_sprd.iloc[i] * 1.2:
        return None

    # Close in LOWER half
    if close_position(row) > 0.5:
        return None

    return {
        "event": "BC",
        "date": str(df.index[i].date()),
        "price": round(float(row["High"]), 2),
        "volume": int(row["Volume"]),
        "bar_index": i
    }


# ─────────────────────────────────────────────────────────────────
#  PHASE DETERMINATION
# ─────────────────────────────────────────────────────────────────

def determine_phase(events: list[dict], df: pd.DataFrame) -> tuple[str, str]:
    """
    Determine the current Wyckoff phase and bias from detected events.
    Returns: (phase, bias)
    """
    event_names = [e["event"] for e in events]

    has_sc = "SC" in event_names
    has_bc = "BC" in event_names
    has_ar = "AR" in event_names
    has_st = "ST" in event_names
    has_spring = "Spring" in event_names
    has_sos = "SOS" in event_names

    if has_sc:
        if has_sos:
            return "E", "BULLISH"
        elif has_spring:
            return "C", "BULLISH"
        elif has_st:
            return "B", "NEUTRAL"
        elif has_ar:
            return "A", "NEUTRAL"
        else:
            return "A", "NEUTRAL"

    if has_bc:
        return "A", "BEARISH"

    # Fallback: check recent price action
    recent = df.tail(20)
    price_change = (recent["Close"].iloc[-1] - recent["Close"].iloc[0]) / recent["Close"].iloc[0]
    if price_change > 0.05:
        return "?", "BULLISH"
    elif price_change < -0.05:
        return "?", "BEARISH"
    return "?", "NEUTRAL"


# ─────────────────────────────────────────────────────────────────
#  WAVE ANALYSIS (Section 5M)
# ─────────────────────────────────────────────────────────────────

def analyze_waves(df: pd.DataFrame) -> dict:
    """
    Wave Analysis Rules (from Section 5M):
    - Reaction less than 50% = strong trend
    - Volume expands on trend waves, shrinks on corrections
    """
    closes = df["Close"].values
    volumes = df["Volume"].values

    # Simple swing detection using rolling window
    window = 5
    highs, lows = [], []

    for i in range(window, len(closes) - window):
        if closes[i] == max(closes[i-window:i+window+1]):
            highs.append((i, closes[i]))
        if closes[i] == min(closes[i-window:i+window+1]):
            lows.append((i, closes[i]))

    trend_strength = "Unknown"
    if len(highs) >= 2 and len(lows) >= 2:
        # Check if highs and lows are rising
        last_highs = [h[1] for h in highs[-3:]]
        last_lows = [l[1] for l in lows[-3:]]
        if all(last_highs[i] > last_highs[i-1] for i in range(1, len(last_highs))):
            if all(last_lows[i] > last_lows[i-1] for i in range(1, len(last_lows))):
                trend_strength = "Strong Uptrend — Rising Highs & Lows"
        elif all(last_highs[i] < last_highs[i-1] for i in range(1, len(last_highs))):
            if all(last_lows[i] < last_lows[i-1] for i in range(1, len(last_lows))):
                trend_strength = "Strong Downtrend — Falling Highs & Lows"
        else:
            trend_strength = "Sideways / Transition"

    # Volume trend on recent bars
    recent_vol = volumes[-20:]
    vol_trend = "Increasing" if recent_vol[-1] > np.mean(recent_vol) else "Decreasing"

    return {
        "trend_strength": trend_strength,
        "recent_volume_trend": vol_trend,
        "swing_highs_count": len(highs),
        "swing_lows_count": len(lows)
    }


# ─────────────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────

def detect_wyckoff_events(df: pd.DataFrame) -> dict:
    """
    Main function: runs all detectors and returns full analysis.
    Based exclusively on Wyckoff SMI Course rules.
    """
    events = []
    support_lines = []
    trend_lines = []
    zones = []

    print("🔍 Running Wyckoff event detection...")

    # Detect accumulation events
    sc = find_SC(df)
    if sc:
        events.append(sc)
        print(f"  ✅ SC found: {sc['date']} @ ${sc['price']}")

        ar = find_AR(df, sc)
        if ar:
            events.append(ar)
            print(f"  ✅ AR found: {ar['date']} @ ${ar['price']}")

            sts = find_ST(df, sc, ar)
            for st in sts:
                events.append(st)
                print(f"  ✅ ST found: {st['date']} @ ${st['price']}")

            spring = find_Spring(df, sc, ar)
            if spring:
                events.append(spring)
                print(f"  ✅ Spring found: {spring['date']} @ ${spring['price']}")

            sos = find_SOS(df, sc, ar)
            if sos:
                events.append(sos)
                print(f"  ✅ SOS found: {sos['date']} @ ${sos['price']}")

            # Support lines
            support_lines.append({
                "y": sc["price"],
                "label": f"SC Low ${sc['price']}",
                "color": "#e03c3c"
            })
            support_lines.append({
                "y": ar["price"],
                "label": f"AR High ${ar['price']}",
                "color": "#4fc3f7"
            })

            # Trading range zone
            zones.append({
                "x0": sc["date"],
                "x1": df.index[-1].strftime("%Y-%m-%d"),
                "y0": sc["price"] * 0.98,
                "y1": ar["price"] * 1.01,
                "name": "Trading Range",
                "color": "green"
            })

    # Detect distribution events
    else:
        bc = find_BC(df)
        if bc:
            events.append(bc)
            print(f"  ✅ BC found: {bc['date']} @ ${bc['price']}")
            support_lines.append({
                "y": bc["price"],
                "label": f"BC High ${bc['price']}",
                "color": "#e03c3c"
            })

    # Phase + bias
    phase, bias = determine_phase(events, df)

    # Wave analysis
    waves = analyze_waves(df)

    # Build summary
    event_names = [e["event"] for e in events]
    if event_names:
        summary = f"Events: {', '.join(event_names)}"
    else:
        summary = "No clear Wyckoff events detected"

    print(f"  📊 Phase: {phase} | Bias: {bias}")
    print(f"  🌊 Trend: {waves['trend_strength']}")

    return {
        "events": events,
        "support_lines": support_lines,
        "trend_lines": trend_lines,
        "zones": zones,
        "trading_ranges": [],
        "markup_lines": [],
        "phase": phase,
        "bias": bias,
        "wave_analysis": waves,
        "summary_ar": summary,
        "events_found": len(events) > 0
    }
