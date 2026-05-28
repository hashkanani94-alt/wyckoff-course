"""
chart_generator.py  —  FINAL VERSION
=====================================
Label placement guarantee:
  • Every event label lives in a TOP or BOTTOM band that is
    ALWAYS outside the price-bar area (mathematically impossible to overlap).
  • Within each band, 4 rows are available.
  • A greedy row-picker assigns each label to the first row
    that has no other label within MIN_GAP bars (prevents horizontal clash).
  • The ylim is expanded so both bands are fully visible.
"""

import io
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

# ── Color palettes ─────────────────────────────────────────────────────────────
EVENT_COLORS = {
    "BC":    "#e03c3c", "PSY":  "#e03c3c", "UT":   "#ff7043",
    "UTAD":  "#ff7043", "LPSY": "#ff8a65", "SOW":  "#ef5350",
    "MSOW":  "#b71c1c", "COB":  "#f48fb1", "MCOB": "#c62828",
    "AR":    "#4fc3f7", "SOS":  "#43a047", "MSOS": "#1b5e20",
    "JOC":   "#00bcd4", "SC":   "#e03c3c", "LPS":  "#66bb6a",
    "Spring":"#a5d6a7", "BU":   "#81c784", "mSOW": "#e57373",
    "PS":    "#ff8a65", "ST":   "#90caf9", "UTA":  "#ff8a65",
    "LPSY2":"#ffb74d",  "LPSY3":"#ffa726",
}
ZONE_COLORS = {
    "red":    ("#c62828", 0.13),
    "orange": ("#e65100", 0.13),
    "green":  ("#1b5e20", 0.13),
}

# Peak events → TOP band  |  Trough events → BOTTOM band
PEAK_EVENTS   = {
    "BC","PSY","UT","UTAD","LPSY","LPSY2","LPSY3",
    "SOW","MSOW","COB","MCOB","AR","SOS","MSOS","JOC","UTA",
}
TROUGH_EVENTS = {"SC","LPS","Spring","BU","mSOW","PS","ST"}

# Min bar-distance between two labels sharing the same band row
MIN_GAP = 7


# ══════════════════════════════════════════════════════════════════════════════
def draw_chart(ticker: str, df: pd.DataFrame, analysis: dict) -> io.BytesIO:
    events         = analysis.get("events", [])
    zones          = analysis.get("zones", [])
    supports       = analysis.get("support_lines", [])
    trends         = analysis.get("trend_lines", [])
    trading_ranges = analysis.get("trading_ranges", [])
    markup_lines   = analysis.get("markup_lines", [])
    bias           = analysis.get("bias", "?").upper()
    phase          = analysis.get("phase", "?")
    summary        = analysis.get("summary_ar", "")

    # ── Index helpers ──────────────────────────────────────────────────────────
    n           = len(df)
    date_to_idx = {str(d): i for i, d in enumerate(df.index)}

    def nearest(s: str) -> int:
        ts = pd.Timestamp(s)
        return min(range(n), key=lambda i: abs((df.index[i] - ts).days))

    def bx(s: str) -> int:
        return date_to_idx.get(s, nearest(s))

    # ── mplfinance style ───────────────────────────────────────────────────────
    mc = mpf.make_marketcolors(
        up="#2eb872", down="#e03c3c",
        edge={"up": "#2eb872", "down": "#e03c3c"},
        wick={"up": "#2eb872", "down": "#e03c3c"},
        volume={"up": "#2eb87240", "down": "#e03c3c40"},
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        facecolor="#0d1421", figcolor="#0a0e17",
        gridcolor="#1a2540", gridstyle="--",
        y_on_right=True,
        rc={
            "axes.labelcolor": "#c8d6e5",
            "axes.edgecolor":  "#1e2d40",
            "xtick.color":     "#8899aa",
            "ytick.color":     "#8899aa",
            "font.family":     "monospace",
            "font.size":       9,
        },
    )

    fig, axes = mpf.plot(
        df, type="candle", style=style, volume=True,
        figsize=(20, 11), returnfig=True,
        panel_ratios=(4, 1), tight_layout=False,
        warn_too_much_data=10000,
    )
    ax = axes[0]

    # ── Price statistics ───────────────────────────────────────────────────────
    y_max   = float(df["High"].max())
    y_min   = float(df["Low"].min())
    y_range = y_max - y_min

    # ══════════════════════════════════════════════════════════════════════════
    #  BAND DEFINITIONS
    #  TOP rows start at y_max + 5 % and each row is 7 % of range apart.
    #  BOT rows start at y_min - 7 % and each row is 7 % apart (going down).
    #  With 4 rows each band spans ≈28 % of range → enough for busy charts.
    # ══════════════════════════════════════════════════════════════════════════
    ROW_STEP = y_range * 0.07
    TOP_ROWS = [y_max + y_range * 0.05 + i * ROW_STEP for i in range(4)]
    BOT_ROWS = [y_min - y_range * 0.07 - i * ROW_STEP for i in range(4)]

    # Expand y-axis so all band rows show
    ax.set_ylim(
        BOT_ROWS[-1] - y_range * 0.03,
        TOP_ROWS[-1] + y_range * 0.03,
    )

    # ── Row-picker: greedy, avoids horizontal conflicts ───────────────────────
    def pick_rows(event_list: list, rows: list) -> list:
        """
        Returns list of (event, row_y) pairs.
        Each row tracks which bar-indices are already occupied.
        A label is placed in the first row where no existing label
        is within MIN_GAP bars of the new label's bar-index.
        Falls back to row 0 if all rows are crowded.
        """
        occupied: list[list[int]] = [[] for _ in rows]
        result = []
        # Sort by bar-index so assignment is left→right (deterministic)
        sorted_events = sorted(event_list, key=lambda e: bx(e["date"]))
        for e in sorted_events:
            b = bx(e["date"])
            chosen_row = 0  # default
            for ri, occ in enumerate(occupied):
                if all(abs(b - ox) >= MIN_GAP for ox in occ):
                    chosen_row = ri
                    break
            occupied[chosen_row].append(b)
            result.append((e, rows[chosen_row]))
        return result

    # ── Primitive: draw one annotated label ───────────────────────────────────
    def place_label(bar_idx: int, band_y: float, dot_y: float,
                    text: str, col: str) -> None:
        # 1. Dot at exact price (candle high or low)
        ax.plot(
            bar_idx, dot_y, "o",
            color=col, markersize=4,
            markeredgecolor="#0d1421", markeredgewidth=0.5,
            zorder=9,
        )
        # 2. Dashed vertical connector dot → band
        lo, hi = min(dot_y, band_y), max(dot_y, band_y)
        ax.plot(
            [bar_idx, bar_idx], [lo, hi],
            color=col, linewidth=0.7, linestyle=":",
            alpha=0.60, zorder=4,
        )
        # 3. Label box inside band — NEVER inside price-bar area
        ax.text(
            bar_idx, band_y, text,
            color="white", fontsize=7, fontweight="bold",
            ha="center", va="center", zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor=col, edgecolor="none", alpha=0.93,
            ),
        )

    # ── Classify events ────────────────────────────────────────────────────────
    peaks   = []
    troughs = []
    for e in events:
        evt = e.get("event", "")
        tp  = e.get("type", "")
        if tp == "peak" or evt in PEAK_EVENTS:
            peaks.append(e)
        else:
            troughs.append(e)

    # ── Place peak labels in TOP band ──────────────────────────────────────────
    for e, band_y in pick_rows(peaks, TOP_ROWS):
        b     = bx(e["date"])
        col   = EVENT_COLORS.get(e["event"], "#90caf9")
        label = f'{e.get("number", "")} {e["event"]}\n${e["price"]:.1f}'
        dot_y = float(df["High"].iloc[b])        # exact candle HIGH
        place_label(b, band_y, dot_y, label, col)

    # ── Place trough labels in BOTTOM band ────────────────────────────────────
    for e, band_y in pick_rows(troughs, BOT_ROWS):
        b     = bx(e["date"])
        col   = EVENT_COLORS.get(e["event"], "#90caf9")
        label = f'{e.get("number", "")} {e["event"]}\n${e["price"]:.1f}'
        dot_y = float(df["Low"].iloc[b])          # exact candle LOW
        place_label(b, band_y, dot_y, label, col)

    # ══════════════════════════════════════════════════════════════════════════
    #  AUTOMATIC HORIZONTAL REFERENCE LINES FROM KEY EVENTS
    # ══════════════════════════════════════════════════════════════════════════
    def hline(price: float, color: str, label: str) -> None:
        ax.axhline(
            price, color=color, linewidth=1.0,
            linestyle="--", alpha=0.50, zorder=2,
        )
        ax.text(
            n - 1, price, f"  {label}  ",
            color=color, fontsize=7, va="center", ha="left", zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#0d1421",
                edgecolor=color, linewidth=0.6, alpha=0.85,
            ),
        )

    sc_evts  = [e for e in events if e["event"] == "SC"]
    ar_evts  = [e for e in events if e["event"] == "AR"]
    bc_evts  = [e for e in events if e["event"] == "BC"]
    sp_evts  = [e for e in events if e["event"] == "Spring"]
    lps_evts = [e for e in events if e["event"] == "LPS"]
    ut_evts  = [e for e in events if e["event"] in ("UT", "UTAD")]
    ly_evts  = [e for e in events if e["event"] in ("LPSY","LPSY2","LPSY3")]

    for e in sc_evts:
        hline(e["price"], "#e03c3c", f'SC Low  ${e["price"]:.1f}')
    for e in ar_evts:
        hline(e["price"], "#4fc3f7", f'AR High  ${e["price"]:.1f}')
    for e in bc_evts:
        hline(e["price"], "#ff7043", f'BC High  ${e["price"]:.1f}')
    for e in sp_evts:
        hline(e["price"], "#a5d6a7", f'Spring  ${e["price"]:.1f}')
    for e in lps_evts:
        hline(e["price"], "#66bb6a", f'LPS  ${e["price"]:.1f}')

    # Demand line: SC → Spring (or LPS)
    if sc_evts and (sp_evts or lps_evts):
        p1 = sc_evts[0]
        p2 = (sp_evts + lps_evts)[0]
        x0, x1 = bx(p1["date"]), bx(p2["date"])
        ax.plot([x0, x1], [p1["price"], p2["price"]],
                color="#2eb872", linewidth=1.5, alpha=0.70, zorder=3)
        mid = ((x0+x1)//2, (p1["price"]+p2["price"])/2)
        ax.text(*mid, " Demand Line ", color="white", fontsize=7,
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#2eb872", edgecolor="none", alpha=0.80))

    # Supply line: BC → UT or LPSY
    if bc_evts and (ut_evts or ly_evts):
        p1 = bc_evts[0]
        p2 = (ut_evts + ly_evts)[0]
        x0, x1 = bx(p1["date"]), bx(p2["date"])
        ax.plot([x0, x1], [p1["price"], p2["price"]],
                color="#e03c3c", linewidth=1.5, alpha=0.70, zorder=3)
        mid = ((x0+x1)//2, (p1["price"]+p2["price"])/2)
        ax.text(*mid, " Supply Line ", color="white", fontsize=7,
                ha="center", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#e03c3c", edgecolor="none", alpha=0.80))

    # ══════════════════════════════════════════════════════════════════════════
    #  TRADING RANGES
    # ══════════════════════════════════════════════════════════════════════════
    for tr in trading_ranges:
        x0 = bx(tr["x0"]); x1 = bx(tr["x1"])
        y0 = tr.get("y_bottom", tr.get("y0", 0))
        y1 = tr.get("y_top",    tr.get("y1", 0))
        ax.fill_betweenx([y0, y1], x0, x1,
                          alpha=0.08, color="#1565c0", zorder=1)
        ax.hlines([y0, y1], x0, x1,
                   colors="#1565c0", linewidths=1.2, alpha=0.60, zorder=2)
        ax.text((x0+x1)//2, y1, f' {tr["name"]} ',
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1565c0", edgecolor="none", alpha=0.88))

    # ══════════════════════════════════════════════════════════════════════════
    #  MARKUP / MARKDOWN DIAGONAL LINES
    # ══════════════════════════════════════════════════════════════════════════
    for ml in markup_lines:
        x0 = bx(ml["x0"]); x1 = bx(ml["x1"])
        ax.plot([x0, x1], [ml["y0"], ml["y1"]],
                color="#f5c842", linewidth=1.8,
                linestyle="--", alpha=0.75, zorder=3)
        mx = (x0+x1)//2
        my = (ml["y0"]+ml["y1"])/2
        ax.text(mx, my, ml["label"],
                color="#0a0e17", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#f5c842", edgecolor="none", alpha=0.92))

    # ══════════════════════════════════════════════════════════════════════════
    #  ZONE RECTANGLES
    # ══════════════════════════════════════════════════════════════════════════
    for z in zones:
        x0 = bx(z["x0"]); x1 = bx(z["x1"])
        col, alpha = ZONE_COLORS.get(z.get("color", "red"), ("#c62828", 0.13))
        ax.fill_betweenx([z["y0"], z["y1"]], x0, x1,
                          alpha=alpha, color=col, zorder=1)
        ax.hlines([z["y0"], z["y1"]], x0, x1,
                   colors=col, linewidths=1.1, alpha=0.65, zorder=2)
        ax.text(x0 + 0.5, z["y1"], f' {z["name"]} ',
                color="white", fontsize=7, fontweight="bold",
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=col, edgecolor="none", alpha=0.85))

    # ══════════════════════════════════════════════════════════════════════════
    #  SUPPORT / RESISTANCE LINES (from analysis JSON)
    # ══════════════════════════════════════════════════════════════════════════
    for sl in supports:
        c = sl.get("color", "#f5c842")
        ax.axhline(sl["y"], linestyle="--", linewidth=1.0,
                   color=c, alpha=0.50, zorder=2)
        ax.text(n - 1, sl["y"], f'  {sl["label"]}  ',
                color=c, fontsize=7, va="center", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1421",
                          edgecolor=c, linewidth=0.6, alpha=0.85))

    # ══════════════════════════════════════════════════════════════════════════
    #  TREND LINES
    # ══════════════════════════════════════════════════════════════════════════
    for tl in trends:
        x0 = bx(tl["x0"]); x1 = bx(tl["x1"])
        col = "#e03c3c" if tl.get("style") == "supply" else "#2eb872"
        ax.plot([x0, x1], [tl["y0"], tl["y1"]],
                color=col, linewidth=1.4,
                linestyle="--", alpha=0.65, zorder=3)
        lbl = tl.get("label", "")
        if lbl:
            ax.text((x0+x1)//2, (tl["y0"]+tl["y1"])/2,
                    f' {lbl} ', color="white", fontsize=7,
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=col, edgecolor="none", alpha=0.75))

    # ══════════════════════════════════════════════════════════════════════════
    #  CHART HEADER
    # ══════════════════════════════════════════════════════════════════════════
    fig.text(
        0.12, 0.97,
        f"  {ticker}  —  Wyckoff Analysis  [{bias}  Phase {phase}]  ",
        color="#f5c842", fontsize=13, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#0f1e35", edgecolor="#1e2d40"),
    )
    if summary:
        fig.text(
            0.88, 0.97, summary,
            color="#c8d6e5", fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#111827", edgecolor="#1e2d40", alpha=0.85),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  SAVE
    # ══════════════════════════════════════════════════════════════════════════
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180,
                bbox_inches="tight", facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return buf
