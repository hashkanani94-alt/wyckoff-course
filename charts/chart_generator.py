import io
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

EVENT_COLORS = {
    "BC":"#e03c3c","PSY":"#e03c3c","UT":"#ff7043","UTAD":"#ff7043",
    "LPSY":"#ff8a65","SOW":"#ef5350","MSOW":"#b71c1c","COB":"#f48fb1",
    "MCOB":"#c62828","AR":"#4fc3f7","SOS":"#43a047","MSOS":"#1b5e20",
    "JOC":"#00bcd4","SC":"#e03c3c","LPS":"#66bb6a","Spring":"#a5d6a7",
    "BU":"#81c784","mSOW":"#e57373","PS":"#ff8a65","ST":"#90caf9",
}
ZONE_COLORS = {
    "red":("#c62828",0.13),"orange":("#e65100",0.13),"green":("#1b5e20",0.13),
}
# Events that go in TOP band (peaks)
TOP_EVENTS = {"BC","PSY","UT","UTAD","LPSY","SOW","MSOW","COB","MCOB","AR","SOS","MSOS","JOC","UTA"}
# Events that go in BOTTOM band (troughs)
BOT_EVENTS = {"SC","LPS","Spring","BU","mSOW","PS","ST"}

def draw_chart(ticker, df, analysis):
    events         = analysis.get("events", [])
    zones          = analysis.get("zones", [])
    supports       = analysis.get("support_lines", [])
    trends         = analysis.get("trend_lines", [])
    trading_ranges = analysis.get("trading_ranges", [])
    markup_lines   = analysis.get("markup_lines", [])
    bias           = analysis.get("bias", "?").upper()
    phase          = analysis.get("phase", "?")
    summary        = analysis.get("summary_ar", "")

    n           = len(df)
    date_to_idx = {str(d): i for i, d in enumerate(df.index)}

    def nearest(s):
        ts = pd.Timestamp(s)
        return min(range(n), key=lambda i: abs((df.index[i] - ts).days))

    def bx(s):
        return date_to_idx.get(s, nearest(s))

    # ── Style ────────────────────────────────────────────────────────────────
    mc = mpf.make_marketcolors(
        up="#2eb872", down="#e03c3c",
        edge={"up":"#2eb872","down":"#e03c3c"},
        wick={"up":"#2eb872","down":"#e03c3c"},
        volume={"up":"#2eb87240","down":"#e03c3c40"})
    style = mpf.make_mpf_style(
        marketcolors=mc, facecolor="#0d1421", figcolor="#0a0e17",
        gridcolor="#1a2540", gridstyle="--", y_on_right=True,
        rc={"axes.labelcolor":"#c8d6e5","axes.edgecolor":"#1e2d40",
            "xtick.color":"#8899aa","ytick.color":"#8899aa",
            "font.family":"monospace","font.size":9})

    fig, axes = mpf.plot(
        df, type="candle", style=style, volume=True,
        figsize=(20,11), returnfig=True, panel_ratios=(4,1),
        tight_layout=False, warn_too_much_data=10000)
    ax = axes[0]

    # ── Price range ──────────────────────────────────────────────────────────
    y_max   = df["High"].max()
    y_min   = df["Low"].min()
    y_range = y_max - y_min

    # ════════════════════════════════════════════════════════════════════════
    #  BAND SYSTEM
    #  TOP  BAND: y_max + 3%  and  y_max + 10%   ← peaks go here
    #  BOT  BAND: y_min - 7%  and  y_min - 14%   ← troughs go here
    #  Chart ylim extended so bands are visible
    #  Price bars are ALWAYS between y_min and y_max → bands never overlap
    # ════════════════════════════════════════════════════════════════════════
    TOP_R0 = y_max + y_range * 0.04    # row 0 top band
    TOP_R1 = y_max + y_range * 0.11   # row 1 top band
    BOT_R0 = y_min - y_range * 0.07   # row 0 bot band
    BOT_R1 = y_min - y_range * 0.14   # row 1 bot band
    ax.set_ylim(BOT_R1 - y_range * 0.03, TOP_R1 + y_range * 0.03)

    def place_label(bar_idx, band_y, dot_y, text, col):
        """Dot on exact price, dashed line, label in band."""
        # dot on the candle high or low
        ax.plot(bar_idx, dot_y, "o", color=col,
                markersize=4, markeredgecolor="#0d1421",
                markeredgewidth=0.5, zorder=9)
        # dashed vertical connector
        y1, y2 = (dot_y, band_y) if dot_y < band_y else (band_y, dot_y)
        ax.plot([bar_idx, bar_idx], [y1, y2],
                color=col, linewidth=0.7, linestyle=":",
                alpha=0.65, zorder=4)
        # label box INSIDE band
        ax.text(bar_idx, band_y, text,
                color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=col, edgecolor="none", alpha=0.93))

    # ── Draw events in bands ─────────────────────────────────────────────────
    peak_evts   = []
    trough_evts = []
    for e in events:
        evt = e.get("event","")
        if e.get("type") == "peak" or evt in TOP_EVENTS:
            peak_evts.append(e)
        else:
            trough_evts.append(e)

    for i, e in enumerate(peak_evts):
        b   = bx(e["date"])
        col = EVENT_COLORS.get(e["event"], "#90caf9")
        lbl = f'{e.get("number","")} {e["event"]}\n${e["price"]:.1f}'
        # Use actual bar HIGH as dot position
        dot_y   = float(df["High"].iloc[b])
        band_y  = TOP_R0 if i % 2 == 0 else TOP_R1
        place_label(b, band_y, dot_y, lbl, col)

    for i, e in enumerate(trough_evts):
        b   = bx(e["date"])
        col = EVENT_COLORS.get(e["event"], "#90caf9")
        lbl = f'{e.get("number","")} {e["event"]}\n${e["price"]:.1f}'
        # Use actual bar LOW as dot position
        dot_y  = float(df["Low"].iloc[b])
        band_y = BOT_R0 if i % 2 == 0 else BOT_R1
        place_label(b, band_y, dot_y, lbl, col)

    # ── Auto horizontal lines from key events ────────────────────────────────
    def hline(price, color, label):
        ax.axhline(price, color=color, linewidth=1.0,
                   linestyle="--", alpha=0.5, zorder=2)
        ax.text(n - 1, price, f"  {label}  ",
                color=color, fontsize=7, va="center", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1421",
                          edgecolor=color, linewidth=0.6, alpha=0.85))

    sc_evts     = [e for e in events if e["event"] == "SC"]
    ar_evts     = [e for e in events if e["event"] == "AR"]
    bc_evts     = [e for e in events if e["event"] == "BC"]
    spring_evts = [e for e in events if e["event"] == "Spring"]
    lps_evts    = [e for e in events if e["event"] == "LPS"]
    ut_evts     = [e for e in events if e["event"] in ["UT","UTAD"]]
    lpsy_evts   = [e for e in events if e["event"] == "LPSY"]

    for e in sc_evts:
        hline(e["price"], "#e03c3c", f'SC Low ${e["price"]:.1f}')
    for e in ar_evts:
        hline(e["price"], "#4fc3f7", f'AR High ${e["price"]:.1f}')
    for e in bc_evts:
        hline(e["price"], "#ff7043", f'BC High ${e["price"]:.1f}')
    for e in spring_evts:
        hline(e["price"], "#a5d6a7", f'Spring ${e["price"]:.1f}')
    for e in lps_evts:
        hline(e["price"], "#66bb6a", f'LPS ${e["price"]:.1f}')

    # Demand line: SC → Spring or LPS
    if sc_evts and (spring_evts or lps_evts):
        p1 = sc_evts[0]
        p2 = (spring_evts + lps_evts)[0]
        x0, x1 = bx(p1["date"]), bx(p2["date"])
        ax.plot([x0, x1], [p1["price"], p2["price"]],
                color="#2eb872", linewidth=1.5, linestyle="-",
                alpha=0.7, zorder=3)
        ax.text((x0+x1)//2, (p1["price"]+p2["price"])/2,
                " Demand Line ", color="white", fontsize=7,
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#2eb872",
                          edgecolor="none", alpha=0.8))

    # Supply line: BC → UT or LPSY
    if bc_evts and (ut_evts or lpsy_evts):
        p1 = bc_evts[0]
        p2 = (ut_evts + lpsy_evts)[0]
        x0, x1 = bx(p1["date"]), bx(p2["date"])
        ax.plot([x0, x1], [p1["price"], p2["price"]],
                color="#e03c3c", linewidth=1.5, linestyle="-",
                alpha=0.7, zorder=3)
        ax.text((x0+x1)//2, (p1["price"]+p2["price"])/2,
                " Supply Line ", color="white", fontsize=7,
                ha="center", va="top", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e03c3c",
                          edgecolor="none", alpha=0.8))

    # ── Trading Ranges ────────────────────────────────────────────────────────
    for tr in trading_ranges:
        x0 = bx(tr["x0"]); x1 = bx(tr["x1"])
        y0 = tr.get("y_bottom", tr.get("y0", 0))
        y1 = tr.get("y_top",    tr.get("y1", 0))
        ax.fill_betweenx([y0, y1], x0, x1,
                          alpha=0.08, color="#1565c0", zorder=1)
        ax.hlines([y0, y1], x0, x1,
                   colors="#1565c0", linewidths=1.2,
                   alpha=0.6, zorder=2)
        ax.text((x0+x1)//2, y1, f' {tr["name"]} ',
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1565c0", edgecolor="none", alpha=0.88))

    # ── Markup lines ──────────────────────────────────────────────────────────
    for ml in markup_lines:
        x0 = bx(ml["x0"]); x1 = bx(ml["x1"])
        ax.plot([x0, x1], [ml["y0"], ml["y1"]],
                color="#f5c842", linewidth=1.8,
                linestyle="--", alpha=0.75, zorder=3)
        ax.text((x0+x1)//2, (ml["y0"]+ml["y1"])/2, ml["label"],
                color="#0a0e17", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#f5c842", edgecolor="none", alpha=0.92))

    # ── Zones ─────────────────────────────────────────────────────────────────
    for z in zones:
        x0 = bx(z["x0"]); x1 = bx(z["x1"])
        col, alpha = ZONE_COLORS.get(z.get("color","red"), ("#c62828",0.13))
        ax.fill_betweenx([z["y0"], z["y1"]], x0, x1,
                          alpha=alpha, color=col, zorder=1)
        ax.hlines([z["y0"], z["y1"]], x0, x1,
                   colors=col, linewidths=1.1, alpha=0.65, zorder=2)
        ax.text(x0 + 0.5, z["y1"], f' {z["name"]} ',
                color="white", fontsize=7, fontweight="bold",
                va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=col, edgecolor="none", alpha=0.85))

    # ── Extra support lines ───────────────────────────────────────────────────
    for sl in supports:
        ax.axhline(sl["y"], linestyle="--", linewidth=1.0,
                   color=sl.get("color","#f5c842"), alpha=0.5, zorder=2)
        ax.text(n - 1, sl["y"], f'  {sl["label"]}  ',
                color=sl.get("color","#f5c842"), fontsize=7,
                va="center", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1421",
                          edgecolor=sl.get("color","#f5c842"),
                          linewidth=0.6, alpha=0.85))

    # ── Trend lines ───────────────────────────────────────────────────────────
    for tl in trends:
        x0 = bx(tl["x0"]); x1 = bx(tl["x1"])
        col = "#e03c3c" if tl.get("style") == "supply" else "#2eb872"
        ax.plot([x0, x1], [tl["y0"], tl["y1"]],
                color=col, linewidth=1.4,
                linestyle="--", alpha=0.65, zorder=3)
        lbl = tl.get("label","")
        if lbl:
            ax.text((x0+x1)//2, (tl["y0"]+tl["y1"])/2,
                    f' {lbl} ', color="white", fontsize=7,
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor=col, edgecolor="none", alpha=0.75))

    # ── Phase Labels (A, B, C, D-E) ──────────────────────────────────────────
    phase_labels = analysis.get("phase_labels", [])
    ax2 = axes[0]
    y_phase = BOT_R1 - y_range * 0.01   # just below bottom band

    for pl in phase_labels:
        x0 = bx(pl["x0"]); x1 = bx(pl["x1"])
        col = pl.get("color", "#4fc3f7")
        xmid = (x0 + x1) // 2
        # Shaded vertical band
        ax2.axvspan(x0, x1, alpha=0.06, color=col, zorder=0)
        # Top divider line
        ax2.axvline(x0, color=col, linewidth=1.0, linestyle=":", alpha=0.5, zorder=2)
        # Phase label at bottom
        ax2.text(xmid, y_phase, pl["text"],
                 color=col, fontsize=8, fontweight="bold",
                 ha="center", va="center", zorder=10,
                 bbox=dict(boxstyle="round,pad=0.35",
                           facecolor="#0d1421", edgecolor=col,
                           linewidth=1.2, alpha=0.95))

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.12, 0.97,
             f"  {ticker}  —  Wyckoff Analysis  [{bias}  Phase {phase}]  ",
             color="#f5c842", fontsize=13, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#0f1e35", edgecolor="#1e2d40"))
    if summary:
        fig.text(0.88, 0.97, summary,
                 color="#c8d6e5", fontsize=9, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="#111827", edgecolor="#1e2d40", alpha=0.85))

    # ── Save ──────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180,
                bbox_inches="tight", facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return buf
