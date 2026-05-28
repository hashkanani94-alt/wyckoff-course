import io
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EVENT_COLORS = {
    "BC":"#e03c3c","PSY":"#ff7043","SOW":"#ef5350",
    "AR":"#4fc3f7","SOS":"#43a047",
    "SC":"#e03c3c","LPS":"#66bb6a","Spring":"#a5d6a7","ST":"#90caf9",
}
ZONE_COLORS = {
    "Accumulation":    ("#4fc3f7", 0.12),
    "Distribution":    ("#e03c3c", 0.12),
    "Re-Accumulation": ("#26c6da", 0.13),
    "Re-Distribution": ("#ff7043", 0.13),
}
TOP_EVENTS = {"BC","PSY","SOW","AR","SOS"}
BOT_EVENTS = {"SC","LPS","Spring","ST"}


def draw_chart(ticker, df, analysis):
    events     = [e for e in analysis.get("events",[])
                  if e["event"] not in ("ReAcc","ReDist")]
    zone_boxes = analysis.get("zone_boxes", [])
    all_events = analysis.get("events", [])
    bias       = analysis.get("bias","?").upper()
    phase      = analysis.get("phase","?")

    n = len(df)
    def nearest(s):
        ts = pd.Timestamp(s)
        return min(range(n), key=lambda i: abs((df.index[i]-ts).days))
    def bx(s):
        key = str(pd.Timestamp(s).date())
        d2i = {str(d.date()): i for i, d in enumerate(df.index)}
        return d2i.get(key, nearest(s))

    # ── Style ─────────────────────────────────────────────────────────────────
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
        figsize=(22,12), returnfig=True, panel_ratios=(4,1),
        tight_layout=False, warn_too_much_data=10000)
    ax = axes[0]

    y_max = df["High"].max(); y_min = df["Low"].min()
    y_range = y_max - y_min
    TOP_R0 = y_max + y_range*0.04;  TOP_R1 = y_max + y_range*0.10
    BOT_R0 = y_min - y_range*0.07;  BOT_R1 = y_min - y_range*0.13
    ax.set_ylim(BOT_R1 - y_range*0.04, TOP_R1 + y_range*0.03)

    # ── Event labels ──────────────────────────────────────────────────────────
    def place_label(bi, band_y, dot_y, text, col):
        ax.plot(bi, dot_y, "o", color=col, markersize=5,
                markeredgecolor="#0d1421", markeredgewidth=0.5, zorder=9)
        ax.plot([bi,bi],[min(dot_y,band_y),max(dot_y,band_y)],
                color=col, lw=0.7, ls=":", alpha=0.6, zorder=4)
        ax.text(bi, band_y, text, color="white", fontsize=7,
                fontweight="bold", ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=col, edgecolor="none", alpha=0.93))

    peak_evts   = [e for e in events if e["event"] in TOP_EVENTS]
    trough_evts = [e for e in events if e["event"] in BOT_EVENTS]

    for i, e in enumerate(peak_evts):
        b = bx(e["date"]); col = EVENT_COLORS.get(e["event"],"#90caf9")
        place_label(b, TOP_R0 if i%2==0 else TOP_R1,
                    float(df["High"].iloc[b]),
                    f'{e["event"]}\n${e["price"]:.0f}', col)

    for i, e in enumerate(trough_evts):
        b = bx(e["date"]); col = EVENT_COLORS.get(e["event"],"#90caf9")
        place_label(b, BOT_R0 if i%2==0 else BOT_R1,
                    float(df["Low"].iloc[b]),
                    f'{e["event"]}\n${e["price"]:.0f}', col)

    # ── Zone BOXES ────────────────────────────────────────────────────────────
    def draw_box(x0, x1, y0, y1, col, alpha, label):
        """Draw box with EXACT highs and lows — no padding."""
        # Recalculate exact high/low from actual price data within x0:x1
        chunk = df.iloc[x0:x1+1]
        exact_high = float(chunk["High"].max())
        exact_low  = float(chunk["Low"].min())
        y0 = exact_low
        y1 = exact_high

        # Fill
        rect = mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                                   linewidth=0, facecolor=col,
                                   alpha=alpha, zorder=1)
        ax.add_patch(rect)
        # Top border — solid line on exact high
        ax.plot([x0,x1],[y1,y1], color=col, lw=2.2, ls="-", alpha=0.90, zorder=3)
        # Bottom border — solid line on exact low
        ax.plot([x0,x1],[y0,y0], color=col, lw=2.2, ls="-", alpha=0.90, zorder=3)
        # Left border
        ax.plot([x0,x0],[y0,y1], color=col, lw=1.5, ls="-", alpha=0.60, zorder=3)
        # Right border
        ax.plot([x1,x1],[y0,y1], color=col, lw=1.5, ls="-", alpha=0.60, zorder=3)
        # Label top-left
        ax.text(x0+(x1-x0)*0.02, y1, f" {label} ",
                color="white", fontsize=8, fontweight="bold",
                va="top", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=col, edgecolor="none", alpha=0.92))
        # Price tags right side
        if x1 - x0 > 10:
            ax.text(x1+2, y1, f"${y1:.0f}", color=col,
                    fontsize=7.5, fontweight="bold", va="center", zorder=5)
            ax.text(x1+2, y0, f"${y0:.0f}", color=col,
                    fontsize=7.5, fontweight="bold", va="center", zorder=5)

    for z in zone_boxes:
        x0 = bx(z["x0"]); x1 = bx(z["x1"])
        col, alpha = ZONE_COLORS.get(z["label"], ("#90caf9", 0.10))
        draw_box(x0, x1, z["y0"], z["y1"], col, alpha, z["label"])

    # ── Demand line inside Accumulation zones ─────────────────────────────────
    sc_e = next((e for e in events if e["event"]=="SC"),     None)
    sp_e = next((e for e in events if e["event"]=="Spring"), None)
    lp_e = next((e for e in events if e["event"]=="LPS"),    None)
    p2   = sp_e or lp_e
    if sc_e and p2:
        x0d, x1d = bx(sc_e["date"]), bx(p2["date"])
        ax.plot([x0d,x1d],[sc_e["price"],p2["price"]],
                color="#2eb872", lw=2.0, ls="-", alpha=0.9, zorder=4)
        ax.text((x0d+x1d)//2,(sc_e["price"]+p2["price"])/2,
                " Demand Line ", color="white", fontsize=7,
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#2eb872", edgecolor="none", alpha=0.88))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [
        ("Accumulation",    "#4fc3f7"),
        ("Distribution",    "#e03c3c"),
        ("Re-Accumulation", "#26c6da"),
        ("Re-Distribution", "#ff7043"),
        ("SC/BC — Climax",  "#e03c3c"),
        ("AR — Auto Rally", "#4fc3f7"),
        ("Spring — Shakeout","#a5d6a7"),
        ("SOS — Breakout",  "#43a047"),
    ]
    for k,(lbl,col) in enumerate(legend):
        fig.text(0.005, 0.96-k*0.028, f"  ■ {lbl}",
                 color=col, fontsize=7.5, transform=fig.transFigure,
                 bbox=dict(boxstyle="square,pad=0.05",
                           facecolor="#0d1421", edgecolor="none", alpha=0))

    fig.text(0.005, 0.97, "  WYCKOFF ZONES  ",
             color="white", fontsize=8, fontweight="bold",
             transform=fig.transFigure,
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#1a2540", edgecolor="#4fc3f7", lw=1.2))

    # ── Header ────────────────────────────────────────────────────────────────
    nm = [e["event"] for e in events]
    summary = ", ".join(dict.fromkeys(nm)) if nm else "No events"
    fig.text(0.35, 0.97,
             f"  {ticker}  —  Wyckoff SMI Analysis  [{bias}  |  Phase {phase}]  ",
             color="#f5c842", fontsize=14, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#0f1e35", edgecolor="#f5c842", lw=1.2))
    fig.text(0.88, 0.97, f"Events: {summary}",
             color="#c8d6e5", fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#111827", edgecolor="#1e2d40", alpha=0.85))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180,
                bbox_inches="tight", facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return buf
