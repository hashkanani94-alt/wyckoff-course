import io
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EVENT_COLORS = {
    "BC":"#e03c3c","PSY":"#ff7043","UT":"#ff7043","UTAD":"#ff7043",
    "LPSY":"#ff8a65","SOW":"#ef5350","AR":"#4fc3f7","SOS":"#43a047",
    "SC":"#e03c3c","LPS":"#66bb6a","Spring":"#a5d6a7","ST":"#90caf9",
    "ReAcc":"#26c6da",
}
TOP_EVENTS = {"BC","PSY","UT","UTAD","LPSY","SOW","AR","SOS"}
BOT_EVENTS = {"SC","LPS","Spring","ST"}


def draw_chart(ticker, df, analysis):
    events  = [e for e in analysis.get("events",[]) if e["event"] != "ReAcc"]
    bias    = analysis.get("bias","?").upper()
    phase   = analysis.get("phase","?")

    n = len(df)
    date_to_idx = {str(d.date()): i for i, d in enumerate(df.index)}

    def nearest(s):
        ts = pd.Timestamp(s)
        return min(range(n), key=lambda i: abs((df.index[i]-ts).days))

    def bx(s):
        key = str(pd.Timestamp(s).date())
        return date_to_idx.get(key, nearest(s))

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

    y_max   = df["High"].max()
    y_min   = df["Low"].min()
    y_range = y_max - y_min

    TOP_R0 = y_max + y_range * 0.04
    TOP_R1 = y_max + y_range * 0.10
    BOT_R0 = y_min - y_range * 0.07
    BOT_R1 = y_min - y_range * 0.13
    ax.set_ylim(BOT_R1 - y_range*0.04, TOP_R1 + y_range*0.03)

    # ── Event labels in bands ─────────────────────────────────────────────────
    def place_label(bar_idx, band_y, dot_y, text, col):
        ax.plot(bar_idx, dot_y, "o", color=col, markersize=5,
                markeredgecolor="#0d1421", markeredgewidth=0.5, zorder=9)
        ax.plot([bar_idx, bar_idx],
                [min(dot_y,band_y), max(dot_y,band_y)],
                color=col, linewidth=0.7, linestyle=":", alpha=0.6, zorder=4)
        ax.text(bar_idx, band_y, text, color="white", fontsize=7,
                fontweight="bold", ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.28",
                          facecolor=col, edgecolor="none", alpha=0.93))

    peak_evts   = [e for e in events if e["event"] in TOP_EVENTS]
    trough_evts = [e for e in events if e["event"] in BOT_EVENTS]

    for i, e in enumerate(peak_evts):
        b   = bx(e["date"])
        col = EVENT_COLORS.get(e["event"], "#90caf9")
        lbl = f'{e["event"]}\n${e["price"]:.0f}'
        place_label(b, TOP_R0 if i%2==0 else TOP_R1,
                    float(df["High"].iloc[b]), lbl, col)

    for i, e in enumerate(trough_evts):
        b   = bx(e["date"])
        col = EVENT_COLORS.get(e["event"], "#90caf9")
        lbl = f'{e["event"]}\n${e["price"]:.0f}'
        place_label(b, BOT_R0 if i%2==0 else BOT_R1,
                    float(df["Low"].iloc[b]), lbl, col)

    # ── TR BOX function (clean, bounded, NO infinite lines) ───────────────────
    def draw_box(x0, x1, y_bot, y_top, col, label, alpha_fill=0.08):
        """Draw a clean bounded rectangle for a Trading Range."""
        # Filled background
        rect = mpatches.Rectangle(
            (x0, y_bot), x1-x0, y_top-y_bot,
            linewidth=0, facecolor=col, alpha=alpha_fill, zorder=1)
        ax.add_patch(rect)
        # Top & bottom dashed borders
        ax.plot([x0, x1], [y_top, y_top], color=col,
                linewidth=1.8, linestyle="--", alpha=0.85, zorder=3)
        ax.plot([x0, x1], [y_bot, y_bot], color=col,
                linewidth=1.8, linestyle="--", alpha=0.85, zorder=3)
        # Left & right solid borders
        ax.plot([x0, x0], [y_bot, y_top], color=col,
                linewidth=1.5, linestyle="-", alpha=0.7, zorder=3)
        ax.plot([x1, x1], [y_bot, y_top], color=col,
                linewidth=1.5, linestyle="-", alpha=0.7, zorder=3)
        # Label top-left
        ax.text(x0 + max((x1-x0)*0.02, 2), y_top,
                f" {label} ", color="white", fontsize=8,
                fontweight="bold", va="top", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=col, edgecolor="none", alpha=0.92))
        # Price tags right side
        ax.text(x1+2, y_top, f"${y_top:.0f}", color=col,
                fontsize=7.5, va="center", ha="left", zorder=5,
                fontweight="bold")
        ax.text(x1+2, y_bot, f"${y_bot:.0f}", color=col,
                fontsize=7.5, va="center", ha="left", zorder=5,
                fontweight="bold")

    # ── Identify key events ───────────────────────────────────────────────────
    sc_e    = next((e for e in events if e["event"]=="SC"),   None)
    ar_e    = next((e for e in events if e["event"]=="AR"),   None)
    sp_e    = next((e for e in events if e["event"]=="Spring"),None)
    sos_e   = next((e for e in events if e["event"]=="SOS"),  None)
    lps_e   = next((e for e in events if e["event"]=="LPS"),  None)
    bc_e    = next((e for e in events if e["event"]=="BC"),   None)
    st_evts = [e for e in events if e["event"]=="ST"]

    # ── Accumulation TR Box ───────────────────────────────────────────────────
    if sc_e and ar_e:
        x0_acc = bx(sc_e["date"])
        # End box at SOS if found, else extend 400 bars
        x1_acc = bx(sos_e["date"]) if sos_e else min(x0_acc+400, n-2)
        y_bot_acc = (sp_e["price"] if sp_e else sc_e["price"]) * 0.995
        y_top_acc = ar_e["price"] * 1.005
        draw_box(x0_acc, x1_acc, y_bot_acc, y_top_acc, "#4fc3f7", "Accumulation TR")

    # ── Distribution TR Box ───────────────────────────────────────────────────
    if bc_e:
        x0_dist = max(bx(bc_e["date"])-15, 0)
        x1_dist = min(bx(bc_e["date"])+150, n-2)
        y_top_dist = bc_e["price"] * 1.005
        y_bot_dist = bc_e["price"] * 0.92
        draw_box(x0_dist, x1_dist, y_bot_dist, y_top_dist, "#e03c3c", "Distribution TR")

    # ── Re-Accumulation boxes ─────────────────────────────────────────────────
    all_evts_raw = analysis.get("events",[])
    for ra in [e for e in all_evts_raw if e["event"]=="ReAcc"]:
        if "x0" in ra and "x1" in ra:
            draw_box(bx(ra["x0"]), bx(ra["x1"]),
                     ra["y0"], ra["y1"], "#26c6da", "Re-Acc", alpha_fill=0.1)

    # ── Markup channel (SOS → present) ───────────────────────────────────────
    if sos_e:
        x0_m = bx(sos_e["date"])
        x1_m = n - 2
        # Draw a simple upward channel shading
        sos_price = sos_e["price"]
        last_high = float(df["High"].iloc[-1])
        ax.fill_betweenx(
            [sos_price, last_high],
            x0_m, x1_m,
            alpha=0.04, color="#43a047", zorder=0)
        ax.plot([x0_m, x1_m], [sos_price, last_high],
                color="#43a047", linewidth=1.4, linestyle=":",
                alpha=0.6, zorder=2)
        ax.text(x0_m+5, sos_price*1.01,
                " Markup Phase E ", color="white", fontsize=8,
                fontweight="bold", va="bottom", ha="left", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#43a047", edgecolor="none", alpha=0.85))

    # ── Demand line (SC → Spring, bounded) ───────────────────────────────────
    p2 = sp_e or lps_e
    if sc_e and p2:
        x0d, x1d = bx(sc_e["date"]), bx(p2["date"])
        ax.plot([x0d, x1d], [sc_e["price"], p2["price"]],
                color="#2eb872", linewidth=2.0, linestyle="-",
                alpha=0.9, zorder=4)
        ax.text((x0d+x1d)//2, (sc_e["price"]+p2["price"])/2,
                " Demand Line ", color="white", fontsize=7,
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#2eb872", edgecolor="none", alpha=0.88))

    # ── Phase labels at bottom of chart ───────────────────────────────────────
    phase_labels = analysis.get("phase_labels", [])
    y_ph = BOT_R1 + y_range * 0.005

    for pl in phase_labels:
        x0p = bx(pl["x0"]); x1p = bx(pl["x1"])
        col = pl.get("color","#4fc3f7")
        # Vertical dividers only (no full span shading — keeps chart clean)
        ax.axvline(x0p, color=col, linewidth=1.0,
                   linestyle=":", alpha=0.45, zorder=2)
        # Phase label
        ax.text((x0p+x1p)//2, y_ph, pl["text"],
                color=col, fontsize=8, fontweight="bold",
                ha="center", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#0d1421", edgecolor=col,
                          linewidth=1.3, alpha=0.95))

    # ── Legend box top-left ───────────────────────────────────────────────────
    legend_items = [
        ("SC",    "Selling Climax — end of downtrend",       "#e03c3c"),
        ("AR",    "Automatic Rally — sets TR top",           "#4fc3f7"),
        ("ST",    "Secondary Test — retests SC area",        "#90caf9"),
        ("Spring","Shakeout below SC — Phase C",             "#a5d6a7"),
        ("SOS",   "Sign of Strength — breaks TR",            "#43a047"),
        ("LPS",   "Last Point of Support — re-entry",        "#66bb6a"),
    ]
    leg_x = 0.01; leg_y = 0.97; step = 0.026
    fig.text(leg_x, leg_y, "  WYCKOFF EVENTS  ",
             color="white", fontsize=8, fontweight="bold",
             transform=fig.transFigure,
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="#1a2540", edgecolor="#4fc3f7",
                       linewidth=1.2, alpha=0.95))
    for k, (ev, desc, col) in enumerate(legend_items):
        fig.text(leg_x, leg_y - (k+1)*step,
                 f"  ■ {ev}: {desc}",
                 color=col, fontsize=7, transform=fig.transFigure,
                 bbox=dict(boxstyle="square,pad=0.1",
                           facecolor="#0d1421", edgecolor="none", alpha=0.0))

    # ── Header ────────────────────────────────────────────────────────────────
    ev_names = [e["event"] for e in events]
    summary  = f"Events: {', '.join(ev_names)}" if ev_names else "No events detected"
    fig.text(0.35, 0.97,
             f"  {ticker}  —  Wyckoff SMI Analysis  [{bias}  |  Phase {phase}]  ",
             color="#f5c842", fontsize=14, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#0f1e35", edgecolor="#f5c842", linewidth=1.2))
    fig.text(0.88, 0.97, summary,
             color="#c8d6e5", fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#111827", edgecolor="#1e2d40", alpha=0.85))

    # ── Save ──────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180,
                bbox_inches="tight", facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return buf
