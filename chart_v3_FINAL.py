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
PEAK_EVENTS   = {"BC","PSY","UT","UTAD","LPSY","SOW","MSOW","COB","MCOB","AR","SOS","MSOS","JOC","UTA"}
TROUGH_EVENTS = {"SC","LPS","Spring","BU","mSOW","PS","ST"}
MIN_GAP = 15

def draw_chart(ticker, df, analysis):
    events         = analysis.get("events", [])
    zones          = analysis.get("zones", [])
    supports       = analysis.get("support_lines", [])
    trends         = analysis.get("trend_lines", [])
    trading_ranges = analysis.get("trading_ranges", [])
    markup_lines   = analysis.get("markup_lines", [])
    bias           = analysis.get("bias","?").upper()
    phase          = analysis.get("phase","?")
    summary        = analysis.get("summary_ar","")

    n           = len(df)
    date_to_idx = {str(d):i for i,d in enumerate(df.index)}

    def nearest(s):
        ts = pd.Timestamp(s)
        return min(range(n), key=lambda i: abs((df.index[i]-ts).days))

    def bx(s):
        return date_to_idx.get(s, nearest(s))

    mc = mpf.make_marketcolors(
        up="#2eb872",down="#e03c3c",
        edge={"up":"#2eb872","down":"#e03c3c"},
        wick={"up":"#2eb872","down":"#e03c3c"},
        volume={"up":"#2eb87240","down":"#e03c3c40"})
    style = mpf.make_mpf_style(
        marketcolors=mc,facecolor="#0d1421",figcolor="#0a0e17",
        gridcolor="#1a2540",gridstyle="--",y_on_right=True,
        rc={"axes.labelcolor":"#c8d6e5","axes.edgecolor":"#1e2d40",
            "xtick.color":"#8899aa","ytick.color":"#8899aa",
            "font.family":"monospace","font.size":9})

    fig,axes = mpf.plot(df,type="candle",style=style,volume=True,
        figsize=(20,11),returnfig=True,panel_ratios=(4,1),
        tight_layout=False,warn_too_much_data=10000)
    ax = axes[0]

    y_max   = float(df["High"].max())
    y_min   = float(df["Low"].min())
    y_range = y_max - y_min

    # ── BAND SYSTEM: 4 rows top + 4 rows bottom ──────────────────────────────
    ROW = y_range * 0.09
    TOP = [y_max + y_range*0.05 + i*ROW for i in range(4)]
    BOT = [y_min - y_range*0.08 - i*ROW for i in range(4)]
    ax.set_ylim(BOT[-1]-y_range*0.03, TOP[-1]+y_range*0.03)

    def pick_row(events_list, rows):
        occupied = [[] for _ in rows]
        result   = []
        for e in sorted(events_list, key=lambda e: bx(e["date"])):
            b = bx(e["date"])
            chosen = 0
            for ri,occ in enumerate(occupied):
                if all(abs(b-ox)>=MIN_GAP for ox in occ):
                    chosen=ri; break
            occupied[chosen].append(b)
            result.append((e, rows[chosen]))
        return result

    def place(bx_i, band_y, dot_y, text, col):
        # Dot on price
        ax.plot(bx_i,dot_y,"o",color=col,markersize=5,
                markeredgecolor="#0d1421",markeredgewidth=0.6,zorder=9)
        # Dashed connector
        lo,hi = min(dot_y,band_y),max(dot_y,band_y)
        ax.plot([bx_i,bx_i],[lo,hi],color=col,
                linewidth=0.8,linestyle=":",alpha=0.6,zorder=4)
        # Label in band
        ax.text(bx_i,band_y,text,color="white",fontsize=7,
                fontweight="bold",ha="center",va="center",zorder=10,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=col,edgecolor="none",alpha=0.93))

    peaks   = [e for e in events if e.get("type")=="peak"   or e.get("event") in PEAK_EVENTS]
    troughs = [e for e in events if e.get("type")=="trough" or e.get("event") in TROUGH_EVENTS]
    peaks   = [e for e in peaks   if e not in troughs]

    for e,band_y in pick_row(peaks,   TOP):
        b=bx(e["date"]); col=EVENT_COLORS.get(e["event"],"#90caf9")
        lbl=f'{e.get("number","")} {e["event"]}\n${e["price"]:.1f}'
        place(b, band_y, float(df["High"].iloc[b]), lbl, col)

    for e,band_y in pick_row(troughs, BOT):
        b=bx(e["date"]); col=EVENT_COLORS.get(e["event"],"#90caf9")
        lbl=f'{e.get("number","")} {e["event"]}\n${e["price"]:.1f}'
        place(b, band_y, float(df["Low"].iloc[b]),  lbl, col)

    # ── KEY EVENT LISTS ───────────────────────────────────────────────────────
    sc_evts  = [e for e in events if e["event"]=="SC"]
    ar_evts  = [e for e in events if e["event"]=="AR"]
    bc_evts  = [e for e in events if e["event"]=="BC"]
    sp_evts  = [e for e in events if e["event"]=="Spring"]
    lps_evts = [e for e in events if e["event"]=="LPS"]
    ut_evts  = [e for e in events if e["event"] in ("UT","UTAD")]
    ly_evts  = [e for e in events if e["event"] in ("LPSY","LPSY2","LPSY3")]

    # ── HORIZONTAL CHANNEL: SC-AR (white) ────────────────────────────────────
    if sc_evts and ar_evts:
        sc_p = sc_evts[0]["price"]
        ar_p = ar_evts[0]["price"]
        ax.fill_betweenx([sc_p,ar_p],0,n,alpha=0.05,color="white",zorder=1)
        ax.hlines([sc_p,ar_p],0,n,colors="white",
                  linewidths=1.3,alpha=0.45,linestyles="--",zorder=2)
        ax.text(2,sc_p,f" SC ${sc_p:.0f} ",color="white",fontsize=7,
                va="top",ha="left",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#e03c3c",
                          edgecolor="none",alpha=0.85))
        ax.text(2,ar_p,f" AR ${ar_p:.0f} ",color="white",fontsize=7,
                va="bottom",ha="left",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#4fc3f7",
                          edgecolor="none",alpha=0.85))

    # ── HORIZONTAL CHANNEL: BC-AR (white) ────────────────────────────────────
    if bc_evts and ar_evts:
        bc_p = bc_evts[0]["price"]
        ar_p = ar_evts[0]["price"]
        ax.fill_betweenx([ar_p,bc_p],0,n,alpha=0.05,color="white",zorder=1)
        ax.hlines([bc_p],0,n,colors="white",
                  linewidths=1.3,alpha=0.45,linestyles="--",zorder=2)
        ax.text(2,bc_p,f" BC ${bc_p:.0f} ",color="white",fontsize=7,
                va="bottom",ha="left",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#ff7043",
                          edgecolor="none",alpha=0.85))

    # ── DIAGONAL CHANNEL (orange dashed) ─────────────────────────────────────
    # Supply line: BC high → next lower high (UT or LPSY), extended to right
    if bc_evts and (ut_evts or ly_evts):
        p1 = bc_evts[0]
        p2 = (ut_evts+ly_evts)[0]
        x0,x1 = bx(p1["date"]),bx(p2["date"])
        y0 = float(df["High"].iloc[x0])
        y1 = float(df["High"].iloc[x1])
        if x1!=x0:
            slope=(y1-y0)/(x1-x0)
            xe=n-1; ye=y1+slope*(xe-x1)
            ax.plot([x0,xe],[y0,ye],color="#ff9800",linewidth=1.8,
                    linestyle="--",alpha=0.8,zorder=3)
            ax.text((x0+x1)//2,(y0+y1)/2," Supply Line ",
                    color="white",fontsize=7,ha="center",va="top",zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="#e65100",
                              edgecolor="none",alpha=0.85))

    # Demand line: SC low → Spring/LPS low, extended to right
    if sc_evts and (sp_evts or lps_evts):
        p1 = sc_evts[0]
        p2 = (sp_evts+lps_evts)[0]
        x0,x1 = bx(p1["date"]),bx(p2["date"])
        y0 = float(df["Low"].iloc[x0])
        y1 = float(df["Low"].iloc[x1])
        if x1!=x0:
            slope=(y1-y0)/(x1-x0)
            xe=n-1; ye=y1+slope*(xe-x1)
            ax.plot([x0,xe],[y0,ye],color="#ff9800",linewidth=1.8,
                    linestyle="--",alpha=0.8,zorder=3)
            ax.text((x0+x1)//2,(y0+y1)/2," Demand Line ",
                    color="white",fontsize=7,ha="center",va="bottom",zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="#2e7d32",
                              edgecolor="none",alpha=0.85))

    # ── TRADING RANGES ────────────────────────────────────────────────────────
    for tr in trading_ranges:
        x0=bx(tr["x0"]); x1=bx(tr["x1"])
        y0=tr.get("y_bottom",tr.get("y0",0))
        y1=tr.get("y_top",   tr.get("y1",0))
        ax.fill_betweenx([y0,y1],x0,x1,alpha=0.08,color="#1565c0",zorder=1)
        ax.hlines([y0,y1],x0,x1,colors="#1565c0",
                  linewidths=1.2,alpha=0.6,zorder=2)
        ax.text((x0+x1)//2,y1,f' {tr["name"]} ',
                color="white",fontsize=8,fontweight="bold",
                ha="center",va="bottom",zorder=6,
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#1565c0",
                          edgecolor="none",alpha=0.88))

    # ── MARKUP LINES ──────────────────────────────────────────────────────────
    for ml in markup_lines:
        x0=bx(ml["x0"]); x1=bx(ml["x1"])
        ax.plot([x0,x1],[ml["y0"],ml["y1"]],color="#f5c842",
                linewidth=1.8,linestyle="--",alpha=0.75,zorder=3)
        ax.text((x0+x1)//2,(ml["y0"]+ml["y1"])/2,ml["label"],
                color="#0a0e17",fontsize=8,fontweight="bold",
                ha="center",va="center",zorder=7,
                bbox=dict(boxstyle="round,pad=0.3",facecolor="#f5c842",
                          edgecolor="none",alpha=0.92))

    # ── ZONES ─────────────────────────────────────────────────────────────────
    for z in zones:
        x0=bx(z["x0"]); x1=bx(z["x1"])
        col,alpha=ZONE_COLORS.get(z.get("color","red"),("#c62828",0.13))
        ax.fill_betweenx([z["y0"],z["y1"]],x0,x1,
                          alpha=alpha,color=col,zorder=1)
        ax.hlines([z["y0"],z["y1"]],x0,x1,
                   colors=col,linewidths=1.1,alpha=0.65,zorder=2)
        ax.text(x0+0.5,z["y1"],f' {z["name"]} ',
                color="white",fontsize=7,fontweight="bold",
                va="bottom",ha="left",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor=col,
                          edgecolor="none",alpha=0.85))

    # ── SUPPORT LINES ─────────────────────────────────────────────────────────
    for sl in supports:
        c=sl.get("color","#f5c842")
        ax.axhline(sl["y"],linestyle="--",linewidth=1.0,color=c,alpha=0.5,zorder=2)
        ax.text(n-1,sl["y"],f'  {sl["label"]}  ',color=c,fontsize=7,
                va="center",ha="left",zorder=5,
                bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d1421",
                          edgecolor=c,linewidth=0.6,alpha=0.85))

    # ── TREND LINES ───────────────────────────────────────────────────────────
    for tl in trends:
        x0=bx(tl["x0"]); x1=bx(tl["x1"])
        col="#e03c3c" if tl.get("style")=="supply" else "#2eb872"
        ax.plot([x0,x1],[tl["y0"],tl["y1"]],color=col,
                linewidth=1.4,linestyle="--",alpha=0.65,zorder=3)
        lbl=tl.get("label","")
        if lbl:
            ax.text((x0+x1)//2,(tl["y0"]+tl["y1"])/2,f' {lbl} ',
                    color="white",fontsize=7,ha="center",va="center",zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2",facecolor=col,
                              edgecolor="none",alpha=0.75))

    # ── HEADER ────────────────────────────────────────────────────────────────
    fig.text(0.12,0.97,
             f"  {ticker}  —  Wyckoff Analysis  ·  {bias}  Phase {phase}  ",
             color="#f5c842",fontsize=13,fontweight="bold",va="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#0f1e35",edgecolor="#1e2d40"))
    if summary:
        fig.text(0.88,0.97,summary,color="#c8d6e5",fontsize=9,
                 va="top",ha="right",
                 bbox=dict(boxstyle="round,pad=0.4",facecolor="#111827",
                           edgecolor="#1e2d40",alpha=0.85))

    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=180,
                bbox_inches="tight",facecolor="#0a0e17")
    plt.close(fig)
    buf.seek(0)
    return buf
