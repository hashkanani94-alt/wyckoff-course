"""
main.py — Wyckoff Analysis App
================================
Usage:
    python main.py AAPL
    python main.py TSLA --period 2y
    python main.py NVDA --period 6mo --interval 1wk
"""

import sys
import os
import argparse
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import fetch_data
from analysis.wyckoff_detector import detect_wyckoff_events
from charts.chart_generator import draw_chart


def print_report(ticker: str, analysis: dict):
    """Print a clean Wyckoff analysis report to the terminal."""

    bias = analysis.get("bias", "?")
    phase = analysis.get("phase", "?")
    events = analysis.get("events", [])
    waves = analysis.get("wave_analysis", {})

    bias_color = {
        "BULLISH": "\033[92m",   # green
        "BEARISH": "\033[91m",   # red
        "NEUTRAL": "\033[93m",   # yellow
    }.get(bias, "\033[0m")
    reset = "\033[0m"

    print("\n" + "═" * 55)
    print(f"  📊  WYCKOFF ANALYSIS — {ticker.upper()}")
    print("═" * 55)
    print(f"  Bias    : {bias_color}{bias}{reset}")
    print(f"  Phase   : {phase}")
    print(f"  Trend   : {waves.get('trend_strength', 'Unknown')}")
    print(f"  Volume  : {waves.get('recent_volume_trend', 'Unknown')}")
    print("─" * 55)

    if events:
        print("  EVENTS DETECTED:")
        for e in events:
            print(f"    [{e['event']:8s}]  {e['date']}  ${e['price']:>10,.2f}")
    else:
        print("  ⚠️  No clear Wyckoff events detected in this period.")
        print("     Try: longer period (--period 2y) or weekly (--interval 1wk)")

    print("═" * 55 + "\n")


def run(ticker: str, period: str = "1y", interval: str = "1d", save: bool = True):
    """
    Full pipeline:
    1. Fetch data
    2. Detect Wyckoff events
    3. Draw chart
    4. Print report
    """
    print(f"\n🚀 Wyckoff Analysis — {ticker.upper()}")
    print("─" * 40)

    # Step 1: Fetch data
    df = fetch_data(ticker, period=period, interval=interval)

    # Step 2: Detect events
    analysis = detect_wyckoff_events(df)

    # Step 3: Print report
    print_report(ticker, analysis)

    # Step 4: Draw and save chart
    print("🎨 Drawing chart...")
    buf = draw_chart(ticker.upper(), df, analysis)

    if save:
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{ticker.upper()}_{interval}_wyckoff.png"
        with open(filename, "wb") as f:
            f.write(buf.read())
        print(f"✅ Chart saved: {filename}")
        return filename

    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wyckoff Chart Analyzer")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL, TSLA, NVDA)")
    parser.add_argument("--period",   default="1y",  help="Data period: 6mo, 1y, 2y, 5y (default: 1y)")
    parser.add_argument("--interval", default="1d",  help="Bar interval: 1d, 1wk (default: 1d)")

    args = parser.parse_args()

    run(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval
    )
