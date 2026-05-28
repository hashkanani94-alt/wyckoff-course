# Wyckoff Analysis App 📊

A Python tool that analyzes stocks using the **original Wyckoff SMI Course** methodology.

---

## 📁 Project Structure

```
wyckoff_app/
├── main.py                  ← Entry point — run this
├── requirements.txt         ← Install dependencies
├── data/
│   └── fetcher.py           ← Downloads stock data (yfinance)
├── analysis/
│   └── wyckoff_detector.py  ← Detects SC, AR, ST, Spring, SOS, BC
├── charts/
│   └── chart_generator.py   ← Draws the annotated chart
└── output/                  ← Charts saved here (auto-created)
```

---

## ⚙️ Installation

```bash
# 1. Clone from GitHub
git clone https://github.com/hashkanani94-alt/wyckoff-course
cd wyckoff-course

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Basic usage (1 year daily chart)
python main.py AAPL

# 2-year analysis
python main.py TSLA --period 2y

# Weekly chart
python main.py NVDA --period 2y --interval 1wk

# 6-month short-term
python main.py MSFT --period 6mo
```

---

## 📌 Wyckoff Events Detected

| Event  | Description |
|--------|-------------|
| SC     | Selling Climax — end of downtrend |
| AR     | Automatic Rally — bounce after SC |
| ST     | Secondary Test — retest of SC area |
| Spring | Shakeout below SC low |
| SOS    | Sign of Strength — breakout |
| BC     | Buying Climax — end of uptrend |

---

## 📄 Course PDFs

All original Wyckoff SMI course PDFs are in the `pdfs/` folder.

---

## ⚠️ Important

All analysis is based **exclusively** on the original Wyckoff SMI Course.
No external indicators or other methodologies are used.
