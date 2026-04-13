# 📡 BIST 50 Technical Signal Radar

A Streamlit dashboard that automatically detects and visualizes technical analysis signals for stocks in Turkey's BIST 50 index.

🔗 **Live Demo:** [bist50radar.streamlit.app](https://bist50radar.streamlit.app/)

## 🎯 Purpose

Tracking 48 stocks simultaneously is impractical. This system calculates 5 technical indicators automatically and highlights strong signals — so investors can focus only on stocks that deserve attention.

> ⚠️ This is not investment advice. Technical signals are a starting point for research, not a standalone buy/sell decision tool.

## 📊 Technical Indicators

| Indicator | Question It Answers | Parameters |
|-----------|-------------------|------------|
| **RSI** | Is the stock overbought or oversold? | 14-day |
| **MACD** | Is momentum increasing or decreasing? | 12/26/9 |
| **Bollinger Bands** | Has price moved outside normal range? | 20-day, 2σ |
| **Moving Average** | What is the overall trend direction? | MA50, MA200 |
| **OBV** | Where is the money flowing? | Cumulative volume |

## 🔢 Strength Score System

Each stock receives a strength score from 0 to 10, based on how many indicators align in the same direction:

- **8-10:** Very strong signal — investigate immediately
- **6-7:** Strong signal — add to watchlist
- **3-5:** Weak/mixed — proceed with caution
- **0-2:** No signal or conflicting indicators

## 🛠️ Features

- **Morning Report:** Daily signal summary for all BIST 50 stocks
- **Stock Detail:** Per-stock technical analysis with interactive charts
- **Handbook:** Built-in reference guide explaining each indicator

## ⚙️ Setup

```bash
git clone https://github.com/gizembal/bist_radar.git
cd bist_radar
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Requirements

- Python 3.9+
- Streamlit
- yfinance (live data)
- Pandas, NumPy
- Plotly (charts)

## 🏗️ Architecture

```
bist_radar/
├── app.py              ← Main dashboard (Streamlit)
├── main.py             ← Helper functions
├── requirements.txt
└── README.md
```

- Live data fetched via **yfinance** — no CSV dependency
- `@st.cache_data(ttl=3600)` for 1-hour caching
- Rule-based signal detection (RSI, MACD, Bollinger Bands, OBV, Golden/Death Cross)

## 👤 Developer

**Gizem Bal**
- Industrial Engineer | Istanbul Technical University
- [LinkedIn](https://www.linkedin.com/in/balgizem/)
- [GitHub](https://github.com/gizembal)

