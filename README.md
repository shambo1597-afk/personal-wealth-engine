# Personal Wealth Engine 🚀

A comprehensive, institutional-grade Personal Wealth Management and Quantitative Portfolio Optimization Engine tailored for Indian and global financial markets. Built with **Python**, **Streamlit**, and **Plotly**.

---

## 🌟 Key Features

### 1. 📊 Portfolio Tracking & Wealth Ledger
- Track multi-asset classes: **Indian Equities (NSE/BSE)**, **Mutual Funds**, **Fixed Deposits**, **Gold/Commodities**, **Real Estate**, and **Cash/Liquid Assets**.
- Real-time valuation updates with price history, asset allocation breakdowns, and visual portfolio health meters.
- Automated local SQLite database with robust backup mechanisms and atomic transactions.

### 2. 🧮 Quantitative Finance & Risk Engine (`quant_engine.py`)
- **Modern Portfolio Theory (Markowitz)**: Efficient Frontier simulation, Maximum Sharpe Ratio, Minimum Volatility, and Custom Target Return optimizations.
- **Risk Metrics**: Value-at-Risk (VaR Historical & Parametric), Conditional VaR (CVaR / Expected Shortfall), Maximum Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Beta, and Treynor Ratio.
- **Monte Carlo Simulations**: Probabilistic future wealth projections (1 to 30 years) with confidence intervals (5th, 50th, 95th percentiles).
- **Asset Correlation & Risk Decomposition**: Dynamic correlation heatmaps, portfolio variance attribution, and diversification index.

### 3. 🏛️ Indian Tax Planning Engine (`tax_engine.py`)
- **Capital Gains Computation**: Short-Term Capital Gains (STCG) and Long-Term Capital Gains (LTCG) across Equity, Debt, and Real Estate as per latest Indian Income Tax rules.
- **Section 112A & Grandfathering Support**: Optimized tracking of ₹1.25 Lakh LTCG annual equity exemption limit.
- **Tax Harvesting Simulator**: Identify unrealized losses and gains to execute tax-loss harvesting before financial year-end (March 31st).

### 4. 📈 Interactive Analytics & Dashboards (`app.py`)
- Visual interactive charts powered by **Plotly**.
- Scenario stress-testing (Market Crashes, Inflation Surges, Interest Rate Hikes).
- Goal-based financial planning: Retirement, Child Education, Emergency Fund, Home Purchase with inflation adjustments.

---

## 🛠️ Tech Stack & Dependencies

- **UI Framework**: [Streamlit](https://streamlit.io/)
- **Market Data**: `yfinance`, `nselib`, `requests`, `beautifulsoup4`
- **Analytics & Computation**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **Data Visualization**: `plotly`
- **Database**: SQLite3 (with WAL mode)

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+** (Python 3.13 supported)

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Personal_Wealth_Engine.git
cd Personal_Wealth_Engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
- **On Windows**: Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```
- **On Linux / macOS**:
```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## 🔒 Security & Privacy
- **100% Local**: All portfolio data, transactions, and personal financial numbers remain securely on your local machine in SQLite.
- Sensitive files (`*.db`, `*.wal`, `.env`) are excluded from Git via `.gitignore`.

---

## 📄 License
This project is licensed under the MIT License.
