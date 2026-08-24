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
- ⚠️ **Surcharge is not modeled.** Every tax figure includes the 4% Health & Education Cess but not the income-based surcharge (10%/15%/25% slabs above ₹50L/1Cr/2Cr total income). This engine has no way to know your total income, so numbers above the ₹50L threshold understate your real liability — treat them as a floor, not a final number, and verify with a CA.

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
- **100% Local**: this app never sends your portfolio data anywhere over the network -- it's a claim about network exposure, not encryption. All portfolio data, transactions, and personal financial numbers are stored in `wealth_ledger.db`, an **unencrypted** SQLite file. Anyone with access to that file -- another user on a shared machine, a cloud-sync tool (Dropbox/OneDrive/Google Drive) if the app's folder sits inside a synced directory, or someone who gets hold of the machine -- can read it in plaintext. Keep it out of synced folders, or encrypt the disk/folder yourself if that's a real threat for you.
- Sensitive files (`*.db`, `*.wal`, `.env`) are excluded from Git via `.gitignore`.
- Zerodha Kite API credentials are entered per-session via a password-masked field and are never written to disk or the database.

---

## 📄 License
This project is licensed under the MIT License.
