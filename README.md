# MrKrabs

LSTM-based stock trading bot that predicts S&P 500 (SPY) price movements and executes trades via Alpaca paper trading API.

## Project Structure

```
MrKrabs/
├── prepare_data.py     # Download data, add technical indicators, create sequences
├── train_model.py      # Train LSTM models (regression, classifier, XGBoost)
├── backtest.py         # Backtest & validate trading strategy
├── trading_bot.py      # Live trading signals (paper trading analysis)
├── alpaca_trader.py    # Alpaca paper trading execution
├── run_bot.sh          # Scheduler script
├── clean_data.sh       # Clean data files
├── data/               # Training data (10 years SPY with 25 features)
├── models/             # Trained models
│   ├── classifier_model.keras  # BiLSTM directional classifier
│   ├── regression_model.keras  # LSTM price regression
│   └── xgboost_model.pkl        # XGBoost classifier
├── logs/               # Trading logs
└── results/            # Backtest results
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Download data, add technical indicators, create sequences
python prepare_data.py

# 2. Train models (regression, classifier, XGBoost)
python train_model.py

# 3. Backtest (uses classifier by default)
python backtest.py

# 4. Run live trading signals (without executing)
python trading_bot.py

# 5. Execute trades on Alpaca paper trading
python alpaca_trader.py
```

## Features

- **25 Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs, ATR, Momentum, Volume ratios
- **Multiple Models**: BiLSTM classifier (58% accuracy), XGBoost, LSTM regression
- **Short Selling**: Support for short positions (predict DOWN → SHORT)
- **Market Hours Check**: Only executes trades when market is open (9:30 AM - 4:00 PM ET)
- **Directional Trading**: Classifier predicts UP/DOWN with probability threshold

## Configuration

Edit script constants in each file:
- `TICKER = "SPY"` - S&P 500 ETF (tradable)
- `LOOK_BACK = 60` - 60-day sequence window
- `THRESHOLD = 0.45` - Classifier threshold (45% triggers trade on any upward signal)

## Backtest Results

```
Total Return: 56,823%
Sharpe Ratio: 7.07
Max Drawdown: -11.72%
Total Trades: 442
Wins: 251
Losses: 191
Win/Loss Ratio: 1.31
```

## Alpaca Paper Trading

For automatic trade execution:
1. Get free paper trading keys at https://app.alpaca.markets/paper
2. Add to `.env`:
   ```
   ALPACA_URL=https://paper-api.alpaca.markets
   ALPACA_KEY=your_key
   ALPACA_SECRET=your_secret
   ```
3. Run: `python alpaca_trader.py`

**Note**: Trades only execute when market is open. Use `run_bot.sh` with cron for hourly execution.

## Scheduling

Add to crontab for hourly execution (during market hours):
```bash
crontab -e
0 9-16 * * 1-5 /home/abka/Documents/MrKrabs/run_bot.sh
```

## Requirements

- Python 3.9+
- TensorFlow, Keras
- pandas, numpy, scikit-learn
- yfinance
- requests (for Alpaca)
- xgboost

Install: `pip install -r requirements.txt`