# MrKrabs

LSTM-based stock trading bot that predicts S&P 500 price movements and executes trades via Alpaca paper trading API.

## Project Structure

```
MrKrabs/
├── prepare_data.py     # Download data & create sequences
├── train_model.py      # Train LSTM model
├── backtest.py         # Backtest & validate
├── trading_bot.py      # Live trading (signals only)
├── alpaca_trader.py    # Alpaca paper trading execution
├── run_bot.sh          # Scheduler script
├── clean_data.sh       # Clean data files
├── data/               # Training data
├── models/             # Trained models
├── logs/               # Trading logs
└── results/            # Backtest results
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# 1. Download data & preprocess
python prepare_data.py

# 2. Train model
python train_model.py

# 3. Backtest
python backtest.py

# 4. Run live bot
python trading_bot.py
```

## Configuration

Edit script constants:
- `TICKER = "^GSPC"` - S&P 500
- `LOOK_BACK = 60` - 60-day sequence window
- `THRESHOLD = 0.005` - Buy/sell threshold (0.5%)

## Scheduling

Add to crontab for hourly execution:
```bash
crontab -e
0 * * * * /home/abka/Documents/MrKrabs/run_bot.sh
```

## Alpaca Trading

For automatic trade execution:
1. Get free paper trading keys at https://app.alpaca.markets/paper
2. Add to `.env`:
   ```
   ALPACA_URL=https://paper-api.alpaca.markets
   ALPACA_KEY=your_key
   ALPACA_SECRET=your_secret
   ```
3. Run: `python alpaca_trader.py`

## Requirements

- Python 3.9+
- TensorFlow, Keras
- pandas, numpy, scikit-learn
- yfinance
- requests (for Alpaca)

Install: `pip install -r requirements.txt`