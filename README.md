# MrKrabs

LSTM-based multi-ticker stock trading bot that predicts price movements for multiple stocks (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) and executes trades via Alpaca paper trading API.

## Features

- **Multi-Ticker Support**: Trade multiple stocks simultaneously with ticker-specific data and models
- **Multithreaded Execution**: Run all tickers in parallel for faster execution
- **25 Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs, ATR, Momentum, Volume ratios
- **Multiple Models**: BiLSTM classifier (58% accuracy), XGBoost, LSTM regression
- **Short Selling**: Equity shorts only when Alpaca marks the asset shortable and easy-to-borrow
- **Options Mode**: Optional single-leg call/put trading using Alpaca option contracts
- **Market Hours Check**: Only executes trades when market is open (9:30 AM - 4:00 PM ET)
- **Directional Trading**: Classifier predicts UP/DOWN with probability threshold
- **Automatic Trading**: Runs on cron every 5 minutes during market hours
- **Multiple Trades Per Day**: No limit on trades per day

## Project Structure

```
MrKrabs/
├── prepare_data.py     # Download data, add technical indicators, create sequences
├── train_model.py      # Train LSTM models (regression, classifier, XGBoost)
├── backtest.py         # Backtest & validate trading strategy
├── trading_bot.py      # Trading signals bot (paper trading analysis)
├── alpaca_trader.py    # Alpaca paper trading execution
├── run_bot.sh          # Scheduler script (multithreaded)
├── run_all.py          # Full pipeline orchestrator
├── config.py           # Centralized configuration
├── data/               # Training data (ticker-specific subdirectories)
├── models/             # Trained models (ticker-specific subdirectories)
├── logs/               # Trading logs
└── results/            # Backtest results
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Option 1: Full pipeline for all tickers (recommended)
python run_all.py --all

# Option 2: Full pipeline for single ticker
python run_all.py --ticker SPY

# Option 3: Run individual steps
python prepare_data.py --all          # Prepare data for all tickers
python train_model.py --all           # Train models for all tickers
python backtest.py --all              # Backtest all tickers
python alpaca_trader.py --all         # Trade all tickers as equities
python alpaca_trader.py --all --mode option  # Trade all tickers with options

# Or specific ticker:
python alpaca_trader.py --ticker SPY  # Trade specific ticker
python alpaca_trader.py --ticker SPY --mode option
```

## Trading Bot (trading_bot.py)

The `trading_bot.py` script analyzes market data and generates trading signals without executing live trades. It's useful for testing and analyzing strategy performance.

### Usage

```bash
# Run for all tickers from config
python trading_bot.py --all

# Run for specific ticker
python trading_bot.py --ticker SPY

# Default (no args) runs all tickers
python trading_bot.py
```

### Features

- Fetches recent market data for each ticker
- Applies 25 technical indicators
- Generates BUY/SELL/HOLD signals based on classifier probability
- Tracks portfolio value and position
- Logs all activity to `logs/trading_bot.log`

### Output

The bot provides:
- Current price for each ticker
- Probability of UP/DOWN movement
- Trading actions taken (BUY, SELL, SHORT, COVER, or HOLD)
- Portfolio summary with cash, positions, and total return

## Configuration

Edit [config.py](config.py) for centralized settings:

```python
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"]
THRESHOLD = 0.45      # Classifier threshold (45% triggers trade)
POSITION_SIZE = 0.15  # 15% of cash per trade
LOOK_BACK = 60        # 60-day sequence window
TRADE_MODE = "equity" # or "option" or "auto"
ALLOW_SHORTS = True
OPTIONS_POSITION_SIZE = 0.05
OPTIONS_MIN_DTE = 7
OPTIONS_MAX_DTE = 45
AUTO_MODE_MIN_CONFIDENCE_GAP = 0.05
```

## Ticker-Specific Directories

Each ticker has its own data and model directories:
```
data/
├── SPY/
│   ├── raw_data.csv
│   ├── scaler.pkl
│   ├── X_train.npy
│   └── ...
├── QQQ/
│   ├── raw_data.csv
│   └── ...
models/
├── SPY/
│   ├── classifier_model.keras
│   ├── regression_model.keras
│   └── xgboost_model.pkl
└── QQQ/
    └── ...
```

## Usage Examples

### Single Ticker Pipeline
```bash
# Prepare data for SPY
python prepare_data.py --ticker SPY

# Train models for SPY
python train_model.py --ticker SPY

# Backtest SPY
python backtest.py --ticker SPY

# Trade SPY as equity
python alpaca_trader.py --ticker SPY

# Trade SPY with options
python alpaca_trader.py --ticker SPY --mode option
```

### All Tickers Pipeline
```bash
# Full pipeline for all tickers
python run_all.py --all

# Or individual steps with --all flag
python prepare_data.py --all
python train_model.py --all
python backtest.py --all
python alpaca_trader.py --all
python alpaca_trader.py --all --mode option
python alpaca_trader.py --all --mode auto
```

### Live Trading
```bash
# Run trading bot for all tickers as equities
python alpaca_trader.py --all

# Or options mode
python alpaca_trader.py --all --mode option
python alpaca_trader.py --all --mode auto

# Or for specific ticker
python alpaca_trader.py --ticker SPY
python alpaca_trader.py --ticker SPY --mode option
python alpaca_trader.py --ticker SPY --mode auto
```

### Trading Bot Output Example

```
2024-01-15 10:30:00 - INFO - Running strategy for SPY
2024-01-15 10:30:00 - INFO - Current price: $512.34
2024-01-15 10:30:00 - INFO - Probability UP: 62.45%
2024-01-15 10:30:00 - INFO - Probability DOWN: 37.55%
2024-01-15 10:30:00 - INFO - BUY (LONG) 10 shares of SPY at $512.34
2024-01-15 10:30:00 - INFO - Portfolio Value: $9847.66
2024-01-15 10:30:00 - INFO -   Cash: $4847.66
2024-01-15 10:30:00 - INFO -   Position: 10.00 shares
```

## Scheduling

Add to crontab for execution every 5 minutes during market hours:
```bash
crontab -e
*/5 9-16 * * 1-5 /home/abka/Documents/MrKrabs/run_bot.sh
```

This runs the bot every 5 minutes from 9 AM to 4 PM on weekdays (Mon-Fri) for ALL tickers in parallel.

### Manual Run

To test without cron:
```bash
./run_bot.sh
```

For automatic trade execution:
1. Get free paper trading keys at https://app.alpaca.markets/paper
2. Add to `.env`:
   ```
   ALPACA_URL=https://paper-api.alpaca.markets
   ALPACA_KEY=your_key
   ALPACA_SECRET=your_secret
   ```
3. Run: `python alpaca_trader.py`

**Note**: Trades only execute when market is open. Use `run_bot.sh` with cron for 5-minute execution.

## Trade Modes

- `equity` mode is the default. Bullish signals buy shares, bearish signals can short only when Alpaca reports the asset as shortable and easy-to-borrow.
- `option` mode uses the same directional model to buy single-leg calls on bullish signals or puts on bearish signals.
- `auto` mode estimates the edge of an equity trade versus a candidate option contract and chooses the higher-scoring path.
- After you run `backtest.py`, `auto` mode will prefer a trained selector from `models/<ticker>/mode_selector.pkl` when available and fall back to the heuristic only when no selector artifact exists.
- Option contracts are selected from Alpaca's options contracts API using the configured expiry window and a strike near the underlying price.
- Existing option positions for a ticker are closed when the signal flips to the opposite direction.

## Learned Auto Mode

Run a backtest after training to build the mode selector artifact:

```bash
python backtest.py --ticker SPY
python backtest.py --all
```

This now produces, per ticker:

- `results/<ticker>/mode_selector_dataset.csv`
- `results/<ticker>/mode_selector_predictions.csv`
- `results/<ticker>/mode_selector_metrics.json`
- `models/<ticker>/mode_selector.pkl`

The selector is trained from historical test-window signals using realized equity outcomes and a synthetic option payoff proxy derived from the same underlying move/volatility data. The repo still does not contain historical option chain data, so the option side of the selector is an approximation rather than a true contract-by-contract historical options backtest.

## Requirements

- Python 3.9+
- TensorFlow, Keras
- pandas, numpy, scikit-learn
- yfinance
- requests (for Alpaca)
- xgboost

Install: `pip install -r requirements.txt`

## Alpaca Paper Trading

For automatic trade execution:
1. Get free paper trading keys at https://app.alpaca.markets/paper
2. Add to `.env`:
   ```
   ALPACA_URL=https://paper-api.alpaca.markets
   ALPACA_KEY=your_key
   ALPACA_SECRET=your_secret
   ```
3. Run: `python alpaca_trader.py --ticker SPY` or `python alpaca_trader.py --all`
