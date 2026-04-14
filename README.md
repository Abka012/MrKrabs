# MrKrabs

LSTM-based multi-ticker stock trading bot that predicts price movements for multiple stocks (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) and executes trades via Alpaca paper trading API.

## Features

- **Multi-Ticker Support**: Trade multiple stocks simultaneously with ticker-specific data and models
- **Multithreaded Execution**: Run all tickers in parallel for faster execution
- **25 Technical Indicators**: RSI, MACD, Bollinger Bands, SMAs, ATR, Momentum, Volume ratios
- **BiLSTM Classifier**: Directional model predicting UP/DOWN (~55-58% accuracy)
- **Learned Auto Mode**: Per-ticker mode selector (equity vs option) trained on historical signals
- **Short Selling**: Equity shorts only when Alpaca marks the asset shortable and easy-to-borrow
- **Options Mode**: Optional single-leg call/put trading using Alpaca option contracts
- **One Execution Per Day**: Bot places at most one order per ticker per day
- **All Tickers Meeting Threshold**: Executes ALL tickers that meet the confidence threshold

## Project Structure

```
MrKrabs/
├── prepare_data.py     # Download data, add technical indicators, create sequences
├── train_model.py      # Train LSTM models (regression, classifier, XGBoost)
├── backtest.py         # Backtest & validate trading strategy

├── alpaca_trader.py    # Alpaca paper trading execution
├── run_all.py          # Full pipeline orchestrator
├── config.py           # Centralized configuration
├── .env                # API keys (Alpaca)
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

## Configuration

Edit [config.py](config.py) for centralized settings:

```python
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"]
DEFAULT_TICKER = "SPY"

# Entry thresholds
LONG_ENTRY_THRESHOLD = 0.52    # Prob(UP) threshold for long entries
SHORT_ENTRY_THRESHOLD = 0.48   # Prob(UP) threshold for short entries
MIN_CONFIDENCE_GAP = 0.02      # Minimum confidence gap required

# Risk management
POSITION_SIZE = 0.05           # 5% of cash per trade
STOP_LOSS_PCT = 0.03           # 3% stop-loss
TAKE_PROFIT_PCT = 0.06         # 6% take-profit
MAX_HOLD_DAYS = 20             # Max days to hold a position

# Trade mode
TRADE_MODE = "equity"          # "equity", "option", or "auto"
ALLOW_SHORTS = True
USE_TREND_FILTER = False

# Options
OPTIONS_POSITION_SIZE = 0.05   # 5% of options buying power per trade
OPTIONS_MIN_DTE = 7
OPTIONS_MAX_DTE = 45
OPTIONS_STRIKE_WINDOW = 0.08   # +/- 8% around spot

# Auto mode
AUTO_MODE_MIN_CONFIDENCE_GAP = 0.05
AUTO_MODE_ATR_MULTIPLIER = 1.0
AUTO_MODE_OPTION_COST_PENALTY = 0.015
AUTO_MODE_MAX_OPTION_LEVERAGE = 4.0

# Model
LOOK_BACK = 60                 # 60-day sequence window
PERIOD = "10y"                 # 10 years of historical data
EPOCHS = 50
BATCH_SIZE = 32
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

### Local Crontab (Recommended)

The bot runs locally using crontab at **14:00 UTC (10:00 AM ET)** — 30 minutes after market open, Mon-Fri.

**Setup:**

```bash
# Edit crontab
crontab -e

# Add this line:
0 14 * * 1-5 /home/abka/Documents/MrKrabs/run_bot.sh
```

This runs the bot once daily at 14:00 UTC. The bot places at most one order per ticker per day.

### Manual Execution

```bash
# Run trading bot
python alpaca_trader.py --all --mode auto

# Or for specific ticker
python alpaca_trader.py --ticker SPY --mode auto
```

**Note**: The bot places at most one order per ticker per day. Subsequent runs on the same day will skip with "Already placed a non-canceled order for this ticker today." Risk exits (stop-loss, take-profit, max hold) are still checked on every run.

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

### Risk Management

The bot manages risk through:
- **Stop-Loss**: 3% adverse move closes the position
- **Take-Profit**: 6% favorable move locks in gains
- **Max Hold Days**: Positions auto-close after 20 trading days
- **One Order Per Day**: Prevents overtrading; signal resets daily
- **Trend Filter** (optional): Requires SMA alignment when `USE_TREND_FILTER = True`

### Per-Ticker Tuned Thresholds

After running `backtest.py`, the bot auto-tunes per-ticker entry thresholds and stores them in `models/<ticker>/tuned_thresholds.json`. These override the global `LONG_ENTRY_THRESHOLD` and `MIN_CONFIDENCE_GAP` when present.
