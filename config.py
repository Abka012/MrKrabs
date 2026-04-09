# MrKrabs Configuration
# Centralized settings for all tickers

# Supported tickers for trading
TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
]

# Default ticker to use when no specific ticker is provided
DEFAULT_TICKER = "SPY"

# Trading settings
POSITION_SIZE = 0.05  # 5% of cash per trade
THRESHOLD = 0.45  # Legacy threshold retained for compatibility
LONG_ENTRY_THRESHOLD = 0.52
SHORT_ENTRY_THRESHOLD = 0.48
MIN_CONFIDENCE_GAP = 0.02

# Risk management
STOP_LOSS_PCT = 0.03  # 3% stop-loss
TAKE_PROFIT_PCT = 0.06  # 6% take-profit
MAX_HOLD_DAYS = 20  # Max days to hold a position

# Auto-tuning settings
AUTO_TUNE_ENABLED = True
MIN_SIGNALS_PER_TICKER = 30
TARGET_WIN_RATE = 0.50
TRADE_MODE = "equity"  # equity, option, or auto
ALLOW_SHORTS = True
USE_TREND_FILTER = False

# Options settings
OPTIONS_POSITION_SIZE = 0.05  # 5% of options buying power per trade
OPTIONS_ENABLED_UNDERLYINGS = TICKERS
OPTIONS_MIN_DTE = 7
OPTIONS_MAX_DTE = 45
OPTIONS_STRIKE_WINDOW = 0.08  # +/- 8% around spot when picking a contract

# Auto mode settings
AUTO_MODE_MIN_CONFIDENCE_GAP = 0.05
AUTO_MODE_ATR_MULTIPLIER = 1.0
AUTO_MODE_OPTION_COST_PENALTY = 0.015
AUTO_MODE_MAX_OPTION_LEVERAGE = 4.0

# Model settings
LOOK_BACK = 60  # 60-day sequence window
PERIOD = "10y"  # 10 years of historical data
EPOCHS = 50
BATCH_SIZE = 32

# Paths (derived from script location)
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")


def get_data_dir(ticker):
    """Get data directory for a specific ticker"""
    return os.path.join(DATA_DIR, ticker)


def get_model_dir(ticker):
    """Get model directory for a specific ticker"""
    return os.path.join(MODEL_DIR, ticker)


def ensure_dirs(ticker):
    """Ensure data and model directories exist for a ticker"""
    os.makedirs(get_data_dir(ticker), exist_ok=True)
    os.makedirs(get_model_dir(ticker), exist_ok=True)
