#!/bin/bash

PROJECT_DIR="/home/abka/Documents/MrKrabs"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
PYTHON="$PROJECT_DIR/venv/bin/python"
BOT_SCRIPT="$PROJECT_DIR/alpaca_trader.py"

cd "$PROJECT_DIR"

echo "Running bot at 14:00 UTC (10:00 AM ET)..."
source "$VENV_ACTIVATE"
python "$BOT_SCRIPT" --all --mode auto