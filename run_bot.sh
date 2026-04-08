#!/bin/bash

# MrKrabs Trading Bot - Scheduler Script
# Run this script to schedule periodic execution of the trading bot

PROJECT_DIR="/home/abka/Documents/MrKrabs"
VENV_ACTIVATE="$PROJECT_DIR/venv/bin/activate"
PYTHON="$PROJECT_DIR/venv/bin/python"
BOT_SCRIPT="$PROJECT_DIR/alpaca_trader.py"

cd "$PROJECT_DIR"

# Schedule: Run every 5 minutes (customize with crontab)
# */5 9-16 * * 1-5 /home/abka/Documents/MrKrabs/run_bot.sh

echo "To schedule the bot, add this line to your crontab:"
echo "*/5 9-16 * * 1-5 /home/abka/Documents/MrKrabs/run_bot.sh"
echo ""
echo "Or run manually:"
echo "source $VENV_ACTIVATE && python $BOT_SCRIPT"

# Run once for testing
echo ""
echo "Running bot now..."
source $VENV_ACTIVATE
python $BOT_SCRIPT
