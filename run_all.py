#!/usr/bin/env python3
"""
MrKrabs - Full Pipeline Script
Runs: prepare_data -> train_model -> backtest -> (optional) trade

Usage:
    python run_all.py --ticker SPY    # Single ticker
    python run_all.py --all           # All tickers from config
    python run_all.py --all --trade   # All tickers + start trading
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run full pipeline: prepare + train + backtest + trade"
    )
    parser.add_argument("--ticker", type=str, default=None, help="Specific ticker")
    parser.add_argument(
        "--all", action="store_true", help="Process all tickers from config"
    )
    parser.add_argument(
        "--trade", action="store_true", help="Also run live trading after backtest"
    )
    args = parser.parse_args()

    tickers = []

    if args.all:
        tickers = config.TICKERS
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [config.DEFAULT_TICKER]

    print(f"\n{'#' * 60}")
    print(f"# MrKrabs Full Pipeline")
    print(f"# Tickers: {tickers}")
    print(f"# Trade: {args.trade}")
    print(f"{'#' * 60}")

    all_results = []

    for ticker in tickers:
        print(f"\n\n{'#' * 60}")
        print(f"# PROCESSING: {ticker}")
        print(f"{'#' * 60}")

        # Step 1: Prepare data
        success = run_command(
            [sys.executable, "prepare_data.py", "--ticker", ticker],
            f"1. Preparing data for {ticker}",
        )
        if not success:
            print(f"ERROR: Failed to prepare data for {ticker}")
            continue

        # Step 2: Train models
        success = run_command(
            [sys.executable, "train_model.py", "--ticker", ticker],
            f"2. Training models for {ticker}",
        )
        if not success:
            print(f"ERROR: Failed to train models for {ticker}")
            continue

        # Step 3: Backtest
        success = run_command(
            [sys.executable, "backtest.py", "--ticker", ticker],
            f"3. Backtesting {ticker}",
        )
        if not success:
            print(f"WARNING: Backtest failed for {ticker}")

        all_results.append(ticker)

    # Summary
    print(f"\n\n{'#' * 60}")
    print(f"# PIPELINE COMPLETE")
    print(f"{'#' * 60}")
    print(f"\nProcessed tickers: {all_results}")

    if args.trade:
        print(f"\nStarting live trading...")
        if len(tickers) == 1:
            trade_cmd = [sys.executable, "alpaca_trader.py", "--ticker", tickers[0]]
        else:
            trade_cmd = [sys.executable, "alpaca_trader.py", "--all"]

        run_command(trade_cmd, "Live Trading")
    else:
        print(f"\nTo start live trading later:")
        print(f"  python alpaca_trader.py --ticker {tickers[0]}")
        print(f"  or")
        print(f"  python alpaca_trader.py --all")


if __name__ == "__main__":
    main()
