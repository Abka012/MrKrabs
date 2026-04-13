import os
import sys
import json
from datetime import datetime

import requests

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from alpaca_trader import find_trades, execute_trades

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
}


def get_today_schedule():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"{SUPABASE_URL}/rest/v1/trade_schedule"
    params = {
        "scheduled_date": f"eq.{today}",
        "scheduled_hour_utc": "eq.14",
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code == 200:
        data = resp.json()
        return data[0] if data else None
    return None


def create_schedule():
    """Create a schedule if none exists (backup for create_schedule.yml delays)"""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"{SUPABASE_URL}/rest/v1/trade_schedule"
    data = {
        "ticker": "ALL",
        "scheduled_date": today,
        "scheduled_hour_utc": 14,
        "status": "pending"
    }
    resp = requests.post(url, headers=HEADERS, json=data)
    if resp.status_code in [200, 201]:
        print("Created schedule (backup for delayed create_schedule.yml)")
        return get_today_schedule()
    return None


def update_schedule_status(schedule_id, status):
    url = f"{SUPABASE_URL}/rest/v1/trade_schedule"
    params = {"id": f"eq.{schedule_id}"}
    data = {
        "status": status,
        "executed_at": datetime.utcnow().isoformat() + "Z" if status == "executed" else None,
    }
    resp = requests.patch(url, headers=HEADERS, params=params, json=data)
    return resp.status_code in [200, 204]


def main():
    print(f"\n{'#' * 60}")
    print(f"# Trade Executor - Checking Schedule")
    print(f"{'#' * 60}\n")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("ERROR: SUPABASE_URL or SUPABASE_ANON_KEY not set")
        return

    # Time check: only execute after 14:00 UTC
    now = datetime.utcnow()
    if now.hour < 14:
        print(f"Too early to trade (UTC: {now.hour}:{now.minute:02d})")
        print(f"Trading starts at 14:00 UTC")
        return

    print(f"Current UTC time: {now.hour}:{now.minute:02d}")

    schedule = get_today_schedule()

    # If no schedule exists, create one on-the-fly (backup for delayed create_schedule.yml)
    if not schedule:
        print("No schedule found for today, creating one...")
        schedule = create_schedule()
        if not schedule:
            print("Failed to create schedule")
            return

    print(f"Found schedule: {schedule['id']}")
    print(f"Status: {schedule['status']}")
    print(f"Scheduled for: {schedule['scheduled_date']} at {schedule['scheduled_hour_utc']}:00 UTC")

    # Check if already executed today
    if schedule["status"] == "executed":
        print("\n>>> Trade already executed today, skipping")
        return

    if schedule["status"] == "skipped":
        print("\n>>> Trade was skipped today (no tickers met threshold), skipping")
        return

    # Execute trades
    print("\n--- Finding trades ---")
    trades = find_trades()

    if trades:
        print(f"\n--- Executing {len(trades)} trade(s) ---")
        execute_trades(trades)
        # Update schedule status to executed
        if update_schedule_status(schedule["id"], "executed"):
            print("\nSchedule updated to 'executed'")
    else:
        print("No trades to execute - no tickers met threshold")
        # Update schedule status to skipped
        if update_schedule_status(schedule["id"], "skipped"):
            print("Schedule updated to 'skipped'")


if __name__ == "__main__":
    main()