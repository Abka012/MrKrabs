CREATE TABLE IF NOT EXISTS trade_schedule (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker TEXT NOT NULL,
  scheduled_date DATE NOT NULL,
  scheduled_hour_utc INT DEFAULT 14,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  executed_at TIMESTAMPTZ
);

CREATE INDEX idx_schedule_pending ON trade_schedule(status, scheduled_date, scheduled_hour_utc)
WHERE status = 'pending';
