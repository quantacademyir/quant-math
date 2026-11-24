"""Builds the simulated sample data used by the course "ریاضی و آمار برای Quant Developer".

The numbers are SIMULATED (seeded), not real market data.

Output (committed):
  data/trades_sample.csv   one day of simulated trades (2025-12-01, UTC)
                                             columns: timestamp, side, size

How the trades are made, so the lesson on the Poisson process has something true to find:
  - the trade rate follows a time-of-day curve (quiet at night, busiest around 14:30 UTC);
  - on top of it the rate is multiplied by a random "burst" factor that changes every 20 seconds
    (log-normal, mean 1), so trades arrive in clusters: the counts per minute are more spread out
    than a plain Poisson process would give (over-dispersion);
  - inside each 20-second block the trade times are uniform (a Poisson process given the rate);
  - the side is "buy" with probability 0.55, independently of everything else;
  - the size is log-normal (median about 0.015 BTC).

(data/btc_daily_5y.csv is a copy of the file made by scripts/make-backtest-data.py.)

Run: python3 scripts/make_quant_math_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1405)
BLOCK = 20  # seconds
n_blocks = 24 * 3600 // BLOCK

block_start = np.arange(n_blocks) * BLOCK
hour = (block_start + BLOCK / 2) / 3600
rate_tod = 0.10 + 0.25 * np.exp(-(((hour - 14.5) / 3.5) ** 2))  # trades per second
burst_sigma = 0.5
burst = np.exp(rng.normal(-burst_sigma**2 / 2, burst_sigma, n_blocks))  # mean 1
counts = rng.poisson(rate_tod * burst * BLOCK)

block_of_trade = np.repeat(np.arange(n_blocks), counts)
seconds = block_start[block_of_trade] + rng.uniform(0, BLOCK, len(block_of_trade))
seconds.sort()

ts = pd.Timestamp("2025-12-01", tz="UTC") + pd.to_timedelta(np.round(seconds * 1000).astype(int), unit="ms")
side = np.where(rng.random(len(ts)) < 0.55, "buy", "sell")
size = np.round(rng.lognormal(np.log(0.015), 1.1, len(ts)), 4).clip(min=0.0001)

df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z", "side": side, "size": size})
df.to_csv(OUT / "trades_sample.csv", index=False)
print(len(df), "trades;", f"{(OUT / 'trades_sample.csv').stat().st_size / 1e6:.2f} MB")
