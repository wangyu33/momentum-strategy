# Frontier Factors Research

This directory is for structured exploration of factor layers that are common in
institutional momentum / trend-following stacks but are only partially present
in the current baseline.

Current focus order:

1. `carry / term-structure`
2. `multi-horizon trend consistency`
3. `correlation / crowding proxies`

Scripts write their artifacts to:

- `momentum_backtest/output/research/frontier_factors/`

Scripts:

- `analyze_carry_term_structure.py`
  Builds a factor inventory table and a carry / term-structure feasibility table
  for the current ETF universe.
- `compare_multihorizon_consistency.py`
  Compares the current single-window leader logic against an equal-weight
  multi-horizon consistency score built from `10/20/40/60/120` day momentum.
- `analyze_crowding_proxies.py`
  Builds simple price-based crowding proxies from the current universe, then
  tests whether high-crowding leaders tend to have worse forward returns.

Notes:

- The carry script is intentionally explicit about data gaps. For the current
  repo, the main missing layer is not code complexity but source data.
- The multi-horizon and crowding scripts are immediately runnable from existing
  cached price panels in `output/core/prices.csv`.
