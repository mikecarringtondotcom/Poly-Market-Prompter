Instructions

polymarketprompter.py
─────────────────────────
Finds the common open bets among Polymarket's top leaderboard traders,
ranks the traders by their historical win rate, and runs a weighted
aggregation through an AI agent to produce a ranked list of common bets
with confidence scores.

Pipeline
────────
1. Fetch top-N traders from the leaderboard (by PnL).
2. For each trader, pull ALL positions (open + resolved) in one call.
3. Derive each trader's win history from resolved positions:
     win  = final settlement value $1.00  (or redeemable=true)
     loss = final settlement value $0.00  (resolved + curPrice = 0)
4. Re-rank the leaderboard by smoothed win rate.
5. Aggregate active positions, weighting each trader by:
     weight = (n_traders - new_rank + 1) * (1 + smoothed_win_rate)
6. Send the aggregated data to Claude for analysis.
7. Print the win-rate-sorted leaderboard and the AI-ranked bets.

Usage
─────
    python polymarketprompter.py

Dependencies
────────────
    pip install requests anthropic

Instructions
────────────
- Set the environment variable ANTHROPIC_API_KEY (or put it in a .env file
  next to this script). If neither is set, you'll be prompted at runtime.
"""
