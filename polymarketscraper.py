"""
polymarketscraper.py
─────────────────────────
Finds the common open bets among Polymarket's top leaderboard traders
and runs them through an AI agent to produce a ranked list with
confidence scores.
 
Pipeline
────────
1. Fetch top-N traders from the leaderboard (by PnL, weekly by default)
2. Fetch each trader's open positions
3. Aggregate: score every market by how many top traders hold it,
   weighted by leaderboard rank
4. Send the aggregated data to Claude for analysis
5. Print a ranked list of common bets with AI confidence scores
 
Usage
─────
    python polymarketscraper.py
 
Dependencies
────────────
    pip install requests anthropic

Instructions
────────────
To run python:

- There are two ways to provide your Anthropic API key:
  1. Set the environment variable ANTHROPIC_API_KEY before running the script.

  2. If the environment variable is not set, the script will prompt you to enter your API key securely (input will be hidden).


"""
 
import time
import json
import os
import getpass
import requests
import anthropic


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file next to this script into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def get_anthropic_client() -> anthropic.Anthropic:
    """
    Returns an Anthropic client. Uses ANTHROPIC_API_KEY from .env or env var if set,
    otherwise prompts the user to enter it (input is hidden, never printed).
    """
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nNo ANTHROPIC_API_KEY found in environment or .env file.")
        api_key = getpass.getpass("Enter your Anthropic API key: ").strip()
        if not api_key:
            raise ValueError("API key cannot be empty.")
    return anthropic.Anthropic(api_key=api_key)
 
# ─── Config ────────────────────────────────────────────────────────────────
 
BASE_URL        = "https://data-api.polymarket.com"
LEADERBOARD_TOP = 20      # How many traders to pull from the leaderboard
MIN_OVERLAP     = 2       # Only surface markets held by at least this many top traders
TIME_PERIOD     = "WEEK"  # DAY | WEEK | MONTH | ALL
CATEGORY        = "OVERALL"
SLEEP_BETWEEN   = 0.5     # Seconds between position requests (rate-limit courtesy)
 
 
# ─── Step 1: Leaderboard ───────────────────────────────────────────────────
 
def get_leaderboard_traders(limit: int = LEADERBOARD_TOP) -> list[dict]:
    """Return the top traders from the Polymarket leaderboard."""
    url = f"{BASE_URL}/v1/leaderboard"
    params = {
        "category":   CATEGORY,
        "timePeriod": TIME_PERIOD,
        "orderBy":    "PNL",
        "limit":      limit,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[ERROR] Leaderboard fetch failed: {e}")
        return []
 
 
# ─── Step 2: Active Positions ──────────────────────────────────────────────
 
def get_user_active_positions(address: str) -> list[dict]:
    """Return open positions for a wallet address."""
    url = f"{BASE_URL}/positions"
    params = {
        "user":          address,
        "sizeThreshold": 1,
        "limit":         100,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[WARN] Positions fetch failed for {address}: {e}")
        return []
 
 
# ─── Step 3: Aggregate & Score ─────────────────────────────────────────────
 
def build_smart_money_map(leaderboard: list[dict]) -> dict:
    """
    Collect every open position across top traders and score each market.
 
    Scoring formula (per trader who holds the market):
        score += (total_traders - rank + 1)   ← higher weight for top-ranked traders
 
    Returns a dict keyed by market conditionId (or title fallback):
        {
          market_id: {
            "title":        str,
            "outcome":      str,           # most common outcome held (YES/NO)
            "traders":      [{ rank, username, address, outcome, size, value }],
            "trader_count": int,
            "total_score":  float,
            "avg_value":    float,
          },
          ...
        }
    """
    total = len(leaderboard)
    market_map: dict[str, dict] = {}
 
    for rank, trader in enumerate(leaderboard, start=1):
        address  = trader.get("user") or trader.get("proxyWallet")
        username = trader.get("name") or trader.get("username", "Anonymous")
        pnl      = trader.get("pnl", 0)
 
        if not address:
            continue
 
        print(f"  Fetching positions: #{rank} {username} | PnL ${pnl:,.2f}")
        positions = get_user_active_positions(address)
 
        for pos in positions:
            # Skip resolved markets
            if pos.get("redeemed") or pos.get("closed"):
                continue
 
            market_id = pos.get("conditionId") or pos.get("marketId") or pos.get("title", "unknown")
            title     = pos.get("title", "Unknown Market")
            outcome   = pos.get("outcome", "N/A")
            size      = float(pos.get("tokens", 0) or 0)
            value     = float(pos.get("currentValue", 0) or pos.get("current", 0) or 0)
 
            weight = total - rank + 1  # rank 1 → highest weight
 
            if market_id not in market_map:
                market_map[market_id] = {
                    "title":        title,
                    "outcome":      outcome,
                    "traders":      [],
                    "trader_count": 0,
                    "total_score":  0.0,
                    "total_value":  0.0,
                }
 
            market_map[market_id]["traders"].append({
                "rank":     rank,
                "username": username,
                "address":  address,
                "outcome":  outcome,
                "size":     size,
                "value":    value,
            })
            market_map[market_id]["trader_count"] += 1
            market_map[market_id]["total_score"]  += weight
            market_map[market_id]["total_value"]  += value
 
        time.sleep(SLEEP_BETWEEN)
 
    # Compute avg_value and dominant outcome
    for mid, data in market_map.items():
        n = data["trader_count"]
        data["avg_value"] = data["total_value"] / n if n else 0
 
        # Dominant outcome = most common among holders
        outcomes = [t["outcome"] for t in data["traders"]]
        data["outcome"] = max(set(outcomes), key=outcomes.count) if outcomes else "N/A"
 
    return market_map
 
 
def filter_and_rank(market_map: dict, min_overlap: int = MIN_OVERLAP) -> list[dict]:
    """
    Keep only markets held by >= min_overlap top traders,
    sorted by total_score descending.
    """
    filtered = [
        {"market_id": mid, **data}
        for mid, data in market_map.items()
        if data["trader_count"] >= min_overlap
    ]
    return sorted(filtered, key=lambda x: x["total_score"], reverse=True)
 
 
# ─── Step 4: AI Agent ──────────────────────────────────────────────────────
 
# Compact system prompt — same instructions, ~40% fewer tokens
SYSTEM_PROMPT = (
    "You are a prediction-market analyst. Rank the provided markets by copy-bet quality. "
    "Fields: t=title, o=dominant outcome, n=trader count, s=rank-weighted score, v=avg $ value, split=YES/NO sides mixed. "
    "Return ONLY a JSON array, no markdown:\n"
    '[{"rank":1,"title":"...","outcome":"YES","confidence":85,"rationale":"one sentence","trader_count":3,"split_signal":false}]'
)
 
# Only the 4 fields Claude actually needs — traders array dropped entirely
_PAYLOAD_KEYS = ("title", "outcome", "trader_count", "total_score", "avg_value", "split_signal")
 
 
def _has_split(market: dict) -> bool:
    """True if top traders disagree on YES vs NO."""
    outcomes = {t["outcome"] for t in market.get("traders", [])}
    return len(outcomes) > 1
 
 
def run_ai_agent(ranked_markets: list[dict], top_n: int = 15) -> list[dict]:
    """
    Send the top-N aggregated markets to Claude and return a ranked list
    with confidence scores.
    top_n capped at 15 — beyond that signal quality drops and token cost rises.
    """
    slim = [
        {
            "title":        m["title"],
            "outcome":      m["outcome"],
            "trader_count": m["trader_count"],
            "total_score":  round(m["total_score"], 1),   # 1 decimal is enough
            "avg_value":    round(m["avg_value"], 0),      # whole dollars
            "split_signal": _has_split(m),
        }
        for m in ranked_markets[:top_n]
    ]
 
    client = get_anthropic_client()
 
    print("\n[AI] Sending aggregated positions to Claude for analysis...")
    response = client.messages.create(
        model      = "claude-haiku-4-5-20251001",  # Haiku: ~10x cheaper, plenty for structured JSON
        max_tokens = 1200,                          # JSON for 15 markets needs ~600–900 tokens
        system     = SYSTEM_PROMPT,
        messages   = [
            # Compact JSON (no indent) — same data, ~30% fewer input tokens
            {"role": "user", "content": json.dumps(slim, separators=(",", ":"))}
        ],
    )
 
    raw = response.content[0].text.strip()
 
    # Log token usage so you can monitor costs
    usage = response.usage
    print(f"     Tokens used — input: {usage.input_tokens} | output: {usage.output_tokens}")
 
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
 
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse AI response as JSON: {e}")
        print("Raw response:\n", raw)
        return []
 
 
# ─── Step 5: Display Results ───────────────────────────────────────────────
 
def print_results(ai_results: list[dict]) -> None:
    """Pretty-print the AI-ranked results to the terminal."""
    print("\n" + "═" * 70)
    print("  SMART MONEY CONSENSUS — TOP COMMON OPEN BETS")
    print("═" * 70)
 
    if not ai_results:
        print("  No results to display.")
        return
 
    for bet in ai_results:
        rank         = bet.get("rank", "?")
        title        = bet.get("title", "Unknown")
        outcome      = bet.get("outcome", "?")
        confidence   = bet.get("confidence", 0)
        rationale    = bet.get("rationale", "")
        trader_count = bet.get("trader_count", 0)
        split        = "⚠ SPLIT SIGNAL" if bet.get("split_signal") else ""
 
        bar_len = confidence // 5
        bar     = "█" * bar_len + "░" * (20 - bar_len)
 
        print(f"\n  #{rank}  {title}")
        print(f"      Bet: {outcome}  |  Traders: {trader_count}  {split}")
        print(f"      Confidence: [{bar}] {confidence}/100")
        print(f"      {rationale}")
 
    print("\n" + "═" * 70)
 
 
# ─── Main ──────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 70)
    print("  POLYMARKET SMART MONEY TRACKER")
    print(f"  Leaderboard: top {LEADERBOARD_TOP} traders | Period: {TIME_PERIOD}")
    print("=" * 70)
 
    # 1. Leaderboard
    print("\n[1/4] Fetching leaderboard...")
    leaderboard = get_leaderboard_traders(LEADERBOARD_TOP)
    if not leaderboard:
        print("No leaderboard data. Exiting.")
        return
    print(f"      Found {len(leaderboard)} traders.")
 
    # 2 & 3. Positions + Aggregation
    print(f"\n[2/4] Fetching open positions for each trader...")
    market_map = build_smart_money_map(leaderboard)
    print(f"      Discovered {len(market_map)} unique markets across all traders.")
 
    print(f"\n[3/4] Filtering to markets with >= {MIN_OVERLAP} top-trader overlap...")
    ranked = filter_and_rank(market_map, min_overlap=MIN_OVERLAP)
    print(f"      {len(ranked)} markets meet the overlap threshold.")
 
    if not ranked:
        print("\nNo overlapping bets found. Try lowering MIN_OVERLAP or expanding the leaderboard.")
        return
 
    # 4. AI Analysis
    print("\n[4/4] Running AI agent analysis...")
    ai_results = run_ai_agent(ranked)
 
    # 5. Display
    print_results(ai_results)
 
    # Optionally save raw JSON output
    out_file = "smart_money_output.json"
    with open(out_file, "w") as f:
        json.dump({"ranked_markets": ranked, "ai_analysis": ai_results}, f, indent=2)
    print(f"\n  Full output saved to: {out_file}")
 
 
if __name__ == "__main__":
    main()
 