"""
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

import time
import json
import os
import getpass
from datetime import datetime, timezone
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
    """Returns an Anthropic client, prompting for an API key only if needed."""
    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nNo ANTHROPIC_API_KEY found in environment or .env file.")
        api_key = getpass.getpass("Enter your Anthropic API key: ").strip()
        if not api_key:
            raise ValueError("API key cannot be empty.")
    return anthropic.Anthropic(api_key=api_key)


# ─── Config ────────────────────────────────────────────────────────────────

BASE_URL             = "https://data-api.polymarket.com"
LEADERBOARD_TOP      = 50      # How many traders to pull from the leaderboard
MIN_OVERLAP          = 2       # Only surface markets held by at least this many top traders
TIME_PERIOD          = "MONTH" # DAY | WEEK | MONTH | ALL
CATEGORY             = "OVERALL"
SLEEP_BETWEEN        = 0.5     # Seconds between position requests (rate-limit courtesy)
ACTIVE_SIZE_MIN      = 1.0     # Minimum token size for an open position to count
POSITION_FETCH_LIMIT = 500     # Per-trader cap for /positions (covers open + resolved)


# ─── Small helpers ─────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    """Coerce to float, returning `default` on None / non-numeric."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_end_date(value):
    """Parse Polymarket ISO timestamps. Returns aware datetime or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return None


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
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.RequestException as e:
        print(f"[ERROR] Leaderboard fetch failed: {e}")
        return []


# ─── Step 2: Positions (open + resolved, one call per trader) ──────────────

def get_user_positions(address: str) -> list[dict]:
    """
    Return ALL positions for a wallet — open and resolved.
    sizeThreshold=0 keeps zero-size (already-settled / fully-redeemed) entries
    in the response so we can compute win history from the same payload.
    """
    url = f"{BASE_URL}/positions"
    params = {
        "user":          address,
        "sizeThreshold": 0,
        "limit":         POSITION_FETCH_LIMIT,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.RequestException as e:
        print(f"[WARN] Positions fetch failed for {address}: {e}")
        return []


def split_positions(positions: list[dict]) -> tuple[list[dict], dict]:
    """
    Split a user's positions into:
      - active:   unresolved positions with size >= ACTIVE_SIZE_MIN
      - history:  {wins, losses, total_resolved, raw_win_rate, smoothed_win_rate}

    Win  = redeemable=True, OR resolved with curPrice >= 0.99 ($1.00 settlement).
    Loss = resolved with curPrice <= 0.01 ($0.00 settlement).
    Resolved = redeemable OR endDate has passed.

    Laplace smoothing (wins+1)/(total+2) is applied to the win rate so that
    small samples (e.g. 1/1) don't dominate the rank-weighting.
    """
    now = datetime.now(timezone.utc)
    active: list[dict] = []
    wins = 0
    losses = 0

    for pos in positions:
        if not isinstance(pos, dict):
            continue

        redeemable = bool(pos.get("redeemable"))
        end_dt     = _parse_end_date(pos.get("endDate"))
        is_past    = end_dt is not None and end_dt <= now
        resolved   = redeemable or is_past

        if resolved:
            cur_price = _safe_float(pos.get("curPrice"), default=-1.0)
            if redeemable or cur_price >= 0.99:
                wins += 1
            elif 0.0 <= cur_price <= 0.01:
                losses += 1
            # else: ambiguous (refunded, mid-resolve, etc.) — skip
        else:
            if _safe_float(pos.get("size")) >= ACTIVE_SIZE_MIN:
                active.append(pos)

    total = wins + losses
    raw_rate      = (wins / total) if total > 0 else 0.0
    smoothed_rate = (wins + 1) / (total + 2)

    return active, {
        "wins":              wins,
        "losses":            losses,
        "total_resolved":    total,
        "raw_win_rate":      raw_rate,
        "smoothed_win_rate": smoothed_rate,
    }


def fetch_and_enrich_traders(leaderboard: list[dict]) -> list[dict]:
    """
    For each trader on the leaderboard, fetch positions once, derive win
    history from the resolved subset, and keep the open subset for
    aggregation. Returns enriched trader dicts.
    """
    enriched: list[dict] = []
    for pnl_rank, trader in enumerate(leaderboard, start=1):
        if not isinstance(trader, dict):
            continue
        address  = trader.get("user") or trader.get("proxyWallet")
        username = trader.get("name") or trader.get("username") or "Anonymous"
        pnl      = _safe_float(trader.get("pnl"))

        if not address:
            continue

        print(f"  Fetching positions: PNL #{pnl_rank} {username} | PnL ${pnl:,.2f}")
        positions = get_user_positions(address)
        active, history = split_positions(positions)

        enriched.append({
            "address":           address,
            "username":          username,
            "pnl":               pnl,
            "pnl_rank":          pnl_rank,
            "wins":              history["wins"],
            "losses":            history["losses"],
            "total_resolved":    history["total_resolved"],
            "raw_win_rate":      history["raw_win_rate"],
            "smoothed_win_rate": history["smoothed_win_rate"],
            "active":            active,
        })

        time.sleep(SLEEP_BETWEEN)
    return enriched


# ─── Step 3: Aggregate & Score ─────────────────────────────────────────────

def build_smart_money_map(enriched: list[dict]) -> dict:
    """
    Collect every open position across traders and score each market.

    Weight formula per trader who holds the market:
        base   = (total_traders - new_rank + 1)      # new_rank uses win-rate order
        weight = base * (1 + smoothed_win_rate)      # bigger boost for proven winners

    `enriched` must already be sorted by descending smoothed_win_rate, so
    list index + 1 == new rank.

    Returns a dict keyed by market conditionId (or title fallback):
        {
          market_id: {
            "title":               str,
            "outcome":             str,           # dominant outcome (YES/NO)
            "traders":             [{ rank, username, address, outcome, size, value, win_rate, weight }],
            "trader_count":        int,
            "total_score":         float,
            "avg_value":           float,
            "avg_holder_win_rate": float,         # smoothed; passed to the AI
          },
          ...
        }
    """
    total = len(enriched)
    market_map: dict[str, dict] = {}

    for new_rank, trader in enumerate(enriched, start=1):
        address  = trader["address"]
        username = trader["username"]
        sm_rate  = trader["smoothed_win_rate"]
        raw_rate = trader["raw_win_rate"]
        base     = total - new_rank + 1
        weight   = base * (1.0 + sm_rate)

        for pos in trader["active"]:
            market_id = pos.get("conditionId") or pos.get("marketId") or pos.get("title", "unknown")
            title     = pos.get("title", "Unknown Market")
            outcome   = pos.get("outcome", "N/A")
            size      = _safe_float(pos.get("tokens"))
            value     = _safe_float(pos.get("currentValue"), _safe_float(pos.get("current")))

            if market_id not in market_map:
                market_map[market_id] = {
                    "title":            title,
                    "outcome":          outcome,
                    "traders":          [],
                    "trader_count":     0,
                    "total_score":      0.0,
                    "total_value":      0.0,
                    "total_holder_wr":  0.0,
                }

            market_map[market_id]["traders"].append({
                "rank":     new_rank,
                "username": username,
                "address":  address,
                "outcome":  outcome,
                "size":     size,
                "value":    value,
                "win_rate": round(raw_rate, 3),
                "weight":   round(weight, 2),
            })
            market_map[market_id]["trader_count"]    += 1
            market_map[market_id]["total_score"]     += weight
            market_map[market_id]["total_value"]     += value
            market_map[market_id]["total_holder_wr"] += sm_rate

    for data in market_map.values():
        n = data["trader_count"]
        data["avg_value"]           = data["total_value"] / n if n else 0.0
        data["avg_holder_win_rate"] = data["total_holder_wr"] / n if n else 0.0
        outcomes = [t["outcome"] for t in data["traders"]]
        data["outcome"] = max(set(outcomes), key=outcomes.count) if outcomes else "N/A"

    return market_map


def filter_and_rank(market_map: dict, min_overlap: int = MIN_OVERLAP) -> list[dict]:
    """Keep markets held by >= min_overlap traders, sorted by total_score desc."""
    filtered = [
        {"market_id": mid, **data}
        for mid, data in market_map.items()
        if data["trader_count"] >= min_overlap
    ]
    return sorted(filtered, key=lambda x: x["total_score"], reverse=True)


# ─── Step 4: AI Agent ──────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a prediction-market analyst. Rank the provided markets by copy-bet quality. "
    "Fields: title, outcome=dominant side, trader_count=holders, "
    "total_score=rank-weighted score (higher = more conviction from high-win-rate traders), "
    "avg_value=avg $ position size, whr=avg holder win rate (0-1, Laplace-smoothed), "
    "split_signal=YES/NO sides mixed. "
    "Treat high `whr` as a stronger signal even when `trader_count` is small. "
    "Return ONLY a JSON array, no markdown:\n"
    '[{"rank":1,"title":"...","outcome":"YES","confidence":85,"rationale":"one sentence","trader_count":3,"split_signal":false}]'
)


def _has_split(market: dict) -> bool:
    """True if top traders disagree on YES vs NO."""
    outcomes = {t["outcome"] for t in market.get("traders", [])}
    return len(outcomes) > 1


def _recover_truncated_json_array(raw: str):
    """
    Salvage a JSON array of objects that was cut off mid-item (e.g. when the
    model hit max_tokens). Walks the string with brace-depth tracking and
    truncates after the last fully-balanced top-level object, then closes
    the array. Returns the parsed list or None if recovery isn't possible.
    """
    s = raw.lstrip()
    if not s.startswith("["):
        return None
    depth = 0
    in_string = False
    escape = False
    last_complete_end = -1  # index (exclusive) just after the last balanced top-level }
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete_end = i + 1
    if last_complete_end == -1:
        return None
    repaired = s[:last_complete_end] + "]"
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def run_ai_agent(ranked_markets: list[dict], top_n: int = 15) -> list[dict]:
    """
    Send the top-N aggregated markets to Claude and return a ranked list
    with confidence scores. top_n capped at 15 — beyond that signal quality
    drops and token cost rises.
    """
    slim = [
        {
            "title":        m["title"],
            "outcome":      m["outcome"],
            "trader_count": m["trader_count"],
            "total_score":  round(m["total_score"], 1),
            "avg_value":    round(m["avg_value"], 0),
            "whr":          round(m["avg_holder_win_rate"], 2),
            "split_signal": _has_split(m),
        }
        for m in ranked_markets[:top_n]
    ]

    client = get_anthropic_client()

    print("\n[AI] Sending aggregated positions to Claude for analysis...")
    response = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 2500,   # 15 markets w/ rationales ≈ 1.5–2k tokens; headroom for safety
        system     = SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": json.dumps(slim, separators=(",", ":"))}
        ],
    )

    raw = response.content[0].text.strip()
    usage = response.usage
    print(f"     Tokens used — input: {usage.input_tokens} | output: {usage.output_tokens}")

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        recovered = _recover_truncated_json_array(raw)
        if recovered is not None:
            stop = getattr(response, "stop_reason", None)
            print(
                f"[WARN] AI response was truncated (stop_reason={stop}); "
                f"recovered {len(recovered)} complete bets. Consider raising max_tokens."
            )
            return recovered
        print(f"[ERROR] Failed to parse AI response as JSON: {e}")
        print("Raw response:\n", raw)
        return []


# ─── Step 5: Display Results ───────────────────────────────────────────────

def print_win_rate_leaderboard(enriched: list[dict]) -> None:
    """Print the leaderboard re-sorted by historical win rate."""
    print("\n" + "═" * 80)
    print("  LEADERBOARD SORTED BY HISTORICAL WIN RATE (from resolved positions)")
    print("═" * 80)
    print(f"  {'WR#':<5}{'PNL#':<6}{'Trader':<28}{'W':>5}{'L':>5}{'Rate':>9}{'PnL ($)':>17}")
    print("  " + "─" * 75)
    for wr_rank, t in enumerate(enriched, start=1):
        rate_str = f"{t['raw_win_rate']*100:5.1f}%" if t["total_resolved"] > 0 else "  n/a"
        username = (t["username"] or "")[:26]
        print(
            f"  {wr_rank:<5}{t['pnl_rank']:<6}{username:<28}"
            f"{t['wins']:>5}{t['losses']:>5}{rate_str:>9}"
            f"{t['pnl']:>17,.0f}"
        )
    print("═" * 80)


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
        confidence   = int(_safe_float(bet.get("confidence")))
        rationale    = bet.get("rationale", "")
        trader_count = bet.get("trader_count", 0)
        split        = "⚠ SPLIT SIGNAL" if bet.get("split_signal") else ""

        bar_len = max(0, min(20, confidence // 5))
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
    print("\n[1/5] Fetching leaderboard by PNL...")
    leaderboard = get_leaderboard_traders(LEADERBOARD_TOP)
    if not leaderboard:
        print("No leaderboard data. Exiting.")
        return
    print(f"      Found {len(leaderboard)} traders.")

    # 2. Positions + win history (one /positions call per trader)
    print(f"\n[2/5] Fetching positions and computing win history per trader...")
    enriched = fetch_and_enrich_traders(leaderboard)
    if not enriched:
        print("No trader data could be enriched. Exiting.")
        return

    # 3. Re-sort by smoothed win rate (tiebreak: original PNL rank)
    enriched.sort(key=lambda t: (-t["smoothed_win_rate"], t["pnl_rank"]))
    print_win_rate_leaderboard(enriched)

    # 4. Aggregate (weights now incorporate each trader's win rate)
    print(f"\n[3/5] Aggregating common open bets (weighted by win rate)...")
    market_map = build_smart_money_map(enriched)
    print(f"      Discovered {len(market_map)} unique markets across all traders.")

    print(f"\n[4/5] Filtering to markets with >= {MIN_OVERLAP} top-trader overlap...")
    ranked = filter_and_rank(market_map, min_overlap=MIN_OVERLAP)
    print(f"      {len(ranked)} markets meet the overlap threshold.")

    if not ranked:
        print("\nNo overlapping bets found. Try lowering MIN_OVERLAP or expanding the leaderboard.")
        return

    # 5. AI Analysis
    print("\n[5/5] Running AI agent analysis...")
    ai_results = run_ai_agent(ranked)

    print_results(ai_results)

    # Save full output (strip raw active positions from the saved leaderboard
    # — they're already in ranked_markets via traders[])
    out_file = "smart_money_output.json"
    traders_out = [
        {k: v for k, v in t.items() if k != "active"}
        for t in enriched
    ]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "win_rate_leaderboard": traders_out,
            "ranked_markets":       ranked,
            "ai_analysis":          ai_results,
        }, f, indent=2)
    print(f"\n  Full output saved to: {out_file}")


if __name__ == "__main__":
    main()
