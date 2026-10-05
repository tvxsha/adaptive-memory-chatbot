"""Manual smoke test for a RUNNING server (uvicorn server:app --port 8000).

Sends a fixed set of varied messages through /add_message, prints what the
pipeline decided for each one next to what we'd expect, then runs a few
/retrieve queries. Use it to eyeball categorization quality and to check that
everything (Groq + HF embeddings + SQLite) still works after a long break.

Usage (server must already be running in another terminal):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --keep            # don't delete the test data afterwards
    python scripts/smoke_test.py --url http://localhost:8000
"""
import argparse
import sys
import time

import httpx

# (message, category we'd expect, note)
# "expect" is only a guide for eyeballing: categorization is an LLM judgment,
# so a mismatch isn't automatically a bug. Consistent mismatches on one kind
# of message are what's worth investigating.
MESSAGES = [
    ("I live in Delhi and I work as a data analyst.",       "fact",           ""),
    ("My favorite food is biryani.",                         "preference",     ""),
    ("I hate cilantro in my food.",                          "preference",     ""),
    ("I'm trying to learn Spanish this year.",               "goal",           ""),
    ("I've decided to go with the blue laptop.",             "decision",       ""),
    ("I already booked my flight to Goa.",                   "completed_task", ""),
    ("lol okay thanks!",                                     "recent_chat",    "filler: should be skipped"),
    ("hmm",                                                  "recent_chat",    "filler: should be skipped"),
    ("My favorite food is biryani.",                         "preference",     "exact repeat: should be skipped"),
    ("Actually I just moved to Mumbai last week.",           "fact",           "contradicts 'I live in Delhi': should REPLACE"),
    ("I live in Mumbai and I work as a data analyst.",       "fact",           "near-repeat of the Mumbai fact: skip or replace"),
    ("My sister is getting married in December.",            "fact",           ""),
]

QUERIES = [
    "where does the user live?",
    "what food does the user like?",
    "what is the user studying?",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--keep", action="store_true", help="keep the test conversation in the database")
    args = parser.parse_args()

    conv_id = f"smoke-{int(time.time())}"
    client = httpx.Client(base_url=args.url, timeout=60)

    try:
        health = client.get("/health").json()
    except httpx.HTTPError as e:
        print(f"Could not reach the server at {args.url}: {e}")
        print("Start it first in another terminal:  uvicorn server:app --reload --port 8000")
        return 1
    print(f"Server health: {health}\n")

    def request(method: str, path: str, **kwargs):
        """Like client.request, but returns None (and prints why) on a network
        error instead of crashing the whole script with a traceback."""
        try:
            return client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            print(f"   network error on {method} {path}: {e}")
            return None

    print(f"=== /add_message  (conversation: {conv_id}) ===")
    print(f"{'#':<3} {'got category':<15} {'expected':<15} {'action':<9} message")
    print("-" * 100)
    failures = 0
    for i, (text, expected, note) in enumerate(MESSAGES, start=1):
        resp = request(
            "POST", "/add_message",
            json={"conversation_id": conv_id, "role": "user", "text": text},
        )
        if resp is None:
            failures += 1
            continue
        if resp.status_code != 200:
            failures += 1
            print(f"{i:<3} HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        data = resp.json()
        got = data.get("category") or "-"
        flag = "" if got == expected else "  <-- differs"
        print(f"{i:<3} {got:<15} {expected:<15} {data['action']:<9} {text[:48]}{flag}")
        if note:
            print(f"{'':<3} {'':<15} {'':<15} {'':<9} ({note})")
        reason = data.get("reasoning") or data.get("reason")
        if reason:
            print(f"{'':<3} {'':<15} {'':<15} {'':<9} -> {reason}")
            time.sleep(3)  # up to 4 Groq calls per message; stay under the free-tier rate limit

    print(f"\n=== /retrieve ===")
    for q in QUERIES:
        resp = request("GET", "/retrieve", params={"conversation_id": conv_id, "query": q, "top_k": 3})
        if resp is None:
            failures += 1
            continue
        if resp.status_code != 200:
            failures += 1
            print(f"Q: {q}\n   HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        print(f"Q: {q}")
        for r in resp.json()["results"]:
            print(f"   [{r['category']:<14}] sim={r['similarity']:<6} score={r['score']:<6} {r['text']}")

    print("\n=== /export ===")
    resp = request("GET", "/export", params={"conversation_id": conv_id})
    if resp is None or resp.status_code != 200:
        failures += 1
        print("   export failed")
    else:
        snapshot = resp.json()
        print(f"{len(snapshot['memory'])} memory item(s) in snapshot:")
        for item in snapshot["memory"]:
            print(f"   {item['score']:<6} [{item['category']}] {item['text']}")

    if not args.keep:
        request("DELETE", f"/conversation/{conv_id}")
        print(f"\n(test conversation {conv_id} deleted; use --keep to retain it)")

    print("\nDONE" if failures == 0 else f"\nFINISHED WITH {failures} HTTP ERROR(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())