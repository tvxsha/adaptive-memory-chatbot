"""Evaluates contradiction handling on a labelled set of (old, new) statement pairs.

The decision that matters: should the NEW statement REPLACE the old memory?
    updates / contradicts -> replace          consistent / unrelated -> keep both

Two parts:
  1. GATE (free, no Groq): is the pair similar enough to reach the LLM judge at all
     (cosine >= SIMILARITY_THRESHOLD)? A real update that fails the gate can never be
     replaced. Also prints a threshold sweep.
  2. JUDGE (--judge, uses Groq, ~1 call per gated pair): runs the real judge and prints
     a confusion matrix plus precision / recall of "replace".

    python evaluation/contradiction_eval.py              # gate only, free
    python evaluation/contradiction_eval.py --judge      # + the LLM judge

NOTE: the pairs below were written by the project authors, not by independent
annotators, so treat the numbers as indicative, not as a benchmark.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory_engine import contradiction  # noqa: E402
from memory_engine.embeddings import cosine_similarity, embed  # noqa: E402

# (old, new, label)
PAIRS = [
    # ---- updates: the old fact is no longer true ----
    ("I live in Delhi and work as a data analyst.", "I just moved to Mumbai last month.", "updates"),
    ("I work at Infosys.", "I quit Infosys and joined Google last week.", "updates"),
    ("My exam is on Friday.", "The exam got postponed to next Tuesday.", "updates"),
    ("I'm single.", "I got engaged yesterday!", "updates"),
    ("I drive a Honda Civic.", "I sold my Civic and bought a Tesla.", "updates"),
    ("I'm vegetarian.", "I started eating meat again after five years.", "updates"),
    ("My phone is an iPhone 12.", "I switched to a Pixel 8 last week.", "updates"),
    ("I'm studying computer science at VIT.", "I dropped out of VIT and I'm working full time now.", "updates"),
    ("The team meeting is at 3 PM on Monday.", "The team meeting has been moved to 5 PM.", "updates"),
    ("I'm planning to learn guitar.", "I gave up on guitar, I'm learning piano instead.", "updates"),
    ("My sister lives with me.", "My sister moved out to her own apartment.", "updates"),
    ("I have a cat named Luna.", "Luna passed away last night.", "updates"),
    ("I'm training for a marathon in March.", "I tore my ligament, so the marathon is off.", "updates"),
    ("My budget for the laptop is 60000 rupees.", "I raised my laptop budget to 90000 rupees.", "updates"),
    ("I'm a junior developer.", "I got promoted to senior developer.", "updates"),
    # ---- contradictions: both cannot be true ----
    ("I love spicy food.", "I can't stand spicy food.", "contradicts"),
    ("I have never been abroad.", "I visited Japan and Korea last year.", "contradicts"),
    ("I prefer working in the morning.", "I'm a night owl and do my best work after midnight.", "contradicts"),
    ("I don't drink coffee.", "I drink three cups of coffee every day.", "contradicts"),
    ("I'm allergic to peanuts.", "I'm not allergic to anything.", "contradicts"),
    ("I hate cricket.", "Cricket is my favourite sport.", "contradicts"),
    ("My favourite language is Python.", "I really dislike Python and avoid it whenever I can.", "contradicts"),
    ("I'm an only child.", "I have two older brothers.", "contradicts"),
    ("I always take the metro to work.", "I never take the metro, I drive everywhere.", "contradicts"),
    ("I'm right-handed.", "I'm left-handed.", "contradicts"),
    # ---- consistent: old fact still true (elaboration, repeat, other person) ----
    ("I have a dog.", "I have a golden retriever named Max.", "consistent"),
    ("I work as a data analyst.", "I mostly use SQL and Python for my analysis work.", "consistent"),
    ("I live in Delhi.", "I live in Delhi near Connaught Place.", "consistent"),
    ("I like hiking.", "I hiked the Kedarkantha trek last winter and loved it.", "consistent"),
    ("I'm studying computer science.", "I'm specialising in machine learning this semester.", "consistent"),
    ("My exam is on Friday.", "I've started revising for the Friday exam.", "consistent"),
    ("I enjoy cooking.", "I made pasta from scratch yesterday.", "consistent"),
    ("I have a younger brother.", "My brother just turned sixteen.", "consistent"),
    ("I want to learn Spanish.", "I downloaded Duolingo to practise Spanish.", "consistent"),
    ("I drive a Honda Civic.", "My Civic needs a service next month.", "consistent"),
    ("I'm vegetarian.", "I tried a new vegetarian restaurant yesterday.", "consistent"),
    ("I live in Mumbai.", "My friend Arjun lives in Pune.", "consistent"),
    ("I work at Google.", "My cousin works at Amazon.", "consistent"),
    ("I play football on weekends.", "My team won the match on Sunday.", "consistent"),
    ("I'm saving up for a new laptop.", "I've saved about half of the laptop money.", "consistent"),
    ("I like jazz music.", "I like classical music too.", "consistent"),
    ("I read a lot of science fiction.", "I just finished reading Dune.", "consistent"),
    ("I have a meeting on Monday.", "I have a dentist appointment on Tuesday.", "consistent"),
    ("I'm learning to swim.", "I can now swim two lengths of the pool.", "consistent"),
    ("I prefer tea over coffee.", "I usually have chai in the evening.", "consistent"),
    # ---- unrelated ----
    ("I live in Delhi.", "My favourite colour is blue.", "unrelated"),
    ("I have a dog named Max.", "I'm preparing for my database systems exam.", "unrelated"),
    ("I work as a data analyst.", "I watched a great movie last night.", "unrelated"),
    ("I'm vegetarian.", "My laptop battery drains too fast.", "unrelated"),
    ("I play football on weekends.", "I need to renew my passport.", "unrelated"),
    ("My exam is on Friday.", "I like listening to podcasts while commuting.", "unrelated"),
    ("I love spicy food.", "I'm applying for an internship at a startup.", "unrelated"),
    ("I drive a Honda Civic.", "I'm thinking about adopting a cat.", "unrelated"),
    ("I want to learn Spanish.", "The wifi in my hostel keeps disconnecting.", "unrelated"),
    ("I'm a junior developer.", "I booked a flight to Goa for December.", "unrelated"),
    ("I like jazz music.", "I have a dentist appointment on Tuesday.", "unrelated"),
    ("My phone is an iPhone 12.", "I'm reading a book about the Roman Empire.", "unrelated"),
    ("I enjoy cooking.", "My sister is getting married in June.", "unrelated"),
    ("I live in Mumbai.", "I should start going to the gym again.", "unrelated"),
    ("I'm studying computer science.", "My favourite biryani place just closed down.", "unrelated"),
]
REPLACE = {"updates", "contradicts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the real LLM judge (uses Groq)")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between judge calls")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "figures" / "contradiction_eval.csv"))
    args = ap.parse_args()

    rows = []
    for old, new, label in PAIRS:
        sim = cosine_similarity(embed(old), embed(new))
        rows.append({"old": old, "new": new, "label": label, "expected": "replace" if label in REPLACE else "keep",
                     "similarity": round(sim, 3)})

    print(f"{len(rows)} pairs: {dict(Counter(r['label'] for r in rows))}\n")
    thr = contradiction.SIMILARITY_THRESHOLD
    print(f"== GATE (threshold {thr}) ==")
    for label in ("updates", "contradicts", "consistent", "unrelated"):
        sub = [r for r in rows if r["label"] == label]
        passed = sum(r["similarity"] >= thr for r in sub)
        mean = sum(r["similarity"] for r in sub) / len(sub)
        print(f"  {label:<12} n={len(sub):<3} reach the judge: {passed}/{len(sub)}   mean similarity {mean:.2f}")
    print("\n  threshold sweep      replace-pairs that reach judge   non-replace pairs that reach judge (= LLM calls)")
    for t in (0.15, 0.25, 0.35, 0.45, 0.55):
        rep = [r for r in rows if r["expected"] == "replace"]
        keep = [r for r in rows if r["expected"] == "keep"]
        print(f"  {t:<20.2f} {sum(r['similarity'] >= t for r in rep)}/{len(rep):<27} {sum(r['similarity'] >= t for r in keep)}/{len(keep)}")

    if args.judge:
        print("\n== LLM JUDGE (pairs that pass the gate) ==")
        for r in rows:
            if r["similarity"] < thr:
                r["relation"], r["decision"] = "(gated out)", "keep"
                continue
            res = contradiction.check_relation(r["old"], r["new"])
            if res.get("error"):
                print(f"  judge error: {res['error'][:200]}\n  Stopping; run again later. Partial results below.")
                r["relation"], r["decision"] = "ERROR", "keep"
                break
            r["relation"] = res["relation"]
            r["decision"] = "replace" if res["relation"] in REPLACE else "keep"
            time.sleep(args.sleep)
        done = [r for r in rows if "decision" in r and r["relation"] != "ERROR"]
        tp = sum(r["decision"] == "replace" and r["expected"] == "replace" for r in done)
        fp = sum(r["decision"] == "replace" and r["expected"] == "keep" for r in done)
        fn = sum(r["decision"] == "keep" and r["expected"] == "replace" for r in done)
        tn = sum(r["decision"] == "keep" and r["expected"] == "keep" for r in done)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n  end to end (gate + judge) on {len(done)} pairs:")
        print(f"    correct replaces      {tp}")
        print(f"    wrong replaces        {fp}   <- a correct memory deleted (worst error)")
        print(f"    missed replaces       {fn}   <- a stale fact kept")
        print(f"    correct keeps         {tn}")
        print(f"    precision of replace  {prec:.2f}")
        print(f"    recall of replace     {rec:.2f}")
        for r in done:
            if r["decision"] != r["expected"]:
                kind = "WRONG REPLACE" if r["decision"] == "replace" else "MISSED"
                print(f"    {kind}: [{r['label']} sim {r['similarity']}] {r['old']!r} -> {r['new']!r} (judge: {r['relation']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-pair results: {args.out}")


if __name__ == "__main__":
    main()