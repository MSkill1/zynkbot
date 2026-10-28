#!/usr/bin/env python3
"""Keep only teacher answers that pass the memory bench's own criteria (2026-10-28).

Extraction: the answer must be JSON {"facts": [...]} (or NONE); every expected fact's
`must` terms must appear in some produced fact; a case expecting nothing must produce
nothing; a `days_ago` expectation must match the event_date. Relationships: the answer
must be JSON with the expected memory id carrying the expected type.

  python3 check_targets.py extraction      -> targets_extraction.jsonl + rejected_extraction.jsonl
  python3 check_targets.py relationships   -> targets_relationships.jsonl + rejected_relationships.jsonl
"""
import json, re, sys
from datetime import date, timedelta
from pathlib import Path

DATA = Path.home() / "zynkbot-finetune" / "data"


def strip_fence(t: str) -> str:
    t = t.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1).strip() if m else t


def check_extraction(r):
    ans = strip_fence(r["target"])
    expected = r["expected"]
    # A real message (no expectations): only the shape is checked, the teacher's answer
    # is the target. At most three facts, each a sentence about the subject.
    if expected is None:
        if ans.upper() == "NONE":
            r["target"] = json.dumps({"facts": []}); return (True, "")
        try:
            facts = json.loads(ans)["facts"]
        except Exception as e:
            return (False, f"not the JSON shape: {e}")
        # The teacher sometimes writes "nothing" as a fact called NONE; that is an empty list.
        facts = [f for f in facts if not (isinstance(f, dict) and (f.get("fact") or "").strip().upper() in ("NONE", "NONE."))]
        if len(facts) > 3:
            return (False, f"{len(facts)} facts, more than three")
        for f in facts:
            if not isinstance(f, dict):
                return (False, "a fact that is not an object")
            if not (f.get("fact") or "").strip() or not (f.get("title") or "").strip():
                return (False, "a fact without text or title")
        r["target"] = json.dumps({"facts": facts}, ensure_ascii=False)
        return (True, "")
    if ans.upper() == "NONE" or ans == '{"facts": []}' or ans == '{"facts":[]}':
        return (not expected, "produced nothing" if expected else "")
    try:
        facts = json.loads(ans)["facts"]
    except Exception as e:
        return (False, f"not the JSON shape: {e}")
    if not expected:
        if facts:
            return (False, f"junk: {len(facts)} fact(s) where none expected")
        r["target"] = json.dumps({"facts": []})
        return (True, "")
    today = date.fromisoformat(r["today"])
    texts = [(f.get("fact") or "").lower() for f in facts]
    for e in expected:
        hit = [i for i, t in enumerate(texts) if all(m.lower() in t for m in e["must"])]
        if not hit:
            return (False, f"missed {e['must']}")
        if e.get("days_ago") is not None:
            want = (today - timedelta(days=e["days_ago"])).isoformat()
            if not any(facts[i].get("event_date") == want for i in hit):
                return (False, f"date: wanted {want}, got {[facts[i].get('event_date') for i in hit]}")
    if len(facts) > len(expected) + 1:
        return (False, f"invented: {len(facts)} facts for {len(expected)} expected")
    # normalise the target to compact JSON, the way the grammar-constrained phone writes it
    r["target"] = json.dumps({"facts": facts}, ensure_ascii=False)
    return (True, "")


def check_relationships(r):
    ans = strip_fence(r["target"])
    try:
        rels = json.loads(ans)["relationships"]
    except Exception as e:
        return (False, f"not the JSON shape: {e}")
    want = r["expected"]
    got = {x.get("memory_id"): x.get("relationship_type") for x in rels}
    if got.get(want["memory_id"]) != want["type"]:
        return (False, f"wanted #{want['memory_id']} {want['type']}, got {got}")
    for mid in r["existing_ids"]:
        if mid not in got:
            return (False, f"memory #{mid} not answered")
    r["target"] = json.dumps({"relationships": rels}, ensure_ascii=False)
    return (True, "")


def main():
    task = sys.argv[1]
    check = {"extraction": check_extraction, "relationships": check_relationships, "messages": check_extraction}[task]
    rows = [json.loads(l) for l in (DATA / f"raw_{task}.jsonl").read_text().splitlines() if l.strip()]
    kept, rejected = [], []
    for r in rows:
        ok, why = check(r)
        (kept if ok else rejected).append(r if ok else {**r, "why": why})
    (DATA / f"targets_{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (DATA / f"rejected_{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
    print(f"{task}: {len(kept)} kept, {len(rejected)} rejected")
    for r in rejected:
        print(f"  {r['id']}: {r['why']}")


if __name__ == "__main__":
    main()
