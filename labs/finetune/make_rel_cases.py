#!/usr/bin/env python3
"""Relationship-step training cases (2026-10-28): fresh cases written by the teacher, never
the bench's own. Writes prompts_relgen.jsonl (generation requests). After the teacher test
runs them (FINETUNE_TASK=relgen), build_rel_cases.py turns the answers into a cases file
in the memory bench's format, and the exporter writes prompts with the app's own links
prompt text.
"""
import json
from pathlib import Path
OUT = Path.home() / "zynkbot-finetune" / "data" / "prompts_relgen.jsonl"
RULES = """Write test cases for the step of a personal-memory app that decides how a NEW FACT about the user relates to EXISTING MEMORIES (facts stored earlier). The four types, pick the first that fits:
- "updates": the new fact replaces an older state of the same thing and the wording signals the change ("moved", "now", "switched to", "no longer"). The old memory is kept as history.
- "contradicts": same attribute, different values, with no sign that one came after the other; or opposite claims on the same topic.
- "related": the same specific person, pet, place, thing or habit, adding to it or confirming it.
- "none": different things. Two facts about the same user are not related just because they are both about the user; a shared broad topic is not enough.
Each case: a new fact (one sentence about "the user", whose name is Matt), and 1 to 3 existing memories (one sentence each, ids 101, 102, 103), and the expected type for ONE named memory id (the others in the case should be 'none' unless obviously related). Everyday life topics, varied: pets, family, home, work, health, food, money, hobbies, cars, devices, places, plans. Make the hard ones hard: near-misses that are 'none', changes with no signal word that are 'contradicts', gradual updates. Never use the name Alexandra. Answer with JSON only, an array of 6 objects: [{"new_fact": "...", "existing": [{"id": 101, "content": "..."}], "expected": {"memory_id": 101, "type": "updates"}}]"""
rows = []
areas = ["pets and animals", "family and friends", "home and property", "work and money", "health and body", "food and diet", "hobbies and games", "cars, devices and services", "places, travel and plans"]
for t in ["updates", "contradicts", "related", "none"]:
    for ai, area in enumerate(areas):
        for part in "ab":
            rows.append({"id": f"rg-{t}-{ai}{part}", "task": "relgen", "prompt": RULES + f"\n\nAll 6 cases must have expected type \"{t}\". Topic area: {area}. (Batch {part}: use different situations from the other batch.)"})
OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), "requests ->", OUT)
