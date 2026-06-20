#!/usr/bin/env python3
"""
One-time script: reads Einstein entity data from the DB and bakes it into einstein_seed.sql.

Run AFTER clicking "Reinstall Einstein" in the app (which populates entities_detected).
Once the seed file is updated and committed, the NER bootstrap block in apply_einstein_seed
can be removed.

Usage:
    python3 scripts/db/patch_einstein_entities.py
"""
import sqlite3
import json
import re
import os

DB_PATH = os.path.expanduser('~/.local/share/zynkbot/zynkbot.db')
SEED_PATH = os.path.join(os.path.dirname(__file__), 'einstein_seed.sql')


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT title, entities_detected FROM memories WHERE session_id = 'einstein-demo-session'"
    ).fetchall()
    conn.close()

    if not rows:
        print("❌ No Einstein demo memories found. Run 'Reinstall Einstein' in the app first.")
        return

    populated = [(t, e) for t, e in rows if e and e != '[]']
    print(f"Found {len(rows)} Einstein memories, {len(populated)} with entities.")

    if not populated:
        print("❌ No entities found. Make sure the app has run NER on Einstein memories first.")
        return

    # Build title → SQL-safe JSON string mapping (only non-empty entities)
    all_db_titles = {title for title, _ in rows}
    entity_map = {}
    for title, entities_json in rows:
        if entities_json and entities_json != '[]':
            # Validate JSON and re-serialize compactly
            try:
                parsed = json.loads(entities_json)
                compact = json.dumps(parsed, separators=(',', ':'))
                # Escape single quotes for SQL string literal
                entity_map[title] = compact.replace("'", "''")
            except json.JSONDecodeError:
                print(f"  ⚠️  Bad JSON for '{title}', skipping")

    with open(SEED_PATH, 'r') as f:
        lines = f.readlines()

    updated = 0
    skipped = 0
    new_lines = []
    for line in lines:
        if line.startswith("INSERT INTO memories"):
            # Extract the title — first VALUES argument
            m = re.search(r"VALUES \('((?:[^'\\]|''|\\.)*)'", line)
            if m:
                raw_title = m.group(1).replace("''", "'")
                if raw_title in entity_map:
                    old_fragment = "'[]'::jsonb"
                    new_fragment = f"'{entity_map[raw_title]}'::jsonb"
                    if old_fragment in line:
                        line = line.replace(old_fragment, new_fragment, 1)
                        updated += 1
                    else:
                        skipped += 1  # already patched
                elif raw_title not in all_db_titles:
                    print(f"  ⚠️  Title not found in DB at all: '{raw_title}'")
        new_lines.append(line)

    with open(SEED_PATH, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Patched {updated} INSERT statements in einstein_seed.sql")
    if skipped:
        print(f"   ({skipped} already had entity data)")
    print(f"\nNext step: remove the NER bootstrap block from apply_einstein_seed in lib.rs")
    print(f"           (the block marked '// Extract entities for all Einstein memories')")


if __name__ == '__main__':
    main()
