#!/usr/bin/env python3
"""Rescore the MMLU (generative) samples by the first answer letter.

lm_eval's exact_match wants the bare letter: "D. 0,4" scores 0 even when D is right.
The base model mostly writes the bare letter and Sophro mostly writes "D. 0,4", so the
stock number compares answer formats, not knowledge. This takes the first A-D at the
start of the reply, which is what a reader would take. Usage: rescore_mmlu.py results/base results/sophro
"""
import glob, json, re, sys, collections
LETTER = re.compile(r'^\s*\(?([ABCD])(?![A-Za-z])')
for d in sys.argv[1:]:
    right = total = bare = none = 0
    per = collections.defaultdict(lambda: [0, 0])
    for f in glob.glob(f"{d}/samples_mmlu_*_generative_*.jsonl"):
        subj = re.sub(r'.*samples_mmlu_(.*)_generative_.*', r'\1', f)
        for line in open(f):
            r = json.loads(line); out = r['resps'][0][0]
            m = LETTER.match(out); total += 1
            if m is None: none += 1
            elif out.strip() == m.group(1): bare += 1
            ok = m is not None and m.group(1) == r['target'].strip()
            right += ok; per[subj][0] += ok; per[subj][1] += 1
    print(f"{d}: {right}/{total} = {100*right/total:.1f}% by first letter; bare-letter replies {bare}, no letter {none}")
