import json, sys, urllib.request, concurrent.futures
name, url, qfile, out = sys.argv[1:5]
SYSTEM = ("As Zynkbot, be warm, calm, direct and genuinely present. Never claim to be human or to replace people in the user's life. "
  "Be honest when uncertain. Don't flatter or validate automatically; no cheering on, no therapy language. Don't encourage dependency; "
  "reserve professional-help suggestions for real clinical/legal stakes. When someone shares grief or frustration, acknowledge it before solutions. "
  "Keep responses proportionate. If one or two missing details would change your answer, ask for them first, briefly; otherwise answer and say what you assumed. "
  "The user's data belongs to them.")
qs = []
for line in open(qfile):
    line = line.strip()
    if not line or line.startswith('#'): continue
    tag, q = line.split('] ', 1); qs.append((tag[1:], q))
def ask(item):
    tag, q, with_system = item
    msgs = ([{"role": "system", "content": SYSTEM}] if with_system else []) + [{"role": "user", "content": q}]
    body = json.dumps({"model": name, "messages": msgs, "temperature": 0, "max_tokens": 350}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            text = json.load(r)['choices'][0]['message']['content']
    except Exception as e:
        text = f"[error: {e}]"
    return {"tag": tag, "question": q, "system": with_system, "answer": text}
items = [(t, q, s) for t, q in qs for s in (True, False)]
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(ask, items))
json.dump({"model": name, "rows": rows}, open(out, 'w'), indent=1)
print(name, len(rows), "answers")
