# Check-in questions: "did you ever…?" (design note, 2026-10-29)

**Status:** built 2026-10-29 as an experimental beta 5 feature (switch under System Controls → Experimental); the question is a template, no model call; `memory_checkins` is per device, not synced. Agreed with Matt 2026-10-29: build this as app
logic first; whether Sophro is later trained to ask questions is a separate decision
(round 2 or 3 of the fine-tune).

## The problem

"What do I have coming up?" on 2026-10-29 listed the glasses, the mudroom steps and the
kitchen-sink purifier. All three were done or dropped weeks earlier; Matt had never told
Zynkbot. The memories were right when stored and nobody closed them. Today's partial fix:
the answer gives the date each item was mentioned, so a stale plan at least looks stale.

Making intentions expire on their own was rejected (Matt, same day): "I want to retire by
sixty" and "I'm going to cut the grocery budget" are intentions too, and losing them after
two weeks would be a real failure. A classifier cannot be trusted to tell an errand from a
life plan every time. So the user closes them, and Zynkbot's only job is to ask, rarely.

## What it does

When a conversation starts, if there is a stale intention, Zynkbot adds one line before
anything else:

> A while back you mentioned getting a water filter for the kitchen sink. Did that happen?

The answer closes the loop:

| answer | effect |
|---|---|
| done ("yes", "got it last week") | the memory is marked completed: it leaves recall like a superseded memory, stays in Memory Manager as history |
| still planned ("not yet") | the memory's date moves to today, so it is not asked about again for another window |
| never mind ("forget it", "I dropped that") | the memory is deleted (with the usual tombstone so sync removes it everywhere) |
| anything else, or no answer | nothing changes; the question is not repeated for seven days |

## Limits that keep it from being annoying

- At most **one question a day**, and only at the start of a conversation, never in the
  middle of one.
- Only about things the user said they would **do** (buy, fix, call, book, finish). Never
  about feelings, people, health or anything the model has to infer.
- A memory is asked about **once**. If the answer is not clear, it is not asked again.
- **Off by default?** Matt to decide. Proposed: on, with a switch on the System Controls
  page ("Ask about old plans"), because a user who never sees a question never discovers
  the switch.
- Containment: not in Child mode. In HIPAA mode only if the memory is not health-related
  (the containment check already classifies text; reuse it).

## How an intention is recognised

No new field is needed. Candidates are memories that:

1. carry `passing = true` from extraction (the flag added 2026-10-26: "over in days or
   weeks: an errand, where something is parked, an appointment, a trip"), and
2. whose `original_text` is in the first person future or obligation: "I need to", "I'm
   going to", "I have to", "I should", "I want to get", "I'll", "going to pick up", and
3. are **at their expiry** (passing facts leave recall after about two weeks), and
4. have no later memory linked to them as "updates" or "related" (a later mention means
   the user is still on it, no need to ask).

Life plans fail test 1 (retirement and budgets are not marked passing) and so are never
asked about. That is the whole safeguard, and it is the one Matt already accepted for
parking spots. The question moment is the expiry moment: "before I let this go, did it
happen?" Nothing new has to be scheduled.

## Pseudocode

```
on conversation start (desktop: app open; phone: app to the front), once per day:
    if setting "ask about old plans" is off: return
    if containment mode is child: return
    candidate = first memory where
        passing = true
        and expires_at within the last 7 days or the next 2 days
        and original_text matches an intention pattern
        and not yet asked (checkin_asked_at on the memory is empty)
        and no newer memory links to it (updates / related)
        and (mode != hipaa or not health-related)
    if none: return
    set checkin_asked_at = now on the memory (syncs with it)
    question = model phrases it from a template:
        "A while back you mentioned {original_text, trimmed}. Did that happen?"
        (template used as-is when the model is unavailable)
    show the question as Zynkbot's first line; remember "pending check-in = memory_id"

on the user's next message, if a check-in is pending:
    classify the reply with the small yes / not-yet / never / other prompt
        (same shape as the Remember/MEMORY: marker, one word)
    yes      -> set temporal_status = 'completed' (new value beside 'superseded');
                recall skips it like superseded; Memory Manager shows "Done"
    not yet  -> expires_at = now + 14 days; created_at unchanged (history stays honest)
    never    -> delete_memory (tombstone, sync)
    other    -> leave it; the message is then handled as a normal message
    clear the pending check-in either way
```

New storage: two columns on the memory row, `checkin_asked_at` and `checkin_answer`
(moved there on 2026-10-30 from a per-device table, so the mark syncs and a question is
asked once across all devices), and one new `temporal_status` value, `completed`. Both
sync like any other memory change. Recall,
Memory Manager and the "coming up" answer treat `completed` exactly like `superseded`,
with a different label.

## What it is not

- Not a reminder system. It never asks about a date in the future; the roadmap item
  "startup date reminder surfacing" covers that and the two should share the start-of-
  conversation slot, with the reminder taking priority.
- Not an interview. One question, about one thing the user said, then silence. A
  journaling or therapy snap-in may ask more, under its own rules (see
  labs/snapin_platform).

## Open questions for Matt

1. On or off by default?
2. Should "done" keep the memory visible in "What do I have coming up?" as "done on the
   29th" for a week, or drop it at once? Proposed: drop at once; it is in Memory Manager.
3. Phone wording: the question costs one extra model call per day on the local model.
   Acceptable, or template only on the phone?
