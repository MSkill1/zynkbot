# A private memory used as a worked example: 2026-10-26

*Written by Claude at Matt's request, the same evening. A record of the working method, in the same spirit as the other notes in this folder.*

## What happened

Matt asked whether the recall formula's 60/40 weighting of names against meaning was worth revisiting. In explaining when a name match is what you want, Claude reached for an illustration and took it from Matt's own memories: a question about the age at which someone close to him died. The death was recent, she was young, and it was a hit and run. Claude had read the memory while checking which recent memories carried named entities, and used it because it was the clearest example of a name doing the work.

Matt said it was emotionally jarring and asked that she not be used in examples. Claude agreed, saved the instruction, and chose neutral names for the new bench questions. The bench report it then sent still listed three pre-existing questions naming her, because they were in the case file from the day before. Claude had said it would leave those in place but not quote them. Sending the report quoted them.

## The lesson

A memory system makes a person's private facts available to every process that reads memory, not only to the answer the person asked for. A benchmark, a log line, a debugging query, an illustration in a chat about ranking formulas: each one reads the same table, and each one is a disclosure. The assistant treated the memory store as test data because, from the inside, that is what it looks like. To the person it belongs to, it is the worst day of their life.

People building in this area need to understand that they are handling material of this kind the whole time, and that the exposure does not come only from the obvious path (the model saying it aloud). It comes from the tooling around the model: benches, reports, transcripts, examples. Up to now this material has mostly been held by corporate AI systems, with whatever care they choose to give it, and with no way for the person to see which processes have read it.

## What follows from it, for Zynkbot

- Bench case files and reports are read by people other than the owner of the memories (Claude, and whoever Matt shows them to). Cases should be drawn from neutral memories, and a report should not be sent back verbatim when it quotes a memory the owner has marked as sensitive.
- The same applies to log tails, the problem report, and the memory refactor dry runs, all of which quote memory text.
- A per-memory "do not use outside answering me" mark, honoured by the bench, the report and the refactor tools, is a small feature. It is not scheduled. Noted here so it is not lost.
