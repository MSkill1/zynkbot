# Collaboration history study (idea, not scheduled)

*Noted 2026-09-09 from a remark by Matt; written by Claude. Not on the roadmap.*

**The question.** The git history of this repository (main, `voice`, `memory` and the earlier branches) is a record of how a non-programmer product owner and an AI coding agent solved problems together over more than a year: which difficulties came up, how each was worked through, and which techniques recurred (two-model design comparison as in `../AI_Collaboration_Case_Study_ZynkBot_mTLS.md`, "which machine is writing this" for networking bugs, build-and-test loops on real phones, documentation audits against the code). Could that history be mined into a description of the working method itself, for a book, a course or a talk?

**What is possible.** Yes. The material is all there and dated: commit messages, the diffs, `docs/KNOWN_ISSUES.md` (each entry has a cause and a fix), `CHANGELOG.md`, the labs notes, and the GitHub issues. A pass would group commits into episodes (a difficulty, the attempts, the resolution), then look across episodes for the repeated moves. Commit messages written by the agent are detailed enough to reconstruct most episodes without the chat transcripts; where they are not, the Claude Code session files on Matt's machine hold the conversation.

**What it would need.** A day of agent time to build the episode list from `git log -p` and the known-issues file, then Matt's own reading to say which episodes were actually turning points. Output: a document in this folder, or the raw material for the "reading, not writing" thesis in the case study's working notes.
