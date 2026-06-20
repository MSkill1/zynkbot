# 🧭 Personal Snap-ins

Zynkbot's personal Snap-ins are opt-in tools for everyday life. They cover the full range of personal needs — from basic organization to reflective analysis — with privacy at the core. Your data stays on your device, not in a cloud service's hands.

These Snap-ins are scoped for individual use and can be toggled on or off at any time. Third-party development for personal use is supported and encouraged.  Personal Snap-ins can be free and open-source or be developed for profit with a license.

---

## ⚙️ How Personal Snap-ins Work

Personal Snap-ins can be built in a number of ways depending on what the snap-in needs to do. Two common patterns could be:

- **On-demand:** The user opens the snap-in to log, view, or interact with their data (calendar, journal, lists)
- **Retrospective analysis:** The user triggers an analysis of past memory or stored data over a chosen time window (e.g. "summarize my last 3 months of journal entries") — this queries existing memory, runs a single analysis pass, and returns a summary with no per-message overhead

Other approaches are possible, including background processes or event-driven hooks, depending on what the snap-in requires and how the developer chooses to build it.

---

## 🧩 Structural Considerations

- All personal Snap-ins must **respect Mode constraints** (e.g., Guardian Mode limits inference scope).
- Snap-ins should **never manipulate**, only reflect or assist.
- Every Snap-in must expose its memory footprint and allow full deletion or export.
- Documentation must define **input/output scope**, failure behavior, and consent prompts.

---

## 🪴 Personal Snap-in Categories

### 1. 📅 Personal Organization
The most common use case. Snap-ins that help manage everyday life privately:

- **Calendar & Appointments** — track personal appointments, reminders, and recurring events without syncing to Google or Apple
- **Shopping & Household Lists** — persistent lists with memory context ("we're out of X again")
- **Meal Planning & Recipes** — store recipes, plan weekly meals, generate shopping lists from a meal plan
- **Gift & Birthday Tracker** — remember what you gave, what was appreciated, upcoming dates

### 2. ✍️ Journal & Notes
Long-form journaling that integrates with Zynkbot's memory system. Because journal entries are indexed alongside your conversation history, you can ask Zynkbot about past entries in plain language — "what was I worried about in February?" or "have I written about this before?" — and it will surface relevant content without you needing to search manually. You can also tag entries by topic or mood so that related entries are easier to retrieve later.

### 3. 🏃 Health & Personal Tracking
Private logging for health-related data you don't want living in a corporate app:

- Medication schedules and reminders
- Fitness and activity logging
- Sleep notes
- Symptom tracking over time

### 4. 🧸 Parenting Companion
Assists with communication, tone-checking, and long-range consistency across parenting moments.

🔗 [View use case](../case_studies/child_mode.md)

### 5. 🐾 Pet Care Journal
Track vet visits, medications, behavioral notes, and health history for one or more pets. Useful for multi-pet households where care schedules overlap, or for owners managing chronic conditions.

### 6. 🏠 Home Improvement Log
Logs repairs, contractor visits, appliance ages, warranty info, and maintenance schedules. Ask "when did I last service the HVAC?" or "who did the roof work in 2023?" — the history is searchable and local.

### 7. ✈️ Travel Companion
Pre-trip planning, packing lists, and local notes. Post-trip: index what you saw, what you ate, and what you'd do differently. Builds a searchable personal travel log over time.

---

## 🌱 Hobbyist Snap-ins

Personal snap-ins for specific interests where accumulated knowledge and personal history make an AI genuinely useful. These work the same way as other snap-ins — local storage, semantic search, no cloud dependency.

- **Horticulture Journal** — Log planting dates, soil conditions, treatments, and seasonal observations per bed or plot. Ask "what did I plant here last year?" or "when did that pest issue first show up?" without digging through notes.

- **Artist's Studio Log** — Track works in progress, material experiments, reference sources, and reflections on what worked. Useful for building a personal creative history that's searchable over time.

- **Reading Log** — Record books read, notes on them, and connections between ideas across titles. Ask "what did I think about that book on habits?" or "have I read anything else by this author?" — without relying on a third-party service.

- **Home Lab Companion** — For hobbyists running local servers, electronics projects, or hardware experiments. Tracks configurations, build notes, and troubleshooting history across projects.

---

## 🔍 Reflection & Behavioral Snap-ins

These snap-ins work retrospectively — analyzing stored memory and conversation history over a chosen time window rather than running on every message.

### ⏳ Procrastination Loop Breaker
Recognizes repeat avoidance patterns in your notes, journal, and conversations. Surfaces them without judgment and helps identify what's actually blocking forward motion.

### 🧠 Mood & Tone Drift
Looks for subtle shifts in language, priorities, or emotional tone across a time window. Useful for noticing burnout, grief phases, or positive momentum you might not consciously register.

### 🪞 Other Behavioral Patterns
The retrospective model generalizes to any pattern worth surfacing: recurring topics, unresolved decisions, goals you keep mentioning but not acting on, relationship dynamics. These can be built as lightweight analysis prompts over existing memory with no new data collection required.

---

## 🧭 Notes for Contributors

- Lead with practical utility. The most valuable personal snap-ins solve everyday friction, not just deep introspection.
- Case studies should reflect realistic use, not idealized behavior.
- Each Snap-in should link to related documentation in `/docs/` where applicable.
- If you're building a snap-in for your own use and don't see it listed here, open a discussion or submit a PR — this list is meant to grow.
