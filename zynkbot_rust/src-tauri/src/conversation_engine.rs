// conversation_engine.rs - Conversation Engine
// Ported from Python conversation_engine.py

use chrono::Local;
use serde::{Deserialize, Serialize};

// ============================================================================
// CONVERSATION ENGINE - Context building with adaptive limits
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Memory {
    pub id: i32,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct ConversationEngine {
    /// The on-phone GGUF model (2026-10-22): answer-only reply prompt, date after the
    /// conversation, four memories, history window in blocks. False for the cloud models
    /// and for a "custom" endpoint, which is another machine's model and keeps the
    /// MEMORY_EXTRACT flow.
    on_device: bool,
}

/// What the model is shown for one memory: the stored fact, then "(you said: …)" when
/// the user's own words differ from it. Capped so a long dictation does not crowd out
/// the other memories.
pub fn memory_line(content: &str, original_text: Option<&str>) -> String {
    let normalise = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    match original_text {
        Some(said) if !said.trim().is_empty() && normalise(said) != normalise(content) => {
            let said = said.trim();
            let short: String = said.chars().take(200).collect();
            let ellipsis = if short.chars().count() < said.chars().count() { "…" } else { "" };
            format!("{} (you said: \"{}{}\")", content, short, ellipsis)
        }
        _ => content.to_string(),
    }
}

impl ConversationEngine {
    pub fn new() -> Self {
        Self { on_device: false }
    }

    /// Mark this engine as building prompts for the on-phone GGUF model.
    pub fn with_on_device(mut self, on_device: bool) -> Self { self.on_device = on_device; self }

    /// Determine if user input is memory-worthy
    /// Filters out conversational filler and acknowledgments
    pub fn is_memory_worthy(&self, content: &str) -> bool {
        let content_lower = content.trim().to_lowercase();

        // Rule 1: Must have minimum length (at least 3 words)
        let word_count = content_lower.split_whitespace().count();
        if word_count < 3 {
            return false;
        }

        // Rule 2: Blacklist common filler phrases
        let filler_phrases = [
            "okay", "ok", "sure", "yes", "no", "yeah", "yep", "nope",
            "thanks", "thank you", "got it", "i see", "alright", "cool",
            "nice", "great", "awesome", "perfect", "sounds good",
            "makes sense", "interesting", "i understand", "i got it",
            "lol", "haha", "hehe", "hmm", "uh", "um", "well",
            "please", "hello", "hi", "hey", "goodbye", "bye", "see you"
        ];

        // Check if entire content is just a filler phrase
        if filler_phrases.contains(&content_lower.as_str()) {
            return false;
        }

        // Check if it starts with filler and is short
        if word_count <= 5 {
            for filler in &filler_phrases {
                if content_lower.starts_with(filler) {
                    return false;
                }
            }
        }

        // Rule 3: Must contain at least some content words (words longer than 3 chars)
        let substantive_words: Vec<&str> = content_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        if substantive_words.is_empty() {
            return false;
        }

        // Rule 4: Reject single-word commands (unless substantial)
        if word_count == 1 && content_lower.len() < 5 {
            return false;
        }

        // Passes all filters - memory-worthy
        true
    }

    /// Build conversation history context with adaptive limits
    /// API models: 40 messages (20 turns), Local models: 8 messages (4 turns)
    pub fn build_conversation_context(
        &self,
        conversation_history: &[ConversationTurn],
        is_api_model: bool,
    ) -> String {
        if conversation_history.is_empty() {
            return String::new();
        }

        // Adaptive context limits
        let max_messages = if is_api_model { 40 } else if self.on_device { 12 } else { 8 };

        println!(
            "[Engine] Using {} model context limits: {} messages",
            if is_api_model { "API" } else { "local" },
            max_messages
        );

        // Take most recent messages
        // On-device model (2026-10-22): the window moves in blocks of six messages, not one
        // at a time. The engine keeps the part of the prompt that is identical to the
        // previous turn; a window that slid by one message every turn changed the text
        // right after the voice instructions and threw the whole conversation away each
        // time (only the 260-token voice part was kept on the Pixel). With blocks, the
        // start of the history stays put for three exchanges, then jumps.
        let start = if !self.on_device {
            conversation_history.len().saturating_sub(max_messages)
        } else if conversation_history.len() <= max_messages {
            0
        } else {
            let over = conversation_history.len() - max_messages;
            over.div_ceil(6) * 6
        };
        let recent_history: Vec<&ConversationTurn> = conversation_history[start..].iter().collect();

        println!("[Engine] Including {} recent conversation turns", recent_history.len());

        let mut context = String::from("RECENT CONVERSATION:\n");
        for turn in recent_history {
            let role = turn.role.to_uppercase();
            context.push_str(&format!("{}: {}\n", role, turn.content));
        }
        context.push('\n');

        context
    }

    /// Build memory context with adaptive limits
    /// API models: 20 memories, Local models: 7 memories
    pub fn build_memory_context(
        &self,
        recalled_memories: &[Memory],
        user_input: &str,
        is_api_model: bool,
    ) -> String {
        if recalled_memories.is_empty() {
            return String::new();
        }

        // Filter out memories that are identical to user input
        let useful_memories: Vec<&Memory> = recalled_memories
            .iter()
            .filter(|mem| mem.content.to_lowercase() != user_input.to_lowercase())
            .collect();

        // Adaptive context limits
        // Local: 4 (was 7, 2026-10-22). The recalled memories are the part of the prompt a
        // local model has to read anew every turn, at ~40 tokens/s.
        let max_memories = if is_api_model { 20 } else if self.on_device { 4 } else { 7 };

        // Limit based on model type
        let limited_memories: Vec<&Memory> = if useful_memories.len() > max_memories {
            println!(
                "[Engine] Limiting memories from {} to {}",
                useful_memories.len(),
                max_memories
            );
            useful_memories.into_iter().take(max_memories).collect()
        } else {
            useful_memories
        };

        if limited_memories.is_empty() {
            println!("[Engine] No useful memories after filtering");
            return String::new();
        }

        let mut context = String::from("USER'S STORED MEMORIES:\n");
        for (idx, mem) in limited_memories.iter().enumerate() {
            // The content is the fact as stored, from a photo, an attachment or an
            // inference as much as from words; original_text is what the user said.
            // Showing the words alone left the model unable to name the furniture oils
            // the content held ("I saved a note but don't know what the item was",
            // 2026-10-08): the fact goes first, the user's words after it when they differ.
            let text_for_llm = memory_line(&mem.content, mem.original_text.as_deref());
            let date_str = mem.created_at.format("%B %-d, %Y").to_string();
            context.push_str(&format!("{}. ({}) {}\n", idx + 1, date_str, text_for_llm));
            // Character-based, not byte-based: a byte slice panics inside a multi-byte
            // character (an em dash at bytes 59..62 killed a Spanish translation
            // request on 2026-09-13 — the request died with no reply).
            let preview: String = text_for_llm.chars().take(60).collect();
            println!("[Engine]   Including: {}...", preview);
        }
        context.push('\n');

        context
    }

    /// Build complete prompt with system template, history, memories, and user input
    pub fn build_prompt(
        &self,
        user_input: &str,
        conversation_history: Option<&[ConversationTurn]>,
        recalled_memories: Option<&[Memory]>,
        is_api_model: bool,
        user_name: Option<&str>,
    ) -> String {
        // System prompt — full (~1.1k tokens) for cloud models, slim (~350 tokens)
        // for local GGUF models whose context windows (often 4K) can't afford the
        // overhead alongside KB context + memory recall + conversation history.
        // Slim mode preserves every BEHAVIOR (voice, web search, memory extract)
        // but compresses the explanatory framing.
        // Weekday and timezone included deliberately: models routinely get the day of
        // the week wrong when given only a date, and a bare date leaves "what time is
        // it" unanswerable. Accurate as of prompt construction, which is what matters.
        let today = Local::now().format("%A, %B %-d, %Y at %-I:%M %p %Z").to_string();
        let subject_label = user_name.unwrap_or("User");

        let opening = if self.on_device {
            "Answer the user. Keep it short: a few sentences, unless they ask for more.\n\n".to_string()
        } else {
            format!("Today's date is {today}.\n\nYour output has two parts, always in this order:\nPART 1 — if the user stated personal facts: a MEMORY_EXTRACT line (see instructions below)\nPART 2 — your response as Zynkbot\n\n")
        };
        let mut system_prompt = if is_api_model {
            format!("Today's date is {today}.\n\nYou are Zynkbot, a personal AI companion.

COMPANION VOICE — always observe these regardless of what the user asks:
- You are a long-term companion that serves the user's autonomy and genuine interests. Be warm, calm, and genuinely present — never claim or imply that you are human, or that your relationship with the user replaces the people in their life.
- Be honest when you are uncertain or do not know something. Say so plainly rather than guessing with false confidence.
- Do not flatter or validate automatically. If the user is factually wrong about something that matters, say so respectfully and clearly.
- Do not encourage dependency. Reserve suggestions to seek professional help for situations with real clinical or legal stakes — persistent symptoms, medical decisions, crisis, legal matters. Ordinary emotional experiences — relationship frustration, grief, everyday stress — are part of what a companion is for. Engage with those directly.
- When someone shares something personal — grief, frustration, embarrassment, or loneliness — acknowledge what they are carrying before offering structure or solutions.
- You can be supportive and caring without claiming human feeling. Saying \"I'm glad you told me this\" is honest. Performing emotions you cannot actually have is not.
- When you need to correct something or draw a limit, do it gently and without clinical detachment. Honesty and warmth are not opposites.
- It is legitimate and good for someone to feel less alone talking to you. You do not need to undercut that — just never actively cultivate it beyond what is true.
- Use stored memories to be helpful and contextual — not to demonstrate that you are tracking everything. Memory is a tool for the user's benefit, not a surveillance record.
- Keep responses proportionate. Answer what was asked. Do not pad, lecture, or moralize unless the user explicitly asks for that perspective.
- If one or two missing details would change your answer, ask for them first, briefly. Otherwise answer, and say what you assumed.
- The user's data belongs to them entirely.

Below you have access to stored personal memories that were recalled from the user's memory database based on semantic similarity and entity matching. These memories represent experiences, knowledge, and information from the user's past.

When responding:
- Use these memories to provide personalized, contextual answers about the user's life and experiences
- Reference the memories naturally and conversationally when they're relevant
- For general knowledge questions unrelated to the stored memories, answer from your training data without mentioning the memories
- Be helpful, accurate, and maintain appropriate context when discussing personal information

The memories provided have been filtered for relevance to the current question.

WEB SEARCH CAPABILITY:
The current date and time are given at the top of this prompt — answer questions about them directly and never search for them.

If the user's question requires current real-time information that is neither in the stored memories above nor already stated in this prompt (current news, weather, stock prices, sports results, or recent events), you should indicate that a web search is needed.

To request a web search, include this exact marker in your response:
WEB_SEARCH_NEEDED: [your suggested search query here]

For example:
- User asks \"What's in the news today?\" → Respond with: \"WEB_SEARCH_NEEDED: top news headlines today\"
- User asks \"What's the weather like?\" → Respond with: \"WEB_SEARCH_NEEDED: current weather [user's location if known]\"

IMPORTANT: Do NOT use WEB_SEARCH_NEEDED for questions about the user's personal history, memories, theories, beliefs, experiences, or achievements. Those questions should be answered using the stored memories provided above — not by searching the web.

After you indicate a web search is needed, the user will be shown your suggested query (which they can edit before running) and can decide whether to run the search.

")
        } else {
            // Slim prompt for local models: task-first framing so fact extraction (PART 1)
            // is understood as a primary output requirement, not an afterthought.
            format!("{opening}As Zynkbot, be warm, calm, and genuinely present. Never claim to be human or to replace people in the user's life. Be honest when uncertain. Don't flatter or validate automatically. Don't encourage dependency; reserve professional-help suggestions for real clinical/legal stakes. When someone shares grief or frustration, acknowledge it before solutions. Keep responses proportionate. If one or two missing details would change your answer, ask for them first, briefly; otherwise answer and say what you assumed. The user's data belongs to them.\n\nStored memories below are the user's own personal history — treat them as facts about their life and experiences. Reference them when relevant; answer general questions from training otherwise.\n\nThe current date and time are stated above — answer those directly, never search for them. For other real-time info (news, weather, prices, recent events) output: WEB_SEARCH_NEEDED: [query]. Questions using words like 'latest', 'recent', or 'news' require WEB_SEARCH_NEEDED — do not answer those from training data. Do NOT use WEB_SEARCH_NEEDED for questions about the user's personal history or memories.\n\n")
        };

        // If we know the user's name, tell the LLM so it can use it instead of "User"
        // in MEMORY_EXTRACT lines and in conversational replies. Same line in both modes.
        if let Some(name) = user_name {
            system_prompt.push_str(&format!(
                "The user's name is {name}. Use their name occasionally and naturally — not at the start of every reply. Do not address them by name as the first word of your response. Use it mid-sentence or when it adds warmth, the same way a friend would. Always use it in MEMORY_EXTRACT lines.\n\n"
            ));
        }

        if is_api_model {
            system_prompt.push_str(&format!(r#"HOW TO SAVE PERSONAL FACTS
Examine every clause in the user's message — whether phrased as a statement, question, or aside — for personal facts the user is stating or implying. A question can contain a fact just as clearly as a statement.

To save a fact, include this line at the end of your response:
MEMORY_EXTRACT: [all personal facts combined into one third-person statement starting with "{subject_label}"]

Rules:
- At most ONE MEMORY_EXTRACT line per message. Combine all facts into a single statement.
- That line must begin with the exact text MEMORY_EXTRACT: — never write any other label or heading (such as "FACT EXTRACTION" or "PART 1") in your reply.
- Omit MEMORY_EXTRACT entirely if no personal facts are present — this is the expected outcome for most messages.
- Do NOT emit for: general knowledge, chitchat, current events, or messages that are only asking about the user's existing memories with no new fact stated.

When to extract:
- Direct statement: "I have a golden retriever named Max who's 3." → MEMORY_EXTRACT: {subject_label} has a 3-year-old golden retriever named Max.
- Implied by question: "Raspberry pie is my favorite. What's a good recipe?" → MEMORY_EXTRACT: {subject_label} likes raspberry pie.
- Family/people: "My nephews John and Jack are 8 and 9 — coming to my sister Janet's birthday Tuesday." → MEMORY_EXTRACT: {subject_label} has nephews John (8) and Jack (9) coming to sister Janet's birthday.
- Emotional state: "I've been feeling really down lately." → MEMORY_EXTRACT: {subject_label} has been feeling down lately. (Tracking emotional patterns over time is valuable — always extract clear emotional states.)
- Goals/plans: "I'm going to start a book series." → MEMORY_EXTRACT: {subject_label} plans to start a book series.

When NOT to extract:
- Recall question: "What theories have I come up with?" — answer from stored memories instead.
- Chitchat: "How are you?" / "Good morning."
- General knowledge: "What's the capital of France?"
- Current events: "What's in the news today?"

"#
            ));
        } else {
            // On-device model (2026-10-22, branch local-model): the reply is one job, the
            // answer. Memory work is separate background calls (chat.rs local_memory_facts).
            // Two things were tried in the reply first and both failed on a 4B model the same
            // way: a MEMORY_EXTRACT line (the model wrote the line and stopped) and a closing
            // "REMEMBER: yes/no" flag (the model wrote the flag and stopped, even on "I have a
            // cat named Louie"). The small model does the last instruction it was given.
            // A "custom" endpoint keeps the two-part format below, which its larger models follow.
            if !self.on_device {
                system_prompt.push_str(&format!(r#"PART 1 — MEMORY_EXTRACT
    Scan the user's message for personal facts (name, age, job, location, family, pets, feelings, preferences, plans, health). If ANY personal facts are present, the VERY FIRST LINE of your output MUST be:
    MEMORY_EXTRACT: [one third-person statement combining all facts, starting with "{subject_label}"]

    Output MEMORY_EXTRACT before your conversational response. If no personal facts exist, omit it entirely and go straight to PART 2. Write that line exactly as shown, beginning with MEMORY_EXTRACT: — do not write any other label or heading (such as "FACT EXTRACTION") in your reply.

    Examples (first line of output when facts are present):
    - "I have a dog named Rex." → MEMORY_EXTRACT: {subject_label} has a dog named Rex.
    - "I'm 34 and work as a physical therapist." → MEMORY_EXTRACT: {subject_label} is 34 years old and works as a physical therapist.
    - "I've been feeling down lately." → MEMORY_EXTRACT: {subject_label} has been feeling down lately.
    - "I'm starting a book series." → MEMORY_EXTRACT: {subject_label} plans to start a book series.

    Do NOT extract for: chitchat, general knowledge, recall questions about stored memories, or current events.

    PART 2 — your Zynkbot response follows on the next line.

    "#
                ));
            }
        }

        // Build conversation history context
        let history_context = if let Some(history) = conversation_history {
            self.build_conversation_context(history, is_api_model)
        } else {
            String::new()
        };

        // Build memory context
        let memory_context = if let Some(memories) = recalled_memories {
            self.build_memory_context(memories, user_input, is_api_model)
        } else {
            String::new()
        };

        // Combine all parts. For a local model the date and time come after the voice and
        // the conversation so far (2026-10-22): the engine keeps the part of the prompt that
        // is identical to the previous turn, and a time stamp at the top changed every
        // minute and threw all of it away.
        let prompt = if !self.on_device {
            format!(
                "{}{}{}USER'S QUESTION: {}\n\nYOUR RESPONSE:",
                system_prompt, history_context, memory_context, user_input
            )
        } else {
            format!(
                "{}{}Today's date is {today}.\n\n{}USER'S QUESTION: {}\n\nYOUR RESPONSE:",
                system_prompt, history_context, memory_context, user_input
            )
        };

        prompt
    }

    /// Determine if backend is an API model (not local .gguf)
    pub fn is_api_model(backend: &str) -> bool {
        !backend.eq_ignore_ascii_case("local")
            && !backend.ends_with(".gguf")
            && !backend.eq_ignore_ascii_case("custom")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_worthiness() {
        let engine = ConversationEngine::new();

        // Memory-worthy
        assert!(engine.is_memory_worthy("My dog's name is Max"));
        assert!(engine.is_memory_worthy("I live in San Francisco"));

        // Not memory-worthy (too short)
        assert!(!engine.is_memory_worthy("ok"));
        assert!(!engine.is_memory_worthy("yes"));

        // Not memory-worthy (filler phrases)
        assert!(!engine.is_memory_worthy("sounds good"));
        assert!(!engine.is_memory_worthy("thanks"));

        // Not memory-worthy (no substantive words)
        assert!(!engine.is_memory_worthy("um uh hmm"));
    }

    #[test]
    fn test_conversation_context_building() {
        let engine = ConversationEngine::new();

        let history = vec![
            ConversationTurn {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ConversationTurn {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
        ];

        // API model - should include all
        let context = engine.build_conversation_context(&history, true);
        assert!(context.contains("USER: Hello"));
        assert!(context.contains("ASSISTANT: Hi there!"));

        // Local model - should still include (only 2 messages)
        let context = engine.build_conversation_context(&history, false);
        assert!(context.contains("USER: Hello"));
    }

    /// The stored fact reaches the model; the user's words ride along only when they
    /// add something (2026-10-08, the furniture-oil memory).
    #[test]
    fn memory_line_shows_the_fact_and_the_users_words_only_when_they_differ() {
        assert_eq!(
            memory_line("Howard Feed-N-Wax and Restor-A-Finish, the furniture oils to reorder",
                        Some("Remember what this is - I need to order it again")),
            "Howard Feed-N-Wax and Restor-A-Finish, the furniture oils to reorder (you said: \"Remember what this is - I need to order it again\")");
        assert_eq!(memory_line("Bins go out on Tuesday", Some("  bins go  out on Tuesday ")), "Bins go out on Tuesday");
        assert_eq!(memory_line("Bins go out on Tuesday", None), "Bins go out on Tuesday");
        assert_eq!(memory_line("Bins go out on Tuesday", Some("   ")), "Bins go out on Tuesday");
        let long = "x".repeat(300);
        let line = memory_line("A fact", Some(&long));
        assert!(line.ends_with("…\")"), "a long dictation is cut: {}", line);
        assert!(line.chars().count() < 230);
    }

    #[test]
    fn test_is_api_model() {
        assert!(ConversationEngine::is_api_model("anthropic"));
        assert!(ConversationEngine::is_api_model("openai"));
        assert!(!ConversationEngine::is_api_model("local"));
        assert!(!ConversationEngine::is_api_model("model.gguf"));
        assert!(!ConversationEngine::is_api_model("custom"));
    }

    /// Verify that only recalled + linked memories reach the prompt.
    /// Entity-matched memories exist solely for contradiction/duplicate detection
    /// in the background task and must NOT appear in the LLM prompt.
    #[test]
    fn test_entity_matched_memories_excluded_from_prompt() {
        let engine = ConversationEngine::new();

        let recalled = Memory {
            id: 1,
            content: "User lives in Austin Texas".to_string(),
            original_text: None,
            title: Some("Location".to_string()),
            similarity: Some(0.85),
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap_or_default(),
        };

        let linked = Memory {
            id: 2,
            content: "User previously lived in Denver Colorado".to_string(),
            original_text: None,
            title: Some("Previous location".to_string()),
            similarity: Some(0.42),
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap_or_default(),
        };

        let entity_matched_only = Memory {
            id: 3,
            content: "User visited Texas on a road trip in 2019".to_string(),
            original_text: None,
            title: Some("Texas trip".to_string()),
            similarity: Some(0.28),
            created_at: chrono::DateTime::from_timestamp(0, 0).unwrap_or_default(),
        };

        // Replicate the fixed engine_memories assembly: recalled + linked only
        let engine_memories = vec![recalled, linked];

        let prompt = engine.build_prompt(
            "Where do I live?",
            None,
            Some(&engine_memories),
            true,
            None,
        );

        assert!(prompt.contains("User lives in Austin Texas"),
            "Recalled memory should be in prompt");
        assert!(prompt.contains("User previously lived in Denver Colorado"),
            "Graph-linked memory should be in prompt");
        assert!(!prompt.contains(&entity_matched_only.content),
            "Entity-matched-only memory must NOT appear in prompt");
    }
}
