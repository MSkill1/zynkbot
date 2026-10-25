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

/// On-device prompt budget (2026-10-24). The phone reads the prompt at ~37 tokens/s on a
/// Pixel 10 and ~50 on a OnePlus 12R, so every token in front of the question is time.
pub const ON_DEVICE_MESSAGE_CHARS: usize = 400;
pub const ON_DEVICE_MEMORY_CHARS: usize = 300;
pub const ON_DEVICE_HISTORY_BLOCK: usize = 4;

/// The first `max` characters of `text`, with an ellipsis when something was cut.
/// Character-based, never a byte slice (a multi-byte character would panic).
/// A stored reply without its memory marker: own-line "MEMORY: no" lines dropped, a
/// marker glued to the last sentence cut off.
pub fn strip_memory_marker(text: &str) -> String {
    let mut lines: Vec<String> = text.lines().filter(|l| {
        let t = l.trim().trim_matches(|c: char| matches!(c, '*' | '`' | '_' | '#' | '-' | ' ')).to_lowercase();
        let rest = t.strip_prefix("memory").or_else(|| t.strip_prefix("remember")).map(|r| r.trim_start_matches(|c: char| matches!(c, ':' | ' ' | '-' | '—')).trim_end_matches(|c: char| matches!(c, '.' | '!' | ' ')));
        !matches!(rest, Some("yes") | Some("no"))
    }).map(str::to_string).collect();
    if let Some(last) = lines.iter().rposition(|l| !l.trim().is_empty()) {
        if let Some((_, rest)) = crate::commands::chat::split_inline_memory_marker(&lines[last]) { lines[last] = rest; }
    }
    lines.join("\n").trim_end().to_string()
}

pub fn cut_chars(text: &str, max: usize) -> String {
    let t = text.trim();
    if t.chars().count() <= max { return t.to_string(); }
    format!("{}…", t.chars().take(max).collect::<String>().trim_end())
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
        // On-device: 8 messages (was 12, 2026-10-24), each cut to ON_DEVICE_MESSAGE_CHARS.
        // Nothing capped a message's length before; a thread with a few long replies put
        // 2,617 tokens in front of a one-line question on the Pixel (70 s at 37 tokens/s).
        let max_messages = if is_api_model { 40 } else if self.on_device { 8 } else { 8 };

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
        // Blocks of four since 2026-10-24 (were six with a 12-message window): the window
        // holds five to eight messages and its start stays put for two exchanges.
        let start = if !self.on_device {
            conversation_history.len().saturating_sub(max_messages)
        } else if conversation_history.len() <= max_messages {
            0
        } else {
            let over = conversation_history.len() - max_messages;
            over.div_ceil(ON_DEVICE_HISTORY_BLOCK) * ON_DEVICE_HISTORY_BLOCK
        };
        let recent_history: Vec<&ConversationTurn> = conversation_history[start..].iter().collect();

        println!("[Engine] Including {} recent conversation turns", recent_history.len());

        let mut context = String::from("RECENT CONVERSATION:\n");
        for turn in recent_history {
            let role = turn.role.to_uppercase();
            // A reply that kept its hidden "MEMORY: yes|no" (stored before the inline
            // fix of 2026-10-25) must not teach the next model to write one.
            let content = if turn.role.eq_ignore_ascii_case("assistant") { strip_memory_marker(&turn.content) } else { turn.content.clone() };
            let content = if self.on_device { cut_chars(&content, ON_DEVICE_MESSAGE_CHARS) } else { content };
            context.push_str(&format!("{}: {}\n", role, content));
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
        // On-device 4, the fact only: the "you said" text doubled each line and the fact is
        // what the model needs. (3 for an afternoon on 2026-10-24: the fourth memory held
        // the answer to the second test question, and 100 tokens more is fine; Matt.)
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

        // Facts that came from the same message are written as one block: the message the
        // user said, then its facts (Matt, 2026-10-24: "if multiple memories tie to one
        // thing, it's worth the AI getting the full context it came from"). Since call 2
        // makes one memory per fact, three siblings used to carry the same "you said" quote
        // three times; now once. A fact with no recalled sibling is a line as before.
        let normalise = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut groups: Vec<(Option<String>, Vec<&Memory>)> = Vec::new();
        for mem in &limited_memories {
            let key = mem.original_text.as_deref().filter(|t| !t.trim().is_empty()).map(normalise);
            match key.as_ref().and_then(|k| groups.iter_mut().find(|(gk, _)| gk.as_ref() == Some(k))) {
                Some((_, members)) => members.push(mem),
                None => groups.push((key, vec![mem])),
            }
        }
        let quote_chars = if self.on_device { ON_DEVICE_MEMORY_CHARS } else { 400 };
        let mut context = String::from("USER'S STORED MEMORIES:\n");
        let mut idx = 0;
        for (_, members) in &groups {
            idx += 1;
            if members.len() >= 2 {
                let first = members[0];
                let date_str = first.created_at.format("%B %-d, %Y").to_string();
                let said = cut_chars(first.original_text.as_deref().unwrap_or(""), quote_chars);
                context.push_str(&format!("{}. ({}) You said: \"{}\"\n", idx, date_str, said));
                for mem in members {
                    context.push_str(&format!("   - {}\n", cut_chars(&mem.content, ON_DEVICE_MEMORY_CHARS)));
                }
                println!("[Engine]   Including a group of {} facts from one message", members.len());
                continue;
            }
            let mem = members[0];
            // The content is the fact as stored, from a photo, an attachment or an
            // inference as much as from words; original_text is what the user said.
            // Showing the words alone left the model unable to name the furniture oils
            // the content held ("I saved a note but don't know what the item was",
            // 2026-10-08): the fact goes first, the user's words after it when they differ.
            let text_for_llm = if self.on_device { cut_chars(&mem.content, ON_DEVICE_MEMORY_CHARS) } else { memory_line(&mem.content, mem.original_text.as_deref()) };
            let date_str = mem.created_at.format("%B %-d, %Y").to_string();
            context.push_str(&format!("{}. ({}) {}\n", idx, date_str, text_for_llm));
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
            // "Background, not the topic" (Matt, 2026-10-25): the phone's model answered
            // "his most notable achievement" about France because a recalled memory
            // about France sat in the prompt and it took that for the subject.
            "Answer the newest message. The recent conversation and the memories below are background; use them only when they bear on the question. Keep it short: a few sentences, unless they ask for more.\n\n".to_string()
        } else {
            format!("Today's date is {today}.\n\n")
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
        } else if self.on_device {
            // On-device (2026-10-24): about 85 words. The companion voice in four sentences,
            // the memories line and the web-search rule; the explanations around them went,
            // and the "ask for missing details first" line (2026-10-20, written for GPT) is
            // left out so a 4B model answers rather than questions back.
            format!("{opening}You are Zynkbot, a warm, honest companion. Never claim to be human or to replace the people in the user's life. Say so when you are unsure. Do not flatter. When someone shares grief or frustration, acknowledge it before solutions.\n\nStored memories below are the user's own history; treat them as facts about their life.\n\nFor news, weather, prices or recent events, output: WEB_SEARCH_NEEDED: [query]. Never for the date or time, or for the user's own history.\n\n")
        } else {
            // Slim prompt for local models: task-first framing so fact extraction (PART 1)
            // is understood as a primary output requirement, not an afterthought.
            format!("{opening}As Zynkbot, be warm, calm, and genuinely present. Never claim to be human or to replace people in the user's life. Be honest when uncertain. Don't flatter or validate automatically. Don't encourage dependency; reserve professional-help suggestions for real clinical/legal stakes. When someone shares grief or frustration, acknowledge it before solutions. Keep responses proportionate. If one or two missing details would change your answer, ask for them first, briefly; otherwise answer and say what you assumed. The user's data belongs to them.\n\nStored memories below are the user's own personal history — treat them as facts about their life and experiences. Reference them when relevant; answer general questions from training otherwise. You cannot change a memory yourself: when the user corrects one, say the correction will be checked against the stored memories and that they can edit any memory in Memory Manager; never say you have updated it.\n\nThe current date and time are stated above — answer those directly, never search for them. For other real-time info (news, weather, prices, recent events) output: WEB_SEARCH_NEEDED: [query]. Questions using words like 'latest', 'recent', or 'news' require WEB_SEARCH_NEEDED — do not answer those from training data. Do NOT use WEB_SEARCH_NEEDED for questions about the user's personal history or memories.\n\n")
        };

        // If we know the user's name, tell the LLM so it can use it instead of "User"
        // in MEMORY_EXTRACT lines and in conversational replies. Same line in both modes.
        if let Some(name) = user_name {
            system_prompt.push_str(&format!(
                "The user's name is {name}. Use their name occasionally and naturally — not at the start of every reply. Do not address them by name as the first word of your response. Use it mid-sentence or when it adds warmth, the same way a friend would.{extract_note}\n\n",
                extract_note = ""
            ));
        }

        // Call 1 of the memory pipeline (2026-10-24, Matt's design): the reply says only
        // whether the message holds anything to store. On yes, a separate call lists the
        // facts (memory_prompts.rs); on no, nothing runs. Until today the cloud reply carried
        // the fact itself as a MEMORY_EXTRACT line and a second call decided again on the
        // raw message, storing it verbatim when the line was missing. The on-device model
        // gets no marker (it stops answering when asked for one); its call 2 decides.
        if !self.on_device {
            system_prompt.push_str(&format!(r#"MEMORY LINE
End every reply with one final line, exactly "MEMORY: yes" or "MEMORY: no", and nothing else on that line.
- yes: the user's message states a personal fact about {subject_label}: their life, people, pets, work, home, health, feelings, plans, possessions, likes or dislikes. A fact stated inside a question counts ("Raspberry pie is my favorite. What's a good recipe?" is yes).
- no: a question alone, chit-chat, general knowledge, a request about their stored memories, text they ask you to translate or that they quote from someone else, or a message only about Zynkbot itself.
Do not write the fact in your reply; the line is only yes or no.

"#, subject_label = subject_label));
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
            // Phone order (Matt, 2026-10-25): memories before the conversation, so what the
            // model reads last before the question is the thread, not a recalled fact. A
            // recalled France memory next to "his most notable achievement" had won over the
            // Obama turns. The engine's prompt cache now keeps only the instructions between
            // turns (the memories change with every question); measured on the test phones
            // before this stays.
            format!(
                "{}{}{}Today's date is {today}.\n\nUSER'S QUESTION: {}\n\nYOUR RESPONSE:",
                system_prompt, memory_context, history_context, user_input
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

#[cfg(test)]
mod on_device_prompt_tests {
    use super::*;

    fn turns(n: usize) -> Vec<ConversationTurn> {
        (0..n).map(|i| ConversationTurn {
            role: if i % 2 == 0 { "user".into() } else { "assistant".into() },
            content: format!("message {i}"),
        }).collect()
    }
    fn first_included(ctx: &str) -> usize {
        // "USER: message 7" -> 7
        let i = ctx.find("message ").unwrap() + "message ".len();
        ctx[i..].split_whitespace().next().unwrap().parse().unwrap()
    }

    #[test]
    fn on_device_history_window_moves_in_blocks_of_four() {
        let e = ConversationEngine::new().with_on_device(true);
        assert_eq!(first_included(&e.build_conversation_context(&turns(8), false)), 0);
        // One over the window: jump a whole block, not one message.
        assert_eq!(first_included(&e.build_conversation_context(&turns(9), false)), 4);
        assert_eq!(first_included(&e.build_conversation_context(&turns(12), false)), 4);
        assert_eq!(first_included(&e.build_conversation_context(&turns(13), false)), 8);
    }
    #[test]
    fn on_device_messages_are_cut_to_400_characters_custom_endpoint_not() {
        let long = "x".repeat(1000);
        let t = vec![ConversationTurn { role: "assistant".into(), content: long.clone() }];
        let on = ConversationEngine::new().with_on_device(true).build_conversation_context(&t, false);
        assert!(on.contains(&format!("{}…", "x".repeat(400))));
        assert!(!on.contains(&"x".repeat(401)));
        let custom = ConversationEngine::new().build_conversation_context(&t, false);
        assert!(custom.contains(&long));
    }
    #[test]
    fn on_device_memory_lines_carry_the_fact_only() {
        let mems = vec![Memory { id: 1, content: "Matt has a cat named Louie.".into(), original_text: Some("I have a cat named Louie and he is orange".into()), title: None, similarity: Some(0.9), created_at: chrono::Utc::now() }];
        let on = ConversationEngine::new().with_on_device(true).build_memory_context(&mems, "q", false);
        assert!(on.contains("Matt has a cat named Louie.") && !on.contains("you said"));
        let custom = ConversationEngine::new().build_memory_context(&mems, "q", false);
        assert!(custom.contains("you said"));
    }
    #[test]
    fn facts_from_one_message_are_written_as_one_block_with_the_quote_once() {
        let said = "I have a cat named Louie and I'm flying to Denver Thursday for my nephew's wedding.";
        let m = |id: i32, content: &str, said: Option<&str>| Memory { id, content: content.into(), original_text: said.map(str::to_string), title: None, similarity: Some(0.9), created_at: chrono::Utc::now() };
        let mems = vec![
            m(1, "Matt has a cat named Louie.", Some(said)),
            m(2, "Matt's dentist is Dr. Patel.", Some("I switched my dentist to Dr. Patel last week.")),
            m(3, "Matt is flying to Denver on Thursday for his nephew's wedding.", Some(said)),
        ];
        for engine in [ConversationEngine::new(), ConversationEngine::new().with_on_device(true)] {
            let ctx = engine.build_memory_context(&mems, "q", false);
            assert_eq!(ctx.matches("Denver Thursday for my nephew's wedding").count(), 1, "the quote appears once: {}", ctx);
            assert!(ctx.contains("   - Matt has a cat named Louie.") && ctx.contains("   - Matt is flying to Denver"), "{}", ctx);
            assert!(ctx.contains("Matt's dentist is Dr. Patel."), "the single fact stays a line: {}", ctx);
            assert!(ctx.contains("1. ") && ctx.contains("2. ") && !ctx.contains("3. "), "two items, not three: {}", ctx);
        }
    }

    #[test]
    fn history_never_carries_the_marker() {
        assert_eq!(strip_memory_marker("Fine answer.\n\nMEMORY: no"), "Fine answer.");
        assert_eq!(strip_memory_marker("It can take you places. MEMORY: no"), "It can take you places.");
        assert_eq!(strip_memory_marker("Plain reply with no marker."), "Plain reply with no marker.");
    }

    #[test]
    fn on_device_instructions_stay_short() {
        let e = ConversationEngine::new().with_on_device(true);
        let p = e.build_prompt("hello", Some(&turns(2)), None, false, Some("Matt"));
        let instructions = &p[..p.find("RECENT CONVERSATION").unwrap()];
        // 140 until 2026-10-25; the "background, not the topic" line (Matt) added about twenty words on purpose.
        assert!(instructions.split_whitespace().count() < 165, "{} words", instructions.split_whitespace().count());
        assert!(p.contains("WEB_SEARCH_NEEDED"));
    }
    #[test]
    fn custom_endpoint_history_window_slides_by_one() {
        let e = ConversationEngine::new();
        assert_eq!(first_included(&e.build_conversation_context(&turns(9), false)), 1);
        assert_eq!(first_included(&e.build_conversation_context(&turns(10), false)), 2);
    }
    #[test]
    fn on_device_prompt_asks_only_for_the_answer_and_dates_after_the_conversation() {
        let e = ConversationEngine::new().with_on_device(true);
        let p = e.build_prompt("What is the capital of France?", Some(&turns(4)), None, false, Some("Matt"));
        assert!(!p.contains("MEMORY_EXTRACT"), "no extraction instruction for the on-device model");
        assert!(!p.contains("PART 1"));
        assert!(!p.contains("REMEMBER:"));
        assert!(p.contains("Keep it short"));
        let date = p.find("Today's date is").expect("date present");
        let history = p.find("RECENT CONVERSATION").expect("history present");
        assert!(date > history, "the date comes after the conversation so the kept context survives the clock");
    }
    #[test]
    fn custom_and_cloud_prompts_ask_for_the_memory_line_only() {
        let e = ConversationEngine::new();
        let p = e.build_prompt("I have a dog named Rex.", None, None, false, Some("Matt"));
        assert!(p.contains("MEMORY: yes"), "the yes/no marker instruction is missing");
        assert!(!p.contains("MEMORY_EXTRACT"), "the fact line must not be asked for any more");
        assert!(p.starts_with("Today's date is"));
        let api = e.build_prompt("I have a dog named Rex.", None, None, true, Some("Matt"));
        assert!(api.contains("MEMORY: yes") && !api.contains("MEMORY_EXTRACT"));
        let on = ConversationEngine::new().with_on_device(true).build_prompt("hi", None, None, false, None);
        assert!(!on.contains("MEMORY:"), "the on-device model is not asked for a marker");
    }
    #[test]
    fn on_device_recalls_four_memories() {
        let e = ConversationEngine::new().with_on_device(true);
        let mems: Vec<Memory> = (0..7).map(|i| Memory { id: i, content: format!("fact number {i}"), original_text: None, title: None, similarity: Some(0.9), created_at: chrono::Utc::now() }).collect();
        let ctx = e.build_memory_context(&mems, "question", false);
        assert_eq!(ctx.matches("fact number").count(), 4);
        assert_eq!(ConversationEngine::new().build_memory_context(&mems, "question", false).matches("fact number").count(), 7);
    }
}
