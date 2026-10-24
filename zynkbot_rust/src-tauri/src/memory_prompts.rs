//! The memory pipeline's prompts, in one place (2026-10-24).
//!
//! After a message the app runs up to three calls: 1) does the message hold anything to
//! store (on the cloud, a yes/no line at the end of the reply; on the phone, folded into
//! call 2); 2) the facts, each with its title, date, tags, tone and named things (this
//! file); 3) how each fact relates to similar memories (lib.rs ask_llm_for_relationships).
//! The "Remember:" command bypasses all three and stores the words as said.
//!
//! The wording is scored by the memory bench (labs/memory_bench) before it reaches a phone.

use chrono::NaiveDate;
use crate::memory_extras::{self, MemoryExtras};

/// Memories one message may produce at most.
pub const MAX_FACTS: usize = 3;

/// Call 2: the facts in `message` about `subject`, as JSON. `recent_turns` is the last
/// exchange or two, so "she" and "it" resolve; facts are taken from the message only.
pub fn extraction_prompt(message: &str, subject: &str, recent_turns: Option<&str>, today: NaiveDate) -> String {
    extraction_prompt_with(message, subject, recent_turns, today, false)
}

/// `slim` (the phone, 2026-10-24): fact, title and date only. The 4B writes JSON at about
/// three tokens a second under the grammar, and tags, tone and named things cost it sixty
/// tokens a fact; the on-device name finder (nlp_enhancer) fills those instead.
pub fn extraction_prompt_with(message: &str, subject: &str, recent_turns: Option<&str>, today: NaiveDate, slim: bool) -> String {
    let namespaces = memory_extras::CANONICAL_NAMESPACES.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ");
    let context = match recent_turns.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("The conversation just before this message, so you know who or what the message refers to. List only facts the message itself states.\n{t}\n\n"),
        None => String::new(),
    };
    format!(
"Today's date is {today}.

{context}Read {subject}'s message below. List the personal facts about {subject} in it, up to {max}, one sentence per fact, keeping the message's own words and every detail (names, places, dates, numbers): the people in their life, where they live and go, their work and past work, health, how they are feeling, what they own and the services they use, routines and habits, plans and events, likes and dislikes, names and numbers they ask to keep.

Rules: only what the message states, nothing guessed. What {subject} asks is not a fact, but what {subject} states while asking is: 'Raspberry pie is my favorite. What's a good recipe?' holds one fact, that raspberry pie is {subject}'s favorite, because {subject} states it. 'Could I put salt in my coffee?' holds none, because {subject} only asks; a question says nothing about what {subject} does or has. Nothing from text {subject} quotes, pastes or asks to translate. Greetings and thanks are not facts. If the message holds no fact about {subject}, write NONE.

For each fact give:
- \"fact\": one sentence in the third person, starting with \"{subject}\"
- \"title\": up to 50 characters
- \"event_date\": when it happened, as \"YYYY-MM-DD\", only if the message says or clearly implies it (\"yesterday\", \"last Tuesday\", \"in 2019\"), resolved against today's date; otherwise null. Never guess a year.
{extra_fields}
Answer with JSON only, in this shape:
{shape}

{subject}'s message: \"{message}\"",
        today = today.format("%Y-%m-%d"),
        max = MAX_FACTS,
        message = message.replace('"', "'"),
        extra_fields = if slim { String::new() } else { format!("- \"namespace\": one of {namespaces}; \"personal\" only when nothing more specific fits\n- \"tags\": 1 to 3 short lowercase topic words\n- \"tone\": {subject}'s tone in the message, one of \"positive\", \"neutral\", \"negative\", \"frustrated\", \"anxious\", \"excited\"\n- \"entities\": the people, places, organisations and named things in the fact, each as {{\"name\": \"...\", \"kind\": \"person|place|org|thing\"}}; [] if none\n") },
        shape = if slim { "{\"facts\": [{\"fact\": \"...\", \"title\": \"...\", \"event_date\": null}]}" } else { "{\"facts\": [{\"fact\": \"...\", \"title\": \"...\", \"event_date\": null, \"namespace\": \"personal\", \"tags\": [\"...\"], \"tone\": \"neutral\", \"entities\": []}]}" },
    )
}

/// The slim schema: fact, title, date.
pub const EXTRACTION_SCHEMA_SLIM: &str = r#"{
  "type": "object",
  "properties": {
    "facts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "fact": {"type": "string"},
          "title": {"type": "string"},
          "event_date": {"type": ["string", "null"]}
        },
        "required": ["fact", "title"]
      }
    }
  },
  "required": ["facts"]
}"#;

/// JSON schema for the phone's constrained decoder: the model cannot answer outside it.
pub const EXTRACTION_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "facts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "fact": {"type": "string"},
          "title": {"type": "string"},
          "event_date": {"type": ["string", "null"]},
          "namespace": {"type": "string"},
          "tags": {"type": "array", "items": {"type": "string"}},
          "tone": {"type": "string"},
          "entities": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string"}}, "required": ["name"]}}
        },
        "required": ["fact", "title"]
      }
    }
  },
  "required": ["facts"]
}"#;

/// One memory-to-be from call 2.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct ExtractedFact {
    pub fact: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(flatten)]
    pub extras: MemoryExtras,
}

#[derive(serde::Deserialize)]
struct FactsEnvelope { #[serde(default)] facts: Vec<ExtractedFact> }

/// Words too common to show that a fact came from the message.
const FILLER: &[&str] = &["the", "and", "but", "for", "nor", "not", "has", "had", "have", "was", "were", "are", "his", "her", "its", "him", "she", "they", "them", "our", "you", "your", "who", "what", "how", "why", "when", "that", "this", "with", "from", "into", "about", "just", "now", "then", "than", "too", "very", "can", "will", "all", "any", "one", "out", "get", "got", "did", "does", "also", "been", "being", "some", "only", "like", "user", "matt"];

/// A content word: three letters or more and not filler, or a number of two digits or more
/// ("I'm 52", "my son Tom" — the four-letter rule used until 2026-10-24 dropped "son" and "Tom").
fn is_content_word(w: &str) -> bool {
    (w.len() >= 3 && !FILLER.contains(&w)) || (w.len() > 1 && w.chars().all(|c| c.is_ascii_digit()))
}

/// Does `fact` share at least one content word with `message`? The guard against a
/// model inventing a fact the message never stated (memory bench, 2026-10-22).
pub fn shares_content_word(fact: &str, message: &str) -> bool {
    let msg: std::collections::HashSet<String> = message.to_lowercase()
        .split(|c: char| !c.is_alphanumeric()).filter(|w| is_content_word(w)).map(str::to_string).collect();
    fact.to_lowercase().split(|c: char| !c.is_alphanumeric()).any(|w| is_content_word(w) && msg.contains(w))
}

/// The JSON object in a model's answer, with code fences and chatter around it removed.
fn json_body(text: &str) -> &str {
    let t = text.trim();
    let t = if let Some(i) = t.find("```") {
        let after = &t[i + 3..];
        let after = after.strip_prefix("json").unwrap_or(after);
        after.split("```").next().unwrap_or(after)
    } else { t };
    // Whichever opens first, an object or an array, is the answer.
    let obj = t.find('{'); let arr = t.find('[');
    match (obj, arr) {
        (_, Some(a)) if obj.map_or(true, |o| a < o) => match t.rfind(']') { Some(b) if b > a => &t[a..=b], _ => t.trim() },
        (Some(a), _) => match t.rfind('}') { Some(b) if b > a => &t[a..=b], _ => t.trim() },
        _ => t.trim(),
    }
}

/// The facts in call 2's answer that pass the guards: three or more words, at least one
/// content word shared with the message, no duplicates, at most MAX_FACTS. Titles are
/// trimmed to 50 characters; a missing title is the start of the fact. "NONE" or an
/// empty list means nothing to store.
pub fn parse_extraction(text: &str, message: &str) -> Vec<ExtractedFact> {
    if text.trim().eq_ignore_ascii_case("none") { return Vec::new(); }
    let body = json_body(text);
    let parsed: Vec<ExtractedFact> = if body.starts_with('[') {
        serde_json::from_str::<Vec<ExtractedFact>>(body).unwrap_or_default()
    } else {
        match serde_json::from_str::<FactsEnvelope>(body) {
            Ok(env) => env.facts,
            Err(_) => serde_json::from_str::<ExtractedFact>(body).map(|f| vec![f]).unwrap_or_default(),
        }
    };
    let mut out: Vec<ExtractedFact> = Vec::new();
    for mut f in parsed {
        let fact = f.fact.trim().trim_end_matches('.').trim().to_string();
        if fact.split_whitespace().count() < 3 { continue; }
        if !shares_content_word(&fact, message) {
            println!("[Memory] call 2: dropped a fact that shares no words with the message: {}", fact);
            continue;
        }
        if out.iter().any(|o| o.fact.eq_ignore_ascii_case(&(fact.clone() + "."))) { continue; }
        f.fact = fact + ".";
        f.title = Some(clean_title(f.title.as_deref(), &f.fact));
        out.push(f);
        if out.len() == MAX_FACTS { break; }
    }
    out
}

/// A title of at most 50 characters: the model's if it gave one, else the fact's opening.
pub fn clean_title(title: Option<&str>, fact: &str) -> String {
    let t = title.map(str::trim).filter(|t| !t.is_empty()).unwrap_or_else(|| fact.trim().trim_end_matches('.'));
    if t.chars().count() <= 50 { t.to_string() } else { format!("{}…", t.chars().take(49).collect::<String>()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MSG: &str = "I have a cat named Louie, and I'm planning a trip to Indiana in April.";

    #[test]
    fn facts_with_titles_and_extras_parse() {
        let text = r#"{"facts": [{"fact": "Matt has a cat named Louie", "title": "Cat Louie", "event_date": null, "namespace": "personal", "tags": ["cat"], "tone": "neutral", "entities": [{"name": "Louie", "kind": "thing"}]},
            {"fact": "Matt plans a trip to Indiana in April.", "title": "Indiana trip in April", "namespace": "travel", "tags": ["travel"], "tone": "excited", "entities": [{"name": "Indiana", "kind": "place"}]}]}"#;
        let out = parse_extraction(text, MSG);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].fact, "Matt has a cat named Louie.");
        assert_eq!(out[0].title.as_deref(), Some("Cat Louie"));
        assert_eq!(out[1].extras.namespace.as_deref(), Some("travel"));
        assert_eq!(out[1].extras.entities.as_ref().unwrap()[0].name, "Indiana");
    }

    #[test]
    fn code_fences_and_chatter_are_ignored() {
        let text = "Here you go:\n```json\n{\"facts\": [{\"fact\": \"Matt has a cat named Louie.\", \"title\": \"Louie\"}]}\n```\nHope that helps.";
        assert_eq!(parse_extraction(text, MSG).len(), 1);
    }

    #[test]
    fn empty_list_and_none_mean_nothing() {
        assert!(parse_extraction(r#"{"facts": []}"#, MSG).is_empty());
        assert!(parse_extraction("NONE", MSG).is_empty());
        assert!(parse_extraction("", MSG).is_empty());
    }

    #[test]
    fn invented_fact_is_dropped() {
        let text = r#"{"facts": [{"fact": "Matt lives in Boston.", "title": "Home"}, {"fact": "Matt has a cat named Louie.", "title": "Louie"}]}"#;
        let out = parse_extraction(text, MSG);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fact, "Matt has a cat named Louie.");
    }

    #[test]
    fn at_most_three_and_no_duplicates() {
        let text = r#"{"facts": [{"fact": "Matt has a cat named Louie.", "title": "a"}, {"fact": "Matt has a cat named Louie", "title": "b"},
            {"fact": "Matt plans a trip to Indiana.", "title": "c"}, {"fact": "Matt travels in April.", "title": "d"}, {"fact": "Matt's cat is Louie.", "title": "e"}]}"#;
        assert_eq!(parse_extraction(text, MSG).len(), 3);
    }

    #[test]
    fn title_falls_back_and_is_cut_to_fifty() {
        let text = r#"{"facts": [{"fact": "Matt has a cat named Louie.", "title": ""}, {"fact": "Matt plans a trip to Indiana in April.", "title": "A very long title that goes on and on well past the fifty character mark"}]}"#;
        let out = parse_extraction(text, MSG);
        assert_eq!(out[0].title.as_deref(), Some("Matt has a cat named Louie"));
        assert_eq!(out[1].title.as_ref().unwrap().chars().count(), 50);
    }

    #[test]
    fn short_names_count_as_shared_words() {
        let text = r#"{"facts": [{"fact": "Matt has a son named Tom.", "title": "Son Tom"}]}"#;
        assert_eq!(parse_extraction(text, "My son Tom is turning 16 next month. Any gift ideas?").len(), 1);
        assert!(!shares_content_word("Matt has a cat.", "The and his are not it."));
    }

    #[test]
    fn slim_prompt_asks_for_fact_title_and_date_only() {
        let full = extraction_prompt("I'm 52.", "Matt", None, NaiveDate::from_ymd_opt(2026, 10, 24).unwrap());
        let slim = extraction_prompt_with("I'm 52.", "Matt", None, NaiveDate::from_ymd_opt(2026, 10, 24).unwrap(), true);
        assert!(full.contains("\"tags\"") && full.contains("\"entities\""));
        assert!(!slim.contains("\"tags\"") && !slim.contains("\"entities\"") && slim.contains("\"event_date\""));
        assert!(slim.len() < full.len());
    }

    #[test]
    fn bare_array_is_accepted() {
        let text = r#"[{"fact": "Matt has a cat named Louie.", "title": "Louie"}]"#;
        assert_eq!(parse_extraction(text, MSG).len(), 1);
    }

    #[test]
    fn prompt_carries_message_context_and_date() {
        let p = extraction_prompt("I'm 52.", "Matt", Some("Zynkbot: How old are you now?"), NaiveDate::from_ymd_opt(2026, 10, 24).unwrap());
        assert!(p.contains("Today's date is 2026-10-24."));
        assert!(p.contains("How old are you now?"));
        assert!(p.contains("Matt's message: \"I'm 52.\""));
        assert!(p.contains("What Matt asks is not a fact, but what Matt states while asking is"));
    }
}
