//! What the memory-decision call returns beyond "remember? / title / links",
//! and the validation that keeps model output honest before it is stored.
//!
//! Added 2026-09-07 after an analyst pass over three weeks of real data found
//! `event_date` empty or a Jan-1 placeholder, `namespace` wrong a quarter of the
//! time, sentiment never computed, and entities kept as a raw name-finder blob.
//! All of these are asked for in the same LLM call that already decides whether
//! to remember, so there is no extra request.

use chrono::{Datelike, NaiveDate};

/// Namespaces the UI knows how to show. Anything else the model proposes is
/// replaced by the caller's fallback (the NLP enhancer's guess).
pub const CANONICAL_NAMESPACES: [&str; 15] = [
    "personal", "work", "career", "health", "family", "education", "technology",
    "science", "philosophy", "politics", "travel", "achievements", "biography",
    "kitchen", "hobbies",
];

/// One named thing in a memory, as the model reports it.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize, PartialEq)]
pub struct EntityOut {
    pub name: String,
    #[serde(default)]
    pub kind: String, // person | place | org | thing
}

/// Extra fields from the decision call. Every field is optional so an older or
/// weaker model that returns only the original shape still parses.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct MemoryExtras {
    #[serde(default)]
    pub event_date: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub tone: Option<String>,
    #[serde(default)]
    pub entities: Option<Vec<EntityOut>>,
}

/// Prompt text describing the extra fields. Shared by every backend.
pub fn prompt_instructions(today: NaiveDate) -> String {
    format!(
        r#"Today's date is {today}. Also fill in:
- "event_date": the date the thing described HAPPENED (not today), as "YYYY-MM-DD", only when the message states or clearly implies it ("yesterday", "last Tuesday", "in 2019", "on my birthday, March 4"). Resolve relative words against today's date. If the date is not stated or implied, use null. Never guess a year or default to January 1.
- "namespace": one of {namespaces}. Pick the one a person would file this under; "personal" only when nothing more specific fits.
- "tags": 1 to 3 short lowercase topic words (e.g. ["keto", "chili"], ["elden ring"]).
- "tone": the user's tone in this message, one of "positive", "neutral", "negative", "frustrated", "anxious", "excited".
- "entities": the people, places, organisations and named things mentioned, each as {{"name": "...", "kind": "person|place|org|thing"}}; an empty list if none."#,
        today = today.format("%Y-%m-%d"),
        namespaces = CANONICAL_NAMESPACES.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", "),
    )
}

/// JSON-schema fragment for the local-model constrained decoder.
pub const SCHEMA_PROPERTIES: &str = r#"
    "event_date": {"type": ["string", "null"]},
    "namespace": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "tone": {"type": "string"},
    "entities": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"type": "string"}}, "required": ["name"]}}"#;

/// A model-supplied event date is kept only if it parses, is not absurd, and is
/// not the Jan-1 placeholder unless the text itself talks about January 1 or New Year.
pub fn validate_event_date(raw: Option<&str>, text: &str, today: NaiveDate) -> Option<NaiveDate> {
    let s = raw?.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return None;
    }
    let d = NaiveDate::parse_from_str(&s[..s.len().min(10)], "%Y-%m-%d").ok()?;
    if d.year() < 1900 || d > today + chrono::Duration::days(366 * 2) {
        return None;
    }
    if d.month() == 1 && d.day() == 1 {
        let t = text.to_ascii_lowercase();
        if !(t.contains("january 1") || t.contains("jan 1") || t.contains("new year")) {
            return None;
        }
    }
    Some(d)
}

/// The namespace to store: the model's if it is one we know, else the fallback.
pub fn resolve_namespace(proposed: Option<&str>, fallback: &str) -> String {
    match proposed.map(|p| p.trim().to_ascii_lowercase()) {
        Some(p) if CANONICAL_NAMESPACES.contains(&p.as_str()) => p,
        _ => fallback.to_string(),
    }
}

/// Lower-cased, trimmed, de-duplicated, at most five, none longer than 30 chars.
pub fn clean_tags(tags: Option<&[String]>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in tags.unwrap_or(&[]) {
        let t = t.trim().to_ascii_lowercase();
        if t.is_empty() || t.chars().count() > 30 || out.contains(&t) {
            continue;
        }
        out.push(t);
        if out.len() == 5 {
            break;
        }
    }
    out
}

/// Tone word → (label, score). Unknown or missing tone is neutral.
pub fn tone_to_sentiment(tone: Option<&str>) -> (&'static str, f64) {
    match tone.map(|t| t.trim().to_ascii_lowercase()).as_deref() {
        Some("positive") => ("positive", 0.6),
        Some("excited") => ("positive", 0.8),
        Some("negative") => ("negative", -0.6),
        Some("frustrated") => ("negative", -0.7),
        Some("anxious") => ("negative", -0.5),
        _ => ("neutral", 0.0),
    }
}

/// Entities worth a row: 2+ characters, not a pronoun/contraction, known kind.
pub fn clean_entities(entities: Option<&[EntityOut]>) -> Vec<EntityOut> {
    const SKIP: [&str; 12] = ["i", "me", "you", "we", "they", "it", "user", "the user", "i'm", "i've", "he", "she"];
    let mut out: Vec<EntityOut> = Vec::new();
    for e in entities.unwrap_or(&[]) {
        let name = e.name.trim();
        let canon = name.to_ascii_lowercase();
        if name.chars().count() < 2 || SKIP.contains(&canon.as_str()) || name.contains('\'') {
            continue;
        }
        if out.iter().any(|o| o.name.eq_ignore_ascii_case(name)) {
            continue;
        }
        let kind = match e.kind.trim().to_ascii_lowercase().as_str() {
            "person" | "place" | "org" | "thing" => e.kind.trim().to_ascii_lowercase(),
            "organisation" | "organization" | "company" => "org".to_string(),
            "location" | "city" | "country" => "place".to_string(),
            _ => "thing".to_string(),
        };
        out.push(EntityOut { name: name.to_string(), kind });
        if out.len() == 12 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn today() -> NaiveDate { NaiveDate::from_ymd_opt(2026, 9, 7).unwrap() }

    #[test]
    fn event_date_accepts_real_dates_and_rejects_placeholders() {
        assert_eq!(validate_event_date(Some("2026-09-06"), "I made chili yesterday", today()), NaiveDate::from_ymd_opt(2026, 9, 6));
        assert_eq!(validate_event_date(Some("2026-09-06T00:00:00"), "yesterday", today()), NaiveDate::from_ymd_opt(2026, 9, 6));
        assert_eq!(validate_event_date(Some("2025-01-01"), "I started keto this year", today()), None);
        assert_eq!(validate_event_date(Some("2025-01-01"), "On January 1 I started keto", today()), NaiveDate::from_ymd_opt(2025, 1, 1));
        assert_eq!(validate_event_date(Some("null"), "no date", today()), None);
        assert_eq!(validate_event_date(None, "no date", today()), None);
        assert_eq!(validate_event_date(Some("1850-03-04"), "absurd", today()), None);
        assert_eq!(validate_event_date(Some("2031-03-04"), "too far", today()), None);
        assert_eq!(validate_event_date(Some("not a date"), "x", today()), None);
    }

    #[test]
    fn namespace_falls_back_when_unknown() {
        assert_eq!(resolve_namespace(Some("Kitchen"), "personal"), "kitchen");
        assert_eq!(resolve_namespace(Some("cooking"), "personal"), "personal");
        assert_eq!(resolve_namespace(None, "work"), "work");
    }

    #[test]
    fn tags_are_cleaned_and_capped() {
        let raw = vec!["Keto".into(), " chili ".into(), "keto".into(), "".into(), "a".repeat(40), "b".into(), "c".into(), "d".into()];
        assert_eq!(clean_tags(Some(&raw)), vec!["keto", "chili", "b", "c", "d"]);
        assert!(clean_tags(None).is_empty());
    }

    #[test]
    fn tone_maps_to_label_and_score() {
        assert_eq!(tone_to_sentiment(Some("Frustrated")), ("negative", -0.7));
        assert_eq!(tone_to_sentiment(Some("excited")), ("positive", 0.8));
        assert_eq!(tone_to_sentiment(Some("weird")), ("neutral", 0.0));
        assert_eq!(tone_to_sentiment(None), ("neutral", 0.0));
    }

    #[test]
    fn entities_drop_pronouns_and_normalise_kinds() {
        let raw = vec![
            EntityOut { name: "Vermont".into(), kind: "location".into() },
            EntityOut { name: "I".into(), kind: "person".into() },
            EntityOut { name: "Laurimar".into(), kind: "Person".into() },
            EntityOut { name: "vermont".into(), kind: "place".into() },
            EntityOut { name: "Elden Ring".into(), kind: "game".into() },
        ];
        let out = clean_entities(Some(&raw));
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], EntityOut { name: "Vermont".into(), kind: "place".into() });
        assert_eq!(out[1].kind, "person");
        assert_eq!(out[2].kind, "thing");
    }
}
