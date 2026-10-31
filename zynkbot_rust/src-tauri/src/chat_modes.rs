//! Conversation modes (2026-10-28, Matt): a short block of instructions the user switches
//! on for a conversation, put in front of the prompt. Prompt text only; no code runs.
//! Not the containment modes (containment.rs), which are an enforcement layer.

// Companion was a mode until 2026-10-29: the probe showed Normal already is the companion
// voice, so its two extra phrases (direct; no cheering on, no therapy language) moved into
// the Normal prompt and the mode went.
pub const MODES: &[&str] = &["normal", "tool", "debate", "premise", "editing"];

/// The block for a mode, or None for normal and unknown names.
pub fn block(mode: Option<&str>) -> Option<&'static str> {
    match mode.map(|m| m.trim().to_lowercase()).as_deref() {
        Some("tool") => Some("Mode: Tool. Do what is asked and nothing more: no commentary, no suggestions, no questions back unless the task cannot be done without an answer."),
        Some("debate") => Some("Mode: Debate. Challenge weak reasoning in what the user says and name the weakest point plainly. When asked, argue the other side at full strength, without a both-sides closer."),
        Some("premise") => Some("Mode: Premise. Take what the user states as true for this conversation and reason from it. Do not fact-check or qualify the premise unless asked to."),
        Some("editing") => Some("Mode: Editing. When given text, fix grammar, spelling and clarity only. Keep the wording, tone, profanity and meaning. Return the text and nothing else."),
        _ => None,
    }
}

/// The prompt with the mode's block in front of it, when there is one.
pub fn apply(mode: Option<&str>, prompt: String) -> String {
    match block(mode) {
        Some(b) => format!("{}\n\n{}", b, prompt),
        None => prompt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_and_unknown_leave_the_prompt_alone() {
        assert_eq!(apply(None, "p".into()), "p");
        assert_eq!(apply(Some("normal"), "p".into()), "p");
        assert_eq!(apply(Some("banana"), "p".into()), "p");
    }

    #[test]
    fn a_mode_goes_in_front_and_every_listed_mode_but_normal_has_a_block() {
        assert!(apply(Some("Editing"), "p".into()).starts_with("Mode: Editing."));
        assert!(apply(Some("debate"), "p".into()).ends_with("\n\np"));
        for m in MODES.iter().filter(|m| **m != "normal") { assert!(block(Some(m)).is_some(), "{m}"); }
    }
}
