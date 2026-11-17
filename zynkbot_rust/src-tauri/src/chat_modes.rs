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

/// Whether a conversation in this mode teaches Zynkbot about the user (2026-11-17, Matt).
/// Editing: the text is someone's words to be fixed, not the user's beliefs (a memory
/// "uses editing mode for dictation" came from a pasted paragraph). Tool: the task says
/// little about the user. Premise: the user states things as true for the sake of the
/// argument, so storing them as beliefs is exactly wrong. Debate and Normal record. An
/// explicit "Remember:" is honoured in every mode (chat.rs checks that first).
pub fn records_memory(mode: Option<&str>) -> bool {
    !matches!(mode.map(|m| m.trim().to_lowercase()).as_deref(), Some("tool") | Some("premise") | Some("editing"))
}

/// Editing gets its own short prompt instead of the companion prompt (2026-11-17): no
/// memories, no earlier turns beyond the last reply, no MEMORY line, no search marker,
/// no pictures from earlier turns. On the Pixel (2026-11-16) three Editing requests took
/// 52 to 87 s; the models reasoned over 30 recalled memories, 8 turns and a prompt that
/// asked for a MEMORY line and "the text and nothing else" at once.
pub fn is_lean(mode: Option<&str>) -> bool {
    matches!(mode.map(|m| m.trim().to_lowercase()).as_deref(), Some("editing"))
}

/// The Editing prompt. `last_reply` is the text returned on the previous turn, so
/// "shorter" or "now make it formal" has something to work on.
/// With a picture attached the text to fix is in the picture (a screenshot of a message)
/// and the typed part is at most an instruction: without that line one run in three of
/// gpt-5.4 corrected the two typed words and ignored the picture (2026-11-17).
pub fn editing_prompt(message: &str, last_reply: Option<&str>, with_picture: bool) -> String {
    let mut p = String::from("You are an editor. Fix grammar, spelling and clarity in the text below. Keep the wording, tone, profanity and meaning. Return the corrected text and nothing else: no preamble, no notes, no explanation. If the message is an instruction about the text you returned last time rather than new text, follow it and return the result the same way.\n\n");
    if with_picture {
        p.push_str("The text to fix is in the attached picture; the message below is an instruction about it, if there is one, and otherwise the picture's text is what to return corrected.\n\n");
    }
    if let Some(prev) = last_reply.map(str::trim).filter(|t| !t.is_empty()) {
        p.push_str("THE TEXT YOU RETURNED LAST TIME:\n");
        p.push_str(prev);
        p.push_str("\n\n");
    }
    p.push_str("TEXT:\n");
    p.push_str(message.trim());
    p
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
    fn normal_and_debate_record_memories_and_the_task_modes_do_not() {
        assert!(records_memory(None));
        assert!(records_memory(Some("normal")));
        assert!(records_memory(Some("debate")));
        assert!(records_memory(Some("banana")));
        for m in ["tool", "premise", "editing", "Editing", " editing "] { assert!(!records_memory(Some(m)), "{m}"); }
    }

    #[test]
    fn only_editing_is_lean_and_its_prompt_holds_the_text_and_the_last_reply() {
        assert!(is_lean(Some("Editing")));
        for m in ["normal", "tool", "debate", "premise"] { assert!(!is_lean(Some(m)), "{m}"); }
        let p = editing_prompt("  there going home ", None, false);
        assert!(p.ends_with("TEXT:\nthere going home"));
        assert!(!p.contains("LAST TIME"));
        assert!(!p.contains("MEMORY"));
        let p = editing_prompt("shorter", Some("They are going home."), false);
        assert!(p.contains("THE TEXT YOU RETURNED LAST TIME:\nThey are going home.\n\nTEXT:\nshorter"));
        assert!(!editing_prompt("x", Some("   "), false).contains("LAST TIME"));
        assert!(!p.contains("attached picture"));
        assert!(editing_prompt("fix this", None, true).contains("The text to fix is in the attached picture"));
    }

    #[test]
    fn a_mode_goes_in_front_and_every_listed_mode_but_normal_has_a_block() {
        assert!(apply(Some("Editing"), "p".into()).starts_with("Mode: Editing."));
        assert!(apply(Some("debate"), "p".into()).ends_with("\n\np"));
        for m in MODES.iter().filter(|m| **m != "normal") { assert!(block(Some(m)).is_some(), "{m}"); }
    }
}
