//! What the model is told about a Knowledge Base search, and what the UI is told.
//!
//! Before 2026-09-07 the chat command told the model "you MUST use the
//! information below" and "answer using ONLY the information above" no matter
//! what the search returned. When nothing scored above the threshold it sent
//! the five best chunks anyway under the same wording, and when the search
//! returned nothing at all it added no instruction, so the model answered from
//! general knowledge as if it were the user's documents. A tester removed a
//! CSV from the knowledge base, asked the same question, and got a confident
//! invented answer that was then stored as a memory (GitHub #17, #22).
//!
//! This module decides, from the search results alone, which of three
//! situations the model is in and words the prompt block accordingly. It has
//! no database access so it can be unit-tested.

use crate::kb_rag::KBSearchResult;

/// How well the knowledge base answered the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KbOutcome {
    /// At least one chunk scored above the relevance threshold.
    Found,
    /// The search returned chunks, but none above the threshold.
    WeakOnly,
    /// The search returned nothing (empty knowledge base, or no chunks at all).
    Nothing,
}

/// The prompt block plus what the UI should show for it.
pub struct KbPrompt {
    /// Text prepended to the model prompt. Never empty when the KB button is on.
    pub context: String,
    pub outcome: KbOutcome,
    /// Short line for the chat UI when the answer is not grounded in a document.
    /// `None` when real matches were found.
    pub note: Option<String>,
}

impl KbPrompt {
    /// True only when the reply can be trusted to come from the user's documents.
    /// Memory extraction is skipped for ungrounded replies so an invented answer
    /// cannot become a stored fact.
    pub fn grounded(&self) -> bool {
        self.outcome == KbOutcome::Found
    }
}

pub const NOTE_NOTHING: &str = "Knowledge base searched — no matching documents";
pub const NOTE_WEAK: &str = "Knowledge base searched — no strong match; answer may not come from your documents";

/// Build the prompt block for an explicit (button-on) knowledge base search.
///
/// `threshold` is the similarity score a chunk must exceed to count as a match;
/// `weak_limit` is how many of the best chunks are still shown when none does.
pub fn build(results: &[KBSearchResult], threshold: f32, weak_limit: usize) -> KbPrompt {
    let relevant: Vec<&KBSearchResult> = results
        .iter()
        .filter(|r| r.similarity_score > threshold)
        .collect();

    let mut ctx = String::new();
    ctx.push_str("\n\n=== KNOWLEDGE BASE SEARCH (user pressed the KB button) ===\n");

    if !relevant.is_empty() {
        ctx.push_str("The user asked you to answer from their indexed documents. The passages below matched their question.\n");
        ctx.push_str("Answer from these passages. Quote the exact row or line you relied on.\n");
        ctx.push_str("If the passages do not actually contain the answer, say plainly that their knowledge base has no record of it. Do not fill the gap from general knowledge and do not guess.\n");
        ctx.push_str("Do not suggest a web search.\n\n");
        push_docs(&mut ctx, &relevant, "Document");
        ctx.push_str("=== END OF KNOWLEDGE BASE SEARCH ===\n\n");
        return KbPrompt { context: ctx, outcome: KbOutcome::Found, note: None };
    }

    if !results.is_empty() {
        let weak: Vec<&KBSearchResult> = results.iter().take(weak_limit).collect();
        ctx.push_str("The user asked you to answer from their indexed documents, but NOTHING in the knowledge base clearly matched the question.\n");
        ctx.push_str("The weak matches below are shown only in case one happens to contain the answer.\n");
        ctx.push_str("If one does, answer from it and quote the exact row or line. If none does, say plainly that their knowledge base has no record of this.\n");
        ctx.push_str("Do not answer from general knowledge as if it came from their documents. Do not guess.\n\n");
        push_docs(&mut ctx, &weak, "Weak match");
        ctx.push_str("=== END OF KNOWLEDGE BASE SEARCH ===\n\n");
        return KbPrompt { context: ctx, outcome: KbOutcome::WeakOnly, note: Some(NOTE_WEAK.to_string()) };
    }

    ctx.push_str("The user asked you to answer from their indexed documents, but the knowledge base search returned NOTHING.\n");
    ctx.push_str("Tell the user plainly that their knowledge base has no record of this. Do not invent an answer.\n");
    ctx.push_str("If you can add something from general knowledge, keep it short and label it clearly as not from their documents.\n");
    ctx.push_str("=== END OF KNOWLEDGE BASE SEARCH ===\n\n");
    KbPrompt { context: ctx, outcome: KbOutcome::Nothing, note: Some(NOTE_NOTHING.to_string()) }
}

fn push_docs(ctx: &mut String, docs: &[&KBSearchResult], label: &str) {
    for (idx, r) in docs.iter().enumerate() {
        ctx.push_str(&format!(
            "📄 {} {}: {} (similarity: {:.1}%)\n{}\n\n",
            label,
            idx + 1,
            r.file_name,
            r.similarity_score * 100.0,
            r.content
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(score: f32, content: &str) -> KBSearchResult {
        KBSearchResult {
            chunk_id: 1,
            document_id: 1,
            file_name: "netflixratings.csv".into(),
            file_path: "/kb/netflixratings.csv".into(),
            chunk_index: 0,
            content: content.into(),
            similarity_score: score,
        }
    }

    #[test]
    fn real_matches_are_grounded_and_still_forbid_guessing() {
        let p = build(&[chunk(0.42, "The Matrix,nf,4,2,")], 0.15, 5);
        assert_eq!(p.outcome, KbOutcome::Found);
        assert!(p.grounded());
        assert!(p.note.is_none());
        assert!(p.context.contains("Document 1: netflixratings.csv"));
        assert!(p.context.contains("The Matrix,nf,4,2,"));
        assert!(p.context.contains("has no record of it"));
        assert!(!p.context.contains("ONLY"));
    }

    #[test]
    fn weak_matches_are_labelled_and_not_grounded() {
        let results = vec![chunk(0.09, "Alien,nf,5,2,"), chunk(0.05, "Heat,nf,4,2,")];
        let p = build(&results, 0.15, 5);
        assert_eq!(p.outcome, KbOutcome::WeakOnly);
        assert!(!p.grounded());
        assert_eq!(p.note.as_deref(), Some(NOTE_WEAK));
        assert!(p.context.contains("NOTHING in the knowledge base clearly matched"));
        assert!(p.context.contains("Weak match 1"));
        assert!(p.context.contains("Weak match 2"));
        assert!(!p.context.contains("Document 1"));
    }

    #[test]
    fn weak_limit_caps_how_many_chunks_are_shown() {
        let results: Vec<_> = (0..8).map(|i| chunk(0.1, &format!("row {}", i))).collect();
        let p = build(&results, 0.15, 3);
        assert!(p.context.contains("Weak match 3"));
        assert!(!p.context.contains("Weak match 4"));
    }

    #[test]
    fn empty_search_tells_the_model_to_say_so() {
        let p = build(&[], 0.15, 5);
        assert_eq!(p.outcome, KbOutcome::Nothing);
        assert!(!p.grounded());
        assert_eq!(p.note.as_deref(), Some(NOTE_NOTHING));
        assert!(p.context.contains("returned NOTHING"));
        assert!(p.context.contains("Do not invent an answer"));
        assert!(!p.context.contains("📄"));
    }
}
