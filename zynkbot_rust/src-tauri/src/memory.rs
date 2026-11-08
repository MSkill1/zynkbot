use std::cmp::Ordering;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use chrono::{DateTime, Utc};

// ============================================================================
// STRUCTS
// ============================================================================

/// Memory relationship/link between two memories
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
#[allow(dead_code)]
pub struct MemoryLink {
    pub id: i32,
    pub source_memory_id: i32,
    pub target_memory_id: i32,
    pub relation_type: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
    pub created_by: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Memory {
    pub id: i32,
    pub title: Option<String>,
    pub content: String,
    pub source_type: Option<String>,
    pub session_id: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    #[sqlx(skip)]
    pub embedding: Option<Vec<f32>>,
    pub parent_scroll_id: Option<i32>,
    pub chunk_index: Option<i32>,
    pub user_id: Option<String>,
    pub namespace: String,
    pub is_syncable: bool,
    pub is_shareable: bool,
    pub event_type: Option<String>,
    pub event_date: Option<DateTime<Utc>>,
    pub link_count: i32,
    pub is_ephemeral: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub entities_detected: Option<serde_json::Value>,
    pub original_text: Option<String>,
    /// JSON array of lowercase topic words from the decision call (migration 0011).
    #[sqlx(default)]
    pub tags: Option<String>,
    /// NULL for a current memory; "completed" once the user answered a check-in question
    /// with "done" (labs/check_in, 2026-10-29); "superseded" once an "updates" link replaced it with a
    /// newer memory (migration 0022, 2026-10-23). A superseded memory leaves recall and
    /// stays visible in Memory Manager as history, with a way back.
    #[sqlx(default)]
    pub temporal_status: Option<String>,
    #[sqlx(default)]
    pub similarity: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct MemoryRelationship {
    pub memory_id: i32,
    pub related_memory_id: i32,
    pub relation_type: String,
    pub confidence: f32,
    pub direction: String,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
    pub created_by: String,
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/// Row type for fetching just id + embedding blob (used by vector/hybrid search)
#[derive(sqlx::FromRow)]
struct EmbeddingRow {
    id: i32,
    embedding: Option<Vec<u8>>,
}

/// Row type for fetching id + embedding + entities (used by hybrid search)
#[derive(sqlx::FromRow)]
struct HybridRow {
    id: i32,
    embedding: Option<Vec<u8>>,
    entities_detected: Option<String>,
    namespace: String,
    tags: Option<String>,
}

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn blob_to_f32_pub(blob: &[u8]) -> Vec<f32> {
    blob_to_f32(blob)
}

/// The text a memory's embedding is made from: the title and the fact together
/// ("Weight 305 this morning. User's weight was exactly 305 this morning.") since
/// 2026-10-25. The recall bench on a real database found the right memory in the
/// phone's top four for 21/29 questions this way against 19/29 for the fact alone
/// (labs/recall_bench). A title the content already starts with is not repeated.
pub const EMBEDDING_SCHEME: &str = "title_fact_v1";
pub fn embedding_text(title: Option<&str>, content: &str) -> String {
    match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) if !content.trim().to_lowercase().starts_with(&t.to_lowercase()) => format!("{}. {}", t.trim_end_matches('.'), content.trim()),
        _ => content.trim().to_string(),
    }
}

/// Once per database: when the embedding scheme changed, every memory with a title is
/// re-embedded in the background so recall and the new memories agree. The scheme in
/// use is kept in app_settings; nothing runs when it already matches.
pub async fn reembed_if_scheme_changed(pool: &SqlitePool) -> Result<usize, String> {
    let current: Option<String> = sqlx::query_scalar("SELECT value FROM app_settings WHERE key = 'memory_embedding_scheme'")
        .fetch_optional(pool).await.map_err(|e| e.to_string())?;
    if current.as_deref() == Some(EMBEDDING_SCHEME) { return Ok(0); }
    let rows: Vec<(i32, Option<String>, String)> = sqlx::query_as(
        "SELECT id, title, content FROM memories WHERE embedding IS NOT NULL AND title IS NOT NULL AND title != ''")
        .fetch_all(pool).await.map_err(|e| e.to_string())?;
    println!("[Memory] Embedding scheme {} -> {}: re-embedding {} memories", current.unwrap_or_else(|| "none".into()), EMBEDDING_SCHEME, rows.len());
    let mut done = 0usize;
    for chunk in rows.chunks(32) {
        let texts: Vec<String> = chunk.iter().map(|(_, t, c)| embedding_text(t.as_deref(), c)).collect();
        let embs = tokio::task::spawn_blocking(move || crate::llm::local_embeddings::generate_local_embeddings_batch(texts, Some(32)))
            .await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        for ((id, _, _), e) in chunk.iter().zip(embs) {
            let blob: Vec<u8> = e.iter().flat_map(|f| f.to_le_bytes()).collect();
            sqlx::query("UPDATE memories SET embedding = ? WHERE id = ?").bind(blob).bind(id).execute(pool).await.map_err(|e| e.to_string())?;
            done += 1;
        }
    }
    sqlx::query("INSERT INTO app_settings (key, value, updated_at) VALUES ('memory_embedding_scheme', ?, datetime('now')) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
        .bind(EMBEDDING_SCHEME).execute(pool).await.map_err(|e| e.to_string())?;
    println!("[Memory] Re-embedded {} memories", done);
    Ok(done)
}

#[cfg(test)]
mod embedding_text_tests {
    #[test]
    fn title_joins_the_fact_once() {
        use super::embedding_text as e;
        assert_eq!(e(Some("Weight 305 this morning"), "User's weight was exactly 305 this morning."), "Weight 305 this morning. User's weight was exactly 305 this morning.");
        assert_eq!(e(Some("Keto bread: 1g carb per slice"), "Keto bread: 1g carb per slice of the brand I use."), "Keto bread: 1g carb per slice of the brand I use.");
        assert_eq!(e(None, "I park on level 3"), "I park on level 3");
        assert_eq!(e(Some("  "), "x"), "x");
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Return the id of any existing memory for `user_id` whose embedding has cosine similarity
/// greater than `threshold` to `query_embedding`, or None. Replacement for the pgvector
/// `<=>` operator query — SQLite has no native vector ops, so we score in Rust.
#[allow(dead_code)]
pub async fn find_near_duplicate(
    pool: &SqlitePool,
    user_id: &str,
    query_embedding: &[f32],
    threshold: f32,
) -> Option<i32> {
    let rows: Vec<(i32, Option<Vec<u8>>)> = sqlx::query_as(
        "SELECT id, embedding FROM memories WHERE user_id = ? AND embedding IS NOT NULL",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    .ok()?;

    for (id, blob) in rows {
        if let Some(b) = blob {
            if cosine_similarity(query_embedding, &blob_to_f32(&b)) > threshold {
                return Some(id);
            }
        }
    }
    None
}

/// Return the user's preferred name from user_profile.json, falling back to full_name.
/// Returns None if onboarding hasn't occurred yet.
pub async fn get_user_display_name(_pool: &SqlitePool, _user_id: &str) -> Option<String> {
    let path = crate::db::get_user_profile_path();
    let profile: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));

    let preferred = profile.get("preferred_name").and_then(|v| v.as_str()).unwrap_or("");
    if !preferred.is_empty() {
        return Some(preferred.to_string());
    }

    let full = profile.get("full_name").and_then(|v| v.as_str()).unwrap_or("");
    if !full.is_empty() {
        Some(full.to_string())
    } else {
        None
    }
}

/// Fraction of query_entities that fuzzy-match any stored entity word.
fn entity_overlap_score(entities_json: &Option<String>, query_entities: &[String]) -> f64 {
    if query_entities.is_empty() {
        return 0.0;
    }
    let stored: Vec<String> = entities_json
        .as_ref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    e.get("word")
                        .and_then(|w| w.as_str())
                        .map(|s| s.to_lowercase())
                })
                .collect()
        })
        .unwrap_or_default();

    let matches = query_entities
        .iter()
        .filter(|qe| {
            stored
                .iter()
                .any(|se| se.contains(qe.as_str()) || qe.contains(se.as_str()))
        })
        .count();

    (matches as f64) / (query_entities.len() as f64)
}

// ============================================================================
// QUERY HINTS: what a question says about where to look (2026-11-04)
// ============================================================================

/// Beyond its meaning (the embedding) and the names in it (the name finder), a question
/// carries two signals the extractor already writes on every memory: the namespace its
/// words point at ("my family" → family, "my weight" → health) and its content words,
/// which the tags on a memory often repeat ("parents", "funeral"). Neither was read at
/// recall before 2026-11-04, so "What can you tell me about my family?" scored the family
/// facts under the 0.35 line: the word "family" is in neither "Albert's parents were
/// Hermann Einstein and Pauline Koch" nor its names. Built once per question.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueryHints {
    /// Canonical namespaces the question's words point at; empty when it names no topic.
    pub namespaces: Vec<String>,
    /// The question's content words, lowercased and stemmed, matched against tags.
    pub words: Vec<String>,
}

/// What a namespace hint and a tag match add to the hybrid score. A hint is a nudge, not
/// a filter: the extractor files some family facts under "personal", and what someone eats
/// is filed under health as often as kitchen. Set from the hint bench (labs/recall_bench,
/// 2026-11-04): tags at 0.25 lifted seven of 46 questions on a real database and hurt one
/// by a place; the namespace boost is what finds the Einstein demo's family facts for
/// "What can you tell me about my family?" (not returned before, 2nd with 0.05, 1st with
/// 0.15), but at 0.15 it also pushed a health memory from 21st to 47th for a question
/// that read as kitchen, past the cloud limit of 30, so 0.05.
pub const NAMESPACE_BOOST: f64 = 0.05;
pub const TAG_WEIGHT: f64 = 0.25;

/// Words that mark a question as being about a namespace. Plurals are listed, the words
/// are matched whole, and a question can point at more than one namespace.
const NAMESPACE_WORDS: &[(&str, &[&str])] = &[
    ("family", &["family", "families", "mother", "mom", "mum", "father", "dad", "parent", "parents",
        "son", "sons", "daughter", "daughters", "child", "children", "kid", "kids", "brother", "brothers",
        "sister", "sisters", "sibling", "siblings", "wife", "husband", "spouse", "grandmother", "grandma",
        "grandfather", "grandpa", "grandparents", "granddaughter", "grandson", "grandchildren", "aunt",
        "uncle", "cousin", "cousins", "nephew", "niece", "relative", "relatives", "in-laws", "married",
        "marriage", "wedding", "divorce", "divorced"]),
    ("health", &["health", "healthy", "doctor", "doctors", "medical", "medicine", "medication",
        "medications", "prescription", "weight", "diet", "keto", "blood", "glucose", "ketone", "ketones",
        "symptom", "symptoms", "sick", "illness", "disease", "hospital", "therapy", "therapist", "sleep",
        "exercise", "fitness", "pain", "injury", "allergy", "allergies", "dentist",
        // what someone eats is health as much as kitchen; the extractor files it both ways
        "eat", "ate", "eating", "food", "foods", "meal", "meals", "carbs", "calories", "nutrition"]),
    ("kitchen", &["recipe", "recipes", "cook", "cooking", "cooked", "bake", "baking", "baked", "meal",
        "meals", "breakfast", "lunch", "dinner", "supper", "eat", "ate", "eating", "food", "foods",
        "grocery", "groceries", "ingredient", "ingredients", "bread", "snack", "snacks", "kitchen",
        "oven", "fridge", "carbs", "calories"]),
    ("work", &["job", "project", "projects", "meeting", "meetings", "client", "clients", "business",
        "deadline", "deadlines", "company", "boss", "colleague", "colleagues", "coworker", "coworkers",
        "office", "customer", "customers", "workplace"]),
    ("career", &["career", "hired", "hiring", "salary", "promotion", "promoted", "employer", "resume",
        "interview", "interviews", "profession", "retire", "retired", "retirement"]),
    ("education", &["school", "schools", "university", "college", "degree", "studied", "studying",
        "class", "classes", "course", "courses", "teacher", "teachers", "student", "students", "lecture",
        "exam", "exams", "homework", "graduated", "graduation"]),
    ("travel", &["travel", "traveling", "travelling", "trip", "trips", "vacation", "journey", "flight",
        "flights", "hotel", "visited", "abroad", "airport", "passport", "itinerary"]),
    ("hobbies", &["hobby", "hobbies", "game", "games", "gaming", "guitar", "piano", "band", "sailing",
        "fishing", "hunting", "garden", "gardening", "hiking", "camping", "painting", "photography",
        "chess", "knitting", "woodworking", "collecting", "collection"]),
    ("technology", &["computer", "computers", "software", "hardware", "app", "apps", "laptop", "code",
        "coding", "programming", "server", "servers", "network", "wifi", "router", "printer", "password",
        "passwords", "linux", "windows", "android"]),
    ("politics", &["politics", "political", "election", "elections", "vote", "voting", "government",
        "president", "congress", "senate", "policy"]),
    ("science", &["science", "scientific", "physics", "chemistry", "biology", "experiment", "research",
        "quantum", "relativity"]),
    ("philosophy", &["philosophy", "philosophical", "belief", "beliefs", "religion", "religious", "faith",
        "ethics", "ethical", "spiritual"]),
    ("achievements", &["award", "awards", "prize", "prizes", "achievement", "achievements",
        "accomplishment", "accomplishments", "medal", "trophy"]),
    ("biography", &["born", "childhood", "hometown", "ancestry", "ancestors", "heritage", "funeral",
        "died", "upbringing"]),
];

/// Words a question is made of that say nothing about its topic.
const QUESTION_STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "from", "by",
    "about", "into", "out", "up", "down", "over", "again", "than", "then", "there", "here", "this",
    "that", "these", "those", "it", "its", "i", "me", "my", "mine", "myself", "you", "your", "yours",
    "we", "our", "us", "he", "she", "they", "them", "his", "her", "their", "what", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing", "have", "has", "had", "having", "can", "could", "would",
    "should", "will", "shall", "may", "might", "must", "tell", "told", "know", "knew", "remember",
    "recall", "think", "thought", "say", "said", "like", "want", "need", "please", "something",
    "anything", "everything", "nothing", "thing", "things", "stuff", "still", "also", "really",
    "just", "much", "many", "more", "most", "any", "some", "all", "both", "each", "few", "other",
    "such", "no", "not", "nor", "only", "own", "same", "so", "too", "very", "yet", "ever", "never",
    "always", "now", "get", "got", "give", "let", "make", "made", "use", "used", "going", "go",
    "went", "come", "came", "one", "ones", "way", "lot", "lots", "kind", "sort", "far",
];

/// A word reduced for matching: lowercased, punctuation stripped, a plural ending removed.
fn stem_word(word: &str) -> String {
    let w: String = word
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '\'')
        .collect::<String>()
        .to_lowercase();
    let w = w.trim_matches(|c| c == '-' || c == '\'').to_string();
    let w = w.strip_suffix("'s").map(str::to_string).unwrap_or(w);
    let n = w.chars().count();
    if n > 4 && w.ends_with("ies") {
        format!("{}y", &w[..w.len() - 3])
    } else if n > 4 && (w.ends_with("ses") || w.ends_with("xes") || w.ends_with("shes") || w.ends_with("ches")) {
        w[..w.len() - 2].to_string()
    } else if n > 3 && w.ends_with('s') && !w.ends_with("ss") {
        w[..w.len() - 1].to_string()
    } else {
        w
    }
}

/// Read a question for the namespaces it points at and the words worth matching.
pub fn query_hints(question: &str) -> QueryHints {
    let raw: Vec<String> = question
        .split(|c: char| c.is_whitespace() || matches!(c, ',' | '.' | '?' | '!' | ';' | ':' | '"' | '(' | ')' | '/'))
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();

    let mut namespaces: Vec<String> = Vec::new();
    for (ns, markers) in NAMESPACE_WORDS {
        let hit = raw.iter().any(|t| {
            let t = t.strip_suffix("'s").unwrap_or(t);
            markers.contains(&t)
        });
        if hit && !namespaces.iter().any(|n| n == ns) {
            namespaces.push((*ns).to_string());
        }
    }

    let mut words: Vec<String> = Vec::new();
    for t in &raw {
        if QUESTION_STOP_WORDS.contains(&t.as_str()) {
            continue;
        }
        let w = stem_word(t);
        if w.chars().count() >= 3 && !QUESTION_STOP_WORDS.contains(&w.as_str()) && !words.contains(&w) {
            words.push(w);
        }
    }
    QueryHints { namespaces, words }
}

/// Words that count as each other when a question's words are matched against tags
/// (2026-11-08). The extractor tags a memory with the words of the fact ("house",
/// "address"), and a question often asks with a different one: "Where do I live?" has no
/// word any tag on the home facts repeats, so on the recall bench it ranked 69th with the
/// tag match of 2026-11-04 in place. Each row is one group; a question word in a group
/// matches a tag word anywhere in the same group. Words are in their stemmed form (see
/// `stem_word`): singular, lowercase. Kept small and plain on purpose: a wide table would
/// match memories that share a theme but not the fact, and the bench is the judge.
const TAG_SYNONYMS: &[&[&str]] = &[
    &["live", "living", "lived", "home", "house", "housing", "address", "mailing", "residence", "reside", "residing", "apartment"],
    &["birthday", "birthdate", "born", "birth"],
    &["phone", "telephone", "cellphone", "smartphone", "mobile"],
    &["job", "employer", "employment", "occupation", "profession", "workplace"],
    &["car", "vehicle", "truck", "automobile"],
];

/// The words that count as `word` for tag matching: itself, plus its synonym group.
fn tag_equivalents(word: &str) -> Vec<&str> {
    let mut out = vec![word];
    if let Some(group) = TAG_SYNONYMS.iter().find(|g| g.contains(&word)) {
        out.extend(group.iter().copied().filter(|w| *w != word));
    }
    out
}

/// Fraction of the question's words that appear in a memory's tags (a tag is one or
/// more words; each is stemmed the same way as the question). A word also counts when a
/// tag carries one of its synonyms (`TAG_SYNONYMS`).
fn tag_overlap_score(tags_json: &Option<String>, words: &[String]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let tag_words: Vec<String> = tags_json
        .as_ref()
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .map(|tags| {
            tags.iter()
                .flat_map(|t| t.split(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/').map(stem_word).collect::<Vec<_>>())
                .filter(|w| !w.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if tag_words.is_empty() {
        return 0.0;
    }
    let matches = words
        .iter()
        .filter(|w| tag_equivalents(w.as_str()).iter().any(|eq| tag_words.iter().any(|t| t == eq)))
        .count();
    (matches as f64) / (words.len() as f64)
}

/// The hybrid score of one memory for one question. `semantic` is the cosine between the
/// embeddings, `entity` the share of the question's names found in the memory (60/40 blend
/// when any name matches, meaning alone otherwise, as since 2025), `namespace_hit` whether
/// the memory sits in a namespace the question points at, `tag_overlap` the share of the
/// question's words found in the memory's tags. The two hint terms are added on top, so a
/// memory just under the threshold on meaning gets in when the question names its topic.
pub fn hybrid_score(semantic: f64, entity: f64, namespace_hit: bool, tag_overlap: f64, namespace_boost: f64, tag_weight: f64) -> f64 {
    let base = if entity > 0.0 { (entity * 0.6) + (semantic * 0.4) } else { semantic };
    base + if namespace_hit { namespace_boost } else { 0.0 } + tag_weight * tag_overlap
}

/// For the bench: the share of a question's words found in these tags.
#[cfg(test)]
pub fn tag_overlap_score_pub(tags_json: &Option<String>, words: &[String]) -> f64 {
    tag_overlap_score(tags_json, words)
}

/// Fetch full Memory rows for a list of IDs (similarity field left as None, set by caller).
async fn fetch_memories_by_ids(
    pool: &SqlitePool,
    ids: &[i32],
) -> Result<Vec<Memory>, sqlx::Error> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at, tags,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type,
                CASE WHEN length(event_date) = 10 THEN event_date || 'T00:00:00+00:00' ELSE event_date END AS event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text, temporal_status
         FROM memories WHERE id IN ({})",
        placeholders
    );
    let mut query = sqlx::query_as::<_, Memory>(&sql);
    for id in ids {
        query = query.bind(id);
    }
    query.fetch_all(pool).await
}

// ============================================================================
// VECTOR SEARCH
// ============================================================================

/// Search for similar memories using vector similarity (Rust-side cosine computation).
/// Returns only memories with similarity >= 0.50.
#[allow(dead_code)]
pub async fn vector_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
    limit: i32,
) -> Result<Vec<Memory>, sqlx::Error> {
    const MIN_SIMILARITY: f64 = 0.50;

    println!(
        "[VectorSearch] MIN_SIMILARITY={} (Rust-side cosine)",
        MIN_SIMILARITY
    );

    // Step 1: fetch (id, embedding_blob) for all candidates matching filters
    let mut sql =
        String::from("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL");
    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }

    let mut q = sqlx::query_as::<_, EmbeddingRow>(&sql);
    if let Some(uid) = user_id {
        q = q.bind(uid);
    }
    if let Some(sid) = session_id {
        q = q.bind(sid);
    }
    if let Some(ns) = namespace {
        q = q.bind(ns);
    }
    let rows = q.fetch_all(pool).await?;

    // Step 2: score in Rust
    let mut scored: Vec<(i32, f64)> = rows
        .iter()
        .filter_map(|r| {
            let blob = r.embedding.as_ref()?;
            let sim = cosine_similarity(&query_embedding, &blob_to_f32(blob)) as f64;
            if sim >= MIN_SIMILARITY { Some((r.id, sim)) } else { None }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(limit as usize);

    // Step 3: fetch full rows for top IDs
    let id_scores: HashMap<i32, f64> = scored.iter().cloned().collect();
    let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();

    let mut memories = fetch_memories_by_ids(pool, &ids).await?;
    for m in &mut memories {
        m.similarity = id_scores.get(&m.id).copied();
    }
    memories.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));

    println!("[VectorSearch] Returned {} memories:", memories.len());
    for (i, m) in memories.iter().enumerate() {
        if let Some(sim) = m.similarity {
            println!(
                "  [{}] Memory {} - Similarity: {:.3} ({}%)",
                i + 1,
                m.id,
                sim,
                (sim * 100.0) as i32
            );
        }
    }

    Ok(memories)
}

// ============================================================================
// HYBRID SEARCH
// ============================================================================

/// Hybrid search combining entity matching + semantic similarity (Rust-side).
///
/// Scoring: entity_overlap * 0.6 + semantic_similarity * 0.4 when entities match,
/// pure semantic similarity otherwise.  Threshold: 0.35.
///
/// Vector search is a linear scan: all candidate rows are fetched from SQLite and
/// cosine similarity is computed in Rust. This is correct and fast at typical personal
/// memory counts. At very large scale (hundreds of thousands of memories), sqlite-vec
/// would replace this with an indexed ANN search — see ROADMAP.md.
#[allow(dead_code)]
pub async fn hybrid_search(
    pool: &SqlitePool,
    query_embedding: Vec<f32>,
    query_entities: Vec<String>,
    hints: &QueryHints,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
    limit: i32,
) -> Result<Vec<Memory>, sqlx::Error> {
    const MIN_SIMILARITY: f64 = 0.35;

    let query_entities_lower: Vec<String> =
        query_entities.iter().map(|e| e.to_lowercase()).collect();

    // Step 1: fetch candidates that have an embedding or entity data
    let mut sql = String::from(
        "SELECT id, embedding, entities_detected, namespace, tags FROM memories \
         WHERE (embedding IS NOT NULL OR (entities_detected IS NOT NULL AND entities_detected != '[]')) \
         AND (temporal_status IS NULL OR temporal_status NOT IN ('superseded', 'completed')) \
         AND (expires_at IS NULL OR substr(expires_at, 1, 10) >= date('now'))",
    );
    // Exclude system memories EXCEPT when explicitly searching the system user.
    // Zynkbot system-doc queries pass user_id = "system"; without this guard the
    // blanket `user_id != 'system'` contradicted the caller's `user_id = 'system'`
    // filter and returned 0 rows for every Zynkbot question.
    if user_id != Some("system") {
        sql.push_str(" AND user_id != 'system'");
    }
    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }

    let mut q = sqlx::query_as::<_, HybridRow>(&sql);
    if let Some(uid) = user_id {
        q = q.bind(uid);
    }
    if let Some(sid) = session_id {
        q = q.bind(sid);
    }
    if let Some(ns) = namespace {
        q = q.bind(ns);
    }
    let rows = q.fetch_all(pool).await?;

    // Step 2: score in Rust
    let mut scored: Vec<(i32, f64)> = rows
        .iter()
        .filter_map(|r| {
            let semantic = r
                .embedding
                .as_ref()
                .map(|b| cosine_similarity(&query_embedding, &blob_to_f32(b)) as f64)
                .unwrap_or(0.0);

            let entity =
                entity_overlap_score(&r.entities_detected, &query_entities_lower);
            let namespace_hit = hints.namespaces.iter().any(|n| *n == r.namespace);
            let tags = tag_overlap_score(&r.tags, &hints.words);
            let score = hybrid_score(semantic, entity, namespace_hit, tags, NAMESPACE_BOOST, TAG_WEIGHT);

            if score >= MIN_SIMILARITY || entity > 0.0 {
                Some((r.id, score))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(limit as usize);

    // Step 3: fetch full rows for top IDs
    let id_scores: HashMap<i32, f64> = scored.iter().cloned().collect();
    let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();

    let mut memories = fetch_memories_by_ids(pool, &ids).await?;
    for m in &mut memories {
        m.similarity = id_scores.get(&m.id).copied();
    }
    memories.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(Ordering::Equal));

    println!("[HybridSearch] Returned {} memories:", memories.len());
    for (i, m) in memories.iter().enumerate() {
        if let Some(sim) = m.similarity {
            println!(
                "  [{}] Memory {} - Score: {:.3} ({}%) - {}",
                i + 1,
                m.id,
                sim,
                (sim * 100.0) as i32,
                m.title.as_ref().unwrap_or(&"(no title)".to_string())
            );
        }
    }

    Ok(memories)
}

// ============================================================================
// CRUD — list / get / delete
// ============================================================================

/// List all memories (no vector search, just filtering).
#[allow(dead_code)]
pub async fn list_memories(
    pool: &SqlitePool,
    user_id: Option<&str>,
    session_id: Option<&str>,
    namespace: Option<&str>,
) -> Result<Vec<Memory>, sqlx::Error> {
    let mut sql = String::from(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at, tags,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type,
                CASE WHEN length(event_date) = 10 THEN event_date || 'T00:00:00+00:00' ELSE event_date END AS event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text, temporal_status
         FROM memories WHERE user_id != 'system'",
    );

    if user_id.is_some() {
        sql.push_str(" AND user_id = ?");
    }
    if session_id.is_some() {
        sql.push_str(" AND session_id = ?");
    }
    if namespace.is_some() {
        sql.push_str(" AND namespace = ?");
    }
    sql.push_str(" ORDER BY datetime(created_at) DESC");

    let mut query = sqlx::query_as::<_, Memory>(&sql);
    if let Some(uid) = user_id {
        query = query.bind(uid);
    }
    if let Some(sid) = session_id {
        query = query.bind(sid);
    }
    if let Some(ns) = namespace {
        query = query.bind(ns);
    }

    query.fetch_all(pool).await
}

/// Get a single memory by ID.
pub async fn get_memory(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Option<Memory>, sqlx::Error> {
    sqlx::query_as::<_, Memory>(
        "SELECT id, title, content, source_type, session_id, created_at, updated_at, tags,
                parent_scroll_id, chunk_index, user_id, namespace,
                is_syncable, is_shareable, event_type,
                CASE WHEN length(event_date) = 10 THEN event_date || 'T00:00:00+00:00' ELSE event_date END AS event_date,
                link_count, is_ephemeral, expires_at, entities_detected, original_text, temporal_status
         FROM memories WHERE id = ?",
    )
    .bind(memory_id)
    .fetch_optional(pool)
    .await
}

/// Delete a memory by ID.
#[allow(dead_code)]
pub async fn delete_memory(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM memories WHERE id = ?")
        .bind(memory_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

// ============================================================================
// INSERT / UPDATE
// ============================================================================

/// Insert a new memory into the database.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub async fn insert_memory(
    pool: &SqlitePool,
    title: Option<&str>,
    content: &str,
    source_type: Option<&str>,
    session_id: Option<&str>,
    embedding: Option<Vec<f32>>,
    parent_scroll_id: Option<i32>,
    chunk_index: Option<i32>,
    user_id: Option<&str>,
    namespace: &str,
    is_syncable: bool,
    is_shareable: bool,
    entities_detected: Option<serde_json::Value>,
    event_type: Option<&str>,
    event_date: Option<DateTime<Utc>>,
    original_text: Option<&str>,
) -> Result<i32, sqlx::Error> {
    let embedding_vec: Option<Vec<u8>> = embedding
        .map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());

    let result = sqlx::query_scalar::<_, i32>(
        "INSERT INTO memories (
            title, content, source_type, session_id, embedding,
            parent_scroll_id, chunk_index, user_id, namespace,
            is_syncable, is_shareable, entities_detected, event_type, event_date,
            original_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id",
    )
    .bind(title)
    .bind(content)
    .bind(source_type)
    .bind(session_id)
    .bind(embedding_vec)
    .bind(parent_scroll_id)
    .bind(chunk_index)
    .bind(user_id)
    .bind(namespace)
    .bind(is_syncable)
    .bind(is_shareable)
    .bind(entities_detected)
    .bind(event_type)
    .bind(event_date)
    .bind(original_text)
    .fetch_one(pool)
    .await?;

    Ok(result)
}

/// Update an existing memory's title and/or content.
#[allow(dead_code)]
pub async fn update_memory(
    pool: &SqlitePool,
    memory_id: i32,
    title: Option<&str>,
    content: Option<&str>,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE memories
         SET title = COALESCE(?, title),
             content = COALESCE(?, content)
         WHERE id = ?",
    )
    .bind(title)
    .bind(content)
    .bind(memory_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

// ============================================================================
// MEMORY LINKS — Semantic Relationship Graph
// ============================================================================

/// Get all links for a specific memory (bidirectional, via memory_relationships view).
pub async fn get_memory_relationships(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Vec<MemoryRelationship>, sqlx::Error> {
    sqlx::query_as::<_, MemoryRelationship>(
        "SELECT memory_id, related_memory_id, relation_type,
                confidence, direction, created_at, notes, created_by
         FROM memory_relationships
         WHERE memory_id = ?
         ORDER BY datetime(created_at) DESC",
    )
    .bind(memory_id)
    .fetch_all(pool)
    .await
}

/// Get all direct links for a memory (raw memory_links table, both directions).
#[allow(dead_code)]
pub async fn get_memory_links(
    pool: &SqlitePool,
    memory_id: i32,
) -> Result<Vec<MemoryLink>, sqlx::Error> {
    sqlx::query_as::<_, MemoryLink>(
        "SELECT id, source_memory_id, target_memory_id, relation_type,
                confidence, created_at, notes, created_by
         FROM memory_links
         WHERE source_memory_id = ? OR target_memory_id = ?
         ORDER BY datetime(created_at) DESC",
    )
    .bind(memory_id)
    .bind(memory_id)
    .fetch_all(pool)
    .await
}

/// The relationship types the app stores (migration 0022, 2026-10-23). Each one makes
/// the app do something different: "contradicts" asks the user, "updates" replaces an
/// older memory and keeps it as history, "related" pulls the linked memory into recall,
/// "resolves" is the user's own explanation of a settled conflict.
pub const RELATION_TYPES: [&str; 4] = ["contradicts", "updates", "related", "resolves"];

/// The stored name for a relationship type, including the names used before 0022 (an
/// older device may still send them). `None` for a type that is no longer stored.
pub fn normalize_relation_type(t: &str) -> Option<&'static str> {
    match t {
        "contradicts" => Some("contradicts"),
        "updates" => Some("updates"),
        "related" | "supports" | "elaborates" | "caused_by" => Some("related"),
        "resolves" => Some("resolves"),
        _ => None,
    }
}

/// Mark a memory as replaced by a newer one (`Some("superseded")`) or current again
/// (`None`). Bumps updated_at so the change syncs.
pub async fn set_temporal_status(pool: &SqlitePool, memory_id: i32, status: Option<&str>) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE memories SET temporal_status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
        .bind(status)
        .bind(memory_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// Create a new memory link (semantic relationship).
#[allow(dead_code)]
pub async fn create_memory_link(
    pool: &SqlitePool,
    source_memory_id: i32,
    target_memory_id: i32,
    relation_type: &str,
    confidence: f32,
    notes: Option<&str>,
    created_by: &str,
) -> Result<i32, sqlx::Error> {
    if !RELATION_TYPES.contains(&relation_type) {
        return Err(sqlx::Error::Protocol(format!(
            "Invalid relation_type: {}. Must be one of: {:?}",
            relation_type, RELATION_TYPES
        )));
    }
    if !(0.0..=1.0).contains(&confidence) {
        return Err(sqlx::Error::Protocol(format!(
            "Invalid confidence: {}. Must be between 0.0 and 1.0",
            confidence
        )));
    }

    let link_id = sqlx::query_scalar::<_, i32>(
        "INSERT INTO memory_links
         (source_memory_id, target_memory_id, relation_type, confidence, notes, created_by)
         VALUES (?, ?, ?, ?, ?, ?)
         RETURNING id",
    )
    .bind(source_memory_id)
    .bind(target_memory_id)
    .bind(relation_type)
    .bind(confidence)
    .bind(notes)
    .bind(created_by)
    .fetch_one(pool)
    .await?;

    sqlx::query(
        "UPDATE memories SET updated_at = datetime('now') WHERE id = ?",
    )
    .bind(source_memory_id)
    .execute(pool)
    .await?;

    Ok(link_id)
}

/// Delete a memory link by ID.
#[allow(dead_code)]
pub async fn delete_memory_link(
    pool: &SqlitePool,
    link_id: i32,
) -> Result<u64, sqlx::Error> {
    let result = sqlx::query("DELETE FROM memory_links WHERE id = ?")
        .bind(link_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Get full memory graph for a memory (includes content of related memories).
/// Currently supports depth=1 (direct connections only).
#[allow(dead_code)]
pub async fn get_memory_graph(
    pool: &SqlitePool,
    memory_id: i32,
    _depth: i32,
) -> Result<Vec<(MemoryRelationship, Memory)>, sqlx::Error> {
    let relationships = get_memory_relationships(pool, memory_id).await?;
    let mut graph_data = Vec::new();
    for rel in relationships {
        if let Some(related_memory) = get_memory(pool, rel.related_memory_id).await? {
            graph_data.push((rel, related_memory));
        }
    }
    Ok(graph_data)
}


/// Fields the decision call now supplies (2026-09-07): tags, tone, and a
/// possibly better event date / namespace than the NLP guess. Applied after
/// insert so the many `insert_memory` callers keep their signature.
pub async fn set_memory_extras(
    pool: &SqlitePool,
    memory_id: i32,
    tags: &[String],
    sentiment_label: &str,
    sentiment_score: f64,
) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE memories SET tags = ?, sentiment_label = ?, sentiment_score = ? WHERE id = ?")
        .bind(serde_json::to_string(tags).unwrap_or_else(|_| "[]".to_string()))
        .bind(sentiment_label)
        .bind(sentiment_score)
        .bind(memory_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// One row per named thing in a memory (memory_entities, migration 0011).
pub async fn insert_memory_entities(
    pool: &SqlitePool,
    memory_id: i32,
    entities: &[crate::memory_extras::EntityOut],
) -> Result<usize, sqlx::Error> {
    let mut n = 0;
    for e in entities {
        let canonical = e.name.trim().to_lowercase();
        let r = sqlx::query(
            "INSERT OR IGNORE INTO memory_entities (memory_id, name, canonical, kind) VALUES (?, ?, ?, ?)",
        )
        .bind(memory_id)
        .bind(e.name.trim())
        .bind(&canonical)
        .bind(&e.kind)
        .execute(pool)
        .await?;
        n += r.rows_affected() as usize;
    }
    Ok(n)
}

/// Write a raised card down so it survives the window being gone (migration 0024).
pub async fn insert_memory_card(pool: &SqlitePool, user_id: &str, session_id: &str, payload: &serde_json::Value) -> Result<i64, sqlx::Error> {
    let stored = payload.get("stored_memory_id").and_then(|v| v.as_i64());
    let conflicting = payload.pointer("/memoryA/id").and_then(|v| v.as_i64());
    let r = sqlx::query(
        "INSERT INTO memory_cards (user_id, session_id, stored_memory_id, conflicting_memory_id, payload_json) VALUES (?, ?, ?, ?, ?)")
        .bind(user_id).bind(session_id).bind(stored).bind(conflicting).bind(payload.to_string())
        .execute(pool).await?;
    Ok(r.last_insert_rowid())
}

#[cfg(test)]
mod query_hint_tests {
    use super::*;

    #[test]
    fn a_family_question_points_at_family_and_keeps_its_topic_words() {
        let h = query_hints("What can you tell me about my family?");
        assert_eq!(h.namespaces, vec!["family"]);
        assert_eq!(h.words, vec!["family"]);
        let h = query_hints("Tell me about my parents.");
        assert_eq!(h.namespaces, vec!["family"]);
        assert_eq!(h.words, vec!["parent"]);
    }

    #[test]
    fn a_question_can_point_at_more_than_one_namespace() {
        let h = query_hints("What is my family's religious background?");
        assert!(h.namespaces.contains(&"family".to_string()));
        assert!(h.namespaces.contains(&"philosophy".to_string()));
        assert!(h.words.contains(&"religiou".to_string()) || h.words.contains(&"religious".to_string()));
    }

    #[test]
    fn a_question_with_no_topic_has_no_namespace() {
        let h = query_hints("What did I say yesterday?");
        assert!(h.namespaces.is_empty());
        let h = query_hints("Where do I live?");
        assert!(h.namespaces.is_empty());
        assert_eq!(h.words, vec!["live"]);
    }

    #[test]
    fn tags_match_on_stemmed_words() {
        let tags = Some(r#"["parents","departure"]"#.to_string());
        assert!((tag_overlap_score(&tags, &["parent".to_string()]) - 1.0).abs() < 1e-9);
        let tags = Some(r#"["family background","religion"]"#.to_string());
        let words = query_hints("What is my family's religious background?").words;
        assert!(tag_overlap_score(&tags, &words) > 0.0);
        assert_eq!(tag_overlap_score(&None, &words), 0.0);
        assert_eq!(tag_overlap_score(&Some("[]".into()), &words), 0.0);
        assert_eq!(tag_overlap_score(&tags, &[]), 0.0);
    }

    #[test]
    fn hints_add_to_the_score_and_never_take_from_it() {
        let plain = hybrid_score(0.30, 0.0, false, 0.0, NAMESPACE_BOOST, TAG_WEIGHT);
        assert!((plain - 0.30).abs() < 1e-9);
        let boosted = hybrid_score(0.30, 0.0, true, 0.5, NAMESPACE_BOOST, TAG_WEIGHT);
        assert!(boosted > plain);
        assert!((boosted - (0.30 + NAMESPACE_BOOST + 0.5 * TAG_WEIGHT)).abs() < 1e-9);
        // the 60/40 blend for a name match is unchanged
        let named = hybrid_score(0.30, 1.0, false, 0.0, NAMESPACE_BOOST, TAG_WEIGHT);
        assert!((named - 0.72).abs() < 1e-9);
    }

    #[test]
    fn a_question_word_matches_a_tag_that_says_the_same_thing_another_way() {
        // "Where do I live?" against a memory tagged with the house: the one case the bench
        // still missed after 2026-11-04.
        let words = query_hints("Where do I live?").words;
        assert_eq!(words, vec!["live"]);
        let house = Some(r#"["house","bathroom"]"#.to_string());
        assert!((tag_overlap_score(&house, &words) - 1.0).abs() < 1e-9);
        let address = Some(r#"["address","gift","plush"]"#.to_string());
        assert!((tag_overlap_score(&address, &words) - 1.0).abs() < 1e-9);
        // the other way round, and in the plural
        let words = query_hints("What is my mailing address?").words;
        let homes = Some(r#"["homes"]"#.to_string());
        assert!((tag_overlap_score(&homes, &words) - 1.0).abs() < 1e-9);
        // a tag that shares no group stays a miss
        let bank = Some(r#"["banking","chase"]"#.to_string());
        assert_eq!(tag_overlap_score(&bank, &query_hints("Where do I live?").words), 0.0);
        // a word outside every group matches only itself
        assert_eq!(tag_equivalents("bathroom"), vec!["bathroom"]);
        assert!(tag_equivalents("phone").contains(&"mobile"));
    }

    #[test]
    fn the_synonym_table_is_in_stemmed_form_and_each_word_sits_in_one_group() {
        let mut seen: Vec<&str> = Vec::new();
        for group in TAG_SYNONYMS {
            for w in group.iter() {
                assert_eq!(stem_word(w), *w, "{w} is not in its stemmed form");
                assert!(!seen.contains(w), "{w} appears in two groups");
                seen.push(w);
            }
        }
    }

    #[test]
    fn stemming_folds_plurals_and_possessives() {
        assert_eq!(stem_word("parents"), "parent");
        assert_eq!(stem_word("families"), "family");
        assert_eq!(stem_word("family's"), "family");
        assert_eq!(stem_word("classes"), "class");
        assert_eq!(stem_word("ALS"), "als");
        assert_eq!(stem_word("glass"), "glass");
    }
}
