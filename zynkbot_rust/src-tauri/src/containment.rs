/// Containment Layer - Client-side safety enforcement for Zynkbot
///
/// This is a DEMONSTRATION FRAMEWORK showing that content filtering can be done
/// client-side without involving the LLM. It's designed to be extensible so users
/// can create custom modes for their specific needs.
///
/// Built-in Modes (examples of what's possible):
///
/// - **witness**: No enforcement - all content passes through
///   Use case: Maximum freedom, research, development
///
/// - **sovereign**: Warn but don't block dangerous content
///   Use case: Users who want awareness but not censorship
///
/// - **guardian**: Block harmful instructions (demonstration of corporate-style filtering)
///   Use case: Standard "corporate LLM" interaction without involving the LLM itself
///   Note: This is voluntary - users choose this mode and can switch to Sovereign/Witness anytime
///   Implementation: Pattern matching + TinyBERT toxicity classifier (works offline)
///
/// - **child**: Strict safety for minors (uses OpenAI Moderation API)
///   Use case: Parents who want robust filtering for children
///   Note: Requires internet connection - only mode that calls external API
///
/// - **elder**: Gentle redirection for sensitive topics [STUB — not yet implemented]
///   Use case: Elderly users who may find certain topics distressing
///   Status: Enum variant and match arms exist; no full implementation. Not exposed in UI.
///   Reserved placeholder for future development — do not remove.
///
/// - **hipaa**: Healthcare compliance with PHI protection
///   Use case: Medical contexts where patient data must be protected
///   Implementation: identifiers are removed from everything a model is sent
///   (`redact_phi`, called from the chat and Ensemble paths); requests for a diagnosis,
///   a dose or a treatment decision are refused (`enforce_hipaa`)
///
/// Users can extend this framework to create custom modes for their specific needs
/// (e.g., educational mode, professional mode, etc.)
use serde::{Deserialize, Serialize};
use crate::safety_classifier;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use chrono::Utc;

pub const CHILD_MODE_SYSTEM_PROMPT: &str = "You are Zynkbot in Child Mode - a helpful AI assistant designed specifically for children under 13. \
Your responses must be:\n\
1. Age-appropriate and educational\n\
2. Free from any adult content, violence, or inappropriate topics\n\
3. Protective - refuse requests that could lead to unsafe situations (finding adult content, \
   meeting strangers online, bypassing parental controls, accessing restricted sites, etc.)\n\
4. Encouraging - guide children toward safe, educational, and positive activities\n\
5. Clear - use simple language appropriate for young children\n\n\
If a request is inappropriate, explain why it's not safe in child-friendly language and \
suggest a safer alternative. Never provide information that could help a child access \
inappropriate content or unsafe situations.";

/// One kind of identifier removed in HIPAA mode: what is left in its place, how it is
/// named when the user is told, and the pattern that finds it.
struct PhiKind {
    tag: &'static str,
    one: &'static str,
    many: &'static str,
    re: regex::Regex,
}

/// The identifiers HIPAA mode removes, in the order they are applied. Order matters: a
/// labelled record number and a card number are taken before the shorter number shapes
/// that would otherwise match inside them.
///
/// Every pattern needs a shape or a label, never a bare run of digits: until 2026-11-10
/// any 5-, 9- or 10-digit number counted as a ZIP code, an SSN or a phone number, and
/// "10000 steps" blocked the message. What this cannot find is listed in
/// docs/architecture_and_development/HIPAA_SECURITY_INTEGRATION_GUIDE.md, "Known limits":
/// a name with no "patient" in front of it, an address written in lower case, a date
/// that is not marked as a date of birth, and anything inside a picture.
static PHI_KINDS: once_cell::sync::Lazy<Vec<PhiKind>> = once_cell::sync::Lazy::new(|| {
    const MONTH: &str = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?";
    const STATES: &str = "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY";
    let kind = |tag, one, many, pattern: &str| PhiKind {
        tag, one, many,
        re: regex::Regex::new(pattern).expect("PHI pattern compiles"),
    };
    vec![
        // "MRN 4829201", "member ID: ABC-12345", "policy number is 99812".
        kind("[record number removed]", "record or member number", "record or member numbers",
            r"(?i)\b(?:MRN|medical record(?: number| no\.?)?|patient id|member id|subscriber id|insurance id|policy (?:number|no\.?)|account (?:number|no\.?)|health plan (?:number|id))\s*(?:is|:|#|=)?\s*#?\s*[A-Z0-9][A-Z0-9-]*\d[A-Z0-9-]*"),
        kind("[card number removed]", "card number", "card numbers",
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        // With its dashes, or nine digits after the words.
        kind("[SSN removed]", "Social Security number", "Social Security numbers",
            r"(?i)\b(?:SSN|social security(?: number| no\.?)?)\s*(?:is|:|#)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{4}\b|\b\d{3}-\d{2}-\d{4}\b"),
        // "(555) 123-4567", "555-123-4567", "555.123.4567", "+1 555 123 4567", or ten
        // digits after the word.
        kind("[phone number removed]", "phone number", "phone numbers",
            r"(?i)(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b|\b(?:phone|tel|telephone|cell|mobile|fax)(?: number| no\.?)?\s*(?:is|:|#)?\s*\d{10}\b"),
        kind("[email address removed]", "email address", "email addresses",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        // Only a date marked as a birth date. Other dates stay: the memory system's own
        // date notes and today's date are in every prompt.
        kind("[date of birth removed]", "date of birth", "dates of birth",
            &format!(r"(?i)\b(?:DOB|D\.O\.B\.?|date of birth|birth ?date|born(?: on)?)\s*(?:is|:|=)?\s*(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}}|{m}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{m},?\s+\d{{4}})", m = MONTH)),
        // "123 Main St", "45 North Elm Street": a number, one to four capitalised words,
        // a capitalised street word. Case matters here, so "5 miles down the road" stays.
        kind("[street address removed]", "street address", "street addresses",
            r"\b\d{1,6}\s+(?:[A-Z][A-Za-z'.]*\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd|Way|Place|Terrace|Circle|Highway|Hwy)\b\.?"),
        // After a state's two letters, after the word, or in the nine-digit form.
        kind("[ZIP code removed]", "ZIP code", "ZIP codes",
            &format!(r"\b(?:{s})[ ,]+\d{{5}}(?:-\d{{4}})?\b|(?i:\bzip(?: code)?\s*(?:is|:)?\s*)\d{{5}}(?:-\d{{4}})?\b|\b\d{{5}}-\d{{4}}\b", s = STATES)),
        kind("[IP address removed]", "IP address", "IP addresses",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        // "patient John Smith", "patient: Mrs. Ana Lopez", "the client's name is ...".
        // Two capitalised words at least: one alone is too often not a name.
        kind("[patient name removed]", "patient name", "patient names",
            r"\b(?i:patient|client)(?:'s)?(?:\s+(?i:named|name\s+is|name:|name|is|was)|\s*:)?\s+(?:(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z'-]+){1,2}\b"),
    ]
});

/// What `redact_phi` took out: how each kind is named, and how many of it.
pub type PhiFound = Vec<(&'static str, &'static str, usize)>;

/// HIPAA mode: take the identifiers out of `text`, leaving a short tag where each was.
/// Returns the cleaned text and what was removed. The text removed is returned to no
/// one and logged nowhere.
pub fn redact_phi(text: &str) -> (String, PhiFound) {
    let mut out = text.to_string();
    let mut found: PhiFound = Vec::new();
    for k in PHI_KINDS.iter() {
        let n = k.re.find_iter(&out).count();
        if n > 0 {
            out = k.re.replace_all(&out, k.tag).into_owned();
            found.push((k.one, k.many, n));
        }
    }
    (out, found)
}

/// Add one result of `redact_phi` to a running total.
pub fn merge_phi(total: &mut PhiFound, more: PhiFound) {
    for (one, many, n) in more {
        match total.iter_mut().find(|(o, _, _)| *o == one) {
            Some(entry) => entry.2 += n,
            None => total.push((one, many, n)),
        }
    }
}

/// "1 phone number and 2 email addresses", for the line the user is shown.
pub fn describe_phi(found: &PhiFound) -> String {
    let parts: Vec<String> = found.iter()
        .map(|(one, many, n)| format!("{} {}", n, if *n == 1 { one } else { many }))
        .collect();
    match parts.len() {
        0 => String::new(),
        1 => parts[0].clone(),
        n => format!("{} and {}", parts[..n - 1].join(", "), parts[n - 1]),
    }
}

/// The audit line for a removal: where it happened and how many of each kind. Counts
/// only, never the text.
pub fn audit_phi_removed(place: &str, found: &PhiFound) {
    if found.is_empty() { return; }
    let counts: serde_json::Map<String, serde_json::Value> = found.iter()
        .map(|(one, _, n)| (one.to_string(), json!(n)))
        .collect();
    log_hipaa_audit("phi_removed", json!({ "where": place, "removed": counts }));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentLayer {
    mode: ContainmentMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ContainmentMode {
    Witness,
    Sovereign,
    Guardian,
    Child,
    Elder,  // STUB: reserved for future implementation — see module doc
    Hipaa,
    Default,
}

impl ContainmentMode {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "witness" => Ok(Self::Witness),
            "sovereign" => Ok(Self::Sovereign),
            "guardian" => Ok(Self::Guardian),
            "child" => Ok(Self::Child),
            "elder" => Ok(Self::Elder),
            "hipaa" => Ok(Self::Hipaa),
            "default" => Ok(Self::Default),
            _ => Err(format!("Unknown containment mode: '{}'", s)),
        }
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        match self {
            Self::Witness => "witness".to_string(),
            Self::Sovereign => "sovereign".to_string(),
            Self::Guardian => "guardian".to_string(),
            Self::Child => "child".to_string(),
            Self::Elder => "elder".to_string(),
            Self::Hipaa => "hipaa".to_string(),
            Self::Default => "default".to_string(),
        }
    }
}

/// OpenAI Moderation API response structure
#[derive(Debug, Deserialize)]
struct OpenAIModerationResponse {
    results: Vec<OpenAIModerationResult>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModerationResult {
    flagged: bool,
    categories: OpenAIModerationCategories,
    #[allow(dead_code)]
    category_scores: OpenAIModerationScores,
}

#[derive(Debug, Deserialize)]
struct OpenAIModerationCategories {
    sexual: bool,
    hate: bool,
    harassment: bool,
    #[serde(rename = "self-harm")]
    self_harm: bool,
    #[serde(rename = "sexual/minors")]
    sexual_minors: bool,
    #[serde(rename = "hate/threatening")]
    hate_threatening: bool,
    #[serde(rename = "violence/graphic")]
    violence_graphic: bool,
    #[serde(rename = "self-harm/intent")]
    self_harm_intent: bool,
    #[serde(rename = "self-harm/instructions")]
    self_harm_instructions: bool,
    #[serde(rename = "harassment/threatening")]
    harassment_threatening: bool,
    violence: bool,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct OpenAIModerationScores {
    sexual: f32,
    hate: f32,
    harassment: f32,
    #[serde(rename = "self-harm")]
    self_harm: f32,
    #[serde(rename = "sexual/minors")]
    sexual_minors: f32,
    #[serde(rename = "hate/threatening")]
    hate_threatening: f32,
    #[serde(rename = "violence/graphic")]
    violence_graphic: f32,
    #[serde(rename = "self-harm/intent")]
    self_harm_intent: f32,
    #[serde(rename = "self-harm/instructions")]
    self_harm_instructions: f32,
    #[serde(rename = "harassment/threatening")]
    harassment_threatening: f32,
    violence: f32,
}

impl ContainmentLayer {
    pub fn new(initial_mode: &str) -> Result<Self, String> {
        Ok(Self {
            mode: ContainmentMode::from_str(initial_mode)?,
        })
    }

    #[allow(dead_code)]
    pub fn set_mode(&mut self, mode: &str) -> Result<(), String> {
        self.mode = ContainmentMode::from_str(mode)?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_mode(&self) -> String {
        self.mode.to_string()
    }

    /// Enforce containment rules based on current mode
    ///
    /// Returns:
    /// - None if content is allowed
    /// - Some(message) if content should be blocked/warned
    pub async fn enforce(&self, text: &str) -> Option<String> {
        let lowered = text.to_lowercase().trim().to_string();

        // HIPAA mode: the requests it refuses (a diagnosis, a dose, a treatment decision)
        // are checked first, by themselves. They used to sit behind the allowlist below
        // and behind the classifier, so "my name is ..." in the message, or a classifier
        // that failed to load, skipped them and their audit line (found 2026-11-10).
        if self.mode == ContainmentMode::Hipaa {
            if let Some(hipaa_message) = self.enforce_hipaa(text) {
                return Some(hipaa_message);
            }
        }

        // Allowlist for memory-safe self-referential phrases
        let allowlist_phrases = [
            "my name is",
            "what is my name",
            "what's my name",
            "do you remember my name",
            "i am called",
        ];

        if allowlist_phrases
            .iter()
            .any(|phrase| lowered.contains(phrase))
        {
            return None;
        }

        // Witness mode: no enforcement
        if self.mode == ContainmentMode::Witness {
            return None;
        }

        // Pattern-based instruction harm detection (pre-filter before TinyBERT)
        // This catches instruction requests that TinyBERT misses (it's trained for toxicity, not instruction harm)
        let pattern_flagged = self.detect_harmful_instruction(&lowered);
        if pattern_flagged {
            println!("[Containment] Pattern matching flagged harmful instruction request");
        }

        // Child mode: MUST use OpenAI Moderation API (local models insufficient for child safety)
        if self.mode == ContainmentMode::Child {
            println!("[Containment] Child mode - checking with OpenAI Moderation API");

            // Check if OpenAI API key is available
            if std::env::var("OPENAI_API_KEY").is_err() {
                return Some(
                    "[BLOCK] ⚠️ CHILD MODE REQUIRES OPENAI API KEY\n\n\
                    Child mode requires OpenAI's Moderation API for robust content filtering.\n\
                    Local models are insufficient to guarantee child safety.\n\n\
                    Please set your OPENAI_API_KEY environment variable to use Child mode.\n\n\
                    You can get a key at: https://platform.openai.com/api-keys".to_string()
                );
            }

            // Use OpenAI Moderation API (primary check)
            match self.check_openai_moderation(text).await {
                Ok(Some(block_message)) => {
                    println!("[Containment] OpenAI Moderation blocked content");
                    return Some(format!("[BLOCK] {}", block_message));
                }
                Ok(None) => {
                    println!("[Containment] OpenAI Moderation passed, checking local model as secondary...");
                    // Fall through to local model as secondary check
                }
                Err(e) => {
                    println!("[Containment] OpenAI Moderation API error: {} - Child mode requires working API", e);
                    return Some(format!(
                        "[BLOCK] Child mode safety check failed: {}\n\n\
                        Child mode requires a working OpenAI API connection for safety.\n\
                        Please check your API key and internet connection.",
                        e
                    ));
                }
            }
        }

        // Use Candle-based safety_classifier for all modes (and as secondary check for Child)
        let mode_str = self.mode.to_string();
        println!("[Containment] Using Candle safety classifier for mode: {:?}", self.mode);

        // Fallback: Use Candle-based safety classifier for all modes
        let classification_result = match safety_classifier::should_block(text, &mode_str) {
            Ok((should_block, Some(result))) => {
                println!("[Containment] Safety classification: {:?} (confidence: {:.2})",
                         result.category, result.confidence);
                (should_block, result)
            }
            Ok((_, None)) => {
                println!("[Containment] Warning: No classification result");
                // Still honour pattern matching even when TinyBERT has no result
                if pattern_flagged {
                    return self.fallback_keyword_check(text);
                }
                return None;
            }
            Err(e) => {
                println!("[Containment] Classification error: {}", e);
                // Fallback to keyword detection if classifier fails
                return self.fallback_keyword_check(text);
            }
        };

        let (mut should_block, result) = classification_result;

        // Combine pattern matching with TinyBERT classification
        // Block if EITHER pattern matching OR TinyBERT flags it as harmful
        if pattern_flagged && !should_block {
            println!("[Containment] Pattern matching overriding TinyBERT - blocking harmful instruction");
            should_block = true;
        }

        // Apply mode-specific actions
        match self.mode {
            ContainmentMode::Guardian | ContainmentMode::Default => {
                // Guardian Mode: Demonstration of basic client-side content filtering
                // Enforces "standard corporate LLM" style interactions without involving the LLM itself
                // This is a voluntary filter - users can switch to Sovereign/Witness if they want less filtering
                if should_block {
                    let reason = if pattern_flagged {
                        "Harmful instruction request".to_string()
                    } else {
                        result.category.description().to_string()
                    };
                    Some(format!(
                        "[BLOCK] I'm sorry, I can't help with that. (Detected: {})",
                        reason
                    ))
                } else {
                    None
                }
            }
            ContainmentMode::Sovereign => {
                // Sovereign Mode: Warn but don't block
                // Show warning if either pattern matching OR TinyBERT flags it
                // Raised from 0.3 → 0.85: TinyBERT is demo-grade and false-positives on
                // grief/clinical language. Only warn on very high confidence until we
                // replace the classifier (see ROADMAP → "Real safety classifier").
                if pattern_flagged || (result.category.is_harmful() && result.confidence > 0.85) {
                    let reason = if pattern_flagged {
                        "Harmful instruction request"
                    } else {
                        result.category.description()
                    };
                    // [WARN_ALLOW] prefix tells caller to show warning but still get LLM response
                    Some(format!(
                        "[WARN_ALLOW] ⚠️ WARNING: This request may contain unsafe content ({}). Proceeding with caution.\n\n",
                        reason
                    ))
                } else {
                    None
                }
            }
            ContainmentMode::Hipaa => {
                // The HIPAA refusals ran at the top; this is general safety, as in Guardian.
                if should_block {
                    Some(format!(
                        "[BLOCK] Cannot process this request in HIPAA mode. (Detected: {})",
                        result.category.description()
                    ))
                } else {
                    None
                }
            }
            ContainmentMode::Child => {
                // This is the secondary check (OpenAI Moderation API is primary)
                // Raised from 0.2 → 0.7: 0.2 was aggressive-to-false-positive even for
                // Child mode. Primary safety layer is the moderation API.
                if result.category.is_harmful() && result.confidence > 0.7 {
                    Some(format!(
                        "[BLOCK] I can't answer that in Child Mode. (Secondary check detected: {})",
                        result.category.description()
                    ))
                } else {
                    None
                }
            }
            ContainmentMode::Elder => {
                // Raised from 0.4 → 0.85: Elder mode is meant to be gentle, not
                // restrictive. TinyBERT false-positives on grief language make 0.4
                // interrupt normal conversation.
                if result.category.is_harmful() && result.confidence > 0.85 {
                    Some("[GENTLE] I'm worried this topic might be distressing. Would you like to talk about something else?".to_string())
                } else {
                    None
                }
            }
            ContainmentMode::Witness => None, // Already handled above
        }
    }

    /// HIPAA-specific medication dosing detection (matches Python lines 388-399)
    fn detect_medication_dosing(&self, text: &str) -> bool {
        let lowered = text.to_lowercase();

        let dosing_phrases = [
            "how much should i take",
            "how much",
            "should i take",
            "dosage for",
            "how many pills",
            "what dose",
            "safely",
        ];

        let medication_keywords = vec![
            "mg", "dose", "pill", "tablet", "medication", "medicine", "drug",
            "prescription", "ibuprofen", "acetaminophen", "aspirin", "tylenol",
            "advil", "take", "can i take",
        ];

        let has_dosing_phrase = dosing_phrases.iter().any(|phrase| lowered.contains(phrase));
        let has_medication_context = medication_keywords.iter().any(|keyword| lowered.contains(keyword));

        has_dosing_phrase && has_medication_context
    }

    /// HIPAA-specific diagnosis detection (matches Python lines 367-372)
    fn detect_diagnosis_request(&self, text: &str) -> bool {
        let lowered = text.to_lowercase();

        let diagnostic_phrases = [
            "diagnose me",
            "what do i have",
            "do i have",
            "is it cancer",
            "is it diabetes",
        ];

        diagnostic_phrases.iter().any(|phrase| lowered.contains(phrase))
    }

    /// The requests HIPAA mode refuses: a diagnosis, a dose, a treatment decision.
    fn enforce_hipaa(&self, text: &str) -> Option<String> {
        // Identifiers are no longer a reason to refuse: `redact_phi` takes them out of what
        // the model is sent, and the user is told (Matt, 2026-11-10).

        // Step 2: Check for diagnosis requests
        if self.detect_diagnosis_request(text) {
            log_hipaa_audit("diagnosis_request_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🏥 I cannot provide medical diagnoses. Please consult with a licensed healthcare provider for diagnostic evaluations.".to_string());
        }

        // Step 3: Check for medication dosing questions
        if self.detect_medication_dosing(text) {
            log_hipaa_audit("dosing_request_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🏥 I cannot provide medication dosing advice. Please consult your doctor or pharmacist for accurate dosing information.".to_string());
        }

        // Step 4: Check for treatment planning
        let lowered = text.to_lowercase();
        let treatment_phrases = [
            "should i get surgery",
            "should i have the procedure",
            "should i start treatment",
        ];
        if treatment_phrases.iter().any(|phrase| lowered.contains(phrase)) {
            log_hipaa_audit("treatment_request_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🏥 I cannot advise on treatment decisions. Please discuss treatment options with your healthcare provider.".to_string());
        }

        // Log allowed health conversation
        log_hipaa_audit("hipaa_conversation_allowed", json!({
            "query_length": text.len(),
            "blocked": false
        }));

        None
    }

    /// Child mode: Use OpenAI Chat Completions for both safety check AND response generation
    /// This replaces separate safety check + Claude response with a single OpenAI call
    ///
    /// Returns:
    /// - Ok(response) if successful (OpenAI's child-safe response)
    /// - Err(_) if API call fails (will fall back to toxic-bert + Claude)
    #[allow(dead_code)]
    pub async fn get_child_mode_response(&self, user_message: &str, conversation_history: &str) -> Result<String, String> {
        // Get API key from environment
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set - Child mode requires OpenAI API access".to_string())?;

        // Prepare Chat Completions request
        let client = reqwest::Client::new();

        let system_prompt = CHILD_MODE_SYSTEM_PROMPT;

        // Build messages array with conversation history
        let mut messages = vec![
            json!({
                "role": "system",
                "content": system_prompt
            })
        ];

        // Add conversation history if available
        if !conversation_history.is_empty() {
            // Parse conversation history (format: "User: X\nAssistant: Y\n...")
            for line in conversation_history.lines() {
                if let Some(content) = line.strip_prefix("User: ") {
                    messages.push(json!({"role": "user", "content": content}));
                } else if let Some(content) = line.strip_prefix("Assistant: ") {
                    messages.push(json!({"role": "assistant", "content": content}));
                }
            }
        }

        // Add current user message
        messages.push(json!({
            "role": "user",
            "content": user_message
        }));

        let body = json!({
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        });

        println!("[Containment] Calling OpenAI Chat Completions API for Child mode response...");

        // Call OpenAI Chat Completions API
        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OpenAI API request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("OpenAI API returned error: {}", response.status()));
        }

        // Parse response
        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse OpenAI response: {}", e))?;

        // Extract the assistant's message
        let assistant_message = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| "Failed to extract message content from OpenAI response".to_string())?
            .to_string();

        println!("[Containment] ✅ OpenAI Chat Completions returned child-safe response");
        Ok(assistant_message)
    }

    /// Check content using OpenAI Moderation API (Child mode only)
    ///
    /// Returns:
    /// - Ok(Some(message)) if content should be blocked
    /// - Ok(None) if content is safe
    /// - Err(_) if API call fails (no API key, network error, etc.)
    pub async fn check_openai_moderation(&self, text: &str) -> Result<Option<String>, String> {
        // Get API key from environment
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set - Child mode requires OpenAI API access".to_string())?;

        // Prepare request (use async client, not blocking)
        let client = reqwest::Client::new();
        let body = json!({
            "input": text
        });

        println!("[Containment] Calling OpenAI Moderation API...");

        // Call OpenAI Moderation API (async)
        let response = client
            .post("https://api.openai.com/v1/moderations")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OpenAI API request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("OpenAI API returned error: {}", response.status()));
        }

        let moderation: OpenAIModerationResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse OpenAI response: {}", e))?;

        // Check if content was flagged
        if let Some(result) = moderation.results.first() {
            if result.flagged {
                // Build list of violated categories for logging
                let mut violated = Vec::new();
                if result.categories.sexual { violated.push("sexual"); }
                if result.categories.sexual_minors { violated.push("sexual/minors"); }
                if result.categories.hate { violated.push("hate"); }
                if result.categories.hate_threatening { violated.push("hate/threatening"); }
                if result.categories.harassment { violated.push("harassment"); }
                if result.categories.harassment_threatening { violated.push("harassment/threatening"); }
                if result.categories.self_harm { violated.push("self-harm"); }
                if result.categories.self_harm_intent { violated.push("self-harm/intent"); }
                if result.categories.self_harm_instructions { violated.push("self-harm/instructions"); }
                if result.categories.violence { violated.push("violence"); }
                if result.categories.violence_graphic { violated.push("violence/graphic"); }

                println!("[Containment] OpenAI flagged categories: {:?}", violated);

                return Ok(Some("[BLOCK] I can't answer that in Child Mode. This content was flagged as potentially unsafe.".to_string()));
            }
        }

        Ok(None)  // Content is safe
    }

    /// Fallback keyword-based detection if safety_classifier fails
    fn fallback_keyword_check(&self, text: &str) -> Option<String> {
        let lowered = text.to_lowercase();
        let is_unsafe = self.detect_unsafe_content(&lowered);

        if is_unsafe {
            match self.mode {
                ContainmentMode::Guardian | ContainmentMode::Default | ContainmentMode::Hipaa => {
                    Some("[BLOCK] I'm sorry, I can't help with that.".to_string())
                }
                ContainmentMode::Sovereign => {
                    // [WARN_ALLOW] prefix tells caller to show warning but still get LLM response
                    Some("[WARN_ALLOW] ⚠️ WARNING: This request may be unsafe. Proceeding with caution.\n\n".to_string())
                }
                ContainmentMode::Child => {
                    Some("[BLOCK] I can't answer that in Child Mode.".to_string())
                }
                ContainmentMode::Elder => {
                    Some("[GENTLE] I'm worried this topic might be distressing.".to_string())
                }
                ContainmentMode::Witness => None,
            }
        } else {
            None
        }
    }

    /// Pattern-based harmful instruction detection
    ///
    /// Matches whole words: see `contains_word`.
    ///
    /// Detects requests that ask "how to" do something harmful.
    /// This catches instruction harm that TinyBERT misses (TinyBERT is trained for toxicity, not instruction harm).
    ///
    /// Works by checking:
    /// 1. Does this request ask for instructions? ("how to", "teach me", "show me how")
    /// 2. Is the subject potentially harmful? ("bomb", "weapon", "poison")
    ///
    /// Examples:
    /// - "How to make a bomb" → true (instruction + harmful subject)
    /// - "Bombs in WWII" → false (not an instruction request)
    /// - "Murder, She Wrote" → false (not an instruction request)
    fn detect_harmful_instruction(&self, text: &str) -> bool {
        // Step 1: Check if this is an instruction request
        let instruction_patterns = [
            "how to", "how do i", "how can i", "how would i", "how should i",
            "teach me", "show me how", "tell me how", "explain how to",
            "make a", "make an", "create a", "create an", "build a", "build an",
            "steps to", "way to", "method to", "methods to", "ways to",
            "instructions for", "guide to", "tutorial on",
        ];

        let is_instruction = instruction_patterns
            .iter()
            .any(|pattern| contains_word(text, pattern));

        if !is_instruction {
            return false; // Not asking for instructions
        }

        // Step 2: Check if the subject is potentially harmful
        // These are subjects that shouldn't have instruction requests
        let harmful_subjects = [
            // Weapons & explosives
            "bomb", "explosive", "weapon", "gun", "firearm", "ammunition",
            "grenade", "dynamite", "tnt", "c4", "rpg", "missile",
            // Violence & harm
            "kill", "murder", "assassinate", "poison", "harm", "hurt", "attack",
            "torture", "suicide", "self-harm", "overdose",
            // Illegal activities
            "hack", "crack", "exploit", "malware", "virus", "ransomware",
            "steal", "rob", "fraud", "forge", "counterfeit",
            "meth", "cocaine", "heroin", "fentanyl",
        ];

        harmful_subjects
            .iter()
            .any(|subject| contains_word(text, subject))
    }

    /// Simple keyword-based unsafe content detection (fallback if safety_classifier fails)
    #[allow(dead_code)]
    fn detect_unsafe_content(&self, text: &str) -> bool {
        let unsafe_keywords = vec![
            "hack",
            "exploit",
            "illegal",
            "malware",
            "virus",
            "crack",
            "pirate",
            "steal",
            "weapon",
            "bomb",
            "drug",
            "suicide",
        ];

        unsafe_keywords.iter().any(|keyword| contains_word(text, keyword))
    }

    /// Stricter detection for child mode
    #[allow(dead_code)]
    fn detect_unsafe_content_strict(&self, text: &str) -> bool {
        let strict_keywords = vec![
            "hack",
            "exploit",
            "illegal",
            "malware",
            "virus",
            "crack",
            "pirate",
            "steal",
            "weapon",
            "bomb",
            "drug",
            "suicide",
            "violence",
            "adult",
            "sex",
            "gore",
            "death",
            "kill",
        ];

        strict_keywords
            .iter()
            .any(|keyword| contains_word(text, keyword))
    }

}

/// Log HIPAA compliance events to the daily audit file,
/// `<app data dir>/logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json`. Until 2026-11-10 the
/// folder was relative to wherever the process happened to be started from.
fn log_hipaa_audit(event_type: &str, metadata: serde_json::Value) {
    // Test binaries share the real data folder with a running app.
    if cfg!(test) {
        return;
    }
    let log_dir = crate::db::get_app_data_dir().join("logs").join("hipaa_audit");
    let log_dir = log_dir.as_path();

    // Create directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(log_dir) {
        eprintln!("[HIPAA Audit] Failed to create audit directory: {}", e);
        return;
    }

    // Create daily log file
    let today = Utc::now().format("%Y-%m-%d").to_string();
    let log_file = log_dir.join(format!("hipaa_audit_{}.json", today));

    // Create log entry
    let log_entry = json!({
        "timestamp": Utc::now().to_rfc3339(),
        "event_type": event_type,
        "metadata": metadata
    });

    // Append to log file
    match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
    {
        Ok(mut file) => {
            if let Err(e) = writeln!(file, "{}", log_entry) {
                eprintln!("[HIPAA Audit] Failed to write log entry: {}", e);
            }
        }
        Err(e) => {
            eprintln!("[HIPAA Audit] Failed to open log file: {}", e);
        }
    }
}

/// `needle` occurs in `haystack` bounded by non-letters on both sides, so a word list
/// cannot match inside another word: "meth" no longer matches "something" (a message
/// about posting a plush toy was flagged as a harmful instruction request on
/// 2026-10-21) and "gun" no longer matches "begun". Multi-word needles such as
/// "how to" still match across their own spaces.
pub(crate) fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() { return false; }
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let begin = start + pos;
        let end = begin + needle.len();
        let before_ok = !haystack[..begin].chars().next_back().map_or(false, |c| c.is_alphanumeric());
        let after_ok = !haystack[end..].chars().next().map_or(false, |c| c.is_alphanumeric());
        if before_ok && after_ok { return true; }
        start = begin + needle.chars().next().map_or(1, |c| c.len_utf8());
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_lists_match_whole_words_only() {
        assert!(contains_word("how to make meth at home", "meth"));
        assert!(!contains_word("maybe she has something small", "meth"));
        assert!(!contains_word("the game has begun", "gun"));
        assert!(contains_word("how to build a gun", "gun"));
        assert!(contains_word("tell me how to do it", "how to"));
        assert!(contains_word("meth", "meth"));
        assert!(contains_word("he took an overdose.", "overdose"));
        assert!(!contains_word("the pharmacy is closed", "harm"));
    }

    #[tokio::test]
    async fn test_witness_mode_allows_everything() {
        let guard = ContainmentLayer::new("witness").unwrap();
        assert_eq!(guard.enforce("how to hack a computer").await, None);
    }

    #[tokio::test]
    async fn test_guardian_mode_blocks_unsafe() {
        let guard = ContainmentLayer::new("guardian").unwrap();
        let result = guard.enforce("how to hack a computer").await;
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("[BLOCK]"));
    }

    #[tokio::test]
    async fn test_child_mode_stricter() {
        let guard = ContainmentLayer::new("child").unwrap();
        let result = guard.enforce("tell me about violence").await;
        assert!(result.is_some());
    }

    #[test]
    fn hipaa_removes_the_guides_own_example() {
        let (out, found) = redact_phi("My patient John Smith, DOB 01/15/1985, MRN 4829201, needs a follow-up.");
        assert!(!out.contains("John Smith"), "{out}");
        assert!(!out.contains("01/15/1985"), "{out}");
        assert!(!out.contains("4829201"), "{out}");
        assert!(out.contains("needs a follow-up"), "{out}");
        assert_eq!(found.len(), 3, "{found:?}");
        assert_eq!(describe_phi(&found), "1 record or member number, 1 date of birth and 1 patient name");
    }

    /// The examples printed in the HIPAA guide ("PHI Detection in Action"). If one of
    /// these changes, the guide changes with it.
    #[test]
    fn the_hipaa_guides_examples_read_as_printed() {
        for (text, cleaned, said) in [
            ("My patient is Mary Johnson. Her social is 234-56-7890 and she has been having chest pains.",
             "My [patient name removed]. Her social is [SSN removed] and she has been having chest pains.",
             "1 Social Security number and 1 patient name"),
            ("I've tried reaching the patient twice at 555-867-5309 with no answer. What should I document in the chart?",
             "I've tried reaching the patient twice at [phone number removed] with no answer. What should I document in the chart?",
             "1 phone number"),
            // No word "patient" in front of the name: it stays. The guide says so.
            ("John Smith, a 45-year-old male, presents with substernal chest pain radiating to the left arm.",
             "John Smith, a 45-year-old male, presents with substernal chest pain radiating to the left arm.",
             ""),
            ("My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with chest pain.",
             "My [patient name removed], [date of birth removed], [record number removed], presents with chest pain.",
             "1 record or member number, 1 date of birth and 1 patient name"),
            ("My name is Mary Johnson and my social is 234-56-7890.",
             "My name is Mary Johnson and my social is [SSN removed].",
             "1 Social Security number"),
        ] {
            let (out, found) = redact_phi(text);
            assert_eq!(out, cleaned);
            assert_eq!(describe_phi(&found), said);
        }
    }

    #[test]
    fn hipaa_removes_each_kind_of_identifier() {
        for (text, gone) in [
            ("call me on (555) 123-4567 tomorrow", "123-4567"),
            ("call me on 555-123-4567 tomorrow", "555-123"),
            ("my cell is 5551234567", "5551234567"),
            ("reach me at +1 555.123.4567", "123.4567"),
            ("SSN 123-45-6789 on file", "123-45-6789"),
            ("her social security number is 123456789", "123456789"),
            ("write to ana.lopez@example.org please", "ana.lopez@example.org"),
            ("she lives at 123 Main St. now", "123 Main"),
            ("moved to 45 North Elm Street last year", "North Elm"),
            ("Portland, OR 97201 is home", "97201"),
            ("zip code 90210", "90210"),
            ("card 4111 1111 1111 1111 expired", "1111 1111"),
            ("the server is 192.168.1.20", "192.168.1.20"),
            ("member ID: ABC-12345 on the card", "ABC-12345"),
            ("date of birth: March 4, 1961", "March 4"),
            ("born on 4 March 1961 in Ohio", "4 March"),
            ("the client's name is Mrs. Ana Lopez", "Ana Lopez"),
        ] {
            let (out, found) = redact_phi(text);
            assert!(!out.contains(gone), "{text:?} gave {out:?}");
            assert!(!found.is_empty(), "{text:?}");
            assert!(out.contains("removed]"), "{out}");
        }
    }

    /// The old check called any 5-, 9- or 10-digit number an identifier. Text is changed
    /// here, not refused, so a wrong match damages a memory or a question.
    #[test]
    fn hipaa_leaves_ordinary_text_alone() {
        for text in [
            "I walked 10000 steps today and 12345 yesterday",
            "What options do I have?",
            "It is 5 miles down the road, 2 blocks past the church",
            "(about November 7, 2026)",
            "Today is Tuesday, November 10, 2026.",
            "She was born in 1985 and moved in 2010",
            "The patient is recovering well",
            "Our budget is 1250000000 this year",
            "Version 2.4 of the app, build 100200300",
            "Dinner with Ana Lopez at 7",
        ] {
            let (out, found) = redact_phi(text);
            assert_eq!(out, text);
            assert!(found.is_empty(), "{text:?}: {found:?}");
        }
    }

    #[test]
    fn phi_totals_add_up_and_read_plainly() {
        let mut total: PhiFound = Vec::new();
        merge_phi(&mut total, redact_phi("call 555-123-4567").1);
        merge_phi(&mut total, redact_phi("or 555-987-6543, or mail a@b.org").1);
        assert_eq!(describe_phi(&total), "2 phone numbers and 1 email address");
        assert_eq!(describe_phi(&Vec::new()), "");
    }

    /// A tag left by one pass must not be matched by a later one: the typed message is
    /// cleaned first and the whole prompt again.
    #[test]
    fn cleaning_twice_changes_nothing_more() {
        let (once, _) = redact_phi("patient John Smith, 555-123-4567, 123 Main St, OR 97201, MRN 4829201");
        let (twice, found) = redact_phi(&once);
        assert_eq!(once, twice);
        assert!(found.is_empty(), "{found:?}");
    }

    /// "my name is" is on the allowlist; it used to skip the HIPAA refusals with it.
    #[tokio::test]
    async fn hipaa_refusals_run_before_the_allowlist() {
        let guard = ContainmentLayer::new("hipaa").unwrap();
        let said = guard.enforce("my name is Ann, how many pills of ibuprofen should I take").await;
        assert!(said.is_some_and(|m| m.contains("dosing")));
    }

    #[tokio::test]
    async fn test_allowlist_phrases() {
        let guard = ContainmentLayer::new("guardian").unwrap();
        assert_eq!(guard.enforce("my name is John").await, None);
        assert_eq!(guard.enforce("what is my name?").await, None);
    }
}
