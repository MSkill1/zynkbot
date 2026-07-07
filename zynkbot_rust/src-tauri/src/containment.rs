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
///
/// Users can extend this framework to create custom modes for their specific needs
/// (e.g., educational mode, professional mode, etc.)
use serde::{Deserialize, Serialize};
use crate::safety_classifier;
use serde_json::json;
use std::path::Path;
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
                if pattern_flagged || (result.category.is_harmful() && result.confidence > 0.3) {
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
                // HIPAA mode: Check PHI, diagnoses, medication dosing FIRST
                if let Some(hipaa_message) = self.enforce_hipaa(text) {
                    return Some(hipaa_message);
                }

                // Then check general safety (same as Guardian)
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
                // Use very strict threshold for child safety
                if result.category.is_harmful() && result.confidence > 0.2 {
                    Some(format!(
                        "[BLOCK] I can't answer that in Child Mode. (Secondary check detected: {})",
                        result.category.description()
                    ))
                } else {
                    None
                }
            }
            ContainmentMode::Elder => {
                if result.category.is_harmful() && result.confidence > 0.4 {
                    Some("[GENTLE] I'm worried this topic might be distressing. Would you like to talk about something else?".to_string())
                } else {
                    None
                }
            }
            ContainmentMode::Witness => None, // Already handled above
        }
    }

    /// HIPAA-specific PHI detection (matches Python implementation)
    fn detect_phi(&self, text: &str) -> bool {
        use regex::Regex;

        // PHI patterns (from Python implementation lines 156-165)
        let patterns = vec![
            (r"\b\d{3}-?\d{2}-?\d{4}\b", "SSN"),              // 123-45-6789
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),       // (555) 123-4567
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
            (r"\b\d{5}(?:-\d{4})?\b", "zip"),                  // 12345 or 12345-6789
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit_card"),
            (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "ip_address"),
            (r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd)\b", "address"),
        ];

        for (pattern, name) in patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(text) {
                    println!("[Containment HIPAA] Detected PHI: {}", name);
                    return true;
                }
            }
        }

        false
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

    /// Log HIPAA compliance events to daily audit log
    fn log_hipaa_audit(&self, event_type: &str, metadata: serde_json::Value) {
        let log_dir = Path::new("logs/hipaa_audit");

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

    /// Enforce HIPAA-specific rules (matches Python lines 357-419)
    fn enforce_hipaa(&self, text: &str) -> Option<String> {
        // Step 1: Check for PHI in input
        if self.detect_phi(text) {
            // Log PHI detection attempt
            self.log_hipaa_audit("phi_detection_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🔒 Please don't share personal health information like SSN, insurance numbers, or member IDs. This system is designed for general health discussions only.".to_string());
        }

        // Step 2: Check for diagnosis requests
        if self.detect_diagnosis_request(text) {
            self.log_hipaa_audit("diagnosis_request_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🏥 I cannot provide medical diagnoses. Please consult with a licensed healthcare provider for diagnostic evaluations.".to_string());
        }

        // Step 3: Check for medication dosing questions
        if self.detect_medication_dosing(text) {
            self.log_hipaa_audit("dosing_request_blocked", json!({
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
            self.log_hipaa_audit("treatment_request_blocked", json!({
                "query_length": text.len(),
                "blocked": true
            }));

            return Some("🏥 I cannot advise on treatment decisions. Please discuss treatment options with your healthcare provider.".to_string());
        }

        // Log allowed health conversation
        self.log_hipaa_audit("hipaa_conversation_allowed", json!({
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
            .any(|pattern| text.contains(pattern));

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
            .any(|subject| text.contains(subject))
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

        unsafe_keywords.iter().any(|keyword| text.contains(keyword))
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
            .any(|keyword| text.contains(keyword))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn test_allowlist_phrases() {
        let guard = ContainmentLayer::new("guardian").unwrap();
        assert_eq!(guard.enforce("my name is John").await, None);
        assert_eq!(guard.enforce("what is my name?").await, None);
    }
}
