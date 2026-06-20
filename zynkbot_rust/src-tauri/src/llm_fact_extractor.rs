// LLM-Based Fact Extraction
// Uses the same LLM that answered the user's question to extract personal facts
// NOTE: This is experimental/unused code kept for future development
#![allow(dead_code)]

use crate::question_extractor::ExtractedFact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMExtractedFact {
    pub content: String,
    pub first_person: String,
    pub title: String,
    pub namespace: String,
    pub tags: Vec<String>,
    pub confidence: f32,
}

pub struct LLMFactExtractor;

impl LLMFactExtractor {
    pub fn new() -> Self {
        LLMFactExtractor
    }

    /// Extract facts from a conversation using the LLM
    ///
    /// # Arguments
    /// * `user_question` - The user's original question/statement
    /// * `llm_response` - The LLM's response to the question
    /// * `backend` - Which backend to use (anthropic, openai, etc.)
    ///
    /// # Returns
    /// Vector of ExtractedFact objects
    pub async fn extract_facts(
        &self,
        user_question: &str,
        llm_response: &str,
        backend: &str,
    ) -> Result<Vec<ExtractedFact>, String> {
        // Build the fact extraction prompt
        let extraction_prompt = self.build_extraction_prompt(user_question, llm_response);

        // Call the appropriate LLM backend
        let facts_json = self.call_llm_for_extraction(&extraction_prompt, backend).await?;

        // Parse the JSON response into ExtractedFact objects
        let facts = self.parse_llm_response(&facts_json)?;

        // Validate and normalize the facts
        let normalized_facts = facts.into_iter()
            .map(|f| self.normalize_fact(f))
            .collect();

        Ok(normalized_facts)
    }

    /// Build a prompt that asks the LLM to extract personal facts
    fn build_extraction_prompt(&self, user_question: &str, llm_response: &str) -> String {
        format!(
            r#"You are a fact extraction system. Analyze the following conversation and extract any personal facts about the user.

CONVERSATION:
USER: {}
ASSISTANT: {}

TASK:
Extract personal facts about the user from their message. A "personal fact" is any statement that reveals:
- Personal information (name, age, location, occupation, etc.)
- Possessions (pets, vehicles, devices, etc.)
- Preferences (likes, dislikes, interests, etc.)
- Experiences (events, travel, achievements, etc.)
- Relationships (family, friends, colleagues, etc.)
- Skills or abilities
- Future plans or goals

IMPORTANT RULES:
1. Only extract facts stated BY THE USER, not general knowledge
2. Focus on facts that are personal to the user
3. Each fact should be a complete, standalone statement
4. Do NOT extract questions - only factual statements
5. If no personal facts are present, return an empty array

OUTPUT FORMAT:
Return ONLY a JSON array of facts. Each fact must have:
- "content": Full sentence describing the fact in third-person (e.g., "User has an RTX 3090 GPU")
- "first_person": Same fact in first-person form (e.g., "I have an RTX 3090" or "My PC has an RTX 3090")
- "title": Short title (max 50 chars, e.g., "RTX 3090 GPU")
- "namespace": One of: personal, work, travel, events, health, education, entertainment
- "tags": Array of relevant keywords (e.g., ["hardware", "gpu", "gaming"])
- "confidence": Float 0.0-1.0 indicating certainty (0.9 for explicit facts, 0.7 for implied, 0.5 for uncertain)

IMPORTANT: The "first_person" field should be a natural first-person statement that preserves the user's tone.
Extract only the factual statement portion, not questions.

EXAMPLE OUTPUT:
[
  {{
    "content": "User has an RTX 3090 graphics card in their PC",
    "first_person": "My PC has an RTX 3090",
    "title": "RTX 3090 GPU",
    "namespace": "personal",
    "tags": ["hardware", "gpu", "rtx3090", "pc"],
    "confidence": 0.95
  }},
  {{
    "content": "User is interested in large language model development",
    "first_person": "I'm interested in large language model development",
    "title": "Interest in LLM Development",
    "namespace": "personal",
    "tags": ["interest", "llm", "machine learning", "development"],
    "confidence": 0.85
  }}
]

Now extract facts from the conversation above. Return ONLY the JSON array, no other text:"#,
            user_question, llm_response
        )
    }

    /// Call the LLM backend to extract facts
    async fn call_llm_for_extraction(
        &self,
        prompt: &str,
        backend: &str,
    ) -> Result<String, String> {
        // Determine which backend to use
        if backend.contains("anthropic") {
            self.call_anthropic(prompt).await
        } else if backend.contains("openai") {
            self.call_openai(prompt).await
        } else if backend.contains("xai") {
            self.call_xai(prompt).await
        } else if backend.ends_with(".gguf") {
            // Local models might not be good at structured output, fall back
            Err("Local GGUF models not supported for fact extraction - use pattern-based fallback".to_string())
        } else {
            Err(format!("Unsupported backend for fact extraction: {}", backend))
        }
    }

    /// Call Anthropic Claude for fact extraction
    async fn call_anthropic(&self, prompt: &str) -> Result<String, String> {
        use crate::llm::anthropic;
        use crate::llm::Message;

        // Get API key from environment
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set".to_string())?;

        let model = "claude-sonnet-4-6";

        println!("[LLM Fact Extraction] Calling Anthropic Claude Haiku...");

        // Build messages array
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        match anthropic::send_message(
            &api_key,
            model,
            messages,
            None, // system_prompt
            Some(2000), // max_tokens
            Some(0.3), // temperature - low for consistent extraction
        ).await {
            Ok(response) => {
                println!("[LLM Fact Extraction] ✓ Received response from Claude");
                Ok(response.content)
            }
            Err(e) => {
                println!("[LLM Fact Extraction] ✗ Anthropic error: {}", e);
                Err(format!("Anthropic API error: {}", e))
            }
        }
    }

    /// Call OpenAI GPT for fact extraction
    async fn call_openai(&self, prompt: &str) -> Result<String, String> {
        use crate::llm::openai;
        use crate::llm::Message;

        // Get API key from environment
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set".to_string())?;

        println!("[LLM Fact Extraction] Calling OpenAI GPT-4o-mini...");

        // Build messages array
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        match openai::send_message(
            &api_key,
            "gpt-4o-mini",
            messages,
            Some(2000), // max_tokens
            Some(0.3), // temperature - low for consistent extraction
        ).await {
            Ok(response) => {
                println!("[LLM Fact Extraction] ✓ Received response from OpenAI");
                Ok(response.content)
            }
            Err(e) => {
                println!("[LLM Fact Extraction] ✗ OpenAI error: {}", e);
                Err(format!("OpenAI API error: {}", e))
            }
        }
    }

    /// Call xAI Grok for fact extraction
    async fn call_xai(&self, prompt: &str) -> Result<String, String> {
        use crate::llm::xai;
        use crate::llm::Message;

        // Get API key from environment
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| "XAI_API_KEY not set".to_string())?;

        println!("[LLM Fact Extraction] Calling xAI Grok...");

        // Build messages array
        let messages = vec![Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        }];

        match xai::send_message(
            &api_key,
            "grok-3",
            messages,
            Some(2000), // max_tokens
            Some(0.3), // temperature - low for consistent extraction
        ).await {
            Ok(response) => {
                println!("[LLM Fact Extraction] ✓ Received response from Grok");
                Ok(response.content)
            }
            Err(e) => {
                println!("[LLM Fact Extraction] ✗ xAI error: {}", e);
                Err(format!("xAI API error: {}", e))
            }
        }
    }

    /// Parse the LLM's JSON response into fact objects
    fn parse_llm_response(&self, json_str: &str) -> Result<Vec<LLMExtractedFact>, String> {
        // Clean up the response - remove markdown code blocks if present
        let cleaned = json_str
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        // Try to parse as JSON array
        match serde_json::from_str::<Vec<LLMExtractedFact>>(cleaned) {
            Ok(facts) => {
                println!("[LLM Fact Extraction] ✓ Parsed {} facts", facts.len());
                Ok(facts)
            }
            Err(e) => {
                println!("[LLM Fact Extraction] ✗ JSON parse error: {}", e);
                println!("[LLM Fact Extraction] Response was: {}", cleaned);
                // Return empty array instead of error - graceful degradation
                Ok(Vec::new())
            }
        }
    }

    /// Normalize and validate a fact to ensure it meets requirements
    fn normalize_fact(&self, fact: LLMExtractedFact) -> ExtractedFact {
        // Validate namespace - use "personal" as default
        let valid_namespaces = vec![
            "personal", "work", "travel", "events", "health", "education", "entertainment"
        ];
        let namespace = if valid_namespaces.contains(&fact.namespace.as_str()) {
            fact.namespace
        } else {
            "personal".to_string()
        };

        // Clamp confidence to 0.0-1.0
        let confidence = fact.confidence.clamp(0.0, 1.0);

        // Ensure title is not too long
        let title = if fact.title.len() > 50 {
            fact.title[..50].to_string()
        } else {
            fact.title
        };

        // Ensure we have at least one tag
        let tags = if fact.tags.is_empty() {
            vec!["personal".to_string()]
        } else {
            fact.tags
        };

        ExtractedFact {
            content: fact.content,
            first_person: fact.first_person,
            title,
            namespace,
            tags,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_fact() {
        let extractor = LLMFactExtractor::new();

        let fact = LLMExtractedFact {
            content: "User has RTX 3090".to_string(),
            first_person: "I have an RTX 3090".to_string(),
            title: "RTX 3090".to_string(),
            namespace: "personal".to_string(),
            tags: vec!["hardware".to_string()],
            confidence: 0.95,
        };

        let normalized = extractor.normalize_fact(fact);
        assert_eq!(normalized.confidence, 0.95);
        assert_eq!(normalized.namespace, "personal");
        assert_eq!(normalized.first_person, "I have an RTX 3090");
    }

    #[test]
    fn test_normalize_invalid_namespace() {
        let extractor = LLMFactExtractor::new();

        let fact = LLMExtractedFact {
            content: "Test".to_string(),
            first_person: "Test".to_string(),
            title: "Test".to_string(),
            namespace: "invalid_namespace".to_string(),
            tags: vec!["test".to_string()],
            confidence: 0.8,
        };

        let normalized = extractor.normalize_fact(fact);
        assert_eq!(normalized.namespace, "personal"); // Should default to personal
    }

    #[test]
    fn test_normalize_confidence_clamping() {
        let extractor = LLMFactExtractor::new();

        let fact = LLMExtractedFact {
            content: "Test".to_string(),
            first_person: "Test".to_string(),
            title: "Test".to_string(),
            namespace: "personal".to_string(),
            tags: vec!["test".to_string()],
            confidence: 1.5, // Invalid - above 1.0
        };

        let normalized = extractor.normalize_fact(fact);
        assert_eq!(normalized.confidence, 1.0); // Should clamp to 1.0
    }
}
