/// Question Memory Extractor
/// Analyzes user questions to extract memory-worthy information using local NER.
/// Works offline, no API calls required.
///
/// Mimics Python's question_memory_extractor.py
use regex::Regex;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};
use crate::nlp_enhancer::{Entity, NLPEnhancer};

/// Result of checking if question contains memory-worthy info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWorthinessCheck {
    pub has_info: bool,
    pub confidence: f32,
    pub signals: Vec<String>,
}

/// Extracted fact from a question
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub content: String,
    pub first_person: String,
    pub title: String,
    pub namespace: String,
    pub tags: Vec<String>,
    pub confidence: f32,
}

pub struct QuestionMemoryExtractor {
    // Patterns compiled once for efficiency
    possessive_words: HashSet<String>,
    nlp_enhancer: NLPEnhancer,
}

impl QuestionMemoryExtractor {
    pub fn new() -> Self {
        let possessive_words: HashSet<String> = vec![
            "my", "mine", "i'm", "i've", "i"
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            possessive_words,
            nlp_enhancer: NLPEnhancer::new(),
        }
    }

    /// Extract named entities using Candle BERT NER (via NLPEnhancer)
    fn extract_entities(&self, text: &str) -> Vec<Entity> {
        self.nlp_enhancer.extract_entities(text)
    }

    /// Convert third-person fact content to first-person form
    /// This preserves user's voice while allowing normalized search
    fn to_first_person(&self, content: &str) -> String {
        content
            .replace("User has a ", "I have a ")
            .replace("User has an ", "I have an ")
            .replace("User has ", "I have ")
            .replace("User's ", "My ")
            .replace("User is a ", "I'm a ")
            .replace("User is an ", "I'm an ")
            .replace("User is ", "I'm ")
            .replace("User ", "I ")
    }

    /// Quick check if question contains extractable user information
    /// Matches Python's contains_memory_worthy_info() (lines 27-100)
    pub fn contains_memory_worthy_info(&self, question: &str) -> MemoryWorthinessCheck {
        let mut signals = Vec::new();
        let question_lower = question.to_lowercase();

        // Check for possessive patterns
        let words: Vec<&str> = question_lower.split_whitespace().collect();
        let has_possessive = words.iter().any(|w| {
            let clean = w.trim_matches(|c: char| !c.is_alphabetic());
            self.possessive_words.contains(clean)
        });

        if has_possessive {
            signals.push("possessive_pronoun".to_string());
        }

        // Check for personal entities using NER
        let entities = self.extract_entities(question);
        let has_person = entities.iter().any(|e| e.label == "PER");
        let has_location = entities.iter().any(|e| e.label == "LOC");
        let has_org = entities.iter().any(|e| e.label == "ORG");
        let has_date = entities.iter().any(|e| e.label == "MISC"); // DATE/TIME often tagged as MISC

        if has_person {
            signals.push("person_name".to_string());
        }
        if has_location {
            signals.push("location".to_string());
        }
        if has_org {
            signals.push("organization".to_string());
        }
        if has_date {
            signals.push("date_time".to_string());
        }

        // Check for personal context patterns
        let personal_patterns = vec![
            r"\bmy (wife|husband|son|daughter|child|kids?|parent|mother|father|mom|dad|brother|sister)\b",
            r"\bmy (dog|cat|pet|car|house|home|apartment|office|job|work|company|business)\b",
            r"\bmy (flight|trip|vacation|appointment|meeting|interview|exam|test)\b",
            r"\bi (have|own|work at|live in|study at|attend|going to|planning to)\b",
            r"\b(i'm|i am) (a |an |the )?(student|teacher|doctor|engineer|developer|designer|manager)\b",
        ];

        for pattern_str in personal_patterns {
            if let Ok(pattern) = Regex::new(pattern_str) {
                if pattern.is_match(&question_lower) {
                    signals.push("personal_context".to_string());
                    break;
                }
            }
        }

        // Calculate confidence based on signals
        let mut confidence: f32 = 0.0;
        if has_possessive {
            confidence += 0.3;
        }
        if has_person || has_location || has_org {
            confidence += 0.4;
        }
        if signals.contains(&"personal_context".to_string()) {
            confidence += 0.3;
        }

        // Must have possessive + something else, OR strong personal context
        let has_info = (has_possessive && signals.len() >= 2)
            || (signals.contains(&"personal_context".to_string()) && signals.len() >= 2);

        MemoryWorthinessCheck {
            has_info,
            confidence: confidence.min(1.0),
            signals,
        }
    }

    /// Extract specific facts from a question
    /// Matches Python's extract_facts() (lines 102-269)
    pub fn extract_facts(&self, question: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        // Extract possession patterns with named entities
        // Pattern: "my [entity] [name]"
        facts.extend(self.extract_possession_facts(question));

        // Extract event organization patterns (BEFORE travel to avoid false occupation matches)
        if let Some(fact) = self.extract_event_facts(question) {
            facts.push(fact);
        }

        // Extract travel/appointment information
        facts.extend(self.extract_travel_facts(question));

        // Extract occupation/role information
        if let Some(fact) = self.extract_occupation_fact(question) {
            facts.push(fact);
        }

        // Extract family relationships
        facts.extend(self.extract_family_facts(question));

        facts
    }

    /// Extract "my [entity] [name]" patterns using BERT NER
    /// Uses ML-based entity extraction to identify proper nouns and their relationships
    fn extract_possession_facts(&self, question: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        let words: Vec<&str> = question.split_whitespace().collect();

        // Extract named entities using BERT NER
        let entities = self.extract_entities(question);

        // Common entity type keywords (for context, not for filtering entities)
        let entity_type_keywords = vec![
            "dog", "puppy", "cat", "kitten", "pet", "car", "wife", "husband",
            "son", "daughter", "child", "friend", "colleague", "partner",
            "flight", "trip", "house", "apartment", "job", "company",
            "retriever", "labrador", "poodle", "beagle", "golden"
        ];

        // Process possessive patterns with NER entities
        for (i, word) in words.iter().enumerate() {
            let word_lower = word.to_lowercase();
            let is_possessive = word_lower == "my" || word_lower == "our";

            // Also check for third-person possession: "She has", "He has", "[Name] has"
            let is_third_person = i + 1 < words.len()
                && (word_lower == "she" || word_lower == "he"
                    || word.chars().next().is_some_and(|c| c.is_uppercase()))
                && words[i + 1].to_lowercase() == "has";

            if is_possessive || is_third_person {
                let owner = if is_third_person {
                    // Check if the owner is a PERSON entity from NER
                    entities.iter()
                        .find(|e| e.label == "PER" && e.word.eq_ignore_ascii_case(word))
                        .map(|e| e.word.clone())
                        .or_else(|| {
                            // If pronoun, try to find the person name in context
                            if word_lower == "she" || word_lower == "he" {
                                entities.iter()
                                    .filter(|e| e.label == "PER")
                                    .last()
                                    .map(|e| e.word.clone())
                            } else {
                                Some(word.to_string())
                            }
                        })
                } else {
                    None
                };

                let start_idx = if is_third_person { i + 2 } else { i + 1 };

                // Look for entity type and name using both keywords and NER
                let mut entity_type: Option<String> = None;
                let mut entity_name: Option<String> = None;

                #[allow(clippy::needless_range_loop)]
                for j in start_idx..std::cmp::min(start_idx + 8, words.len()) {
                    let next_word = words[j];
                    let next_lower = next_word.to_lowercase();

                    // Skip common words
                    if matches!(next_lower.as_str(), "a" | "an" | "the" | "new" | "got") {
                        continue;
                    }

                    // Check if it's an entity type keyword
                    if entity_type_keywords.contains(&next_lower.as_str()) {
                        entity_type = Some(next_lower.clone());
                    }

                    // Check for "named" keyword followed by a PERSON entity from NER
                    if next_lower == "named" && j + 1 < words.len() {
                        let name_word = words[j + 1];
                        // Use NER to validate it's a PERSON
                        if let Some(person_entity) = entities.iter()
                            .find(|e| e.label == "PER" && e.word.eq_ignore_ascii_case(name_word))
                        {
                            entity_name = Some(person_entity.word.clone());
                            break;
                        }
                    }

                    // Check if NER identified this as a PERSON entity (proper noun)
                    if let Some(person_entity) = entities.iter()
                        .find(|e| e.label == "PER" && e.word.eq_ignore_ascii_case(next_word))
                    {
                        if entity_type.is_some() && entity_name.is_none() {
                            entity_name = Some(person_entity.word.clone());
                            break;
                        }
                    }
                }

                // Create fact if we found both type and name
                if let (Some(etype), Some(ename)) = (entity_type, entity_name) {
                    let namespace = self.determine_namespace(&etype);
                    let content = if let Some(ref owner_name) = owner {
                        format!("{} has a {} named {}", owner_name, etype, ename)
                    } else {
                        format!("User has a {} named {}", etype, ename)
                    };

                    let first_person = if owner.is_some() {
                        content.clone()  // Keep third-person as is
                    } else {
                        self.to_first_person(&content)
                    };

                    let fact = ExtractedFact {
                        content: content.clone(),
                        first_person,
                        title: format!("{}: {}", capitalize(&etype), ename),
                        namespace,
                        tags: vec![etype.clone(), "personal".to_string()],
                        confidence: 0.9,  // Higher confidence with NER validation
                    };
                    facts.push(fact);
                }
            }
        }

        facts
    }

    /// Extract event organization patterns
    fn extract_event_facts(&self, question: &str) -> Option<ExtractedFact> {
        let event_patterns = vec![
            // "organizing a conference on nuclear disarmament in Geneva"
            (r"(?:i'm |i am )?(?:organizing|hosting|running|leading)\s+(?:a |an |the )?(\w+(?:\s+\w+)?)\s+(?:on|about)\s+([^,]+?)\s+in\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)", "organizing", 3),
            // "organizing a conference in Geneva"
            (r"(?:i'm |i am )?(?:organizing|hosting|running|leading)\s+(?:a |an |the )?(\w+(?:\s+\w+)?)\s+in\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)", "organizing", 2),
            // "attending/speaking at a conference on X in Y"
            (r"(?:i'm |i am )?(?:attending|speaking at|presenting at)\s+(?:a |an |the )?(\w+(?:\s+\w+)?)\s+(?:on|about)\s+([^,]+?)\s+in\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)", "attending", 3),
            // "attending/speaking at a conference in Y"
            (r"(?:i'm |i am )?(?:attending|speaking at|presenting at)\s+(?:a |an |the )?(\w+(?:\s+\w+)?)\s+in\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)", "attending", 2),
        ];

        for (pattern_str, action, group_count) in event_patterns {
            if let Ok(pattern) = Regex::new(pattern_str) {
                if let Some(captures) = pattern.captures(question) {
                    let (content, title, tags) = if group_count == 3 {
                        let event_type = captures.get(1)?.as_str();
                        let topic = captures.get(2)?.as_str();
                        let location = captures.get(3)?.as_str();
                        (
                            format!("User is {} a {} on {} in {}", action, event_type, topic, location),
                            format!("Event: {} in {}", capitalize(topic), capitalize(location)),
                            vec![action.to_string(), event_type.to_lowercase(), topic.to_lowercase(), location.to_lowercase()],
                        )
                    } else {
                        let event_type = captures.get(1)?.as_str();
                        let location = captures.get(2)?.as_str();
                        (
                            format!("User is {} a {} in {}", action, event_type, location),
                            format!("Event: {} in {}", capitalize(event_type), capitalize(location)),
                            vec![action.to_string(), event_type.to_lowercase(), location.to_lowercase()],
                        )
                    };

                    return Some(ExtractedFact {
                        content: content.clone(),
                        first_person: self.to_first_person(&content),
                        title,
                        namespace: "events".to_string(),
                        tags,
                        confidence: 0.85,
                    });
                }
            }
        }

        None
    }

    /// Extract travel/appointment information
    fn extract_travel_facts(&self, question: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();

        let travel_patterns = vec![
            (r"(?:my )?flight to ([A-Za-z]+(?:\s+[A-Za-z]+)*)", "flight", "travel"),
            (r"(?:my )?trip to ([A-Za-z]+(?:\s+[A-Za-z]+)*)", "trip", "travel"),
            (r"(?:i'm |i am )?(?:flying|going|traveling|travelling) to ([A-Za-z]+(?:\s+[A-Za-z]+)*)", "travel_plan", "travel"),
            (r"visiting ([A-Za-z]+(?:\s+[A-Za-z]+)*)", "visit", "travel"),
            (r"my appointment (?:at|with|in) ([A-Za-z]+(?:\s+[A-Za-z]+)*)", "appointment", "personal"),
        ];

        for (pattern_str, fact_type, namespace) in travel_patterns {
            if let Ok(pattern) = Regex::new(pattern_str) {
                if let Some(captures) = pattern.captures(question) {
                    if let Some(location_match) = captures.get(1) {
                        let location = capitalize(location_match.as_str());
                        let content = if fact_type.ends_with("_plan") {
                            format!("User is {}ing to {}", fact_type.replace('_', " "), location)
                        } else {
                            format!("User has {} to {}", fact_type, location)
                        };

                        let fact = ExtractedFact {
                            content: content.clone(),
                            first_person: self.to_first_person(&content),
                            title: format!("{}: {}", capitalize(&fact_type.replace('_', " ")), location),
                            namespace: namespace.to_string(),
                            tags: vec![fact_type.to_string(), location.to_lowercase()],
                            confidence: 0.80,
                        };
                        facts.push(fact);
                    }
                }
            }
        }

        facts
    }

    /// Extract occupation/role information
    fn extract_occupation_fact(&self, question: &str) -> Option<ExtractedFact> {
        let occupation_pattern = r"i(?:'m| am) (?:a |an |the )?([a-z]+(?:\s+[a-z]+)?)\b";

        if let Ok(pattern) = Regex::new(occupation_pattern) {
            if let Some(captures) = pattern.captures(&question.to_lowercase()) {
                if let Some(occupation_match) = captures.get(1) {
                    let occupation = occupation_match.as_str();

                    // Filter common words AND action verbs (not occupations)
                    let excluded_words = vec![
                        "a", "an", "the", "my", "going", "have", "has",
                        "flying", "traveling", "travelling", "visiting", "planning",
                        "working", "living", "staying", "moving", "heading",
                        "organizing", "hosting", "running", "leading", "attending",
                        "speaking", "presenting"
                    ];

                    let excluded_suffixes = ["ing to", "ed to", "ing a", "ing an"];

                    if !excluded_words.contains(&occupation)
                        && !excluded_suffixes.iter().any(|suffix| occupation.ends_with(suffix))
                    {
                        let content = format!("User is a {}", occupation);
                        return Some(ExtractedFact {
                            content: content.clone(),
                            first_person: self.to_first_person(&content),
                            title: format!("Occupation: {}", capitalize(occupation)),
                            namespace: "personal".to_string(),
                            tags: vec!["occupation".to_string(), occupation.to_string()],
                            confidence: 0.7,
                        });
                    }
                }
            }
        }

        None
    }

    /// Extract family relationships
    fn extract_family_facts(&self, question: &str) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        let entities = self.extract_entities(question);

        let family_patterns = vec![
            (r"my (wife|husband)", "spouse"),
            (r"my (son|daughter)", "child"),
            (r"my (mother|mom|father|dad)", "parent"),
            (r"my (brother|sister)", "sibling"),
        ];

        for (pattern_str, _relation_type) in family_patterns {
            if let Ok(pattern) = Regex::new(pattern_str) {
                if let Some(captures) = pattern.captures(&question.to_lowercase()) {
                    if let Some(relation_match) = captures.get(1) {
                        let relation = relation_match.as_str();

                        // Try to find PERSON entity nearby
                        let person_entities: Vec<&Entity> = entities.iter()
                            .filter(|e| e.label == "PER")
                            .collect();

                        if let Some(person) = person_entities.first() {
                            let content = format!("User's {} is named {}", relation, person.word);
                            let fact = ExtractedFact {
                                content: content.clone(),
                                first_person: self.to_first_person(&content),
                                title: format!("{}: {}", capitalize(relation), person.word),
                                namespace: "personal".to_string(),
                                tags: vec!["family".to_string(), relation.to_string(), person.word.to_lowercase()],
                                confidence: 0.8,
                            };
                            facts.push(fact);
                        }
                    }
                }
            }
        }

        facts
    }

    /// Determine appropriate namespace for entity type
    /// Matches Python's _determine_namespace() (lines 271-282)
    fn determine_namespace(&self, entity_type: &str) -> String {
        match entity_type {
            "dog" | "cat" | "pet" | "house" | "apartment" | "car" => "personal".to_string(),
            "flight" | "trip" | "vacation" => "travel".to_string(),
            "job" | "company" | "office" | "work" => "work".to_string(),
            "appointment" | "meeting" => "events".to_string(),
            _ => "personal".to_string(),
        }
    }
}

impl Default for QuestionMemoryExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Capitalize first letter of a string
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_worthiness_possessive() {
        let extractor = QuestionMemoryExtractor::new();
        let result = extractor.contains_memory_worthy_info("What should I bring to my flight to Paris?");
        assert!(result.has_info);
        assert!(result.confidence > 0.5);
        assert!(result.signals.contains(&"possessive_pronoun".to_string()));
    }

    #[test]
    fn test_memory_worthiness_personal_context() {
        let extractor = QuestionMemoryExtractor::new();
        let result = extractor.contains_memory_worthy_info("I have a dog named Max");
        assert!(result.has_info);
        assert!(result.signals.contains(&"personal_context".to_string()));
    }

    #[test]
    fn test_extract_event_organizing() {
        let extractor = QuestionMemoryExtractor::new();
        let facts = extractor.extract_facts("I'm organizing a conference on nuclear disarmament in Geneva");
        assert!(!facts.is_empty());
        let fact = &facts[0];
        assert!(fact.content.contains("organizing"));
        assert!(fact.content.contains("nuclear disarmament"));
        assert!(fact.content.contains("Geneva"));
        assert_eq!(fact.namespace, "events");
    }

    #[test]
    fn test_extract_travel() {
        let extractor = QuestionMemoryExtractor::new();
        let facts = extractor.extract_facts("What should I pack for my trip to Tokyo?");
        assert!(!facts.is_empty());
        let fact = &facts[0];
        assert!(fact.content.contains("Tokyo"));
        assert_eq!(fact.namespace, "travel");
    }

    #[test]
    fn test_extract_occupation() {
        let extractor = QuestionMemoryExtractor::new();
        let facts = extractor.extract_facts("I'm a scientist at CERN");
        assert!(!facts.is_empty());
        let fact = &facts[0];
        assert!(fact.content.contains("scientist"));
        assert_eq!(fact.namespace, "personal");
    }

    #[test]
    fn test_occupation_filter_excluded_words() {
        let extractor = QuestionMemoryExtractor::new();
        let facts = extractor.extract_facts("I'm flying to Paris");
        // Should NOT extract "flying" as occupation
        assert!(facts.iter().all(|f| f.namespace != "personal" || !f.tags.contains(&"occupation".to_string())));
    }
}
