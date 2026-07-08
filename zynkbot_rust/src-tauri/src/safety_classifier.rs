/// Safety Classification using Candle (pure Rust, no ONNX conflicts)
///
/// This replaces the ONNX-based moderation.rs to resolve Windows linking conflicts.
/// Uses TinyBERT for toxicity classification with Candle framework.
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::sync::Mutex;
use std::path::PathBuf;
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

/// Safety classification categories
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SafetyCategory {
    Toxic,
    SevereToxic,
    Obscene,
    Threat,
    Insult,
    IdentityHate,
    Violence,
    SelfHarm,
    Sexual,
    Safe,  // All scores below threshold
}

impl SafetyCategory {
    pub fn is_harmful(&self) -> bool {
        !matches!(self, Self::Safe)
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Toxic => "Toxic content",
            Self::SevereToxic => "Severely toxic content",
            Self::Obscene => "Obscene content",
            Self::Threat => "Threatening content",
            Self::Insult => "Insulting content",
            Self::IdentityHate => "Identity hate",
            Self::Violence => "Violent content",
            Self::SelfHarm => "Self-harm content",
            Self::Sexual => "Sexual content",
            Self::Safe => "Safe content",
        }
    }
}

/// Classification result with confidence scores
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SafetyResult {
    pub category: SafetyCategory,
    pub confidence: f32,
    pub all_scores: Vec<(SafetyCategory, f32)>,
}

impl SafetyResult {
    #[allow(dead_code)]
    pub fn is_safe(&self) -> bool {
        self.category == SafetyCategory::Safe
    }

    #[allow(dead_code)]
    pub fn max_harmful_score(&self) -> f32 {
        self.all_scores
            .iter()
            .filter(|(cat, _)| cat.is_harmful())
            .map(|(_, score)| *score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }
}

/// Global safety model instance (loaded once, reused)
static SAFETY_MODEL: Lazy<Mutex<Option<SafetyClassifier>>> = Lazy::new(|| Mutex::new(None));

/// Safety classifier using Candle + TinyBERT
#[allow(dead_code)]
struct SafetyClassifier {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pooler_weight: Tensor,  // Pooler dense layer (768 -> 768)
    pooler_bias: Tensor,    // Pooler bias
    classification_head: Tensor,  // Linear layer for classification (768 -> 6)
    classifier_bias: Tensor,      // Classifier bias
}

impl SafetyClassifier {
    /// Load TinyBERT safety model from models/system directory
    fn load() -> Result<Self, String> {
        println!("[Candle Safety] Loading TinyBERT safety classifier from models/system/...");

        // Always use CPU for safety classifier (prevents CUDA context corruption on large inputs)
        // GPU provides no meaningful speedup (safety checks complete in <50ms on CPU)
        // but can corrupt CUDA state when hitting memory limits, breaking main LLM inference
        let device = Device::Cpu;
        println!("[Candle Safety] Using CPU for safety classifier (reliable, <50ms per check)");

        // Get model directory path
        let model_dir = crate::db::get_app_data_dir().join("models/system/toxic-bert");

        if !model_dir.exists() {
            return Err(format!(
                "Model directory not found: {}. Please run the installation script.",
                model_dir.display()
            ));
        }

        // Load model files from local directory
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("model.safetensors");
        let vocab_path = model_dir.join("vocab.txt");

        if !config_path.exists() || !weights_path.exists() || !vocab_path.exists() {
            return Err(format!(
                "Model files incomplete in {}. Please run the installation script.",
                model_dir.display()
            ));
        }

        // Load config
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        // Build BERT tokenizer from vocab.txt
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::normalizers::{BertNormalizer, NormalizerWrapper};
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::processors::bert::BertProcessing;

        let vocab_path_str = vocab_path
            .to_str()
            .ok_or("Failed to convert vocab path to string")?;

        let wordpiece = WordPiece::from_file(vocab_path_str)
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|e| format!("Failed to build WordPiece: {}", e))?;

        let mut tokenizer = Tokenizer::new(wordpiece);
        tokenizer.with_normalizer(Some(NormalizerWrapper::BertNormalizer(BertNormalizer::default())));
        tokenizer.with_pre_tokenizer(Some(tokenizers::PreTokenizerWrapper::BertPreTokenizer(BertPreTokenizer)));
        tokenizer.with_post_processor(Some(tokenizers::PostProcessorWrapper::Bert(
            BertProcessing::new(
                ("[SEP]".to_string(), 102),
                ("[CLS]".to_string(), 101),
            )
        )));

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DTYPE, &device)
                .map_err(|e| format!("Failed to load weights: {}", e))?
        };

        // Create BERT model
        let model = BertModel::load(vb.pp("bert"), &config)
            .map_err(|e| format!("Failed to create model: {}", e))?;

        // Load pooler layer (dense + tanh activation)
        let pooler_weight = vb
            .get((config.hidden_size, config.hidden_size), "bert.pooler.dense.weight")
            .map_err(|e| format!("Failed to load pooler weight: {}", e))?;
        let pooler_bias = vb
            .get(config.hidden_size, "bert.pooler.dense.bias")
            .map_err(|e| format!("Failed to load pooler bias: {}", e))?;

        // Load classification head (toxic-bert has 6 output classes)
        let classification_head = vb
            .get((6, config.hidden_size), "classifier.weight")
            .map_err(|e| format!("Failed to load classifier head: {}", e))?;
        let classifier_bias = vb
            .get(6, "classifier.bias")
            .map_err(|e| format!("Failed to load classifier bias: {}", e))?;

        println!("[Candle Safety] ✅ Safety classifier loaded successfully from {}", model_dir.display());

        Ok(Self {
            model,
            tokenizer,
            device,
            pooler_weight,
            pooler_bias,
            classification_head,
            classifier_bias,
        })
    }

    /// Classify text for safety
    fn classify(&self, text: &str) -> Result<SafetyResult, String> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        let token_ids = Tensor::new(tokens, &self.device)
            .map_err(|e| format!("Failed to create token tensor: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Failed to unsqueeze: {}", e))?;

        // Create attention mask
        let attention_mask = encoding.get_attention_mask();
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| format!("Failed to create attention mask: {}", e))?
            .unsqueeze(0)
            .map_err(|e| format!("Failed to unsqueeze mask: {}", e))?;

        // Run BERT model
        let embeddings = self
            .model
            .forward(&token_ids, &attention_mask_tensor, None)
            .map_err(|e| format!("Model forward pass failed: {}", e))?;

        // Use [CLS] token embedding (first token) for classification
        let cls_embedding = embeddings
            .i((0, 0))
            .map_err(|e| format!("Failed to extract CLS token: {}", e))?  // Shape: [768]
            .unsqueeze(0)
            .map_err(|e| format!("Failed to unsqueeze CLS: {}", e))?;  // Shape: [1, 768]

        // Apply pooler layer: pooled = tanh(cls @ pooler_weight.T + pooler_bias)
        // [1, 768] @ [768, 768].T = [1, 768] + [768] = [1, 768]
        let pooled = cls_embedding
            .matmul(&self.pooler_weight.t().map_err(|e| format!("Failed to transpose pooler: {}", e))?)
            .map_err(|e| format!("Failed to apply pooler matmul: {}", e))?
            .broadcast_add(&self.pooler_bias)
            .map_err(|e| format!("Failed to add pooler bias: {}", e))?
            .tanh()
            .map_err(|e| format!("Failed to apply tanh: {}", e))?;  // Keep as [1, 768]

        // Apply classification head: logits = pooled @ classifier.T + classifier_bias
        // [1, 768] @ [6, 768].T = [1, 6] + [6] = [1, 6], then squeeze to [6]
        let logits = pooled
            .matmul(&self.classification_head.t().map_err(|e| format!("Failed to transpose classifier: {}", e))?)
            .map_err(|e| format!("Failed to apply classification head: {}", e))?
            .broadcast_add(&self.classifier_bias)
            .map_err(|e| format!("Failed to add classifier bias: {}", e))?
            .squeeze(0)
            .map_err(|e| format!("Failed to squeeze logits: {}", e))?;  // [1, 6] -> [6]

        // Convert to Vec<f32>
        let logits_vec = logits
            .to_vec1::<f32>()
            .map_err(|e| format!("Failed to convert logits: {}", e))?;

        // Apply sigmoid (toxic-bert uses sigmoid, not softmax)
        let scores: Vec<f32> = logits_vec.iter().map(|&x| sigmoid(x)).collect();

        // Map scores to categories (toxic-bert has 6 classes)
        let categories = [
            SafetyCategory::Toxic,
            SafetyCategory::SevereToxic,
            SafetyCategory::Obscene,
            SafetyCategory::Threat,
            SafetyCategory::Insult,
            SafetyCategory::IdentityHate,
        ];

        let all_scores: Vec<(SafetyCategory, f32)> = categories
            .iter()
            .zip(scores.iter())
            .map(|(cat, score)| (*cat, *score))
            .collect();

        // Find highest score
        let (max_category, max_score) = all_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(cat, score)| (*cat, *score))
            .unwrap_or((SafetyCategory::Safe, 0.0));

        // Determine final category (Safe if all scores below threshold)
        let category = if max_score < 0.3 {
            SafetyCategory::Safe
        } else {
            max_category
        };

        Ok(SafetyResult {
            category,
            confidence: max_score,
            all_scores,
        })
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Initialize the safety classifier (call once at startup)
pub fn initialize() -> Result<(), String> {
    let mut model_guard = SAFETY_MODEL.lock().unwrap_or_else(|e| e.into_inner());

    if model_guard.is_none() {
        let classifier = SafetyClassifier::load()?;
        *model_guard = Some(classifier);
        println!("[Candle Safety] ✅ Safety classifier initialized");
    }

    Ok(())
}

/// Classify text using the global safety model (auto-initializes if needed)
pub fn classify_text(text: &str) -> Result<SafetyResult, String> {
    // Auto-initialize if not already done
    {
        let model_guard = SAFETY_MODEL.lock().unwrap_or_else(|e| e.into_inner());
        if model_guard.is_none() {
            drop(model_guard); // Release lock before initializing
            println!("[Candle Safety] Auto-initializing safety classifier on first use...");
            initialize()?;
        }
    }

    let model_guard = SAFETY_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let model = model_guard
        .as_ref()
        .ok_or("Safety classifier failed to initialize")?;

    let result = model.classify(text)?;

    Ok(result)
}

/// Determine if content should be blocked based on containment mode
/// Matches Python implementation behavior (only blocks severe harm in Guardian mode)
pub fn should_block(text: &str, mode: &str) -> Result<(bool, Option<SafetyResult>), String> {
    let result = classify_text(text)?;

    let should_block = match mode.to_lowercase().as_str() {
        "witness" => false,  // Never block
        "sovereign" => false,  // Warn but don't block
        "guardian" | "default" => {
            // Python Guardian blocks: S1 (Violent Crimes), S3 (Sex Crimes), S4 (Child Exploitation), S11 (Self-Harm)
            // TinyBERT equivalent: Only block SevereToxic and Threat at high confidence
            // Allow: Toxic (mild), Obscene (adult content OK), Insult (OK), IdentityHate (handled elsewhere)
            match result.category {
                SafetyCategory::SevereToxic => result.confidence > 0.6,  // Violent crimes, severe harm
                SafetyCategory::Threat => result.confidence > 0.6,       // Violent threats, self-harm
                _ => false,  // Allow everything else (including Toxic, Obscene, Insult)
            }
        }
        "hipaa" => {
            // Same as guardian for general safety, but HIPAA-specific checks happen in containment.rs
            match result.category {
                SafetyCategory::SevereToxic => result.confidence > 0.6,
                SafetyCategory::Threat => result.confidence > 0.6,
                _ => false,
            }
        }
        "elder" => {
            // More lenient - only block very severe cases
            result.category.is_harmful() && result.confidence > 0.75
        }
        "child" => {
            // Child mode uses OpenAI API with toxic-bert fallback
            // This is a fallback only - block more aggressively
            result.category.is_harmful() && result.confidence > 0.4
        }
        _ => false,  // Unknown mode - don't block
    };

    Ok((should_block, Some(result)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.01);
        assert!(sigmoid(1.0) > 0.7 && sigmoid(1.0) < 0.75);
        assert!(sigmoid(-1.0) > 0.25 && sigmoid(-1.0) < 0.3);
    }

    #[test]
    fn test_safety_category_is_harmful() {
        assert!(SafetyCategory::Toxic.is_harmful());
        assert!(SafetyCategory::Violence.is_harmful());
        assert!(!SafetyCategory::Safe.is_harmful());
    }
}
