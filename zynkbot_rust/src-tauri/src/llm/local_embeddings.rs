use super::LLMError;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::sync::Mutex;
use std::path::PathBuf;
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

/// Global embedding model instance (loaded once, reused for all requests)
static EMBEDDING_MODEL: Lazy<Mutex<Option<EmbeddingModel>>> = Lazy::new(|| Mutex::new(None));

/// Embedding model using Candle
struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingModel {
    /// Load the all-MiniLM-L6-v2 model from models/system directory
    fn load() -> Result<Self, LLMError> {
        println!("[Candle Embeddings] Loading all-MiniLM-L6-v2 model from models/system/...");

        // Always use CPU for embeddings (prevents CUDA context corruption)
        // Embeddings run AFTER LLM response is sent to user (background processing)
        // CPU speed is sufficient (~100-200ms) and safer than risking CUDA corruption
        // Same Candle BERT tokenizer that crashed safety classifier on large inputs
        let device = Device::Cpu;
        println!("[Candle Embeddings] Using CPU for embeddings (reliable background processing)");

        // Get model directory path
        let model_dir = PathBuf::from("models/system/all-MiniLM-L6-v2");

        if !model_dir.exists() {
            return Err(LLMError::RequestFailed(
                format!("Model directory not found: {}. Please run the installation script.",
                    model_dir.display())
            ));
        }

        // Load model files from local directory
        let config_path = model_dir.join("config.json");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weights_path = model_dir.join("model.safetensors");

        if !config_path.exists() || !tokenizer_path.exists() || !weights_path.exists() {
            return Err(LLMError::RequestFailed(
                format!("Model files incomplete in {}. Please run the installation script.",
                    model_dir.display())
            ));
        }

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to parse config: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to load tokenizer: {}", e)))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DTYPE, &device)
                .map_err(|e| LLMError::RequestFailed(format!("Failed to load weights: {}", e)))?
        };

        // Create model
        let model = BertModel::load(vb, &config)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create model: {}", e)))?;

        println!("[Candle Embeddings] ✅ Model loaded successfully from {} (384 dimensions)", model_dir.display());

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Generate embedding for text using mean pooling
    fn embed(&self, text: &str) -> Result<Vec<f32>, LLMError> {
        // Tokenize with max length truncation (BERT models typically use 512)
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| LLMError::RequestFailed(format!("Tokenization failed: {}", e)))?;

        // Safety check: truncate if too long (BERT max is 512 tokens)
        let tokens_slice = encoding.get_ids();
        let tokens = if tokens_slice.len() > 512 {
            println!("[WARN] Text too long ({} tokens), truncating to 512", tokens_slice.len());
            &tokens_slice[..512]
        } else {
            tokens_slice
        };

        let token_ids = Tensor::new(tokens, &self.device)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create token tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to unsqueeze: {}", e)))?;

        // Create attention mask (truncated to match tokens)
        let attention_mask_slice = encoding.get_attention_mask();
        let attention_mask = if attention_mask_slice.len() > 512 {
            &attention_mask_slice[..512]
        } else {
            attention_mask_slice
        };

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create attention mask: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to unsqueeze mask: {}", e)))?;

        // Run model - candle 0.9 requires attention mask and position IDs
        let embeddings = self
            .model
            .forward(&token_ids, &attention_mask_tensor, None)
            .map_err(|e| LLMError::RequestFailed(format!("Model forward pass failed: {}", e)))?;

        // Mean pooling - average all token embeddings
        let (_batch_size, _seq_len, _hidden_size) = embeddings.dims3()
            .map_err(|e| LLMError::RequestFailed(format!("Failed to get dimensions: {}", e)))?;

        // Expand attention mask to match embedding dimensions
        let attention_mask_expanded = attention_mask_tensor
            .unsqueeze(2)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to expand mask: {}", e)))?
            .expand(embeddings.shape())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to broadcast mask: {}", e)))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert dtype: {}", e)))?;

        // Apply mask and sum
        let masked_embeddings = (embeddings * attention_mask_expanded)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to apply mask: {}", e)))?;

        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to sum embeddings: {}", e)))?;

        // Count non-zero attention mask values and expand to match sum_embeddings shape
        let sum_mask = attention_mask_tensor
            .sum(1)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to sum mask: {}", e)))?
            .to_dtype(sum_embeddings.dtype())  // Convert to same dtype as embeddings (F32)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert sum_mask dtype: {}", e)))?
            .unsqueeze(1)  // Shape: [batch_size, 1]
            .map_err(|e| LLMError::RequestFailed(format!("Failed to unsqueeze sum_mask: {}", e)))?
            .expand(sum_embeddings.shape())  // Broadcast to [batch_size, hidden_size]
            .map_err(|e| LLMError::RequestFailed(format!("Failed to expand sum_mask: {}", e)))?;

        // Average: sum / count (element-wise division)
        let mean_pooled = (sum_embeddings / sum_mask)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to compute mean: {}", e)))?;

        // Convert to Vec<f32>
        let embedding_vec = mean_pooled
            .squeeze(0)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to squeeze: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert to vec: {}", e)))?;

        Ok(embedding_vec)
    }

    /// Generate embeddings for multiple texts at once using TRUE batch processing
    /// This is much faster than calling embed() in a loop because it processes all texts
    /// in a single forward pass through the model.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let encodings: Vec<_> = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| LLMError::RequestFailed(format!("Tokenization failed: {}", e)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Pad all sequences to max_len and create batch tensors
        let mut batch_token_ids = Vec::new();
        let mut batch_attention_masks = Vec::new();

        for encoding in &encodings {
            let mut tokens = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();

            // Pad to max_len
            while tokens.len() < max_len {
                tokens.push(0); // Pad token ID
                attention_mask.push(0); // Pad attention mask
            }

            batch_token_ids.extend(tokens);
            batch_attention_masks.extend(attention_mask);
        }

        // Create batch tensors [batch_size, seq_len]
        let batch_size = texts.len();
        let token_ids_tensor = Tensor::from_vec(batch_token_ids, (batch_size, max_len), &self.device)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create batch token tensor: {}", e)))?;

        let attention_mask_tensor = Tensor::from_vec(batch_attention_masks, (batch_size, max_len), &self.device)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to create batch attention mask: {}", e)))?;

        // Single forward pass for entire batch! This is where the speedup happens
        let embeddings = self
            .model
            .forward(&token_ids_tensor, &attention_mask_tensor, None)
            .map_err(|e| LLMError::RequestFailed(format!("Batch forward pass failed: {}", e)))?;

        // Mean pooling for each sequence in the batch
        // embeddings shape: [batch_size, seq_len, hidden_size]
        let (_batch_size, _seq_len, _hidden_size) = embeddings.dims3()
            .map_err(|e| LLMError::RequestFailed(format!("Failed to get dimensions: {}", e)))?;

        // Expand attention mask to match embedding dimensions
        let attention_mask_expanded = attention_mask_tensor
            .unsqueeze(2)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to expand mask: {}", e)))?
            .expand(embeddings.shape())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to broadcast mask: {}", e)))?
            .to_dtype(embeddings.dtype())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert dtype: {}", e)))?;

        // Apply mask and sum across sequence dimension (dim 1)
        let masked_embeddings = (embeddings * attention_mask_expanded)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to apply mask: {}", e)))?;

        let sum_embeddings = masked_embeddings
            .sum(1)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to sum embeddings: {}", e)))?;

        // Count non-zero attention mask values per sequence
        let sum_mask = attention_mask_tensor
            .sum(1)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to sum mask: {}", e)))?
            .to_dtype(sum_embeddings.dtype())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert sum_mask dtype: {}", e)))?
            .unsqueeze(1)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to unsqueeze sum_mask: {}", e)))?
            .expand(sum_embeddings.shape())
            .map_err(|e| LLMError::RequestFailed(format!("Failed to expand sum_mask: {}", e)))?;

        // Average: sum / count (element-wise division)
        let mean_pooled = (sum_embeddings / sum_mask)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to compute mean: {}", e)))?;

        // Convert to Vec<Vec<f32>> - one embedding per text
        let embeddings_2d = mean_pooled
            .to_vec2::<f32>()
            .map_err(|e| LLMError::RequestFailed(format!("Failed to convert to vec: {}", e)))?;

        Ok(embeddings_2d)
    }
}

/// Initialize the local embedding model
pub fn init_model() -> Result<(), LLMError> {
    let mut model_guard = EMBEDDING_MODEL.lock().unwrap_or_else(|e| e.into_inner());

    if model_guard.is_none() {
        *model_guard = Some(EmbeddingModel::load()?);
    }

    Ok(())
}

/// L2 normalize a vector (required for cosine similarity)
fn normalize_l2(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|x| x / norm).collect()
    }
}

/// Generate embedding using local all-MiniLM-L6-v2 model with Candle
/// Returns a 384-dimensional L2-normalized vector
pub fn generate_local_embedding(text: &str) -> Result<Vec<f32>, LLMError> {
    let start_total = std::time::Instant::now();

    // Ensure model is initialized
    let start_init = std::time::Instant::now();
    init_model()?;
    let init_duration = start_init.elapsed();

    if init_duration.as_millis() > 100 {
        println!("[⏱️ PERF] Model initialization took: {:.2}s (WARNING: Model may not be cached!)", init_duration.as_secs_f32());
    }

    let model_guard = EMBEDDING_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let model = model_guard
        .as_ref()
        .ok_or_else(|| LLMError::RequestFailed("Model not initialized".to_string()))?;

    let start_embed = std::time::Instant::now();
    let embedding = model.embed(text)?;
    let embed_duration = start_embed.elapsed();

    // L2 normalize for cosine similarity
    let normalized = normalize_l2(&embedding);

    let total_duration = start_total.elapsed();
    println!("[⏱️ PERF] Embedding generation: {:.3}s | Total (with init): {:.3}s | Dimension: {}",
        embed_duration.as_secs_f32(),
        total_duration.as_secs_f32(),
        normalized.len()
    );

    Ok(normalized)
}

/// Generate embeddings for multiple texts at once (TRUE batch processing)
/// This is 10-20x faster than calling generate_local_embedding() in a loop
/// because it processes all texts in a single forward pass through the model.
///
/// # Arguments
/// * `texts` - Vector of strings to generate embeddings for
/// * `batch_size` - Optional batch size (default: 32). Larger batches are faster but use more memory.
///
/// # Example
/// ```
/// let texts = vec!["Hello world".to_string(), "How are you?".to_string()];
/// let embeddings = generate_local_embeddings_batch(texts, Some(32))?;
/// // embeddings[0] = embedding for "Hello world"
/// // embeddings[1] = embedding for "How are you?"
/// ```
pub fn generate_local_embeddings_batch(texts: Vec<String>, batch_size: Option<usize>) -> Result<Vec<Vec<f32>>, LLMError> {
    // Ensure model is initialized
    init_model()?;

    let model_guard = EMBEDDING_MODEL.lock().unwrap_or_else(|e| e.into_inner());
    let model = model_guard
        .as_ref()
        .ok_or_else(|| LLMError::RequestFailed("Model not initialized".to_string()))?;

    let batch_size = batch_size.unwrap_or(32); // Default batch size
    let mut all_embeddings = Vec::new();

    // Process in batches to avoid memory issues
    for chunk in texts.chunks(batch_size) {
        let chunk_vec: Vec<String> = chunk.to_vec();
        let chunk_embeddings = model.embed_batch(&chunk_vec)?;

        // L2 normalize each embedding for cosine similarity
        let normalized_embeddings: Vec<Vec<f32>> = chunk_embeddings.iter()
            .map(|emb| normalize_l2(emb))
            .collect();

        all_embeddings.extend(normalized_embeddings);
    }

    Ok(all_embeddings)
}

/// Calculate cosine similarity between two embedding vectors
/// Returns a value between -1 and 1, where 1 means identical, 0 means orthogonal, -1 means opposite
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    // Clamp to [-1.0, 1.0] to handle floating-point precision errors
    let similarity = dot_product / (magnitude_a * magnitude_b);
    similarity.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let result = generate_local_embedding("Hello world");
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384, "all-MiniLM-L6-v2 should produce 384 dimensions");
    }
}
