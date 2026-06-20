/// Whisper STT (Speech-to-Text) using whisper.cpp via whisper-rs
/// Privacy-first local audio transcription
/// 10-20x faster than pure Rust implementation
///
/// Based on: https://github.com/tazz4843/whisper-rs

use super::LLMError;
use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Global Whisper context (loaded once, reused)
static WHISPER_CONTEXT: Lazy<Mutex<Option<WhisperContext>>> = Lazy::new(|| Mutex::new(None));

/// Model path - will be downloaded automatically if not present
fn get_model_path() -> PathBuf {
    // Use project directory to avoid path issues with whisper.cpp on Windows
    // (paths with spaces in AppData can cause issues with whisper.cpp on Windows)
    let project_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("models")
        .join("whisper");

    std::fs::create_dir_all(&project_dir).ok();

    project_dir.join("ggml-base.en.bin")
}

/// Download whisper model if not present
fn ensure_model_exists() -> Result<PathBuf, LLMError> {
    let model_path = get_model_path();

    if model_path.exists() {
        println!("[Whisper] Using cached model: {:?}", model_path);
        return Ok(model_path);
    }

    println!("[Whisper] Downloading whisper base.en model...");
    println!("[Whisper] This is a one-time download (~150MB)");

    // Download from Hugging Face
    let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";

    let response = reqwest::blocking::get(url)
        .map_err(|e| LLMError::RequestFailed(format!("Failed to download model: {}", e)))?;

    let bytes = response.bytes()
        .map_err(|e| LLMError::RequestFailed(format!("Failed to read model bytes: {}", e)))?;

    // Write to temporary file first, then rename (atomic operation on Windows)
    let temp_path = model_path.with_extension("tmp");
    std::fs::write(&temp_path, &bytes)
        .map_err(|e| LLMError::RequestFailed(format!("Failed to write temp model file: {}", e)))?;

    // Rename temp file to final path (ensures file is fully written)
    std::fs::rename(&temp_path, &model_path)
        .map_err(|e| LLMError::RequestFailed(format!("Failed to finalize model file: {}", e)))?;

    println!("[Whisper] ✅ Model downloaded successfully");

    Ok(model_path)
}

/// Initialize Whisper context (call once at startup)
pub fn init_model() -> Result<(), LLMError> {
    let mut context_guard = WHISPER_CONTEXT.lock().unwrap_or_else(|e| e.into_inner());

    if context_guard.is_none() {
        println!("[Whisper] Initializing whisper.cpp context...");

        let model_path = ensure_model_exists()?;

        let ctx_params = WhisperContextParameters::default();
        let model_path_str = model_path.to_str()
            .ok_or_else(|| LLMError::RequestFailed("Whisper model path contains non-UTF8 characters".to_string()))?;
        let ctx = WhisperContext::new_with_params(
            model_path_str,
            ctx_params
        ).map_err(|e| LLMError::RequestFailed(format!("Failed to create whisper context: {}", e)))?;

        *context_guard = Some(ctx);
        println!("[Whisper] ✅ Context initialized successfully");
    }

    Ok(())
}

/// Decode WAV file using hound library
fn load_wav_audio(path: &str) -> Result<Vec<f32>, LLMError> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| LLMError::RequestFailed(format!("Failed to open WAV file: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Whisper expects 16kHz mono
    if sample_rate != 16000 {
        return Err(LLMError::RequestFailed(
            format!("Expected 16kHz audio, got {}Hz", sample_rate)
        ));
    }

    // Read samples and normalize to f32 range [-1.0, 1.0]
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|e| LLMError::RequestFailed(format!("Failed to read WAV samples: {}", e)))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|e| LLMError::RequestFailed(format!("Failed to read WAV samples: {}", e)))?
        }
    };

    // Convert stereo to mono if needed
    let mono_samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|ch| (ch[0] + ch[1]) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok(mono_samples)
}

/// Transcribe audio file to text using local Whisper model
///
/// # Arguments
/// * `audio_path` - Path to WAV audio file (16kHz mono preferred)
///
/// # Returns
/// * Transcribed text string
pub fn transcribe_audio(audio_path: &str) -> Result<String, LLMError> {
    // Ensure model is initialized
    init_model()?;

    println!("[Whisper] Transcribing audio: {}", audio_path);

    // Load audio
    let audio_data = load_wav_audio(audio_path)?;
    println!("[Whisper] Loaded {} samples", audio_data.len());

    // Get context
    let mut context_guard = WHISPER_CONTEXT.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = context_guard
        .as_mut()
        .ok_or_else(|| LLMError::RequestFailed("Whisper context not initialized".to_string()))?;

    // Create state for this transcription
    let mut state = ctx.create_state()
        .map_err(|e| LLMError::RequestFailed(format!("Failed to create whisper state: {}", e)))?;

    // Configure parameters for English transcription
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // Language and settings
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // Run transcription
    state.full(params, &audio_data)
        .map_err(|e| LLMError::RequestFailed(format!("Transcription failed: {}", e)))?;

    // Extract transcribed text
    let num_segments = state.full_n_segments()
        .map_err(|e| LLMError::RequestFailed(format!("Failed to get segment count: {}", e)))?;

    let mut transcription = String::new();

    for i in 0..num_segments {
        let segment_text = state.full_get_segment_text(i)
            .map_err(|e| LLMError::RequestFailed(format!("Failed to get segment text: {}", e)))?;

        transcription.push_str(&segment_text);
        transcription.push(' ');
    }

    let result = transcription.trim().to_string();

    println!("[Whisper] ✅ Transcription complete: '{}' ({} chars)", result, result.len());

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires audio file and model download
    fn test_transcribe() {
        let result = transcribe_audio("test_audio.wav");
        assert!(result.is_ok());
    }
}
