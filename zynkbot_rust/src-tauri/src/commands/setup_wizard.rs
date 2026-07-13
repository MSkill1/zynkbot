use tauri::{AppHandle, Emitter};
use std::path::PathBuf;
use futures::StreamExt;
use std::io::Write;

/// Returns true if first-run setup is needed (required system models missing from all locations)
#[tauri::command]
pub async fn check_needs_setup() -> bool {
    let required = [
        "system/all-MiniLM-L6-v2/model.safetensors",
        "system/bert-base-NER/model.safetensors",
        "system/toxic-bert/model.safetensors",
    ];

    // Check data dir (installed binary location)
    let data_models = crate::db::get_app_data_dir().join("models");
    if required.iter().all(|p| data_models.join(p).exists()) {
        return false;
    }

    // Check dev mode fallback (src-tauri/models/ relative to exe in target/)
    if let Ok(exe) = std::env::current_exe() {
        if exe.to_string_lossy().contains("target") {
            if let Some(exe_dir) = exe.parent() {
                let dev_models = exe_dir.parent()
                    .and_then(|p| p.parent())
                    .unwrap_or(exe_dir)
                    .join("models");
                if required.iter().all(|p| dev_models.join(p).exists()) {
                    return false;
                }
            }
        }
    }

    true
}

async fn download_file(
    app: &AppHandle,
    url: &str,
    dest: &PathBuf,
    progress_event: Option<&str>,
) -> Result<(), String> {
    if dest.exists() {
        if let Some(event) = progress_event {
            app.emit(event, serde_json::json!({ "percent": 100, "downloaded": 0, "total": 0 })).ok();
        }
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let tmp = dest.with_extension("tmp");

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    println!("[Setup] Starting download: {}", url);

    let response = client.get(url)
        .send()
        .await
        .map_err(|e| {
            let msg = format!("Request failed for {}: {}", url, e);
            eprintln!("[Setup] ERROR: {}", msg);
            msg
        })?;

    if !response.status().is_success() {
        let msg = format!("HTTP {} downloading {}", response.status(), url);
        eprintln!("[Setup] ERROR: {}", msg);
        return Err(msg);
    }

    let total = response.content_length().unwrap_or(0);
    println!("[Setup] Download size: {} bytes", total);
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| {
            let msg = format!("Failed to create temp file: {}", e);
            eprintln!("[Setup] ERROR: {}", msg);
            msg
        })?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            let msg = format!("Download error after {} bytes: {}", downloaded, e);
            eprintln!("[Setup] ERROR: {}", msg);
            msg
        })?;
        file.write_all(&chunk)
            .map_err(|e| {
                let msg = format!("Write error: {}", e);
                eprintln!("[Setup] ERROR: {}", msg);
                msg
            })?;
        downloaded += chunk.len() as u64;

        if let Some(event) = progress_event {
            let percent = if total > 0 { (downloaded * 100 / total) as u32 } else { 0 };
            app.emit(event, serde_json::json!({
                "percent": percent,
                "downloaded": downloaded,
                "total": total,
            })).ok();
        }
    }

    std::fs::rename(&tmp, dest)
        .map_err(|e| {
            let msg = format!("Failed to finalize download: {}", e);
            eprintln!("[Setup] ERROR: {}", msg);
            msg
        })?;

    println!("[Setup] Download complete: {}", dest.display());
    Ok(())
}

/// Download all three required system models, emitting per-model progress events
#[tauri::command]
pub async fn download_system_models(app: AppHandle) -> Result<(), String> {
    let models_dir = crate::db::get_models_dir();

    struct ModelSpec {
        id: &'static str,
        base_url: &'static str,
        subdir: &'static str,
        files: &'static [&'static str],
        large_file: &'static str,
    }

    let specs = [
        ModelSpec {
            id: "minilm",
            base_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main",
            subdir: "system/all-MiniLM-L6-v2",
            files: &["config.json", "tokenizer.json", "model.safetensors"],
            large_file: "model.safetensors",
        },
        ModelSpec {
            id: "bert-ner",
            base_url: "https://huggingface.co/dslim/bert-base-NER/resolve/main",
            subdir: "system/bert-base-NER",
            files: &["config.json", "vocab.txt", "model.safetensors"],
            large_file: "model.safetensors",
        },
        ModelSpec {
            id: "toxic-bert",
            base_url: "https://huggingface.co/unitary/toxic-bert/resolve/main",
            subdir: "system/toxic-bert",
            files: &["config.json", "vocab.txt", "model.safetensors"],
            large_file: "model.safetensors",
        },
    ];

    for spec in &specs {
        let model_dir = models_dir.join(spec.subdir);
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| format!("Failed to create model dir: {}", e))?;

        for &filename in spec.files {
            let url = format!("{}/{}", spec.base_url, filename);
            let dest = model_dir.join(filename);
            let event = if filename == spec.large_file {
                Some(format!("setup:progress:{}", spec.id))
            } else {
                None
            };
            download_file(&app, &url, &dest, event.as_deref()).await?;
        }

        app.emit(&format!("setup:complete:{}", spec.id), serde_json::json!({})).ok();
    }

    Ok(())
}

/// Download a single optional user LLM model
#[tauri::command]
pub async fn download_user_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let user_models_dir = crate::db::get_models_dir().join("user");
    std::fs::create_dir_all(&user_models_dir)
        .map_err(|e| format!("Failed to create user models dir: {}", e))?;

    let (url, filename): (&str, &str) = match model_id.as_str() {
        "qwen3-8b" => (
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
        ),
        "deepseek-r1-8b" => (
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        ),
        "llama-lexi-8b" => (
            "https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf",
            "Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf",
        ),
        _ => return Err(format!("Unknown model id: {}", model_id)),
    };

    let dest = user_models_dir.join(filename);
    download_file(&app, url, &dest, Some("setup:llm_progress")).await?;
    app.emit("setup:llm_complete", serde_json::json!({ "model_id": model_id })).ok();

    Ok(())
}
