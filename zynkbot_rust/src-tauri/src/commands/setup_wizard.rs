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

    // A partial file from an earlier attempt continues where it stopped (2026-10-27): a
    // 2.4 GB model on a phone is not downloaded twice because the app was closed once.
    let already: u64 = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    println!("[Setup] Starting download: {}{}", url, if already > 0 { format!(" (resuming after {} bytes)", already) } else { String::new() });

    let mut request = client.get(url);
    if already > 0 {
        request = request.header("Range", format!("bytes={}-", already));
    }
    let response = request
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

    // 206 means the server honoured the range; anything else is the whole file from the start.
    let resumed = already > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let total = response.content_length().unwrap_or(0) + if resumed { already } else { 0 };
    println!("[Setup] Download size: {} bytes{}", total, if resumed { ", resumed" } else { "" });
    let mut downloaded: u64 = if resumed { already } else { 0 };
    let mut stream = response.bytes_stream();

    let mut file = (if resumed {
        std::fs::OpenOptions::new().append(true).open(&tmp)
    } else {
        std::fs::File::create(&tmp)
    }).map_err(|e| {
            let msg = format!("Failed to open temp file: {}", e);
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

pub const ZB_MODEL_ID: &str = "zb-qwen3-4b";
pub const ZB_MODEL_URL: &str = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_0.gguf";
pub const ZB_MODEL_FILE: &str = "Qwen3-4B-Instruct-2507-Q4_0.gguf";
pub const ZB_MODEL_BYTES: u64 = 2_375_773_280;
pub const ZB_MODEL_SHA256: &str = "e0ba675d86ab277c61701c6793659b2ae801d95e3be791464c321e6fbf613be2";

/// The whole file hashed after the download; a mismatch removes the file and says so.
fn verify_sha256(path: &std::path::Path, expected: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open the downloaded file: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("Cannot read the downloaded file: {}", e))?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let got = hex::encode(hasher.finalize());
    if got != expected {
        let _ = std::fs::remove_file(path);
        return Err(format!("The downloaded file did not match its checksum (got {}…, expected {}…); it was removed. Try the download again.", &got[..12], &expected[..12]));
    }
    println!("[Setup] Checksum verified: {}", path.display());
    Ok(())
}

/// Whether the ZB model, or any chat model, is on this device, and how much of a
/// partial download is already here. The page offers the download from this.
#[tauri::command]
pub async fn zb_model_status() -> Result<serde_json::Value, String> {
    let user_dir = crate::db::get_models_dir().join("user");
    let zb = user_dir.join(ZB_MODEL_FILE);
    let partial = std::fs::metadata(zb.with_extension("tmp")).map(|m| m.len()).unwrap_or(0);
    let any_model = std::fs::read_dir(&user_dir)
        .map(|rd| rd.flatten().any(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false)))
        .unwrap_or(false);
    Ok(serde_json::json!({
        "model_id": ZB_MODEL_ID,
        "file": ZB_MODEL_FILE,
        "bytes": ZB_MODEL_BYTES,
        "present": zb.exists(),
        "any_model": any_model,
        "partial_bytes": partial,
    }))
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

    let (url, filename, sha256): (&str, &str, Option<&str>) = match model_id.as_str() {
        // The ZB model (Matt, 2026-10-23): the phone's model, Q4_0 so the Snapdragon GPU and
        // NPU paths can use the same file; the exact unsloth file, a Q4_K_M from elsewhere
        // scored far lower and read prompts 40% slower. 2.4 GB; checksum from the repo.
        "zb-qwen3-4b" => (ZB_MODEL_URL, ZB_MODEL_FILE, Some(ZB_MODEL_SHA256)),
        "qwen3-8b" => (
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
            None,
        ),
        "deepseek-r1-8b" => (
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            None,
        ),
        "llama-lexi-8b" => (
            "https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf",
            "Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf",
            None,
        ),
        _ => return Err(format!("Unknown model id: {}", model_id)),
    };

    let dest = user_models_dir.join(filename);
    download_file(&app, url, &dest, Some("setup:llm_progress")).await?;
    if let Some(expected) = sha256 {
        app.emit("setup:llm_progress", serde_json::json!({ "percent": 100, "downloaded": 0, "total": 0, "verifying": true })).ok();
        verify_sha256(&dest, expected)?;
    }
    app.emit("setup:llm_complete", serde_json::json!({ "model_id": model_id })).ok();

    Ok(())
}

#[cfg(test)]
mod zb_model_tests {
    use super::*;

    #[test]
    fn a_wrong_checksum_removes_the_file_and_a_right_one_keeps_it() {
        let dir = std::env::temp_dir().join(format!("zynk-sha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        std::fs::write(&path, b"not a model").unwrap();
        // sha256("not a model")
        let right = "b8f0ec0a4a3d9c6c9e30ad68a35e7e1ad2f1f3f4a0e1f9a7a4d2f4b9bd1dd6c2";
        let computed = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(b"not a model"))
        };
        assert!(verify_sha256(&path, &computed).is_ok());
        assert!(path.exists());
        let err = verify_sha256(&path, right).unwrap_err();
        assert!(err.contains("did not match"), "{}", err);
        assert!(!path.exists(), "a file with the wrong checksum must not stay");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_zb_model_entry_is_the_exact_unsloth_file() {
        assert!(ZB_MODEL_URL.ends_with("/Qwen3-4B-Instruct-2507-Q4_0.gguf"));
        assert!(ZB_MODEL_URL.contains("unsloth/"));
        assert_eq!(ZB_MODEL_SHA256.len(), 64);
        assert!(ZB_MODEL_BYTES > 2_000_000_000);
    }
}
