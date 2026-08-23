//! Desktop dictation — both Vosk (offline) and OpenAI Whisper (cloud).
//!
//! Both engines share the same mic-capture pipeline via `cpal`, bypassing the
//! WebView `getUserMedia` problem on Linux Tauri. The engine picked at `start`
//! determines what happens on `stop`:
//!   - Vosk: feeds samples into a recognizer live, returns `final_result()`.
//!   - OpenAI: buffers raw i16 samples, encodes to WAV on stop, uploads.
//!
//! One recording at a time (global session mutex).

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use once_cell::sync::Lazy;
use vosk::{Model, Recognizer};

#[derive(Clone, Copy)]
pub enum Engine {
    Vosk,
    OpenAI,
}

static ACTIVE_SESSION: Lazy<Mutex<Option<SessionHandle>>> = Lazy::new(|| Mutex::new(None));
static CACHED_MODEL: Lazy<Mutex<Option<Arc<Model>>>> = Lazy::new(|| Mutex::new(None));

struct SessionHandle {
    stop_tx: Sender<()>,
    result_rx: Receiver<Result<String, String>>,
    thread: thread::JoinHandle<()>,
}

fn find_model_dir() -> Result<PathBuf, String> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidates = [
        PathBuf::from("./models/system/vosk"),
        PathBuf::from("./models/vosk"),
        PathBuf::from(manifest).join("models/system/vosk"),
        PathBuf::from(manifest).join("gen/android/app/src/main/assets/vosk-model"),
    ];
    for p in &candidates {
        if p.is_dir() && p.join("am/final.mdl").is_file() {
            return Ok(p.clone());
        }
    }
    Err(format!("Vosk model not found. Looked in: {:?}", candidates))
}

fn get_model() -> Result<Arc<Model>, String> {
    let mut guard = CACHED_MODEL
        .lock()
        .map_err(|e| format!("Model lock poisoned: {e}"))?;
    if let Some(m) = guard.as_ref() {
        return Ok(Arc::clone(m));
    }
    let path = find_model_dir()?;
    let path_str = path
        .to_str()
        .ok_or_else(|| "Model path is not valid UTF-8".to_string())?;
    println!("[Dictation] Loading Vosk model from {}", path_str);
    let model = Model::new(path_str)
        .ok_or_else(|| format!("Vosk failed to load model at {}", path_str))?;
    let arc = Arc::new(model);
    *guard = Some(Arc::clone(&arc));
    Ok(arc)
}

pub fn start(engine: Engine) -> Result<(), String> {
    let mut guard = ACTIVE_SESSION
        .lock()
        .map_err(|e| format!("Session lock poisoned: {e}"))?;
    if guard.is_some() {
        return Err("A recording is already in progress".to_string());
    }

    // Vosk: preload model on the caller thread so load errors surface here.
    let model = match engine {
        Engine::Vosk => Some(get_model()?),
        Engine::OpenAI => None,
    };

    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let (result_tx, result_rx) = mpsc::channel::<Result<String, String>>();

    let thread = thread::spawn(move || {
        run_audio_thread(engine, model, ready_tx, stop_rx, result_tx);
    });

    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let _ = thread.join();
            return Err(e);
        }
        Err(e) => return Err(format!("Audio thread setup channel closed: {e}")),
    }

    *guard = Some(SessionHandle {
        stop_tx,
        result_rx,
        thread,
    });
    Ok(())
}

pub fn stop() -> Result<String, String> {
    let session = {
        let mut guard = ACTIVE_SESSION
            .lock()
            .map_err(|e| format!("Session lock poisoned: {e}"))?;
        guard.take()
    };
    let session = session.ok_or_else(|| "No recording in progress".to_string())?;

    session
        .stop_tx
        .send(())
        .map_err(|e| format!("Failed to signal stop: {e}"))?;
    let result = session
        .result_rx
        .recv()
        .map_err(|e| format!("Audio thread ended before returning a result: {e}"))?;
    let _ = session.thread.join();
    result
}

fn run_audio_thread(
    engine: Engine,
    model: Option<Arc<Model>>,
    ready_tx: Sender<Result<(), String>>,
    stop_rx: Receiver<()>,
    result_tx: Sender<Result<String, String>>,
) {
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            let _ = ready_tx.send(Err("No default input device found".to_string()));
            return;
        }
    };
    let config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("Failed to read device config: {e}")));
            return;
        }
    };

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let sample_format = config.sample_format();
    println!(
        "[Dictation] Input device: rate={}Hz, channels={}, format={:?}",
        sample_rate, channels, sample_format
    );

    // Set up per-engine state that receives samples from the mic callback.
    let recognizer = match engine {
        Engine::Vosk => {
            let m = model.expect("Vosk engine requires preloaded model");
            match Recognizer::new(&m, sample_rate as f32) {
                Some(r) => Some(Arc::new(Mutex::new(r))),
                None => {
                    let _ = ready_tx.send(Err("Vosk failed to create recognizer".to_string()));
                    return;
                }
            }
        }
        Engine::OpenAI => None,
    };
    let openai_buffer: Option<Arc<Mutex<Vec<i16>>>> = match engine {
        Engine::OpenAI => Some(Arc::new(Mutex::new(Vec::new()))),
        Engine::Vosk => None,
    };

    let stream_config: cpal::StreamConfig = config.clone().into();
    let err_cb = |err| eprintln!("[Dictation] Stream error: {err}");

    let rec_cb = recognizer.clone();
    let buf_cb = openai_buffer.clone();

    let build_result = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let mono = downmix_f32_to_i16(data, channels);
                dispatch(&mono, &rec_cb, &buf_cb);
            },
            err_cb,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let mono = downmix_i16(data, channels);
                dispatch(&mono, &rec_cb, &buf_cb);
            },
            err_cb,
            None,
        ),
        other => {
            let _ = ready_tx.send(Err(format!("Unsupported sample format: {:?}", other)));
            return;
        }
    };

    let stream = match build_result {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("Failed to build input stream: {e}")));
            return;
        }
    };
    if let Err(e) = stream.play() {
        let _ = ready_tx.send(Err(format!("Failed to start stream: {e}")));
        return;
    }

    let _ = ready_tx.send(Ok(()));

    // Block until stop is signaled.
    let _ = stop_rx.recv();

    drop(stream);

    let text = match engine {
        Engine::Vosk => match recognizer.unwrap().lock() {
            Ok(mut r) => {
                let final_result = r.final_result();
                Ok(final_result
                    .single()
                    .map(|s| s.text.trim().to_string())
                    .unwrap_or_default())
            }
            Err(e) => Err(format!("Recognizer lock poisoned: {e}")),
        },
        Engine::OpenAI => {
            let buffer_arc = openai_buffer.expect("OpenAI engine has buffer");
            let pcm_result: Result<Vec<i16>, String> = match buffer_arc.lock() {
                Ok(guard) => Ok(guard.clone()),
                Err(e) => Err(format!("OpenAI buffer lock poisoned: {e}")),
            };
            match pcm_result {
                Ok(pcm) => transcribe_openai(&pcm, sample_rate),
                Err(e) => Err(e),
            }
        }
    };
    let _ = result_tx.send(text);
}

fn dispatch(
    mono: &[i16],
    recognizer: &Option<Arc<Mutex<Recognizer>>>,
    buffer: &Option<Arc<Mutex<Vec<i16>>>>,
) {
    if let Some(r) = recognizer {
        if let Ok(mut rec) = r.lock() {
            let _ = rec.accept_waveform(mono);
        }
    }
    if let Some(b) = buffer {
        if let Ok(mut buf) = b.lock() {
            buf.extend_from_slice(mono);
        }
    }
}

fn downmix_f32_to_i16(data: &[f32], channels: usize) -> Vec<i16> {
    if channels == 1 {
        data.iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect()
    } else {
        data.chunks(channels)
            .map(|frame| (frame[0].clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect()
    }
}

fn downmix_i16(data: &[i16], channels: usize) -> Vec<i16> {
    if channels == 1 {
        data.to_vec()
    } else {
        data.chunks(channels).map(|frame| frame[0]).collect()
    }
}

/// Encode PCM samples as a minimal RIFF/WAV blob and upload to OpenAI Whisper.
fn transcribe_openai(pcm: &[i16], sample_rate: u32) -> Result<String, String> {
    if pcm.is_empty() {
        return Ok(String::new());
    }
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY not set — add it to your .env file".to_string())?;

    let wav = encode_wav(pcm, sample_rate)?;

    // Blocking reqwest, since we're on a dedicated thread already.
    let client = reqwest::blocking::Client::new();
    let part = reqwest::blocking::multipart::Part::bytes(wav)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| format!("MIME error: {e}"))?;
    let form = reqwest::blocking::multipart::Form::new()
        .text("model", "whisper-1")
        .part("file", part);

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .map_err(|e| format!("OpenAI request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {body}"));
    }
    let json: serde_json::Value = response
        .json()
        .map_err(|e| format!("Failed to parse OpenAI response: {e}"))?;
    json["text"]
        .as_str()
        .map(|s| s.trim().to_string())
        .ok_or_else(|| "No text field in OpenAI response".to_string())
}

fn encode_wav(pcm: &[i16], sample_rate: u32) -> Result<Vec<u8>, String> {
    let byte_len = pcm.len() * 2;
    let mut out = Vec::with_capacity(44 + byte_len);
    // RIFF header
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + byte_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt chunk (PCM, mono, 16-bit)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(byte_len as u32).to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}
