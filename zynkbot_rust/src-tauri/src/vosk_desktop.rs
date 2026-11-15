//! Desktop dictation — both Vosk (offline) and OpenAI Whisper (cloud).
//!
//! Both engines share the same mic-capture pipeline via `cpal`, bypassing the
//! WebView `getUserMedia` problem on Linux Tauri. The engine picked at `start`
//! determines what happens on `stop`:
//!   - Vosk: feeds samples into a recognizer live. Each finished utterance (Vosk
//!     ends one at a pause) is kept, and `stop` returns all of them plus the last
//!     one. Until 2026-10-20 only `final_result()` was returned, which is the text
//!     of the current utterance alone — everything said before the last pause was
//!     lost. While recording, the text so far is handed to `on_live` (throttled)
//!     so the page can show the words as they are spoken.
//!   - OpenAI: streams the samples to OpenAI's Realtime transcription session
//!     (`openai_realtime`) so the words show while the user speaks, and keeps the
//!     raw i16 samples too. On stop the session's text is returned; if the session
//!     could not be opened (no network, DICTATION_STREAM=0, server error) the kept
//!     samples are encoded to WAV and uploaded in one go, as before 2026-10-28.
//!
//! One recording at a time (global session mutex).

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;
use once_cell::sync::Lazy;
use vosk::{DecodingState, Model, Recognizer};

/// Receives the text recognised so far, every time it changes (at most a few times a second).
pub type LiveSink = Box<dyn Fn(String) + Send + Sync + 'static>;

/// Called once, with the text so far, when a recording ends by itself (the timers below).
pub type EndSink = Box<dyn Fn(String) + Send + Sync + 'static>;

/// The mic button's own timers (2026-11-15). A mic left on while the user walked away
/// ran until Stop was tapped and then stopped slowly: the live session was asked for
/// the transcript of minutes of nothing, or a clip too large for the one-shot upload
/// went up. Once something has been said, no new words for `BUTTON_SILENCE` ends the
/// recording; nothing said at all for `BUTTON_NO_SPEECH` ends it; `BUTTON_CAP` ends it
/// regardless. The text so far is delivered as if Stop had been tapped. Hands-free on
/// Android has shorter timers of its own (Dictation.kt); these are for typing by voice,
/// where a person stops to think mid-sentence.
pub const BUTTON_SILENCE: Duration = Duration::from_secs(20);
pub const BUTTON_NO_SPEECH: Duration = Duration::from_secs(30);
pub const BUTTON_CAP: Duration = Duration::from_secs(180);

/// What the timers watch: when the recognised text last changed.
struct Activity {
    started: Instant,
    last_change: Option<Instant>,
    last_text: String,
}

/// Why a recording ends by itself, or None to carry on. `text_based` is false when no
/// words can arrive before stop (the OpenAI clip upload): then only the cap applies.
pub fn self_stop_reason(text_based: bool, since_start: Duration, since_change: Option<Duration>) -> Option<&'static str> {
    if text_based {
        match since_change {
            None if since_start >= BUTTON_NO_SPEECH => return Some("nothing said in 30 s"),
            Some(d) if d >= BUTTON_SILENCE => return Some("no new words for 20 s"),
            _ => {}
        }
    }
    if since_start >= BUTTON_CAP { Some("listening cap, 3 minutes") } else { None }
}

/// Vosk recogniser plus the utterances it has already finished.
struct VoskState {
    rec: Recognizer,
    committed: String,
    last_partial: String,
    last_emit: Instant,
}

fn join_text(a: &str, b: &str) -> String {
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{} {}", a, b),
    }
}

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
    // Packaged builds (2026-09-09): the model ships as a Tauri resource
    // (tauri.conf.json bundle.resources). Tauri puts resources at
    // <prefix>/lib/<ProductName>/ next to <prefix>/bin/<exe> on Linux (.deb, .rpm,
    // AppImage) and next to the .exe on Windows, so both are derived from the
    // executable's own location, the same way the rpath finds libvosk.so.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let candidates = [
        exe_dir.join("../lib/Zynkbot/vosk-model"),
        exe_dir.join("vosk-model"),
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

pub fn start(engine: Engine, on_live: Option<LiveSink>, on_ended: Option<EndSink>) -> Result<(), String> {
    let mut guard = ACTIVE_SESSION
        .lock()
        .map_err(|e| format!("Session lock poisoned: {e}"))?;
    if guard.as_ref().map_or(false, |s| !s.thread.is_finished()) {
        return Err("A recording is already in progress".to_string());
    }
    // A session that ended by itself is finished with; let it go.
    if let Some(old) = guard.take() {
        let _ = old.thread.join();
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
        run_audio_thread(engine, model, on_live, on_ended, ready_tx, stop_rx, result_tx);
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

    // A send error means the thread already ended by itself; its result is waiting.
    let _ = session.stop_tx.send(());
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
    on_live: Option<LiveSink>,
    on_ended: Option<EndSink>,
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
                Some(r) => Some(Arc::new(Mutex::new(VoskState {
                    rec: r,
                    committed: String::new(),
                    last_partial: String::new(),
                    last_emit: Instant::now(),
                }))),
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
    // Every change in the recognised text is noted for the timers, whether or not the
    // page asked for live text.
    let activity = Arc::new(Mutex::new(Activity { started: Instant::now(), last_change: None, last_text: String::new() }));
    let activity_sink = activity.clone();
    let on_live_tracked: LiveSink = Box::new(move |text: String| {
        if let Ok(mut a) = activity_sink.lock() {
            if text != a.last_text {
                if !text.trim().is_empty() {
                    a.last_change = Some(Instant::now());
                }
                a.last_text = text.clone();
            }
        }
        if let Some(cb) = &on_live {
            cb(text);
        }
    });
    let live_cb: Option<Arc<LiveSink>> = Some(Arc::new(on_live_tracked));
    let live_cb2 = live_cb.clone();

    // OpenAI: open the live session now; the socket connects in the background while
    // the mic is already running. None here means the one-shot upload runs on stop.
    let (mut feeder, live_session) = match engine {
        Engine::OpenAI => match crate::openai_realtime::Session::open(sample_rate, live_cb.clone()) {
            Ok((f, s)) => (Some(f), Some(s)),
            Err(e) => {
                println!("[Dictation] Live stream not used ({e}); the clip is uploaded on stop");
                (None, None)
            }
        },
        Engine::Vosk => (None, None),
    };
    let mut feeder2 = feeder.clone();

    let build_result = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let mono = downmix_f32_to_i16(data, channels);
                dispatch(&mono, &rec_cb, &buf_cb, &live_cb);
                if let Some(f) = feeder.as_mut() { f.push(&mono); }
            },
            err_cb,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let mono = downmix_i16(data, channels);
                dispatch(&mono, &rec_cb, &buf_cb, &live_cb2);
                if let Some(f) = feeder2.as_mut() { f.push(&mono); }
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

    // Block until stop is signaled, or until the mic button's own timers end it.
    let text_based = matches!(engine, Engine::Vosk) || live_session.is_some();
    let self_ended = loop {
        match stop_rx.recv_timeout(Duration::from_millis(250)) {
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let (since_start, since_change) = match activity.lock() {
                    Ok(a) => (a.started.elapsed(), a.last_change.map(|t| t.elapsed())),
                    Err(_) => (Duration::ZERO, None),
                };
                if let Some(reason) = self_stop_reason(text_based, since_start, since_change) {
                    println!("[Dictation] Ended by itself after {:.0}s: {}", since_start.as_secs_f32(), reason);
                    break true;
                }
            }
            _ => break false,
        }
    };

    drop(stream);

    let text = match engine {
        Engine::Vosk => match recognizer.unwrap().lock() {
            Ok(mut st) => {
                let last = st
                    .rec
                    .final_result()
                    .single()
                    .map(|s| s.text.trim().to_string())
                    .unwrap_or_default();
                Ok(join_text(&st.committed, &last))
            }
            Err(e) => Err(format!("Recognizer lock poisoned: {e}")),
        },
        Engine::OpenAI => {
            let buffer_arc = openai_buffer.expect("OpenAI engine has buffer");
            let pcm_result: Result<Vec<i16>, String> = match buffer_arc.lock() {
                Ok(guard) => Ok(guard.clone()),
                Err(e) => Err(format!("OpenAI buffer lock poisoned: {e}")),
            };
            // The live session's text is the answer when it worked; otherwise the
            // kept clip goes up the old way, so a dropped socket still yields text.
            let streamed = live_session.map(|s| s.finish());
            match (streamed, pcm_result) {
                (Some(Ok(text)), _) => Ok(text),
                (Some(Err(e)), Ok(pcm)) => {
                    eprintln!("[Dictation] Live stream failed ({e}); uploading the clip instead");
                    transcribe_openai(&pcm, sample_rate)
                }
                (None, Ok(pcm)) => transcribe_openai(&pcm, sample_rate),
                (_, Err(e)) => Err(e),
            }
        }
    };
    if self_ended {
        if let Some(cb) = &on_ended {
            let delivered = match &text {
                Ok(t) => t.clone(),
                Err(e) => {
                    eprintln!("[Dictation] Ended by itself but gave no text: {e}");
                    String::new()
                }
            };
            cb(delivered);
        }
    }
    let _ = result_tx.send(text);
}

#[cfg(test)]
mod self_stop_tests {
    use super::*;
    const S: fn(u64) -> Duration = Duration::from_secs;

    #[test]
    fn nothing_said_ends_at_thirty_seconds() {
        assert_eq!(self_stop_reason(true, S(29), None), None);
        assert_eq!(self_stop_reason(true, S(30), None), Some("nothing said in 30 s"));
    }

    #[test]
    fn a_pause_after_words_ends_at_twenty_seconds() {
        assert_eq!(self_stop_reason(true, S(90), Some(S(19))), None);
        assert_eq!(self_stop_reason(true, S(90), Some(S(20))), Some("no new words for 20 s"));
    }

    #[test]
    fn steady_talk_ends_only_at_the_cap() {
        assert_eq!(self_stop_reason(true, S(179), Some(S(1))), None);
        assert_eq!(self_stop_reason(true, S(180), Some(S(1))), Some("listening cap, 3 minutes"));
    }

    #[test]
    fn without_live_text_only_the_cap_applies() {
        assert_eq!(self_stop_reason(false, S(120), None), None);
        assert_eq!(self_stop_reason(false, S(180), None), Some("listening cap, 3 minutes"));
    }
}

fn dispatch(
    mono: &[i16],
    recognizer: &Option<Arc<Mutex<VoskState>>>,
    buffer: &Option<Arc<Mutex<Vec<i16>>>>,
    on_live: &Option<Arc<LiveSink>>,
) {
    if let Some(r) = recognizer {
        if let Ok(mut st) = r.lock() {
            match st.rec.accept_waveform(mono) {
                Ok(DecodingState::Finalized) => {
                    // A pause ended this utterance: keep its text, or the next one
                    // overwrites it inside the recogniser.
                    let text = st
                        .rec
                        .result()
                        .single()
                        .map(|s| s.text.trim().to_string())
                        .unwrap_or_default();
                    if !text.is_empty() {
                        st.committed = join_text(&st.committed, &text);
                    }
                    st.last_partial.clear();
                    if let Some(cb) = on_live {
                        cb(st.committed.clone());
                    }
                    st.last_emit = Instant::now();
                }
                Ok(DecodingState::Running) => {
                    if let Some(cb) = on_live {
                        if st.last_emit.elapsed() >= Duration::from_millis(200) {
                            let partial = st.rec.partial_result().partial.trim().to_string();
                            if partial != st.last_partial {
                                let live = join_text(&st.committed, &partial);
                                st.last_partial = partial;
                                cb(live);
                            }
                            st.last_emit = Instant::now();
                        }
                    }
                }
                _ => {}
            }
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
        .text("model", crate::DICTATION_MODEL)
        .part("file", part);

    let secs = pcm.len() as f32 / sample_rate.max(1) as f32;
    let t_upload = Instant::now();
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
    println!("[⏱️ PERF] Dictation: {:.1}s of audio transcribed in {:.2}s ({})", secs, t_upload.elapsed().as_secs_f32(), crate::DICTATION_MODEL);
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
