//! Live dictation through OpenAI's Realtime transcription session (2026-10-28).
//!
//! The desktop OpenAI path used to hold the whole clip and upload one WAV when the
//! user stopped, so the text arrived after a wait as long as the clip itself. This
//! module opens a WebSocket to `wss://api.openai.com/v1/realtime?intent=transcription`
//! at the start of the recording and sends the microphone audio as it is captured.
//! The server transcribes while the user is still speaking and sends the words back
//! in `...input_audio_transcription.delta` events, which go to the page the same way
//! Vosk's partial text does.
//!
//! It is an addition, not a replacement: `vosk_desktop` still keeps the PCM buffer,
//! and when the socket cannot be opened, the session errors out, or the user sets
//! `DICTATION_STREAM=0`, the old one-shot upload runs on stop exactly as before.
//!
//! The session speaks the "realtime=v1" beta protocol: `transcription_session.update`
//! to configure, `input_audio_buffer.append` with base64 PCM16 at 24 kHz mono, and
//! `input_audio_buffer.commit` on stop. Audio is resampled from the capture rate to
//! 24 kHz here, because the API accepts pcm16 at 24 kHz only.

use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use crate::vosk_desktop::LiveSink;

/// The Realtime API takes pcm16 at this rate only.
pub const TARGET_RATE: u32 = 24_000;
const REALTIME_URL: &str = "wss://api.openai.com/v1/realtime?intent=transcription";
/// How long `stop` waits for the transcript of the last words after committing.
const FINAL_WAIT: Duration = Duration::from_secs(3);
/// Audio is sent in pieces of about this long (100 ms at 24 kHz).
const SEND_CHUNK: usize = (TARGET_RATE / 10) as usize;

/// `DICTATION_STREAM=0` in the environment (or the .env file) turns live streaming off.
pub fn streaming_enabled() -> bool {
    !matches!(
        std::env::var("DICTATION_STREAM").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("no")
    )
}

// ── Resampling ─────────────────────────────────────────────────────────────

/// Linear-interpolation resampler that keeps its place between chunks, so a stream
/// of microphone callbacks resamples the same as the whole clip would. The position
/// is kept in whole units of 1/`to` of an input sample, so it is exact and chunking
/// never shifts it.
#[derive(Clone, Debug)]
pub struct Resampler {
    from: u64,
    to: u64,
    /// Position in the virtual buffer, in units of 1/`to` input samples.
    pos: u64,
    prev: Option<i16>,
}

impl Resampler {
    pub fn new(from: u32, to: u32) -> Self {
        Resampler { from: from.max(1) as u64, to: to.max(1) as u64, pos: 0, prev: None }
    }

    pub fn process(&mut self, input: &[i16]) -> Vec<i16> {
        if self.from == self.to {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }
        // Virtual buffer: the last sample of the previous chunk, then this chunk.
        let mut buf: Vec<i16> = Vec::with_capacity(input.len() + 1);
        if let Some(p) = self.prev {
            buf.push(p);
        }
        buf.extend_from_slice(input);
        let mut out = Vec::with_capacity((input.len() as u64 * self.to / self.from) as usize + 2);
        loop {
            let i = (self.pos / self.to) as usize;
            if i + 1 >= buf.len() {
                break;
            }
            let frac = (self.pos % self.to) as f64 / self.to as f64;
            let a = buf[i] as f64;
            let b = buf[i + 1] as f64;
            out.push((a + (b - a) * frac).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
            self.pos += self.from;
        }
        self.prev = buf.last().copied();
        self.pos -= (buf.len() as u64 - 1) * self.to;
        out
    }
}

// ── Event parsing ──────────────────────────────────────────────────────────

/// The server events this session cares about.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Part of the transcript of the segment being spoken now.
    Delta(String),
    /// The finished transcript of one segment.
    Completed(String),
    /// The server (or our stop) closed a segment; its transcript is still coming.
    Committed,
    /// Transcription of one segment failed; nothing more will come for it.
    Failed(String),
    Error { code: Option<String>, message: String },
    Other(String),
}

pub fn parse_event(json: &str) -> Event {
    let v: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(e) => return Event::Error { code: Some("bad_json".into()), message: e.to_string() },
    };
    let kind = v["type"].as_str().unwrap_or("");
    match kind {
        "conversation.item.input_audio_transcription.delta" => {
            Event::Delta(v["delta"].as_str().unwrap_or("").to_string())
        }
        "conversation.item.input_audio_transcription.completed" => {
            Event::Completed(v["transcript"].as_str().unwrap_or("").to_string())
        }
        "input_audio_buffer.committed" => Event::Committed,
        "conversation.item.input_audio_transcription.failed" => Event::Failed(
            v["error"]["message"].as_str().unwrap_or("transcription failed").to_string(),
        ),
        "error" => Event::Error {
            code: v["error"]["code"].as_str().map(|s| s.to_string()),
            message: v["error"]["message"].as_str().unwrap_or("unknown error").to_string(),
        },
        other => Event::Other(other.to_string()),
    }
}

/// Builds the running transcript from the events: finished segments plus the deltas
/// of the one still being spoken.
#[derive(Debug, Default)]
pub struct Transcript {
    committed: String,
    partial: String,
    /// Segments the server has closed whose transcript has not arrived yet.
    pending: usize,
}

fn join_text(a: &str, b: &str) -> String {
    let (a, b) = (a.trim(), b.trim());
    match (a.is_empty(), b.is_empty()) {
        (true, _) => b.to_string(),
        (_, true) => a.to_string(),
        _ => format!("{} {}", a, b),
    }
}

impl Transcript {
    /// Applies one event; returns true when the text to show has changed.
    pub fn apply(&mut self, ev: &Event) -> bool {
        match ev {
            Event::Delta(d) => {
                if d.is_empty() {
                    return false;
                }
                self.partial.push_str(d);
                true
            }
            Event::Completed(t) => {
                self.pending = self.pending.saturating_sub(1);
                self.committed = join_text(&self.committed, t);
                self.partial.clear();
                true
            }
            Event::Committed => {
                self.pending += 1;
                false
            }
            Event::Failed(_) => {
                self.pending = self.pending.saturating_sub(1);
                let changed = !self.partial.is_empty();
                self.partial.clear();
                changed
            }
            Event::Error { .. } | Event::Other(_) => false,
        }
    }

    pub fn text(&self) -> String {
        join_text(&self.committed, &self.partial)
    }

    pub fn pending(&self) -> usize {
        self.pending
    }
}

// ── The session ────────────────────────────────────────────────────────────

enum Msg {
    Audio(Vec<i16>),
    Commit,
}

/// The microphone side: resamples each callback's samples and hands them to the
/// session thread. Cloneable so either cpal callback flavour can own one.
#[derive(Clone)]
pub struct Feeder {
    resampler: Resampler,
    tx: mpsc::UnboundedSender<Msg>,
}

impl Feeder {
    pub fn push(&mut self, mono: &[i16]) {
        let out = self.resampler.process(mono);
        if !out.is_empty() {
            // Err means the session thread is gone; the WAV fallback still has the clip.
            let _ = self.tx.send(Msg::Audio(out));
        }
    }
}

/// The handle `vosk_desktop` keeps until stop.
pub struct Session {
    tx: mpsc::UnboundedSender<Msg>,
    result_rx: std_mpsc::Receiver<Result<String, String>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Session {
    /// Opens the session on its own thread. Returns the feeder for the mic callback
    /// and the handle for stop. Connecting happens in the background: audio captured
    /// before the socket is up waits in the channel, so recording starts at once.
    pub fn open(capture_rate: u32, on_live: Option<Arc<LiveSink>>) -> Result<(Feeder, Session), String> {
        if !streaming_enabled() {
            return Err("DICTATION_STREAM=0".to_string());
        }
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set — add it to your .env file".to_string())?;
        let (tx, rx) = mpsc::unbounded_channel::<Msg>();
        let (result_tx, result_rx) = std_mpsc::channel::<Result<String, String>>();
        let thread = thread::Builder::new()
            .name("dictation-stream".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = result_tx.send(Err(format!("runtime: {e}")));
                        return;
                    }
                };
                let outcome = rt.block_on(run_session(&api_key, rx, on_live));
                let _ = result_tx.send(outcome);
            })
            .map_err(|e| format!("thread: {e}"))?;
        let feeder = Feeder { resampler: Resampler::new(capture_rate, TARGET_RATE), tx: tx.clone() };
        Ok((feeder, Session { tx, result_rx, thread: Some(thread) }))
    }

    /// Commits the last audio and waits for its transcript. `Ok(text)` is the whole
    /// transcript of the recording; `Err` means the caller should upload the clip.
    pub fn finish(mut self) -> Result<String, String> {
        let _ = self.tx.send(Msg::Commit);
        let result = self
            .result_rx
            .recv_timeout(FINAL_WAIT + Duration::from_secs(3))
            .map_err(|e| format!("stream thread gave no result: {e}"))
            .and_then(|r| r);
        if let Some(t) = self.thread.take() {
            if result.is_ok() {
                let _ = t.join();
            }
        }
        result
    }
}

fn session_update() -> String {
    serde_json::json!({
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": crate::DICTATION_MODEL,
                "language": "en"
            },
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 700
            }
        }
    })
    .to_string()
}

pub fn append_event(pcm: &[i16]) -> String {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    serde_json::json!({
        "type": "input_audio_buffer.append",
        "audio": base64::engine::general_purpose::STANDARD.encode(bytes)
    })
    .to_string()
}

async fn run_session(
    api_key: &str,
    mut rx: mpsc::UnboundedReceiver<Msg>,
    on_live: Option<Arc<LiveSink>>,
) -> Result<String, String> {
    let t0 = Instant::now();
    let mut req = REALTIME_URL
        .into_client_request()
        .map_err(|e| format!("request: {e}"))?;
    let auth = format!("Bearer {api_key}")
        .parse()
        .map_err(|e| format!("auth header: {e}"))?;
    req.headers_mut().insert("Authorization", auth);
    req.headers_mut().insert("OpenAI-Beta", "realtime=v1".parse().map_err(|e| format!("header: {e}"))?);

    let (ws, _) = tokio::time::timeout(Duration::from_secs(5), tokio_tungstenite::connect_async(req))
        .await
        .map_err(|_| "connect timed out".to_string())?
        .map_err(|e| format!("connect: {e}"))?;
    println!("[Dictation] Live stream connected in {:.2}s ({})", t0.elapsed().as_secs_f32(), crate::DICTATION_MODEL);
    let (mut sink, mut source) = ws.split();
    sink.send(Message::text(session_update()))
        .await
        .map_err(|e| format!("session update: {e}"))?;

    let mut transcript = Transcript::default();
    let mut staged: Vec<i16> = Vec::with_capacity(SEND_CHUNK * 2);
    let mut committed_sent = false;
    let mut commit_acked = false;
    let mut deadline: Option<tokio::time::Instant> = None;
    let mut last_error: Option<String> = None;
    let mut stopped_at: Option<Instant> = None;

    loop {
        let wait = match deadline {
            Some(d) => tokio::time::sleep_until(d),
            None => tokio::time::sleep(Duration::from_secs(3600)),
        };
        tokio::select! {
            msg = rx.recv(), if !committed_sent => {
                match msg {
                    Some(Msg::Audio(pcm)) => {
                        staged.extend_from_slice(&pcm);
                        if staged.len() >= SEND_CHUNK {
                            sink.send(Message::text(append_event(&staged))).await.map_err(|e| format!("send audio: {e}"))?;
                            staged.clear();
                        }
                    }
                    Some(Msg::Commit) | None => {
                        if !staged.is_empty() {
                            sink.send(Message::text(append_event(&staged))).await.map_err(|e| format!("send audio: {e}"))?;
                            staged.clear();
                        }
                        sink.send(Message::text(r#"{"type":"input_audio_buffer.commit"}"#.to_string()))
                            .await
                            .map_err(|e| format!("commit: {e}"))?;
                        committed_sent = true;
                        stopped_at = Some(Instant::now());
                        deadline = Some(tokio::time::Instant::now() + FINAL_WAIT);
                    }
                }
            }
            incoming = source.next() => {
                let msg = match incoming {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => {
                        if committed_sent { break; }
                        return Err(format!("socket: {e}"));
                    }
                    None => {
                        if committed_sent { break; }
                        return Err("socket closed by server".to_string());
                    }
                };
                let text = match msg {
                    Message::Text(t) => t.as_str().to_string(),
                    Message::Close(_) => { if committed_sent { break; } return Err("server closed the session".to_string()); }
                    _ => continue,
                };
                let ev = parse_event(&text);
                match &ev {
                    Event::Error { code, message } => {
                        // After our commit, an "empty buffer" error only means the server's
                        // VAD had already closed the last segment: nothing more is coming.
                        let empty_commit = code.as_deref().map(|c| c.contains("commit_empty")).unwrap_or(false);
                        if committed_sent {
                            commit_acked = true;
                            if !empty_commit { last_error = Some(message.clone()); }
                        } else {
                            return Err(format!("server error: {message}"));
                        }
                    }
                    Event::Committed if committed_sent => { commit_acked = true; }
                    Event::Failed(m) => { eprintln!("[Dictation] Live stream segment failed: {m}"); }
                    _ => {}
                }
                if transcript.apply(&ev) {
                    if let Some(cb) = &on_live {
                        cb(transcript.text());
                    }
                }
                if committed_sent && commit_acked && transcript.pending() == 0 {
                    break;
                }
            }
            _ = wait => {
                eprintln!("[Dictation] Live stream: gave up waiting for the last transcript after {:?}", FINAL_WAIT);
                break;
            }
        }
    }

    let _ = tokio::time::timeout(Duration::from_millis(500), sink.send(Message::Close(None))).await;
    let text = transcript.text();
    if text.is_empty() {
        if let Some(e) = last_error {
            return Err(format!("server error: {e}"));
        }
    }
    let after_stop = stopped_at.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
    println!("[⏱️ PERF] Dictation: live stream finished {:.2}s after stop ({})", after_stop, crate::DICTATION_MODEL);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delta_completed_and_error() {
        assert_eq!(
            parse_event(r#"{"type":"conversation.item.input_audio_transcription.delta","item_id":"i1","delta":"Hello "}"#),
            Event::Delta("Hello ".into())
        );
        assert_eq!(
            parse_event(r#"{"type":"conversation.item.input_audio_transcription.completed","item_id":"i1","transcript":"Hello there."}"#),
            Event::Completed("Hello there.".into())
        );
        assert_eq!(
            parse_event(r#"{"type":"error","error":{"type":"invalid_request_error","code":"input_audio_buffer_commit_empty","message":"buffer too small"}}"#),
            Event::Error { code: Some("input_audio_buffer_commit_empty".into()), message: "buffer too small".into() }
        );
        assert_eq!(parse_event(r#"{"type":"input_audio_buffer.committed","item_id":"i1"}"#), Event::Committed);
        assert_eq!(
            parse_event(r#"{"type":"conversation.item.input_audio_transcription.failed","error":{"message":"nope"}}"#),
            Event::Failed("nope".into())
        );
        assert_eq!(parse_event(r#"{"type":"transcription_session.created"}"#), Event::Other("transcription_session.created".into()));
        assert!(matches!(parse_event("not json"), Event::Error { code: Some(c), .. } if c == "bad_json"));
    }

    #[test]
    fn transcript_joins_segments_and_tracks_pending() {
        let mut t = Transcript::default();
        assert!(t.apply(&Event::Delta("Hel".into())));
        assert!(t.apply(&Event::Delta("lo".into())));
        assert_eq!(t.text(), "Hello");
        assert!(!t.apply(&Event::Committed));
        assert_eq!(t.pending(), 1);
        assert!(t.apply(&Event::Completed("Hello.".into())));
        assert_eq!(t.pending(), 0);
        assert_eq!(t.text(), "Hello.");
        t.apply(&Event::Delta(" there".into()));
        assert_eq!(t.text(), "Hello. there");
        t.apply(&Event::Committed);
        t.apply(&Event::Completed("There it is.".into()));
        assert_eq!(t.text(), "Hello. There it is.");
        assert!(!t.apply(&Event::Error { code: None, message: "x".into() }));
        assert!(!t.apply(&Event::Delta(String::new())));
        // A failed segment drops its partial words and its pending slot.
        t.apply(&Event::Delta("junk".into()));
        t.apply(&Event::Committed);
        assert!(t.apply(&Event::Failed("bad".into())));
        assert_eq!(t.text(), "Hello. There it is.");
        assert_eq!(t.pending(), 0);
    }

    #[test]
    fn resampler_halves_and_passes_through() {
        let ramp: Vec<i16> = (0..100).map(|i| i as i16 * 10).collect();
        let mut r = Resampler::new(48_000, 24_000);
        let out = r.process(&ramp);
        assert_eq!(out.len(), 50);
        // Every other sample of a ramp is the ramp at double step.
        assert_eq!(&out[..5], &[0, 20, 40, 60, 80]);

        let mut same = Resampler::new(16_000, 16_000);
        assert_eq!(same.process(&ramp), ramp);

        // Upsampling interpolates between neighbours.
        let mut up = Resampler::new(16_000, 24_000);
        let out = up.process(&[0, 30, 60]);
        assert_eq!(out, vec![0, 20, 40]);
        let out = up.process(&[90, 120]);
        assert_eq!(out, vec![60, 80, 100]);
    }

    #[test]
    fn resampler_chunks_match_whole_clip() {
        let signal: Vec<i16> = (0..4800).map(|i| ((i as f64 * 0.05).sin() * 10000.0) as i16).collect();
        let mut whole = Resampler::new(44_100, 24_000);
        let expected = whole.process(&signal);
        let mut chunked = Resampler::new(44_100, 24_000);
        let mut got = Vec::new();
        for chunk in signal.chunks(441) {
            got.extend(chunked.process(chunk));
        }
        assert_eq!(got, expected);
        let ratio = got.len() as f64 / signal.len() as f64;
        assert!((ratio - 24_000.0 / 44_100.0).abs() < 0.01, "ratio {ratio}");
    }

    #[test]
    fn append_event_is_base64_pcm16() {
        let s = append_event(&[1, -1]);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "input_audio_buffer.append");
        let bytes = base64::engine::general_purpose::STANDARD.decode(v["audio"].as_str().unwrap()).unwrap();
        assert_eq!(bytes, vec![1, 0, 0xFF, 0xFF]);
        let u: serde_json::Value = serde_json::from_str(&session_update()).unwrap();
        assert_eq!(u["session"]["input_audio_format"], "pcm16");
        assert_eq!(u["session"]["input_audio_transcription"]["model"], crate::DICTATION_MODEL);
        assert_eq!(u["session"]["turn_detection"]["silence_duration_ms"], 700);
    }

    #[test]
    fn env_switch_off_values() {
        // Only the value matters; the function reads the process environment, which the
        // other tests do not touch, so set and restore it here.
        std::env::set_var("DICTATION_STREAM", "0");
        assert!(!streaming_enabled());
        std::env::set_var("DICTATION_STREAM", "1");
        assert!(streaming_enabled());
        std::env::remove_var("DICTATION_STREAM");
        assert!(streaming_enabled());
    }
}
