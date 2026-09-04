import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

function normalizeNumbers(text) {
  const words = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15',
    'sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19',
    'twenty':'20','thirty':'30','forty':'40','fifty':'50','sixty':'60',
    'ninety':'90','hundred':'100',
  };
  return text.replace(/\b(twenty|thirty|forty|fifty|sixty|ninety)-?(one|two|three|four|five|six|seven|eight|nine)\b/gi, (_, tens, ones) => {
    return String(parseInt(words[tens.toLowerCase()]) + parseInt(words[ones.toLowerCase()]));
  }).replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|ninety|hundred)\b/gi, m => words[m.toLowerCase()] || m);
}

// Decide whether a reply is spoken. Hands-free requests ("Hey Zynk" while the app
// is not in front) are always spoken — the user cannot see the screen. Anything
// entered with the app open (typed, mic button, or wake word) is text only unless
// the user has opted in with the "Speak replies in the app" toggle.
export function shouldSpeakReply({ handsFree, speakInApp }) {
  return Boolean(handsFree || speakInApp);
}

// Parse a wake-word transcript into a structured voice command.
// Returns { type, ...params } or null if no command matched.
// Vosk produces lowercase, no-punctuation text — regexes are written accordingly.
/**
 * Chat messages for hands-free exchanges the native path finished — only those that
 * belong to the thread on screen (the rest live in Conversation History). Ids are
 * numeric like every other message; derived from the turn's timestamp so a re-drain
 * never duplicates.
 */
export function nativeTurnsToMessages(turns, sessionId) {
  if (!Array.isArray(turns) || !sessionId) return [];
  return turns
    .filter((t) => t && t.sessionId === sessionId && t.question && t.answer)
    .flatMap((t) => {
      const at = Number(t.at) || Date.now();
      const timestamp = new Date(at).toISOString();
      return [
        { id: at, role: 'user', content: t.question, timestamp, source: 'voice' },
        { id: at + 1, role: 'assistant', content: t.answer, timestamp, source: 'voice' },
      ];
    });
}

export function parseVoiceCommand(text) {
  const t = normalizeNumbers(text.toLowerCase().trim());

  const timerRes = [
    /(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)/,
    /(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+timer/,
  ];
  for (const re of timerRes) {
    const m = t.match(re);
    if (m) {
      const val = parseFloat(m[1]);
      const unit = m[2];
      let seconds;
      if (unit.startsWith('hour') || unit.startsWith('hr')) seconds = Math.round(val * 3600);
      else if (unit.startsWith('min')) seconds = Math.round(val * 60);
      else seconds = Math.round(val);
      return { type: 'timer', seconds };
    }
  }

  const alarmM = t.match(
    /(?:set\s+(?:an?\s+)?alarm\s+(?:for|at)|alarm\s+(?:for|at)|wake\s+me\s+up\s+at)\s+(\d{1,2})(?:[:\s](\d{1,2}))?\s*(am|pm)?/
  );
  if (alarmM) {
    let hour = parseInt(alarmM[1]);
    const minute = parseInt(alarmM[2] || '0');
    const ampm = (alarmM[3] || '').toLowerCase();
    if (ampm === 'pm' && hour !== 12) hour += 12;
    if (ampm === 'am' && hour === 12) hour = 0;
    return { type: 'alarm', hour, minute };
  }

  if (/(?:start|begin)\s+(?:the\s+|a\s+)?stopwatch|^stopwatch$/.test(t)) {
    return { type: 'stopwatch' };
  }

  if (/^(?:(?:hey\s+)?zynk(?:bot)?\s+)?stop(?:\s+(talking|dictating|reading|listening))?$/.test(t)) {
    return { type: 'stop_tts' };
  }

  return null;
}

// Manages the wake-word session lifecycle: Hey Zynk detection, TTS playback,
// one-shot voice sessions, voice command execution, and all related bridge callbacks.
// Separate from useVoiceInput, which owns the dictation mic button.
export function useVoiceSession({ setMessages }) {
  const [ttsEnabled, setTtsEnabledRaw] = useState(
    // "Speak replies in the app". Off unless explicitly enabled: with the app open
    // the user is usually reading, often in public. Hands-free replies are spoken
    // regardless of this setting — see shouldSpeakReply().
    () => localStorage.getItem('zynkbot_tts_enabled') === 'true'
  );
  const [heyZynkEnabled, setHeyZynkEnabledRaw] = useState(
    () => localStorage.getItem('zynkbot_hey_zynk_enabled') !== 'false'
  );
  const [voiceInputSource, setVoiceInputSourceRaw] = useState(
    () => localStorage.getItem('zynkbot_voice_input_source') || 'vosk'
  );
  const [keepScreenAwake, setKeepScreenAwakeRaw] = useState(
    // Replaces 'zynkbot_conversation_mode', whose toggle was labelled "Keep screen
    // awake" but also re-opened the microphone after every reply. That loop is gone;
    // the old value is deliberately not carried over.
    () => {
      localStorage.removeItem('zynkbot_conversation_mode');
      return localStorage.getItem('zynkbot_keep_screen_awake') === 'true';
    }
  );
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const [isTtsSpeaking, setIsTtsSpeaking] = useState(false);
  const [isDictating, setIsDictating] = useState(false);
  const [isWakeRecording, setIsWakeRecording] = useState(false);
  const [wakeWordModelReady, setWakeWordModelReady] = useState(
    () => window.WakeWordBridge ? window.WakeWordBridge.isModelReady() : false
  );
  const [wakeWordDownloadProgress, setWakeWordDownloadProgress] = useState(0);
  const [wakeWordDownloadError, setWakeWordDownloadError] = useState('');
  const [wakeWordFlash, setWakeWordFlash] = useState(false);

  // localStorage-persisting setters
  const setTtsEnabled = (val) => {
    setTtsEnabledRaw(val);
    localStorage.setItem('zynkbot_tts_enabled', val);
  };

  const setHeyZynkEnabled = (val) => {
    setHeyZynkEnabledRaw(val);
    localStorage.setItem('zynkbot_hey_zynk_enabled', val);
  };
  const setVoiceInputSource = (val) => {
    setVoiceInputSourceRaw(val);
    localStorage.setItem('zynkbot_voice_input_source', val);
  };
  const setKeepScreenAwake = (val) => {
    setKeepScreenAwakeRaw(val);
    localStorage.setItem('zynkbot_keep_screen_awake', val);
  };

  const silenceTimerRef = useRef(null);
  const wakeTriggeredRef = useRef(false);
  const ttsSourceRef = useRef(null);
  const ttsAudioCtxRef = useRef(null);
  const stopWakeRecordingRef = useRef(null);
  // Updated by App.jsx on every render so wake callbacks never hold a stale closure.
  const handleSendMessageRef = useRef(null);
  // True only for requests captured while the app was not in front (screen-off
  // path). App.jsx reads and clears it to decide whether the reply is spoken.
  const handsFreeRef = useRef(false);
  const endVoiceSessionRef = useRef(null);

  // Restarting the wake-word listener is scattered across this hook (TTS
  // finishing, a session ending, silence, an interrupt). Several of those paths
  // used to call WakeWordBridge.start() directly on a timer with no re-check —
  // if the user turned "Hey Zynk" off while one was pending, it fired anyway.
  // heyZynkEnabledRef holds the live value so a deferred restart checks the
  // toggle at the moment it actually fires, not the moment it was scheduled.
  // armWakeWord() is the only path allowed to call WakeWordBridge.start(): it
  // cancels any restart still pending before scheduling a new one, so overlapping
  // calls collapse into the latest instead of stacking ONNX model reloads.
  const heyZynkEnabledRef = useRef(heyZynkEnabled);
  heyZynkEnabledRef.current = heyZynkEnabled;
  const wakeWordRestartTimerRef = useRef(null);
  const armWakeWord = (delayMs = 0) => {
    clearTimeout(wakeWordRestartTimerRef.current);
    wakeWordRestartTimerRef.current = setTimeout(() => {
      if (heyZynkEnabledRef.current && window.WakeWordBridge?.isModelReady()) {
        window.WakeWordBridge.start(0.72);
      }
    }, delayMs);
  };

  // Native (Kotlin) speech — a locked-phone reply — is invisible to the web TTS
  // state above. Kotlin pushes it via window.__nativeSpeaking so Stop can enable,
  // and stopTts() forwards to the native engine so Stop actually stops it.
  const [isNativeSpeaking, setIsNativeSpeaking] = useState(false);

  const stopTts = () => {
    try { window.WakeWordBridge?.stopSpeaking?.(); } catch (_) {}
    try { ttsSourceRef.current?.stop(); } catch (_) {}
    try { ttsAudioCtxRef.current?.close(); } catch (_) {}
    ttsSourceRef.current = null;
    ttsAudioCtxRef.current = null;
    setIsTtsSpeaking(false);
  };

  const speakResponse = async (text) => {
    if (!text?.trim()) return;
    stopTts();
    try {
      const keys = await invoke('get_api_keys');
      const apiKey = keys['OPENAI_API_KEY'];
      if (!apiKey) return;
      const res = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'tts-1', input: text.slice(0, 4096), voice: 'alloy' }),
      });
      if (!res.ok) return;
      const audioData = await res.arrayBuffer();
      const audioCtx = new AudioContext();
      ttsAudioCtxRef.current = audioCtx;
      const buffer = await audioCtx.decodeAudioData(audioData);
      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      ttsSourceRef.current = source;
      setIsTtsSpeaking(true);
      source.start();
      // The wake detector stays OFF while TTS plays: the speaker output garbles
      // the mic, so "Hey Zynk" can't be heard reliably mid-reply anyway. To stop
      // a reply, tap Stop. It re-arms in onended when playback finishes.
      window.WakeWordBridge?.stop();
      source.onended = () => {
        setIsTtsSpeaking(false);
        ttsSourceRef.current = null;
        ttsAudioCtxRef.current = null;
        // Every interaction is one-shot: after a reply, return to passive
        // wake-word listening. Never open another dictation window on our own.
        armWakeWord();
      };
    } catch (e) {
      console.error('[TTS]', e);
      setIsTtsSpeaking(false);
    }
  };

  const executeVoiceCommand = (cmd, originalText) => {
    const now = new Date().toISOString();
    let confirmation = '';
    if (cmd.type === 'timer') {
      window.VoiceCommandBridge.setTimer(cmd.seconds);
      const h = Math.floor(cmd.seconds / 3600);
      const m = Math.floor((cmd.seconds % 3600) / 60);
      const s = cmd.seconds % 60;
      const parts = [];
      if (h) parts.push(`${h} hour${h > 1 ? 's' : ''}`);
      if (m) parts.push(`${m} minute${m > 1 ? 's' : ''}`);
      if (s && !h) parts.push(`${s} second${s > 1 ? 's' : ''}`);
      confirmation = `⏱️ Timer set for ${parts.join(' and ')}.`;
    } else if (cmd.type === 'alarm') {
      window.VoiceCommandBridge.setAlarm(cmd.hour, cmd.minute, 'Zynkbot');
      const h12 = cmd.hour % 12 || 12;
      const ampm = cmd.hour < 12 ? 'AM' : 'PM';
      const min = cmd.minute.toString().padStart(2, '0');
      confirmation = `⏰ Alarm set for ${h12}:${min} ${ampm}.`;
    } else if (cmd.type === 'stopwatch') {
      window.VoiceCommandBridge.startStopwatch();
      confirmation = '⏱️ Stopwatch started.';
    }
    setMessages(prev => [
      ...prev,
      { id: Date.now(), role: 'user', content: originalText, timestamp: now, recalled_memories: [] },
      { id: Date.now() + 1, role: 'assistant', content: confirmation, timestamp: now, recalled_memories: [], metadata: { model_backend: 'voice command' } },
    ]);
  };

  // Wake word service lifecycle — starts/stops with the Hey Zynk toggle on Android.
  useEffect(() => {
    if (!window.WakeWordBridge) return;

    const SILENCE_MS = 1000;
    const CLOSING_PHRASES = /\b(thank(?:\s+you)?\s+zynk|goodbye\s+zynk|zynk\s+stop|stop\s+listening|that'?s?\s+all|close\s+session)\b/i;

    const stopWakeRecording = () => {
      clearTimeout(silenceTimerRef.current);
      window.__voskPartial = null;
      setIsWakeRecording(false);
      if (!window.VoskBridge) return Promise.resolve('');
      return new Promise((resolve) => {
        window.__voskResult = (text) => {
          window.__voskResult = null;
          window.__voskError = null;
          resolve(text || '');
        };
        window.__voskError = () => {
          window.__voskResult = null;
          window.__voskError = null;
          resolve('');
        };
        window.VoskBridge.stopListening();
      });
    };
    stopWakeRecordingRef.current = stopWakeRecording;

    const endVoiceSession = () => {
      try {
        const audio = new Audio('/wake_chime_close.wav');
        audio.play().catch(() => {});
      } catch (_) {}
      armWakeWord(1500);
    };
    endVoiceSessionRef.current = endVoiceSession;

    const autoSendWake = async () => {
      console.log('[WakeWord] autoSendWake firing');
      const text = await stopWakeRecording();
      console.log('[WakeWord] transcript:', text);

      if (!text) {
        armWakeWord(1000);
        return;
      }

      const NEVERMIND = /^\s*(never\s*mind|cancel|forget\s*it|discard)\s*$/i;
      const STOP_ALONE = /^\s*stop\s*$/i;
      if (CLOSING_PHRASES.test(text) || STOP_ALONE.test(text)) {
        console.log('[WakeWord] stop phrase detected — ending voice session');
        stopTts();
        endVoiceSession();
        return;
      }
      if (NEVERMIND.test(text)) {
        console.log('[WakeWord] nevermind — discarding without sending');
        endVoiceSession();
        return;
      }

      if (text.trim().split(/\s+/).length < 2) {
        console.log('[WakeWord] transcript too short, treating as noise:', text);
        armWakeWord(1000);
        return;
      }

      wakeTriggeredRef.current = true;
      handleSendMessageRef.current?.(text);
    };

    const startListening = () => {
      if (!window.VoskBridge) return;
      if (window.__dictationActive) return;
      const startRecording = () => {
        window.__voskPartial = (partial) => {
          if (partial.trim()) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = setTimeout(autoSendWake, SILENCE_MS);
          }
        };
        window.VoskBridge.startListening();
        setIsWakeRecording(true);
        silenceTimerRef.current = setTimeout(autoSendWake, 8000);
      };
      try {
        const audio = new Audio('/wake_chime.wav');
        audio.onended = startRecording;
        audio.onerror = startRecording;
        audio.play().catch(startRecording);
      } catch (_) {
        startRecording();
      }
    };

    window.__wakeWordDetected = () => {
      if (window.__dictationActive) {
        console.log('[WakeWord] ignored — dictation in progress');
        return;
      }
      if (ttsSourceRef.current) {
        // Detector is stopped during TTS, so this normally can't fire mid-reply.
        // If a stray detection slips through, ignore it rather than interrupt —
        // onended re-arms passive listening when the reply finishes on its own.
        console.log('[WakeWord] detected during TTS — ignoring');
        return;
      }
      console.log('[WakeWord] detected — playing chime then recording');
      window.WakeWordBridge.stop();
      setWakeWordFlash(true);
      setTimeout(() => setWakeWordFlash(false), 2500);
      startListening();
    };
    window.__wakeWordModelReady = () => setWakeWordModelReady(true);
    window.__wakeWordDownloadProgress = (n) => setWakeWordDownloadProgress(n);
    window.__wakeWordDownloadError = (msg) => {
      setWakeWordDownloadError(msg);
      setWakeWordDownloadProgress(0);
    };
    window.__nativeSpeaking = (on) => setIsNativeSpeaking(!!on);
    window.__handleScreenOffTranscript = (transcript) => {
      if (!transcript?.trim()) return;
      console.log('[WakeWord] screen-off transcript received:', transcript);
      // Captured while the app was not in front: the user cannot see the
      // screen, so the reply must be spoken regardless of the in-app setting.
      handsFreeRef.current = true;
      wakeTriggeredRef.current = true;
      setTimeout(() => handleSendMessageRef.current?.(transcript.trim()), 2000);
    };

    if (heyZynkEnabled) {
      armWakeWord();
    } else {
      clearTimeout(wakeWordRestartTimerRef.current);
      window.WakeWordBridge.stop();
    }

    return () => {
      window.__wakeWordDetected = null;
      window.__wakeWordModelReady = null;
      window.__wakeWordDownloadProgress = null;
      window.__wakeWordDownloadError = null;
      window.__handleScreenOffTranscript = null;
      window.__nativeSpeaking = null;
      clearTimeout(wakeWordRestartTimerRef.current);
      if (window.WakeWordBridge) window.WakeWordBridge.stop();
    };
  }, [heyZynkEnabled]);

  // VoiceButton announces dictation active/inactive via custom event. Tracked here
  // because the wake detector is owned by this hook, and VoiceButton's internal
  // state cannot be read from outside it.
  useEffect(() => {
    const onDictation = (e) => setIsDictating(!!e.detail?.active);
    window.addEventListener('zynkbot:dictation', onDictation);
    return () => window.removeEventListener('zynkbot:dictation', onDictation);
  }, []);

  // Pause wake word detection while recording, dictating, or speaking a reply;
  // resume after a delay once all three are clear. The detector is kept off
  // during TTS because the speaker output garbles the mic.
  useEffect(() => {
    if (!window.WakeWordBridge || !heyZynkEnabled) return;
    if (isWakeRecording || isDictating || isTtsSpeaking) {
      window.WakeWordBridge.stop();
    } else {
      armWakeWord(5000);
      return () => clearTimeout(wakeWordRestartTimerRef.current);
    }
  }, [isWakeRecording, heyZynkEnabled, isDictating, isTtsSpeaking]);

  // Restart wake word when app returns to foreground (screen unlock, app switch).
  useEffect(() => {
    if (!window.WakeWordBridge) return;
    const handleVisible = () => {
      // A push sent while the WebView was paused is lost; re-read on resume.
      try { setIsNativeSpeaking(!!window.WakeWordBridge?.isNativeSpeaking?.()); } catch (_) {}
      if (!heyZynkEnabled || isWakeRecording || isDictating) return;
      armWakeWord();
    };
    document.addEventListener('visibilitychange', handleVisible);
    return () => document.removeEventListener('visibilitychange', handleVisible);
  }, [heyZynkEnabled, isWakeRecording, isDictating]);

  // Optional: prevent screen sleep while the app is open for hands-free use.
  // This only holds a screen wake lock; it never affects listening.
  useEffect(() => {
    if (!keepScreenAwake) return;
    let wakeLock = null;
    const acquire = async () => {
      try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (_) {}
    };
    acquire();
    const onVisible = () => { if (document.visibilityState === 'visible') acquire(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      try { wakeLock?.release(); } catch (_) {}
    };
  }, [keepScreenAwake]);

  return {
    // Settings state (with localStorage-persisting setters)
    ttsEnabled, setTtsEnabled,
    heyZynkEnabled, setHeyZynkEnabled,
    voiceInputSource, setVoiceInputSource,
    keepScreenAwake, setKeepScreenAwake,
    showVoiceModal, setShowVoiceModal,
    // Status state
    isTtsSpeaking,
    isNativeSpeaking,
    isDictating,
    isWakeRecording,
    wakeWordModelReady,
    wakeWordDownloadProgress,
    wakeWordDownloadError,
    wakeWordFlash,
    // Functions
    stopTts,
    speakResponse,
    executeVoiceCommand,
    // Refs exposed to App.jsx:
    // - handleSendMessageRef: App.jsx updates this every render so callbacks stay current
    // - wakeTriggeredRef: handleSendMessage reads this for wake-only behaviour (web search auto-run)
    // - handsFreeRef: handleSendMessage reads this to decide whether to speak the reply
    // - stopWakeRecordingRef / endVoiceSessionRef: waveform overlay calls these
    handleSendMessageRef,
    wakeTriggeredRef,
    handsFreeRef,
    stopWakeRecordingRef,
    endVoiceSessionRef,
  };
}
