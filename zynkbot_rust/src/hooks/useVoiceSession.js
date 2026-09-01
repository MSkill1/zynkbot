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

// Parse a wake-word transcript into a structured voice command.
// Returns { type, ...params } or null if no command matched.
// Vosk produces lowercase, no-punctuation text — regexes are written accordingly.
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

  if (/^(zynkbot\s+)?stop(\s+(talking|dictating|reading))?$/.test(t)) {
    return { type: 'stop_tts' };
  }

  return null;
}

// Manages the wake-word session lifecycle: Hey Zynk detection, TTS playback,
// conversation loop, voice command execution, and all related bridge callbacks.
// Separate from useVoiceInput, which owns the dictation mic button.
export function useVoiceSession({ setMessages }) {
  const [ttsEnabled, setTtsEnabledRaw] = useState(
    // Off unless explicitly enabled. Spoken responses are intrusive by default and
    // are wanted mainly for hands-free phone use, so nobody should get audio they
    // did not ask for. Note this only affects installs with no stored value.
    () => localStorage.getItem('zynkbot_tts_enabled') === 'true'
  );
  const [heyZynkEnabled, setHeyZynkEnabledRaw] = useState(
    () => localStorage.getItem('zynkbot_hey_zynk_enabled') !== 'false'
  );
  const [voiceInputSource, setVoiceInputSourceRaw] = useState(
    () => localStorage.getItem('zynkbot_voice_input_source') || 'vosk'
  );
  const [conversationModeEnabled, setConversationModeEnabledRaw] = useState(
    () => localStorage.getItem('zynkbot_conversation_mode') === 'true'
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
  const setConversationModeEnabled = (val) => {
    setConversationModeEnabledRaw(val);
    localStorage.setItem('zynkbot_conversation_mode', val);
  };

  const silenceTimerRef = useRef(null);
  const wakeTriggeredRef = useRef(false);
  const ttsSourceRef = useRef(null);
  const ttsAudioCtxRef = useRef(null);
  const stopWakeRecordingRef = useRef(null);
  // Updated by App.jsx on every render so wake callbacks never hold a stale closure.
  const handleSendMessageRef = useRef(null);
  const conversationLoopActiveRef = useRef(false);
  const endConversationLoopRef = useRef(null);
  const startListeningLoopRef = useRef(null);

  const stopTts = () => {
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
      source.onended = () => {
        setIsTtsSpeaking(false);
        ttsSourceRef.current = null;
        ttsAudioCtxRef.current = null;
        if (conversationLoopActiveRef.current) {
          startListeningLoopRef.current?.();
        }
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

    const endConversationLoop = () => {
      conversationLoopActiveRef.current = false;
      try {
        const audio = new Audio('/wake_chime_close.wav');
        audio.play().catch(() => {});
      } catch (_) {}
      if (window.WakeWordBridge.isModelReady()) {
        setTimeout(() => window.WakeWordBridge.start(0.72), 1500);
      }
    };
    endConversationLoopRef.current = endConversationLoop;

    const autoSendWake = async () => {
      console.log('[WakeWord] autoSendWake firing');
      const text = await stopWakeRecording();
      console.log('[WakeWord] transcript:', text);

      if (!text) {
        conversationLoopActiveRef.current = false;
        if (window.WakeWordBridge.isModelReady()) {
          setTimeout(() => window.WakeWordBridge.start(0.72), 1000);
        }
        return;
      }

      const NEVERMIND = /^\s*(never\s*mind|cancel|forget\s*it|discard)\s*$/i;
      const STOP_ALONE = /^\s*stop\s*$/i;
      if (conversationLoopActiveRef.current && (CLOSING_PHRASES.test(text) || STOP_ALONE.test(text))) {
        console.log('[WakeWord] closing phrase detected — ending conversation loop');
        stopTts();
        endConversationLoop();
        return;
      }
      if (NEVERMIND.test(text)) {
        console.log('[WakeWord] nevermind — discarding without sending');
        endConversationLoop();
        return;
      }

      if (text.trim().split(/\s+/).length < 2) {
        console.log('[WakeWord] transcript too short, treating as noise:', text);
        conversationLoopActiveRef.current = false;
        if (window.WakeWordBridge.isModelReady()) {
          setTimeout(() => window.WakeWordBridge.start(0.72), 1000);
        }
        return;
      }

      wakeTriggeredRef.current = true;
      handleSendMessageRef.current?.(text);
    };

    const startListeningLoop = () => {
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
    startListeningLoopRef.current = startListeningLoop;

    window.__wakeWordDetected = () => {
      if (window.__dictationActive) {
        console.log('[WakeWord] ignored — dictation in progress');
        return;
      }
      if (ttsSourceRef.current) {
        console.log('[WakeWord] detected during TTS — stopping playback and re-listening');
        stopTts();
      }
      console.log('[WakeWord] detected — playing chime then recording');
      window.WakeWordBridge.stop();
      setWakeWordFlash(true);
      setTimeout(() => setWakeWordFlash(false), 2500);
      if (conversationModeEnabled) {
        conversationLoopActiveRef.current = true;
      }
      startListeningLoop();
    };
    window.__wakeWordModelReady = () => setWakeWordModelReady(true);
    window.__wakeWordDownloadProgress = (n) => setWakeWordDownloadProgress(n);
    window.__wakeWordDownloadError = (msg) => {
      setWakeWordDownloadError(msg);
      setWakeWordDownloadProgress(0);
    };
    window.__handleScreenOffTranscript = (transcript) => {
      if (!transcript?.trim()) return;
      console.log('[WakeWord] screen-off transcript received:', transcript);
      conversationLoopActiveRef.current = true;
      wakeTriggeredRef.current = true;
      setTimeout(() => handleSendMessageRef.current?.(transcript.trim()), 2000);
    };

    if (heyZynkEnabled) {
      if (window.WakeWordBridge.isModelReady()) {
        window.WakeWordBridge.start(0.72);
      }
    } else {
      window.WakeWordBridge.stop();
    }

    return () => {
      window.__wakeWordDetected = null;
      window.__wakeWordModelReady = null;
      window.__wakeWordDownloadProgress = null;
      window.__wakeWordDownloadError = null;
      window.__handleScreenOffTranscript = null;
      conversationLoopActiveRef.current = false;
      if (window.WakeWordBridge) window.WakeWordBridge.stop();
    };
  }, [heyZynkEnabled, conversationModeEnabled]);

  // VoiceButton announces dictation active/inactive via custom event. Tracked here
  // because the wake detector is owned by this hook, and VoiceButton's internal
  // state cannot be read from outside it.
  useEffect(() => {
    const onDictation = (e) => setIsDictating(!!e.detail?.active);
    window.addEventListener('zynkbot:dictation', onDictation);
    return () => window.removeEventListener('zynkbot:dictation', onDictation);
  }, []);

  // Pause wake word detection while recording or dictating; resume after a delay.
  // ONNX intentionally stays running during TTS so "Hey Zynk" can interrupt playback.
  useEffect(() => {
    if (!window.WakeWordBridge || !heyZynkEnabled) return;
    if (isWakeRecording || isDictating) {
      window.WakeWordBridge.stop();
    } else if (!conversationLoopActiveRef.current) {
      const t = setTimeout(() => {
        if (window.WakeWordBridge && window.WakeWordBridge.isModelReady()) {
          window.WakeWordBridge.start(0.72);
        }
      }, 5000);
      return () => clearTimeout(t);
    }
  }, [isWakeRecording, heyZynkEnabled, isDictating]);

  // Restart wake word when app returns to foreground (screen unlock, app switch).
  useEffect(() => {
    if (!window.WakeWordBridge) return;
    const handleVisible = () => {
      if (!heyZynkEnabled || isWakeRecording || isDictating) return;
      if (window.WakeWordBridge.isModelReady()) {
        window.WakeWordBridge.start(0.72);
      }
    };
    document.addEventListener('visibilitychange', handleVisible);
    return () => document.removeEventListener('visibilitychange', handleVisible);
  }, [heyZynkEnabled, isWakeRecording, isDictating]);

  // Prevent screen sleep during hands-free conversation sessions.
  useEffect(() => {
    if (!conversationModeEnabled) return;
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
  }, [conversationModeEnabled]);

  return {
    // Settings state (with localStorage-persisting setters)
    ttsEnabled, setTtsEnabled,
    heyZynkEnabled, setHeyZynkEnabled,
    voiceInputSource, setVoiceInputSource,
    conversationModeEnabled, setConversationModeEnabled,
    showVoiceModal, setShowVoiceModal,
    // Status state
    isTtsSpeaking,
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
    // - wakeTriggeredRef: handleSendMessage reads this to decide whether to speak the response
    // - stopWakeRecordingRef / endConversationLoopRef: waveform overlay calls these
    handleSendMessageRef,
    wakeTriggeredRef,
    stopWakeRecordingRef,
    endConversationLoopRef,
  };
}
