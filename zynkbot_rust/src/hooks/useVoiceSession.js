import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import * as desktopTimer from '../utils/zynkTimer';

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
      const pre = (t.preAnswer || '').trim();
      // Sources of an automatic web search, in the shape ChatMessage renders as the
      // "View n search sources" drop-down under a typed search (2026-09-22).
      const sources = t.sources && Array.isArray(t.sources.results) && t.sources.results.length > 0 ? t.sources : null;
      return [
        { id: at, role: 'user', content: t.question, timestamp, source: 'voice' },
        ...(pre ? [{ id: at + 1, role: 'assistant', content: pre, timestamp, source: 'voice' }] : []),
        { id: at + 2, role: 'assistant', content: t.answer, timestamp, source: 'voice', ...(sources ? { web_search_results: sources } : {}) },
      ];
    });
}

/**
 * Text the way a speech engine should receive it: markdown stripped, math symbols as
 * words. Mirrors NativeVoiceAnswerer.cleanForSpeech on the Kotlin side; this copy
 * covers the in-app spoken reply, which goes to OpenAI's speech service.
 */
export function cleanForSpeech(raw) {
  if (!raw) return '';
  let t = String(raw);
  t = t.replace(/```[\s\S]*?```/g, ' ');
  t = t.replace(/`([^`]*)`/g, '$1');
  t = t.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');
  t = t.replace(/^\s{0,3}#{1,6}\s+/gm, '');
  t = t.replace(/^\s*(?:[-*•]|\d+[.)])\s+/gm, '');
  t = t.replace(/(\*\*|__)(.+?)\1/g, '$2');
  t = t.replace(/(^|[^\w*])[*_](?=\S)(.+?)(?<=\S)[*_](?![\w*])/g, '$1$2');
  t = t.replace(/×/g, ' times ').replace(/÷/g, ' divided by ').replace(/−/g, ' minus ')
    .replace(/±/g, ' plus or minus ').replace(/≈/g, ' approximately ')
    .replace(/≠/g, ' is not equal to ').replace(/≤/g, ' is at most ')
    .replace(/≥/g, ' is at least ').replace(/→/g, ' to ');
  t = t.replace(/(?<=\d)\s*=\s*(?=[\d-])/g, ' equals ');
  t = t.replace(/(?<=\d)\s*\*\s*(?=\d)/g, ' times ');
  t = t.replace(/(?<=\d)\s*\/\s*(?=\d)/g, ' divided by ');
  t = t.replace(/[*_#>|~]+/g, ' ');
  return t.replace(/[ \t]{2,}/g, ' ').trim();
}

// Spoken fractions and articles that the number words miss: "two and a half
// minutes", "a minute and a half", "half an hour", "a minute". Without this the
// timer regex saw "2 and a half minutes", matched nothing, the request went to the
// model as chat, and the model answered "timer set" with no timer (Matt, Pixel,
// 2026-10-07; the same shape as KI-027).
export function normalizeFractions(t) {
  return t
    .replace(/(\d+)\s+and\s+a\s+half\b/g, (m, n) => `${n}.5`)
    .replace(/(\d+)\s+and\s+a\s+quarter\b/g, (m, n) => `${n}.25`)
    .replace(/\bhalf\s+an?\s+(hour|minute|second)/g, '0.5 $1')
    .replace(/\ban?\s+(hour|minute|second)\s+and\s+a\s+half\b/g, '1.5 $1')
    .replace(/\ban?\s+(hour|minute|second)\b/g, '1 $1');
}

function toSeconds(value, unit) {
  const v = parseFloat(value);
  if (!(v >= 0)) return 0;
  if (unit.startsWith('hour') || unit.startsWith('hr')) return Math.round(v * 3600);
  if (unit.startsWith('min')) return Math.round(v * 60);
  return Math.round(v);
}
// "1026" -> 10:26, "730" -> 7:30, "7" + "30" + "pm" -> 19:30; null when not a clock time.
function clockFrom(d, m, ampm, now) {
  let digits = d; let minuteText = m || '';
  if (digits.length >= 3 && !minuteText) { minuteText = digits.slice(-2); digits = digits.slice(0, -2); }
  let hour = parseInt(digits, 10);
  const minute = parseInt(minuteText || '0', 10);
  if (!(hour >= 0) || hour > 23 || minute > 59) return null;
  const ap = (ampm || '').toLowerCase();
  if (ap === 'pm' && hour !== 12) hour += 12;
  if (ap === 'am' && hour === 12) hour = 0;
  if (hour > 23) return null;   // "15 pm" is no time at all
  if (!ap && hour >= 1 && hour <= 12 && now) hour = nextUpcoming(hour, minute, now);
  return [hour, minute];
}
// No am/pm said: "8:15" at 7 pm is tonight, at 9 pm tomorrow morning, whichever of the
// two comes first (Matt, Pixel, 2026-10-26; "8:15" alone used to mean 08:15).
export function nextUpcoming(hour12, minute, now) {
  const nowMin = now.getHours() * 60 + now.getMinutes();
  const morning = hour12 % 12;
  const wait = (h) => { const d = (h * 60 + minute - nowMin + 1440) % 1440; return d === 0 ? 1440 : d; };
  return wait(morning) <= wait(morning + 12) ? morning : morning + 12;
}

export function parseVoiceCommand(text, now = new Date()) {
  // Dictation with punctuation gives "Stop, stopwatch." and "stop watch" (Pixel,
  // 2026-10-25): punctuation becomes spaces, "stop watch" one word.
  // "7.30 a.m." (Pixel dictation, 2026-10-25): a.m./p.m. become am/pm, 7.30 becomes 7:30.
  const cleaned = text.toLowerCase()
    .replace(/\ba\.m\.?/g, 'am').replace(/\bp\.m\.?/g, 'pm')
    .replace(/\b(\d{1,2})\.(\d{2})(?!\d)/g, '$1:$2')
    .replace(/[,.!?;:](?!\d)/g, ' ').replace(/\bstop\s+watch\b/g, 'stopwatch').replace(/\s+/g, ' ').trim();
  const t = normalizeFractions(normalizeNumbers(cleaned));
  if (/\bcheck\s+(?:the\s+|my\s+)?alarms?\b|what\s+alarms/.test(t)) return { type: 'check_alarm' };
  const cancelAlarmAt = t.match(/\b(?:cancel|stop|delete)\s+(?:the\s+|my\s+)?(\d{1,4})(?:[:\s](\d{1,2}))?\s*(am|pm)?\s+alarm\b/);
  if (cancelAlarmAt) { const hm = clockFrom(cancelAlarmAt[1], cancelAlarmAt[2], cancelAlarmAt[3], now); if (hm) return { type: 'cancel_alarm', hour: hm[0], minute: hm[1] }; }
  if (/\b(?:cancel|stop|delete)\s+(?:the\s+|my\s+)?alarm\b/.test(t)) return { type: 'cancel_alarm' };
  const cancelTimerLen = t.match(/\b(?:stop|cancel)\s+(?:the\s+|my\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+timer\b/);
  if (cancelTimerLen) return { type: 'cancel_timer', seconds: toSeconds(cancelTimerLen[1], cancelTimerLen[2]) };

  // Control words first: "stop timer" must never read as a timer request. Words fixed
  // by Matt (2026-10-25): "stop timer" / "cancel timer", "check timer", "check stopwatch [time]".
  if (/\b(?:stop|cancel)\s+(?:the\s+|my\s+)?timer\b/.test(t)) return { type: 'cancel_timer' };
  if (/\bcheck\s+(?:the\s+|my\s+)?timer\b|how\s+much\s+time\s+is\s+left|how\s+long\s+(?:is\s+)?left/.test(t)) return { type: 'check_timer' };
  if (/\b(?:stop|pause)\s+(?:the\s+)?stopwatch\b/.test(t)) return { type: 'stopwatch_stop' };
  if (/\b(?:reset|clear)\s+(?:the\s+)?stopwatch\b/.test(t)) return { type: 'stopwatch_reset' };
  if (/\bcheck\s+(?:the\s+)?stopwatch(?:\s+time)?\b/.test(t)) return { type: 'check_stopwatch' };

  // "set a timer for 7:30 am": dictation hears "alarm" as "timer"; a clock time after
  // "timer" (a colon, or am/pm) is an alarm, never a duration.
  const timerAsAlarm = t.match(/time(?:r)?\s+(?:for|at)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s+tomorrow)?\b/);
  if (timerAsAlarm && (timerAsAlarm[2] || timerAsAlarm[3])) {
    const hm = clockFrom(timerAsAlarm[1], timerAsAlarm[2], timerAsAlarm[3], now);
    if (hm) return { type: 'alarm', hour: hm[0], minute: hm[1] };
  }

  const timerRes = [
    // "time" too: dictation drops the r (tester, #29). One misheard word may sit
    // between "timer" and the number: Vosk heard "set timer for fifteen minutes"
    // as "set time or fifteen minutes", nothing matched, the model said it has no
    // clock (Matt, Pixel, 2026-10-12, KI-093). "4" is "four" after number words.
    /(?:set\s+(?:a\s+)?)?time(?:r)?\s+(?:(?:for|four|4|or|of|fer)\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)/,
    /(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+timer/,
  ];
  for (const re of timerRes) {
    const m = t.match(re);
    if (m) {
      const seconds = toSeconds(m[1], m[2]);
      if (seconds > 0) return { type: 'timer', seconds };
    }
  }

  // (\d{1,4}): dictation gives "1026" for ten twenty-six; split below. Tried in order,
  // the first that yields a clock time wins: a misheard "for" ("set an alarm four
  // eight fifteen pm", Matt, Pixel, 2026-10-26; the timer patterns have tolerated the
  // same words since 2026-10-12), then no joining word at all, so "alarm 4 15 pm" (a
  // real four) is not read as a filler plus "15 pm".
  const alarmRes = [
    /(?:set\s+(?:an?\s+)?alarm\s+(?:for|at)|alarm\s+(?:for|at)|wake\s+me\s+up\s+at)\s+(\d{1,4})(?:[:\s](\d{1,2}))?\s*(am|pm)?/,
    /(?:set\s+(?:an?\s+)?alarm|alarm)\s+(?:four|4|or|of|fer)\s+(\d{1,4})(?:[:\s](\d{1,2}))?\s*(am|pm)?/,
    /(?:set\s+(?:an?\s+)?alarm|alarm)\s+(\d{1,4})(?:[:\s](\d{1,2}))?\s*(am|pm)?/,
  ];
  for (const re of alarmRes) {
    const alarmM = t.match(re);
    if (!alarmM) continue;
    const hm = clockFrom(alarmM[1], alarmM[2], alarmM[3], now);
    if (hm) return { type: 'alarm', hour: hm[0], minute: hm[1] };
  }
  // "Set it for 8:15" right after cancelling an alarm (Matt, Pixel, 2026-10-26): a clock
  // time (minutes or am/pm) after "set it for" is an alarm, the timer-as-alarm rule.
  const setIt = t.match(/\bset\s+it\s+(?:for|four|4|at|or|of|fer)\s+(\d{1,2})(?:[:\s](\d{2}))?\s*(am|pm)?\b/);
  if (setIt && (setIt[2] || setIt[3])) {
    const hm = clockFrom(setIt[1], setIt[2], setIt[3], now);
    if (hm) return { type: 'alarm', hour: hm[0], minute: hm[1] };
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
    // Opt-in, not opt-out (beta): without a personal verifier enforcing, the wake
    // classifier alone false-triggers on television and other speech, and the
    // audio loop is a real battery cost. A new user should choose to turn this on
    // rather than meet it firing at the TV. Anyone who already enabled it has
    // 'true' stored and keeps it on.
    () => localStorage.getItem('zynkbot_hey_zynk_enabled') === 'true'
  );
  const [voiceInputSource, setVoiceInputSourceRaw] = useState(
    () => localStorage.getItem('zynkbot_voice_input_source') || 'vosk'
  );
  const [fruitlessTone, setFruitlessToneRaw] = useState(
    () => localStorage.getItem('zynkbot_fruitless_tone') === 'true'
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
  const [wakeWordModelReady, setWakeWordModelReady] = useState(
    () => window.WakeWordBridge ? window.WakeWordBridge.isModelReady() : false
  );
  const [wakeWordDownloadProgress, setWakeWordDownloadProgress] = useState(0);
  // off / loading / listening / dictating / turn / paused, pushed by WakeWordService.
  const [wakeState, setWakeState] = useState(() => {
    try { return window.WakeWordBridge?.wakeState?.() || ''; } catch (_) { return ''; }
  });
  const [wakeWordDownloadError, setWakeWordDownloadError] = useState('');

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
    // The hands-free assistant session runs natively and cannot read localStorage;
    // mirror the choice into Kotlin prefs so it dictates with the same engine (2026-09-08).
    try { window.AndroidPaths?.setVoiceInputSource?.(val); } catch (_) {}
  };
  const setFruitlessTone = (val) => {
    setFruitlessToneRaw(val);
    localStorage.setItem('zynkbot_fruitless_tone', val);
    try { window.AndroidPaths?.setFruitlessTone?.(!!val); } catch (_) {}
  };
  const setKeepScreenAwake = (val) => {
    setKeepScreenAwakeRaw(val);
    localStorage.setItem('zynkbot_keep_screen_awake', val);
  };

  const wakeTriggeredRef = useRef(false);
  const ttsSourceRef = useRef(null);
  const ttsAudioCtxRef = useRef(null);
  // Updated by App.jsx on every render so wake callbacks never hold a stale closure.
  const handleSendMessageRef = useRef(null);
  // True only for a transcript handed to the app by the native last-resort path
  // (native answering failed). App.jsx reads and clears it to decide whether the
  // reply is spoken.
  const handsFreeRef = useRef(false);

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
      if (!apiKey) return false;
      const res = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'tts-1', input: cleanForSpeech(text).slice(0, 4096), voice: 'alloy' }),
      });
      if (!res.ok) return false;
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
      return true;
    } catch (e) {
      console.error('[TTS]', e);
      setIsTtsSpeaking(false);
      return false;
    }
  };

  // Android: VoiceCommandBridge runs the command through VoiceCommands/ZynkTimer and
  // returns the line to show. Desktop: the in-app timer module (utils/zynkTimer.js).
  const runClockCommand = (cmd) => {
    const b = window.VoiceCommandBridge;
    if (b) {
      switch (cmd.type) {
        case 'timer': return b.setTimer(cmd.seconds);
        case 'alarm': return b.setAlarm(cmd.hour, cmd.minute, 'Zynkbot');
        case 'stopwatch': return b.startStopwatch();
        case 'cancel_timer': return b.cancelTimer(cmd.seconds || 0);
        case 'cancel_alarm': return b.cancelAlarm(cmd.hour ?? -1, cmd.minute ?? 0);
        case 'check_alarm': return b.checkAlarm();
        case 'check_timer': return b.checkTimer();
        case 'stopwatch_stop': return b.stopStopwatch();
        case 'stopwatch_reset': return b.resetStopwatch();
        case 'check_stopwatch': return b.checkStopwatch();
        default: return '';
      }
    }
    switch (cmd.type) {
      case 'timer': return desktopTimer.setTimer(cmd.seconds);
      case 'alarm': return 'Alarms are on the phone only for now.';
      case 'stopwatch': return desktopTimer.startStopwatch();
      case 'cancel_timer': return desktopTimer.cancelTimer(cmd.seconds || 0);
      case 'cancel_alarm': return 'Alarms are on the phone only for now.';
      case 'check_alarm': return 'Alarms are on the phone only for now.';
      case 'check_timer': return desktopTimer.checkTimer();
      case 'stopwatch_stop': return desktopTimer.stopStopwatch();
      case 'stopwatch_reset': return desktopTimer.resetStopwatch();
      case 'check_stopwatch': return desktopTimer.checkStopwatch();
      default: return '';
    }
  };

  const executeVoiceCommand = (cmd, originalText) => {
    const now = new Date().toISOString();
    let confirmation = '';
    try { confirmation = String(runClockCommand(cmd) || ''); } catch (e) { confirmation = "I couldn't do that."; console.warn('[voice command]', e); }
    const icon = (cmd.type === 'alarm' || cmd.type === 'cancel_alarm' || cmd.type === 'check_alarm') ? '⏰' : '⏱️';
    setMessages(prev => [
      ...prev,
      { id: Date.now(), role: 'user', content: originalText, timestamp: now, recalled_memories: [] },
      { id: Date.now() + 1, role: 'assistant', content: `${icon} ${confirmation}`, timestamp: now, recalled_memories: [], metadata: { model_backend: 'voice command' } },
    ]);
    refreshTimerState();
  };

  // Timer / stopwatch state for the chip above the message box. Android pushes a
  // change through window.__timerChanged; a 1 s tick keeps the countdown moving.
  const readTimerState = () => {
    try {
      if (window.VoiceCommandBridge?.timerState) return JSON.parse(window.VoiceCommandBridge.timerState());
      return desktopTimer.getState();
    } catch (_) { return null; }
  };
  const [timerState, setTimerState] = useState(() => readTimerState());
  const refreshTimerState = () => setTimerState(readTimerState());
  useEffect(() => {
    window.__timerChanged = refreshTimerState;
    const unsub = desktopTimer.subscribe(() => refreshTimerState());
    return () => { window.__timerChanged = null; unsub(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const timerActive = !!(timerState && ((timerState.timers || []).length > 0 || timerState.sw_running || timerState.ringing));
  const cancelAlarmById = (id) => { try { window.VoiceCommandBridge?.cancelAlarmById?.(id); } catch (_) {} refreshTimerState(); };
  // Desktop: when the in-app timer fires, put the line in the chat and speak it if
  // spoken replies are on (the phone's ring service speaks on its own).
  useEffect(() => {
    desktopTimer.setOnDone((line) => {
      const now = new Date().toISOString();
      setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', content: `🔔 ${line}`, timestamp: now, recalled_memories: [], metadata: { model_backend: 'timer' } }]);
      if (ttsEnabled) speakResponse(line);
    });
    return () => desktopTimer.setOnDone(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ttsEnabled]);
  useEffect(() => {
    if (!timerActive) return;
    const h = setInterval(refreshTimerState, 1000);
    return () => clearInterval(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timerActive]);
  const stopRinging = () => {
    try { window.VoiceCommandBridge?.stopRinging?.(); } catch (_) {}
    desktopTimer.stopRing();
    refreshTimerState();
  };
  const cancelTimerById = (id) => { try { if (window.VoiceCommandBridge?.cancelTimerById) window.VoiceCommandBridge.cancelTimerById(id); else desktopTimer.cancelTimerById(id); } catch (_) {} refreshTimerState(); };
  const stopStopwatch = () => { try { runClockCommand({ type: 'stopwatch_stop' }); } catch (_) {} refreshTimerState(); };
  const resetStopwatch = () => { try { runClockCommand({ type: 'stopwatch_reset' }); } catch (_) {} refreshTimerState(); };

  // Wake word service lifecycle — starts/stops with the Hey Zynk toggle on Android.
  useEffect(() => {
    if (!window.WakeWordBridge) return;

    // Route through the Kotlin bridge as well as the browser console: the WebView's
    // own console output was not reaching logcat, so a real TV-test run (2026-09-05)
    // left nine firings with zero trace of what this in-app path actually did.
    const trace = (m) => { try { window.WakeWordBridge?.log?.(m); } catch (_) {} console.log(m); };

    // There is deliberately no in-app "Hey Zynk" handler any more. Every wake-word
    // trigger, screen on or off, app in front or not, is handled natively by the
    // assistant session (chime, Vosk dictation, listening cap, fragment gate, voice
    // commands, spoken reply, Stop, tap-Z-to-cancel). The finished exchange reaches
    // this thread through window.__nativeTurns. Retired 2026-09-07.
    window.__wakeWordModelReady = () => setWakeWordModelReady(true);
    window.__wakeWordState = (s) => setWakeState(String(s || ''));
    window.__wakeWordDownloadProgress = (n) => setWakeWordDownloadProgress(n);
    window.__wakeWordDownloadError = (msg) => {
      setWakeWordDownloadError(msg);
      setWakeWordDownloadProgress(0);
    };
    window.__nativeSpeaking = (on) => setIsNativeSpeaking(!!on);
    // Last-resort hand-off only: the native path could not answer (no speech
    // engine, model call failed), so it opened the app with the transcript.
    window.__handleScreenOffTranscript = (transcript) => {
      if (!transcript?.trim()) return;
      trace('[WakeWord] transcript handed to the app by the native fallback: ' + transcript);
      // Captured hands-free: the reply must be spoken regardless of the in-app setting.
      handsFreeRef.current = true;
      wakeTriggeredRef.current = true;
      setTimeout(() => handleSendMessageRef.current?.(transcript.trim()), 2000);
    };

    // Native prefs may lag localStorage (older build, cleared app data): push the
    // current engine choice before arming so the session matches the selector.
    try { window.AndroidPaths?.setVoiceInputSource?.(localStorage.getItem('zynkbot_voice_input_source') || 'vosk'); } catch (_) {}
    try { window.AndroidPaths?.setFruitlessTone?.(localStorage.getItem('zynkbot_fruitless_tone') === 'true'); } catch (_) {}

    if (heyZynkEnabled) {
      armWakeWord();
    } else {
      clearTimeout(wakeWordRestartTimerRef.current);
      window.WakeWordBridge.stop();
    }

    return () => {
      window.__wakeWordModelReady = null;
      window.__wakeWordState = null;
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

  // Pause wake word detection while dictating or speaking a reply; resume after
  // a delay once both are clear. The detector is kept off during TTS because the
  // speaker output garbles the mic.
  useEffect(() => {
    if (!window.WakeWordBridge || !heyZynkEnabled) return;
    if (isDictating || isTtsSpeaking) {
      window.WakeWordBridge.stop();
    } else {
      armWakeWord(5000);
      return () => clearTimeout(wakeWordRestartTimerRef.current);
    }
  }, [heyZynkEnabled, isDictating, isTtsSpeaking]);

  // Restart wake word when app returns to foreground (screen unlock, app switch).
  useEffect(() => {
    if (!window.WakeWordBridge) return;
    const handleVisible = () => {
      // A push sent while the WebView was paused is lost; re-read on resume.
      try { setIsNativeSpeaking(!!window.WakeWordBridge?.isNativeSpeaking?.()); } catch (_) {}
      if (!heyZynkEnabled || isDictating) return;
      armWakeWord();
    };
    document.addEventListener('visibilitychange', handleVisible);
    return () => document.removeEventListener('visibilitychange', handleVisible);
  }, [heyZynkEnabled, isDictating]);

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
    fruitlessTone, setFruitlessTone,
    keepScreenAwake, setKeepScreenAwake,
    showVoiceModal, setShowVoiceModal,
    // Status state
    isTtsSpeaking,
    isNativeSpeaking,
    isDictating,
    wakeWordModelReady,
    wakeState,
    wakeWordDownloadProgress,
    wakeWordDownloadError,
    // Functions
    stopTts,
    speakResponse,
    executeVoiceCommand,
    // Timer / stopwatch chip
    timerState,
    stopRinging,
    cancelTimerById,
    cancelAlarmById,
    stopStopwatch,
    resetStopwatch,
    // Refs exposed to App.jsx:
    // - handleSendMessageRef: App.jsx updates this every render so callbacks stay current
    // - wakeTriggeredRef: handleSendMessage reads this for wake-only behaviour (web search auto-run)
    // - handsFreeRef: handleSendMessage reads this to decide whether to speak the reply
    handleSendMessageRef,
    wakeTriggeredRef,
    handsFreeRef,
  };
}
