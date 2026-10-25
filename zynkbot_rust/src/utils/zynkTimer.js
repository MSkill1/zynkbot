// Zynkbot's own timer and stopwatch on the desktop (2026-10-25). The phone has
// ZynkTimer.kt (exact alarms, a ring service); here the app itself counts, so a timer
// only rings while the app is open. State is kept in localStorage so a reload keeps it.
// One timer and one stopwatch at a time, like the phone.
const KEY = 'zynkbot_timer_state';
const RING_SECONDS = 30;

let state = { timers: [], nextId: 1, swStart: 0, swElapsed: 0, ringing: false };
try { const raw = localStorage.getItem(KEY); if (raw) { const p = JSON.parse(raw); state = { ...state, timers: Array.isArray(p.timers) ? p.timers : [], nextId: p.nextId || 1, swStart: p.swStart || 0, swElapsed: p.swElapsed || 0 }; } } catch (_) {}
// A timer that ended while the app was closed is gone; nothing can ring for it now.
state.timers = state.timers.filter((t) => t.end > Date.now());

const listeners = new Set();
const endHandles = new Map();
let ringAudio = null;
let ringLoop = null;
let ringStop = null;
let onDone = null; // (line) => void, set by the voice hook so the line can be spoken

function save() {
  try { localStorage.setItem(KEY, JSON.stringify({ timers: state.timers, nextId: state.nextId, swStart: state.swStart, swElapsed: state.swElapsed })); } catch (_) {}
  listeners.forEach((fn) => { try { fn(getState()); } catch (_) {} });
}

export function durationWords(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const parts = [];
  if (h) parts.push(`${h} hour${h > 1 ? 's' : ''}`);
  if (m) parts.push(`${m} minute${m > 1 ? 's' : ''}`);
  if (s && !h) parts.push(`${s} second${s > 1 ? 's' : ''}`);
  if (!parts.length) parts.push('0 seconds');
  return parts.join(' and ');
}

function armAll() {
  endHandles.forEach((h) => clearTimeout(h));
  endHandles.clear();
  state.timers.forEach((t) => { endHandles.set(t.id, setTimeout(() => fire(t.id), Math.max(0, t.end - Date.now()))); });
}

function fire(id) {
  endHandles.delete(id);
  const t = state.timers.find((x) => x.id === id);
  state.timers = state.timers.filter((x) => x.id !== id);
  const line = `Your ${durationWords(t ? t.total : 0)} timer is done.`;
  startRing(line);
  save();
}

function startRing(line) {
  stopRing();
  state.ringing = true;
  try {
    ringAudio = new Audio(`${process.env.PUBLIC_URL || ''}/wake_chime.wav`);
    const play = () => { try { ringAudio.currentTime = 0; ringAudio.play().catch(() => {}); } catch (_) {} };
    play();
    ringLoop = setInterval(play, 1500);
  } catch (_) {}
  ringStop = setTimeout(stopRing, RING_SECONDS * 1000);
  if (onDone) { try { onDone(line); } catch (_) {} }
}

export function stopRing() {
  if (ringLoop) { clearInterval(ringLoop); ringLoop = null; }
  if (ringStop) { clearTimeout(ringStop); ringStop = null; }
  if (ringAudio) { try { ringAudio.pause(); } catch (_) {} ringAudio = null; }
  if (state.ringing) { state.ringing = false; save(); }
}

export function setOnDone(fn) { onDone = fn; }

export function joinWords(items) {
  if (items.length <= 1) return items[0] || '';
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}
export function setTimer(seconds) {
  const t = { id: state.nextId++, end: Date.now() + seconds * 1000, total: seconds };
  state.timers = [...state.timers, t].sort((a, b) => a.end - b.end);
  armAll();
  save();
  return `Timer set for ${durationWords(seconds)}.`;
}
function removeTimer(id) {
  state.timers = state.timers.filter((x) => x.id !== id);
  armAll();
  stopRing();
  save();
}
export function cancelTimerById(id) { removeTimer(id); }
/** seconds = 0: the only timer, or ask which when there are several. */
export function cancelTimer(seconds = 0) {
  const list = state.timers;
  if (!list.length) return 'No timer is running.';
  if (seconds > 0) {
    const t = list.find((x) => x.total === seconds);
    if (!t) return `There is no ${durationWords(seconds)} timer.`;
    removeTimer(t.id);
    return `The ${durationWords(t.total)} timer is cancelled.`;
  }
  if (list.length === 1) { removeTimer(list[0].id); return 'Timer cancelled.'; }
  return `You have ${list.length} timers: ${joinWords(list.map((t) => durationWords(t.total)))}. Say cancel the ${durationWords(list[0].total)} timer.`;
}
export function secondsLeft(t) { return Math.max(0, Math.ceil((t.end - Date.now()) / 1000)); }
export function checkTimer() {
  const list = state.timers;
  if (!list.length) return 'No timer is running.';
  if (list.length === 1) return `${durationWords(secondsLeft(list[0]))} left.`;
  return list.map((t) => `${durationWords(t.total)} timer: ${durationWords(secondsLeft(t))} left.`).join(' ');
}
export function stopwatchElapsedMs() {
  return state.swStart ? state.swElapsed + (Date.now() - state.swStart) : state.swElapsed;
}
export function startStopwatch() {
  if (!state.swStart) state.swStart = Date.now();
  save();
  return 'Stopwatch started.';
}
export function stopStopwatch() {
  if (state.swStart) { state.swElapsed = stopwatchElapsedMs(); state.swStart = 0; }
  save();
  return `Stopwatch stopped at ${durationWords(Math.floor(state.swElapsed / 1000))}.`;
}
export function resetStopwatch() {
  state.swStart = 0; state.swElapsed = 0;
  save();
  return 'Stopwatch reset.';
}
export function checkStopwatch() {
  const secs = Math.floor(stopwatchElapsedMs() / 1000);
  if (!state.swStart && secs === 0) return 'The stopwatch is not running.';
  return durationWords(secs) + (state.swStart ? '.' : ', stopped.');
}
export function getState() {
  return { timers: state.timers.slice(), alarms: [], sw_running: !!state.swStart, sw_elapsed_ms: stopwatchElapsedMs(), ringing: state.ringing, exact_allowed: true };
}
export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

armAll();
