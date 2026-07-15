const Ctx = window.AudioContext || window.webkitAudioContext;

function drop(ctx, delay, startHz, endHz, vol, dur = 0.28) {
  const osc  = ctx.createOscillator();
  const env  = ctx.createGain();
  const filt = ctx.createBiquadFilter();

  filt.type            = 'bandpass';
  filt.frequency.value = (startHz + endHz) / 2;
  filt.Q.value         = 7;

  osc.connect(filt);
  filt.connect(env);
  env.connect(ctx.destination);

  const t = ctx.currentTime + delay;
  osc.type = 'sine';
  osc.frequency.setValueAtTime(startHz, t);
  osc.frequency.exponentialRampToValueAtTime(endHz, t + 0.06);

  env.gain.setValueAtTime(0, t);
  env.gain.linearRampToValueAtTime(vol, t + 0.007);
  env.gain.exponentialRampToValueAtTime(0.0001, t + dur);

  osc.start(t);
  osc.stop(t + dur + 0.05);
}

export function playNotification(variant = 'ai') {
  if (!Ctx) return;
  try {
    const ctx = new Ctx();
    if (variant === 'ai') {
      // Single soft drop — AI response landed
      drop(ctx, 0,    1100, 240, 0.30);
    } else {
      // Double drop — ZChat message (two ripples)
      drop(ctx, 0,    1450, 360, 0.26, 0.22);
      drop(ctx, 0.14, 1050, 240, 0.20, 0.22);
    }
    setTimeout(() => { try { ctx.close(); } catch (_) {} }, 900);
  } catch (_) { /* no audio context */ }
}
