import React, { useState, useEffect } from 'react';
import PageIntro from './PageIntro';
import { openUrl } from '@tauri-apps/plugin-opener';

// Offline Vosk dictation ships for Android, Linux desktop and Windows desktop.
// macOS is still excluded: libvosk.dylib is not bundled, so the offline option is
// disabled there and OpenAI Whisper is shown/used instead. The matching runtime
// routing lives in useVoiceInput.js (isDesktopLinux / isDesktopWindows) — keep the
// two in step, or the UI will refuse an option the backend actually supports.
// navigator.platform reports "Win32" on 64-bit Windows too.
const voskAvailable =
  !!window.VoskBridge ||
  navigator.platform.toLowerCase().includes('linux') ||
  navigator.platform.toLowerCase().includes('win');

const overlayStyle = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0, 0, 0, 0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 2000,
  padding: '20px',
};

const cardStyle = {
  background: '#282a36',
  border: '1px solid #44475a',
  borderRadius: '12px',
  width: '100%',
  maxWidth: '420px',
  maxHeight: '90vh',
  overflowY: 'auto',
  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  padding: '24px',
  position: 'relative',
};

const sectionHeadingStyle = {
  color: '#8be9fd',
  fontSize: '0.85rem',
  fontWeight: '600',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  margin: '16px 0 8px',
};

const rowStyle = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '10px 0',
  borderBottom: '1px solid #44475a',
};

const labelStyle = {
  color: '#f8f8f2',
  fontSize: '0.9rem',
};

const mutedStyle = {
  color: '#6272a4',
  fontSize: '0.78rem',
  marginTop: '3px',
};

// Pill-shaped toggle switch rendered as a styled checkbox wrapper.
// Matches the checkbox-in-label pattern used in CollapsibleSidebar.
function Toggle({ checked, onChange, id }) {
  return (
    <label
      htmlFor={id}
      style={{
        position: 'relative',
        display: 'inline-block',
        width: '44px',
        height: '24px',
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ opacity: 0, width: 0, height: 0, position: 'absolute' }}
      />
      <span
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: checked ? '#8be9fd' : '#44475a',
          borderRadius: '24px',
          transition: 'background 0.2s',
        }}
      />
      <span
        style={{
          position: 'absolute',
          top: '3px',
          left: checked ? '23px' : '3px',
          width: '18px',
          height: '18px',
          background: checked ? '#282a36' : '#6272a4',
          borderRadius: '50%',
          transition: 'left 0.2s, background 0.2s',
        }}
      />
    </label>
  );
}

const FLAT_COMMANDS = [
  { cmd: '"Hey Zynk"', desc: 'One question per wake. Wait for the tone, then speak. Zynkbot answers, then waits for the next "Hey Zynk" — it never listens on its own.' },
  { cmd: 'Tap the Z', desc: 'If you woke it by accident: tapping the green Z cancels listening or stops the reply without sending anything.' },
  { cmd: '"Stop"', desc: 'Typed or dictated in the app, stops a spoken reply. Hands-free, the microphone is off while Zynkbot speaks, so use the Stop button or tap the Z instead.' },
  { cmd: '"Set a timer for 10 minutes"', desc: 'Zynkbot\'s own timer: it rings and tells you with the phone asleep or the app closed. Any duration in seconds, minutes or hours. Set as many as you like; each shows above the message box.' },
  { cmd: '"Stop timer" / "Cancel timer"', desc: 'Cancels the timer. With several running it names them; say "cancel the 10 minute timer".' },
  { cmd: '"Check timer"', desc: 'Says how much time is left on each.' },
  { cmd: 'When it rings', desc: 'Say "Hey Zynk", tap Stop on the notification, or tap Stop in the app. It rings for 30 seconds, then stops on its own.' },
  { cmd: '"Set an alarm for 7:30 AM"', desc: 'Zynkbot\'s own alarm, same as the timer. If Zynkbot is not allowed exact alarms on this phone, the clock app takes it instead.' },
  { cmd: '"Cancel alarm"', desc: 'Removes the alarm. With several set it names them; say "cancel the 7 30 alarm".' },
  { cmd: '"Check alarms"', desc: 'Says which alarms are set.' },
  { cmd: '"Start stopwatch"', desc: 'Starts Zynkbot\'s stopwatch.' },
  { cmd: '"Stop stopwatch" / "Reset stopwatch"', desc: 'Stops it, keeping the time; reset clears it.' },
  { cmd: '"Check stopwatch" / "Check stopwatch time"', desc: 'Says the time on the stopwatch.' },
];

function VoiceCommandsModal({ onClose }) {
  return (
    <div style={{ ...overlayStyle, zIndex: 2100 }} onClick={onClose}>
      <div style={{ ...cardStyle, maxWidth: '480px' }} onClick={(e) => e.stopPropagation()}>
        <div style={{ marginBottom: '16px' }}>
          <h2 style={{ margin: 0, color: '#8be9fd', fontSize: '1.1rem' }}>📋 Voice Commands</h2>
        </div>

        {FLAT_COMMANDS.map(({ cmd, desc }) => (
          <div key={cmd} style={{ marginBottom: '12px' }}>
            <div style={{
              fontFamily: 'monospace',
              fontSize: '0.88rem',
              color: '#50fa7b',
              background: '#1e1f29',
              border: '1px solid #44475a',
              borderRadius: '5px',
              padding: '5px 10px',
              display: 'inline-block',
              marginBottom: '3px',
            }}>
              {cmd}
            </div>
            <div style={{ color: '#9aa5c4', fontSize: '0.82rem', paddingLeft: '2px' }}>{desc}</div>
          </div>
        ))}

      </div>

      {/* Floating close button - bottom right, like settings panel */}
      <button
        onClick={onClose}
        style={{
          position: 'fixed', bottom: '20px', right: '20px',
          width: '56px', height: '56px', borderRadius: '50%',
          background: '#44475a', color: '#f8f8f2', border: 'none',
          fontSize: '1.5rem', cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          zIndex: 2110, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}
      >✕</button>
    </div>
  );
}

export default function VoiceModal({
  isOpen,
  onClose,
  voiceInputSource,
  onVoiceSourceChange,
  heyZynkEnabled,
  onHeyZynkChange,
  fruitlessTone,
  onFruitlessToneChange,
  wakeWordModelReady,
  wakeState = '',
  wakeWordDownloadProgress,
  wakeWordDownloadError,
  onDownloadModels,
  ttsEnabled,
  onTtsEnabledChange,
  keepScreenAwake,
  onKeepScreenAwakeChange,
}) {
  const [showCommands, setShowCommands] = useState(false);
  const [exactAllowed, setExactAllowed] = useState(true);
  const [startOnBoot, setStartOnBoot] = useState(() => { try { return window.WakeWordBridge?.startOnBoot ? !!window.WakeWordBridge.startOnBoot() : true; } catch (_) { return true; } });
  useEffect(() => {
    if (!isOpen) return;
    try { if (window.VoiceCommandBridge?.canScheduleExact) setExactAllowed(!!window.VoiceCommandBridge.canScheduleExact()); } catch (_) {}
  }, [isOpen]);

  // Voice enrollment card (Android): the wake service's status, re-read while the
  // modal is open so the count moves as the clips come in.
  const [enroll, setEnroll] = useState(null);
  const readEnroll = () => {
    try {
      const raw = window.WakeWordBridge?.enrollmentStatus?.();
      if (raw) setEnroll(JSON.parse(raw));
    } catch (_) {}
  };
  useEffect(() => {
    if (!isOpen || !window.WakeWordBridge?.enrollmentStatus) return;
    readEnroll();
    const t = setInterval(readEnroll, 1500);
    return () => clearInterval(t);
  }, [isOpen]);
  const enrollCall = (fn) => () => { try { fn(); } catch (_) {} readEnroll(); };
  // The explanation shown before teaching starts: 'start' (first time) or 'again'.
  const [teachInfo, setTeachInfo] = useState(null);

  if (!isOpen) return null;

  const effectiveSource = voskAvailable ? (voiceInputSource || 'vosk') : 'openai';

  return (
    <div data-modal="" style={overlayStyle} onClick={onClose}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div style={{ marginBottom: '8px' }}>
          <h2 style={{ margin: 0, color: '#8be9fd', fontSize: '1.2rem' }}>🎙️ Voice Settings</h2>
        </div>
        <PageIntro introKey="voice" accent="#8be9fd">Dictation types what you say into the message box. "Hey Zynk" asks a question hands-free: one question per wake, and it never listens on its own. Spoken replies read the answers aloud. <b>Try this:</b> turn on Hey Zynk and wait a few seconds for the microphone to turn green and shrink to a dot. Then say it, wait for the tone, and ask something short.</PageIntro>

        {/* Hey Zynk first: the switch people open this screen for was below the fold,
            behind the close button (Matt, 2026-09-20). */}
        {/* Hey Zynk — only shown on Android (window.WakeWordBridge present) */}
        {!!window.WakeWordBridge && (
          <>
            <p style={sectionHeadingStyle}>Hey Zynk — Wake Word Detection</p>
            <div style={rowStyle}>
              <span style={labelStyle}>Enable wake word</span>
              <Toggle
                id="hey-zynk-toggle"
                checked={heyZynkEnabled}
                onChange={onHeyZynkChange}
              />
            </div>

            {/* Status line: the models take a few seconds to load after the switch; people
                said "Hey Zynk" at once and thought it was broken (Matt, 2026-10-25). */}
            {heyZynkEnabled && (() => {
              const s = wakeState;
              const line = s === 'listening' ? ['#50fa7b', '● Listening for "Hey Zynk".']
                : (s === 'dictating' || s === 'turn') ? ['#50fa7b', '● Hearing you.']
                : s === 'paused' ? ['#ffb86c', 'Paused: another app is using the microphone.']
                : ['#ffb86c', '⏳ Starting up. Give it a few seconds: the microphone turns green, then shrinks to a dot. Then say "Hey Zynk".'];
              return <p style={{ color: line[0], fontSize: '0.82rem', lineHeight: 1.45, margin: '6px 0 0 0' }}>{line[1]}</p>;
            })()}

            {/* After a restart (2026-10-25): the assistant service tries to bring the
                listener back at boot; if Android refuses, a notification offers a tap. */}
            {!!window.WakeWordBridge?.startOnBoot && (
              <div style={rowStyle}>
                <div>
                  <div style={labelStyle}>Start listening when the phone starts</div>
                  <div style={mutedStyle}>If Hey Zynk was on, it comes back after a restart. If the phone does not allow it, a notification offers a tap instead.</div>
                </div>
                <Toggle
                  id="start-on-boot-toggle"
                  checked={startOnBoot}
                  onChange={(val) => { setStartOnBoot(val); try { window.WakeWordBridge.setStartOnBoot(!!val); } catch (_) {} }}
                />
              </div>
            )}

            {/* How it works, right under the switch (Matt, 2026-10-25). */}
              <div style={{
                background: '#1e1f29',
                border: '1px solid #44475a',
                borderRadius: '8px',
                padding: '12px 14px',
                marginTop: '12px',
                marginBottom: '4px',
              }}>
                <p style={{ ...sectionHeadingStyle, margin: '0 0 8px 0' }}>How It Works</p>
                <ol style={{ margin: 0, paddingLeft: '18px', color: '#9aa5c4', fontSize: '0.84rem', lineHeight: '1.6' }}>
                  <li>Say <span style={{ color: '#50fa7b', fontFamily: 'monospace' }}>"Hey Zynk"</span> and wait for the tone before speaking.</li>
                  <li>Speak your question or command.</li>
                  <li>Stop talking. After 1.5 seconds of silence you'll hear a second tone confirming your message was sent.</li>
                  <li>Say <span style={{ color: '#50fa7b', fontFamily: 'monospace' }}>"Hey Zynk"</span> again after each response to continue the conversation.</li>
                </ol>
              </div>

            <div style={rowStyle}>
              <span style={labelStyle}>Play a tone when a trigger hears nothing</span>
              <Toggle
                id="fruitless-tone-toggle"
                checked={!!fruitlessTone}
                onChange={onFruitlessToneChange}
              />
            </div>
            <p style={{ color: '#9aa5c4', fontSize: '0.8rem', lineHeight: 1.45, margin: '4px 0 0 0' }}>
              If "Hey Zynk" fires but hears nothing, nothing is sent. With this on, a second
              tone tells you it closed; with it off, you only hear the first chime.
            </p>

            {/* Teach Zynkbot your voice (2026-10-22): Start, then say "Hey Zynk" 20 times;
                each one is saved as a real clip and only chimes, and until the twenty are
                in the wake word answers nothing (paused: it does not count either). The
                phone then trains its personal verifier from them and, if it tests well
                enough, enforces it (WakeWordService enrollment, WakeVerifierTrainer). */}
            {heyZynkEnabled && enroll && (() => {
              const s = enroll.state;
              const collecting = s === 'active' || s === 'paused';
              const done = s === 'done' || enroll.enforcing;
              const btn = (label, onClick, primary) => (
                <button onClick={onClick} style={{
                  padding: '8px 14px', borderRadius: '6px', border: '1px solid #6272a4', whiteSpace: 'nowrap',
                  background: primary ? '#50fa7b' : '#44475a', color: primary ? '#282a36' : '#f8f8f2', cursor: 'pointer',
                }}>{label}</button>
              );
              return (
                <div style={{ ...rowStyle, alignItems: 'center', gap: '12px' }}>
                  <div style={{ flex: 1 }}>
                    <span style={labelStyle}>{collecting ? 'Say "Hey Zynk", wait for the count, say it again' : 'Teach Zynkbot your voice'}</span>
                    <div style={mutedStyle}>
                      {collecting
                        ? (s === 'paused'
                          ? 'Paused: nothing counts. Until the twenty are in, "Hey Zynk" does not answer.'
                          : 'Move around: facing the phone, facing away, at arm\'s length, from across the room. Somewhere quiet. Until the twenty are in, "Hey Zynk" only counts; it does not answer.')
                        : done
                          ? (enroll.enforcing ? 'Done. A "Hey Zynk" that does not sound like you is ignored.' : 'Done. It keeps learning as you use it.')
                          : 'Say "Hey Zynk" twenty times so it learns to tell you from the TV. Take your time; the count is kept.'}
                    </div>
                  </div>
                  {collecting && (
                    <div style={{ textAlign: 'center', minWidth: '56px' }}>
                      <div style={{ fontSize: '1.6rem', fontWeight: 'bold', lineHeight: 1, color: '#f8f8f2' }}>{enroll.count}</div>
                      <div style={{ fontSize: '0.7rem', color: '#9aa5c4' }}>of {enroll.needed}</div>
                    </div>
                  )}
                  {s === 'active' && btn('Pause', enrollCall(() => window.WakeWordBridge.pauseEnrollment(true)), false)}
                  {s === 'paused' && btn('Resume', enrollCall(() => window.WakeWordBridge.pauseEnrollment(false)), true)}
                  {!collecting && btn(done ? 'Teach again' : 'Start', () => setTeachInfo(done ? 'again' : 'start'), !done)}
                </div>
              );
            })()}

            {/* What teaching does, before it starts. Shown over the settings; Start begins
                counting (and, for "again", clears the clips already on the phone). */}
            {teachInfo && (
              <div style={{ ...overlayStyle, zIndex: 2100 }} onClick={() => setTeachInfo(null)}>
                <div style={{ ...cardStyle, maxWidth: '380px', padding: '20px' }} onClick={(e) => e.stopPropagation()}>
                  <h3 style={{ margin: '0 0 10px 0', color: '#8be9fd', fontSize: '1.05rem' }}>Teach Zynkbot your voice</h3>
                  <div style={{ color: '#f8f8f2', fontSize: '0.88rem', lineHeight: 1.5 }}>
                    <p style={{ margin: '0 0 10px 0' }}>
                      Say "Hey Zynk" twenty times. Wait for the count to move between each one, and move
                      around: facing the phone, facing away, from across the room.
                    </p>
                    <p style={{ margin: '0 0 10px 0' }}>
                      When the twenty are in, Zynkbot builds a profile of your voice and starts ignoring
                      wake words that do not sound like you. It keeps learning from every "Hey Zynk" you
                      use. A television may still set it off at first; that should fade as it learns.
                    </p>
                    <p style={{ margin: '0 0 10px 0' }}>Until the twenty are in, "Hey Zynk" will not answer questions.</p>
                    {teachInfo === 'again' && (
                      <p style={{ margin: '0 0 10px 0', color: '#ffb86c' }}>Starting over clears the voice clips already on this phone.</p>
                    )}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '14px' }}>
                    <button onClick={() => setTeachInfo(null)} style={{
                      padding: '8px 14px', borderRadius: '6px', border: '1px solid #6272a4', background: '#44475a', color: '#f8f8f2', cursor: 'pointer',
                    }}>Cancel</button>
                    <button onClick={() => { setTeachInfo(null); enrollCall(() => window.WakeWordBridge.startEnrollment())(); }} style={{
                      padding: '8px 14px', borderRadius: '6px', border: '1px solid #6272a4', background: '#50fa7b', color: '#282a36', cursor: 'pointer',
                    }}>Start</button>
                  </div>
                </div>
              </div>
            )}

            {heyZynkEnabled && !wakeWordModelReady && (
              <div style={{
                padding: '12px',
                background: '#1e1f29',
                borderRadius: '8px',
                border: '1px solid #8be9fd',
                marginTop: '8px',
              }}>
                <p style={{ color: '#8be9fd', fontSize: '0.85rem', margin: '0 0 8px 0', fontWeight: 'bold' }}>
                  Voice models needed
                </p>
                {wakeWordDownloadProgress > 0 ? (
                  <div>
                    <div style={{ background: '#44475a', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
                      <div style={{
                        background: '#8be9fd',
                        height: '100%',
                        width: `${wakeWordDownloadProgress}%`,
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                    <p style={{ color: '#9aa5c4', fontSize: '0.8rem', margin: '6px 0 0' }}>
                      Downloading… {wakeWordDownloadProgress}%
                    </p>
                  </div>
                ) : (
                  <>
                    {wakeWordDownloadError && (
                      <p style={{ color: '#ff5555', fontSize: '0.8rem', margin: '0 0 6px 0' }}>
                        {wakeWordDownloadError}
                      </p>
                    )}
                    <button
                      onClick={onDownloadModels}
                      style={{
                        padding: '6px 14px',
                        background: '#8be9fd',
                        color: '#282a36',
                        border: 'none',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        fontWeight: 'bold',
                        cursor: 'pointer',
                      }}
                    >
                      Download (~3 MB)
                    </button>
                  </>
                )}
              </div>
            )}

            {heyZynkEnabled && wakeWordModelReady && (
              <p style={{ color: '#50fa7b', fontSize: '0.82rem', margin: '6px 0 0' }}>
                Models ready ✓
              </p>
            )}
          </>
        )}

        {/* Dictation */}
        <p style={{ ...sectionHeadingStyle, marginTop: '12px' }}>Dictation Engine</p>

        <div
          style={{
            padding: '10px',
            background: effectiveSource === 'vosk' ? '#1e1f29' : 'transparent',
            borderRadius: '6px',
            border: effectiveSource === 'vosk' ? '1px solid #44475a' : '1px solid transparent',
            marginBottom: '6px',
            cursor: voskAvailable ? 'pointer' : 'default',
            opacity: voskAvailable ? 1 : 0.5,
          }}
          onClick={() => voskAvailable && onVoiceSourceChange('vosk')}
        >
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: voskAvailable ? 'pointer' : 'default' }}>
            <input
              type="radio"
              name="voiceSource"
              value="vosk"
              checked={effectiveSource === 'vosk'}
              disabled={!voskAvailable}
              onChange={() => onVoiceSourceChange('vosk')}
              style={{ accentColor: '#8be9fd', cursor: voskAvailable ? 'pointer' : 'default' }}
            />
            <div>
              <div style={{ ...labelStyle, fontWeight: '500' }}>
                Offline — Vosk
                {!voskAvailable && <span style={{ color: '#6272a4', fontWeight: '400' }}> (not available on macOS)</span>}
              </div>
              <div style={mutedStyle}>Runs on device. No internet needed. Produces no punctuation; large language models do not need punctuation to understand a request.</div>
            </div>
          </label>
        </div>

        <div
          style={{
            padding: '10px',
            background: effectiveSource === 'openai' ? '#1e1f29' : 'transparent',
            borderRadius: '6px',
            border: effectiveSource === 'openai' ? '1px solid #44475a' : '1px solid transparent',
            cursor: 'pointer',
          }}
          onClick={() => onVoiceSourceChange('openai')}
        >
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
            <input
              type="radio"
              name="voiceSource"
              value="openai"
              checked={effectiveSource === 'openai'}
              onChange={() => onVoiceSourceChange('openai')}
              style={{ accentColor: '#8be9fd', cursor: 'pointer' }}
            />
            <div>
              <div style={{ ...labelStyle, fontWeight: '500' }}>OpenAI Whisper</div>
              <div style={mutedStyle}>Cloud transcription with punctuation. Requires OPENAI_API_KEY. Audio is sent to OpenAI; its API policy says it is not used for training and the transcription service keeps no audio for abuse monitoring (policy read 2026-10-25).</div>
              <button
                onClick={(e) => { e.preventDefault(); openUrl('https://developers.openai.com/api/docs/guides/your-data'); }}
                style={{ marginTop: '6px', padding: '5px 10px', borderRadius: '6px', border: '1px solid #6272a4', background: 'rgba(98,114,164,0.2)', color: '#8be9fd', fontSize: '0.8rem', cursor: 'pointer' }}
              >Read OpenAI's data retention policy</button>
            </div>
          </label>
        </div>

        {/* Voice Response */}
        <p style={sectionHeadingStyle}>Voice Response</p>
        <div style={rowStyle}>
          <div>
            <div style={labelStyle}>Speak replies in the app</div>
            <div style={mutedStyle}>
              Hands-free "Hey Zynk" replies are always spoken, whether the screen is on
              or off and whether the app is open or not. Turn this on to also hear replies
              to messages you type or dictate in the app.
              {ttsEnabled && ' Uses OpenAI TTS (alloy) — requires OPENAI_API_KEY. Say "stop" or use the Stop button to interrupt.'}
            </div>
          </div>
          <Toggle
            id="tts-toggle"
            checked={ttsEnabled}
            onChange={onTtsEnabledChange}
          />
        </div>


        {/* Web Search: the "Auto-execute in voice sessions" toggle was removed
            2026-09-20 — hands-free always searches now, unconditionally (GitHub
            #26: there was never a way to answer "want me to search?" by voice).
            An "always search, skip the confirmation" setting for typed chat is a
            different, unbuilt feature — see the roadmap. */}

        {/* Screen */}
        <p style={sectionHeadingStyle}>Screen</p>
        <div style={{ ...rowStyle, borderBottom: 'none' }}>
          <div>
            <div style={labelStyle}>Keep screen awake</div>
            <div style={mutedStyle}>Stops the display sleeping while Zynkbot is open. Does not affect listening.</div>
          </div>
          <Toggle
            id="keep-screen-awake-toggle"
            checked={keepScreenAwake}
            onChange={onKeepScreenAwakeChange}
          />
        </div>

        {/* Zynkbot's own timers and alarms need the "Alarms & reminders" switch on
            Android 12+ (denied by default on 14+); without it the clock app takes them. */}
        {!!window.VoiceCommandBridge?.canScheduleExact && !exactAllowed && (
          <div style={{ ...rowStyle, alignItems: 'center', gap: '12px' }}>
            <div style={{ flex: 1 }}>
              <div style={labelStyle}>Let Zynkbot run its own timers and alarms</div>
              <div style={mutedStyle}>Android needs your permission for an app to ring at an exact time. Until then timers and alarms go to the clock app, which cannot ring from the lock screen or say how long is left.</div>
            </div>
            <button onClick={() => { try { window.VoiceCommandBridge.openExactAlarmSettings(); } catch (_) {} }}
              style={{ padding: '8px 14px', borderRadius: '6px', border: 'none', background: '#50fa7b', color: '#282a36', cursor: 'pointer', whiteSpace: 'nowrap', fontWeight: 600 }}>
              Allow
            </button>
          </div>
        )}

        {/* Voice Commands reference */}
        <div style={{ marginTop: '20px', paddingTop: '14px', borderTop: '1px solid #44475a', textAlign: 'center' }}>
          <button
            onClick={() => setShowCommands(true)}
            style={{
              padding: '8px 18px',
              background: 'rgba(98,114,164,0.2)',
              color: '#8be9fd',
              border: '1px solid #6272a4',
              borderRadius: '6px',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: 'pointer',
              width: '100%',
            }}
          >
            📋 View Voice Commands
          </button>
        </div>

      </div>

      {/* Floating close button - bottom right, like settings panel */}
      <button
        onClick={onClose}
        style={{
          position: 'fixed', bottom: '20px', right: '20px',
          width: '56px', height: '56px', borderRadius: '50%',
          background: '#44475a', color: '#f8f8f2', border: 'none',
          fontSize: '1.5rem', cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          zIndex: 1010, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}
      >✕</button>

      {showCommands && <VoiceCommandsModal onClose={() => setShowCommands(false)} />}
    </div>
  );
}
