import React, { useState } from 'react';

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
  { cmd: '"Thank you Zynk" / "Goodbye Zynk"', desc: 'Ends the session and returns to standby.' },
  { cmd: '"Never mind" / "Cancel"', desc: 'If you accidentally woke it — discards your recording without sending.' },
  { cmd: '"Stop"', desc: 'Stops the spoken response.' },
  { cmd: '"Set a timer for 10 minutes"', desc: 'Works with any duration — seconds, minutes, or hours.' },
  { cmd: '"Set an alarm for 7:30 AM"', desc: 'Sets a system alarm.' },
  { cmd: '"Start stopwatch"', desc: 'Starts the device stopwatch.' },
];

function VoiceCommandsModal({ onClose }) {
  return (
    <div style={{ ...overlayStyle, zIndex: 2100 }} onClick={onClose}>
      <div style={{ ...cardStyle, maxWidth: '480px' }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ margin: 0, color: '#8be9fd', fontSize: '1.1rem' }}>📋 Voice Commands</h2>
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: '#6272a4', fontSize: '1.3rem', cursor: 'pointer', padding: '0 4px', lineHeight: 1 }}
          >✕</button>
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

        <p style={{ color: '#6272a4', fontSize: '0.78rem', margin: '12px 0 0', borderTop: '1px solid #44475a', paddingTop: '10px' }}>
          Tap anywhere outside to close.
        </p>
      </div>
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
  wakeWordModelReady,
  wakeWordDownloadProgress,
  wakeWordDownloadError,
  onDownloadModels,
  ttsEnabled,
  onTtsEnabledChange,
  webSearchAutoExecute,
  onWebSearchAutoExecuteChange,
  keepScreenAwake,
  onKeepScreenAwakeChange,
}) {
  const [showCommands, setShowCommands] = useState(false);

  if (!isOpen) return null;

  const effectiveSource = voskAvailable ? (voiceInputSource || 'vosk') : 'openai';

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={cardStyle} onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <h2 style={{ margin: 0, color: '#8be9fd', fontSize: '1.2rem' }}>🎙️ Voice Settings</h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#6272a4',
              fontSize: '1.3rem',
              cursor: 'pointer',
              padding: '0 4px',
              lineHeight: 1,
            }}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* How It Works */}
        {!!window.WakeWordBridge && (
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
              <li>Stop talking. After ~2 seconds you'll hear a second tone confirming your message was sent.</li>
              <li>Say <span style={{ color: '#50fa7b', fontFamily: 'monospace' }}>"Hey Zynk"</span> again after each response to continue the conversation.</li>
            </ol>
          </div>
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
              <div style={mutedStyle}>Runs on device. No internet. No punctuation — LLMs handle it fine.</div>
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
              <div style={mutedStyle}>Cloud transcription with punctuation. Requires OPENAI_API_KEY. OpenAI retains audio 30 days for abuse review only.</div>
            </div>
          </label>
        </div>

        {/* Hey Zynk — only shown on Android (window.WakeWordBridge present) */}
        {!!window.WakeWordBridge && (
          <>
            <p style={sectionHeadingStyle}>Hey Zynk — Always Listening</p>
            <div style={rowStyle}>
              <span style={labelStyle}>Enable wake word</span>
              <Toggle
                id="hey-zynk-toggle"
                checked={heyZynkEnabled}
                onChange={onHeyZynkChange}
              />
            </div>

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

        {/* Voice Response */}
        <p style={sectionHeadingStyle}>Voice Response</p>
        <div style={rowStyle}>
          <div>
            <div style={labelStyle}>Speak replies in the app</div>
            <div style={mutedStyle}>
              Hands-free "Hey Zynk" replies (screen off, app in the background) are
              always spoken. Turn this on to also hear replies while the app is open.
              {ttsEnabled && ' Uses OpenAI TTS (alloy) — requires OPENAI_API_KEY. Say "stop" or use the Stop button to interrupt.'}
            </div>
          </div>
          <Toggle
            id="tts-toggle"
            checked={ttsEnabled}
            onChange={onTtsEnabledChange}
          />
        </div>


        {/* Web Search */}
        <p style={sectionHeadingStyle}>Web Search</p>
        <div style={{ ...rowStyle, borderBottom: 'none' }}>
          <div>
            <div style={labelStyle}>Auto-execute in voice sessions</div>
            <div style={mutedStyle}>When off, Zynkbot pauses and asks before searching the web.</div>
          </div>
          <Toggle
            id="web-search-auto-toggle"
            checked={webSearchAutoExecute}
            onChange={onWebSearchAutoExecuteChange}
          />
        </div>

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

      {showCommands && <VoiceCommandsModal onClose={() => setShowCommands(false)} />}
    </div>
  );
}
