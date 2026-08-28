import React from 'react';

const voskAvailable =
  !!window.VoskBridge || navigator.platform.toLowerCase().includes('linux');

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
  conversationModeEnabled,
  onConversationModeChange,
}) {
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
                {!voskAvailable && <span style={{ color: '#6272a4', fontWeight: '400' }}> (Linux/Android only)</span>}
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
            <div style={labelStyle}>Speak responses aloud</div>
            {ttsEnabled && (
              <div style={mutedStyle}>OpenAI TTS (alloy) — requires OPENAI_API_KEY</div>
            )}
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

        {/* Conversation Mode */}
        <p style={sectionHeadingStyle}>Conversation Mode</p>
        <div style={{ ...rowStyle, borderBottom: 'none' }}>
          <div>
            <div style={labelStyle}>Keep screen awake</div>
            <div style={mutedStyle}>Useful for hands-free use during a session.</div>
          </div>
          <Toggle
            id="conversation-mode-toggle"
            checked={conversationModeEnabled}
            onChange={onConversationModeChange}
          />
        </div>

      </div>
    </div>
  );
}
