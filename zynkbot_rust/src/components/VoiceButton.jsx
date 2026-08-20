import React, { useState } from 'react';
import { useVoiceInput } from '../hooks/useVoiceInput';

/**
 * VoiceButton - Records audio and transcribes via OpenAI Whisper API.
 * Works on both desktop and Android (GrapheneOS-compatible).
 *
 * Props:
 * - onTranscript: (text: string) => void
 * - disabled: boolean
 * - style: object
 */
export default function VoiceButton({ onTranscript, disabled, style }) {
  const [showModal, setShowModal] = useState(false);
  const [hasConsented, setHasConsented] = useState(() =>
    localStorage.getItem('zynkbot_voice_consent_openai') === 'true'
  );
  const { isRecording, isTranscribing, startRecording, stopRecording } = useVoiceInput();

  const handleClick = async () => {
    if (isRecording) {
      const text = await stopRecording();
      if (text && onTranscript) onTranscript(text);
    } else if (hasConsented) {
      await startRecording();
    } else {
      setShowModal(true);
    }
  };

  const handleConsent = async () => {
    localStorage.setItem('zynkbot_voice_consent_openai', 'true');
    setHasConsented(true);
    setShowModal(false);
    await startRecording();
  };

  const getButtonStyle = () => {
    if (isTranscribing) return { background: '#44475a' };
    if (isRecording) return { background: '#ff5555' };
    return { background: '#6272a4' };
  };

  const getButtonText = () => {
    if (isRecording) return '■';
    return '🎤';
  };

  const getTitle = () => {
    if (isTranscribing) return 'Transcribing…';
    if (isRecording) return 'Tap to stop and transcribe';
    return 'Voice input (OpenAI Whisper)';
  };

  return (
    <>
      <style>{`@keyframes zynk-spin { to { transform: rotate(360deg); } }`}</style>
      <button
        onClick={handleClick}
        disabled={disabled || isTranscribing}
        title={getTitle()}
        style={{
          padding: '8px 12px',
          ...getButtonStyle(),
          color: '#f8f8f2',
          border: 'none',
          borderRadius: '4px',
          cursor: (disabled || isTranscribing) ? 'not-allowed' : 'pointer',
          fontSize: '1rem',
          opacity: (disabled || isTranscribing) ? 0.5 : 1,
          minWidth: '48px',
          minHeight: '40px',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'background 0.2s ease',
          ...style
        }}
      >
        {isTranscribing ? (
          <span style={{
            display: 'inline-block',
            width: '14px',
            height: '14px',
            border: '2px solid rgba(248,248,242,0.3)',
            borderTopColor: '#f8f8f2',
            borderRadius: '50%',
            animation: 'zynk-spin 0.7s linear infinite',
          }} />
        ) : getButtonText()}
      </button>

      {showModal && (
        <div
          style={{
            position: 'fixed',
            top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '20px'
          }}
          onClick={() => setShowModal(false)}
        >
          <div
            style={{
              background: '#1e1f29',
              borderRadius: '12px',
              padding: '30px',
              maxWidth: '480px',
              width: '100%',
              border: '1px solid #44475a',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 style={{ color: '#ffb86c', marginBottom: '15px', fontSize: '1.3rem' }}>
              🎤 Voice Input — Privacy Notice
            </h2>
            <div style={{ color: '#f8f8f2', marginBottom: '20px', lineHeight: '1.6' }}>
              <p style={{ marginBottom: '12px' }}>
                Your audio is sent to <strong>OpenAI Whisper</strong> for transcription.
                OpenAI retains audio for up to <strong>30 days</strong> for abuse monitoring
                and does not use it to train models by default.
              </p>
              <p style={{ marginBottom: '0', color: '#bd93f9', fontSize: '0.9rem' }}>
                Speak your message, then tap the red ■ button to transcribe.
              </p>
            </div>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowModal(false)}
                style={{
                  padding: '10px 20px',
                  background: '#44475a',
                  color: '#f8f8f2',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleConsent}
                style={{
                  padding: '10px 20px',
                  background: '#50fa7b',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                I Understand, Record
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
