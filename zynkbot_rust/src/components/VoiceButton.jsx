import React, { useState, useImperativeHandle, forwardRef } from 'react';
import { useVoiceInput } from '../hooks/useVoiceInput';

const VoiceButton = forwardRef(function VoiceButton({ onTranscript, disabled, style }, ref) {
  const isAndroid = !!window.VoskBridge;
  const modelReady = !isAndroid || window.VoskBridge.isModelReady();
  const [noModel, setNoModel] = useState(false);
  const { isRecording, isTranscribing, startRecording, stopRecording } = useVoiceInput();

  // Dictation and wake word share a single microphone, so the wake detector has to be
  // stopped outright while this button is in use — suppressing its callback is not
  // enough, because a detection triggered by the user's own dictation audio can be
  // delivered a moment later. App.jsx owns the detector and listens for this event;
  // VoiceButton's useVoiceInput state is not visible from there.
  const announceDictation = (active) => {
    window.__dictationActive = active;
    window.dispatchEvent(new CustomEvent('zynkbot:dictation', { detail: { active } }));
  };

  useImperativeHandle(ref, () => ({
    triggerRecord: async () => {
      if (!isRecording) {
        announceDictation(true);
        await startRecording();
      }
    },
  }));

  const handleClick = async () => {
    if (isAndroid && !modelReady) {
      setNoModel(true);
      setTimeout(() => setNoModel(false), 2000);
      return;
    }
    if (isRecording) {
      // Stay active until the transcript has been handed over. Clearing this on tap
      // left a window in which the wake detector was unguarded while still holding
      // the user's dictation audio, which started an unrequested wake recording.
      try {
        const text = await stopRecording();
        if (text && onTranscript) onTranscript(text);
      } finally {
        announceDictation(false);
      }
    } else {
      announceDictation(true);
      await startRecording();
    }
  };

  const getBg = () => {
    if (noModel) return '#ff5555';
    if (isTranscribing) return '#44475a';
    if (isRecording) return '#ff5555';
    return '#6272a4';
  };

  // Vector icons instead of an emoji and a text square: emoji render through the
  // phone's font (different on every device) and the square sat off-centre on its
  // text baseline (2026-09-06). These centre exactly and stay crisp at any size.
  const MicIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="17" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
  const StopIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="3" />
    </svg>
  );

  const getLabel = () => {
    if (noModel) return '⚠';
    if (isRecording) return <StopIcon />;
    return <MicIcon />;
  };

  const getTitle = () => {
    if (noModel) return 'Voice model not installed';
    if (isTranscribing) return 'Transcribing…';
    if (isRecording) return 'Tap to stop recording';
    if (isAndroid) return 'Voice input (offline)';
    return 'Voice input';
  };

  return (
    <>
      <style>{`
        @keyframes zynk-spin { to { transform: rotate(360deg); } }
        @keyframes zynk-pulse {
          0% { box-shadow: 0 0 0 0 rgba(255, 85, 85, 0.55); }
          100% { box-shadow: 0 0 0 14px rgba(255, 85, 85, 0); }
        }
        @media (prefers-reduced-motion: reduce) { .zynk-mic-recording { animation: none !important; } }
      `}</style>
      <button
        onClick={handleClick}
        disabled={disabled || isTranscribing}
        title={getTitle()}
        className={isRecording ? 'zynk-mic-recording' : undefined}
        style={{
          padding: '8px 12px',
          background: getBg(),
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
          lineHeight: 0,
          transition: 'background 0.2s ease',
          animation: isRecording ? 'zynk-pulse 1.2s ease-out infinite' : 'none',
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
        ) : getLabel()}
      </button>
    </>
  );
});

export default VoiceButton;
