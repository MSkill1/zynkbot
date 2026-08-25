import React, { useState, useImperativeHandle, forwardRef } from 'react';
import { useVoiceInput } from '../hooks/useVoiceInput';

const VoiceButton = forwardRef(function VoiceButton({ onTranscript, disabled, style }, ref) {
  const isAndroid = !!window.VoskBridge;
  const modelReady = !isAndroid || window.VoskBridge.isModelReady();
  const [noModel, setNoModel] = useState(false);
  const { isRecording, isTranscribing, startRecording, stopRecording } = useVoiceInput();

  useImperativeHandle(ref, () => ({
    triggerRecord: async () => {
      if (!isRecording) await startRecording();
    },
  }));

  const handleClick = async () => {
    if (isAndroid && !modelReady) {
      setNoModel(true);
      setTimeout(() => setNoModel(false), 2000);
      return;
    }
    if (isRecording) {
      window.__dictationActive = false;
      const text = await stopRecording();
      if (text && onTranscript) onTranscript(text);
    } else {
      window.__dictationActive = true; // blocks wake-word from hijacking VoskBridge
      await startRecording();
    }
  };

  const getBg = () => {
    if (noModel) return '#ff5555';
    if (isTranscribing) return '#44475a';
    if (isRecording) return '#ff5555';
    return '#6272a4';
  };

  const getLabel = () => {
    if (noModel) return '⚠';
    if (isRecording) return '■';
    return '🎤';
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
      <style>{`@keyframes zynk-spin { to { transform: rotate(360deg); } }`}</style>
      <button
        onClick={handleClick}
        disabled={disabled || isTranscribing}
        title={getTitle()}
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
        ) : getLabel()}
      </button>
    </>
  );
});

export default VoiceButton;
