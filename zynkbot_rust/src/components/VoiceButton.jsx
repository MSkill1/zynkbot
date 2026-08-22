import React, { useState, useEffect } from 'react';
import { useVoiceInput } from '../hooks/useVoiceInput';

export default function VoiceButton({ onTranscript, disabled, style }) {
  const isAndroid = !!window.VoskBridge;
  const [modelReady, setModelReady] = useState(() => isAndroid ? window.VoskBridge.isModelReady() : true);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [awaitingDownloadConfirm, setAwaitingDownloadConfirm] = useState(false);
  const { isRecording, isTranscribing, startRecording, stopRecording } = useVoiceInput();

  useEffect(() => {
    if (!isAndroid) return;
    window.__voskModelReady = () => { setModelReady(true); setIsDownloading(false); setDownloadProgress(0); setAwaitingDownloadConfirm(false); };
    window.__voskDownloadProgress = (pct) => { setIsDownloading(true); setDownloadProgress(pct); };
    window.__voskDownloadError = (msg) => { setIsDownloading(false); alert('Voice model download failed: ' + msg); };
    return () => {
      window.__voskModelReady = null;
      window.__voskDownloadProgress = null;
      window.__voskDownloadError = null;
    };
  }, [isAndroid]);

  const handleClick = async () => {
    if (isDownloading) return;
    if (isAndroid && !modelReady) {
      if (awaitingDownloadConfirm) {
        setAwaitingDownloadConfirm(false);
        setIsDownloading(true);
        window.VoskBridge.downloadModel();
      } else {
        setAwaitingDownloadConfirm(true);
        setTimeout(() => setAwaitingDownloadConfirm(false), 4000);
      }
      return;
    }
    if (isRecording) {
      const text = await stopRecording();
      if (text && onTranscript) onTranscript(text);
    } else {
      await startRecording();
    }
  };

  const getBg = () => {
    if (isTranscribing || isDownloading) return '#44475a';
    if (isRecording) return '#ff5555';
    return '#6272a4';
  };

  const getLabel = () => {
    if (isDownloading) return `${downloadProgress}%`;
    if (isRecording) return '■';
    if (isAndroid && !modelReady) return awaitingDownloadConfirm ? '✓' : '⬇';
    return '🎤';
  };

  const getTitle = () => {
    if (isDownloading) return `Downloading voice model… ${downloadProgress}%`;
    if (isTranscribing) return 'Transcribing…';
    if (isRecording) return 'Tap to stop recording';
    if (isAndroid && !modelReady) return awaitingDownloadConfirm ? 'Tap again to confirm download (~40 MB)' : 'Tap to download offline voice model (~40 MB)';
    if (isAndroid) return 'Voice input (offline)';
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
          background: getBg(),
          color: '#f8f8f2',
          border: 'none',
          borderRadius: '4px',
          cursor: (disabled || isTranscribing || isDownloading) ? 'not-allowed' : 'pointer',
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
}
