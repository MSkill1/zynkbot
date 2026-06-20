import React, { useState, useRef } from 'react';

/**
 * VoiceButton - Voice input with privacy notice
 *
 * Desktop: Uses Web Speech API (sends audio to Google)
 * Android: Will use native on-device recognition (private)
 *
 * Props:
 * - onTranscript: (text: string) => void - Callback when transcription completes
 * - disabled: boolean - Whether button is disabled
 * - style: object - Optional additional styles
 */
export default function VoiceButton({ onTranscript, disabled, style }) {
  const [showModal, setShowModal] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [hasConsented, setHasConsented] = useState(() => {
    // Check if user has previously consented
    return localStorage.getItem('zynkbot_voice_consent') === 'true';
  });
  const recognitionRef = useRef(null);

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
    } else if (hasConsented) {
      // User already consented, start recording directly
      startWebSpeech();
    } else {
      // Show privacy modal first time
      setShowModal(true);
    }
  };

  const startWebSpeech = () => {
    setShowModal(false);

    // Save consent to localStorage
    if (!hasConsented) {
      localStorage.setItem('zynkbot_voice_consent', 'true');
      setHasConsented(true);
    }

    // Check if Web Speech API is available
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in your browser. Please use Chrome or Edge.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;  // Keep recording until manually stopped
    recognition.interimResults = true;  // Get partial results as user speaks
    recognition.lang = 'en-US';

    let finalTranscript = '';

    recognition.onstart = () => {
      setIsRecording(true);
      finalTranscript = '';
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      // Show interim results in real-time (optional - you could display this)
      console.log('[VoiceButton] Interim:', interimTranscript);
    };

    recognition.onerror = (event) => {
      console.error('[VoiceButton] Speech recognition error:', event.error);
      if (onTranscript && finalTranscript.trim()) {
        onTranscript(finalTranscript.trim());
      }
      setIsRecording(false);
    };

    recognition.onend = () => {
      // Send final transcript when recording ends
      if (onTranscript && finalTranscript.trim()) {
        onTranscript(finalTranscript.trim());
      }
      setIsRecording(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsRecording(false);
  };

  const getButtonStyle = () => {
    if (isRecording) return { background: '#ff5555' };  // Red - recording
    return { background: '#6272a4' };                   // Gray - idle
  };

  const getButtonText = () => {
    if (isRecording) return '■';  // Stop
    return '🎤';                   // Record
  };

  return (
    <>
      <button
        onClick={handleClick}
        disabled={disabled}
        title={isRecording ? 'Click to stop recording' : 'Click for voice input'}
        style={{
          padding: '8px 12px',
          ...getButtonStyle(),
          color: '#f8f8f2',
          border: 'none',
          borderRadius: '4px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          fontSize: '1rem',
          opacity: disabled ? 0.5 : 1,
          minWidth: '48px',
          minHeight: '40px',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'background 0.2s ease',
          ...style
        }}
      >
        {getButtonText()}
      </button>

      {/* Privacy Modal */}
      {showModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
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
              maxWidth: '550px',
              width: '100%',
              border: '1px solid #44475a',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 style={{ color: '#ff5555', marginBottom: '15px', fontSize: '1.3rem' }}>
              ⚠️ Privacy Notice
            </h2>

            <div style={{ color: '#f8f8f2', marginBottom: '20px', lineHeight: '1.6' }}>
              <p style={{ marginBottom: '15px' }}>
                <strong>Desktop limitations:</strong> Rust does not currently have a production-ready
                local speech recognition solution for Windows.
              </p>

              <p style={{ marginBottom: '15px' }}>
                <strong>Good news for Android:</strong> When Zynkbot launches on Android (our primary platform),
                voice input will use native on-device recognition with full privacy - no data leaves your phone.
              </p>

              <p style={{ marginBottom: '15px' }}>
                <strong>Your options on desktop:</strong>
              </p>
              <ul style={{ marginLeft: '20px', marginBottom: '0' }}>
                <li style={{ marginBottom: '8px' }}>
                  <strong style={{ color: '#50fa7b' }}>Type your message</strong> - Complete privacy, no API calls
                </li>
                <li>
                  <strong style={{ color: '#ffb86c' }}>Use Web Speech (Google)</strong> - Audio sent to Google servers (not private)
                </li>
              </ul>
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
                  fontWeight: 'bold',
                  fontSize: '0.95rem'
                }}
              >
                I'll Type Instead
              </button>
              <button
                onClick={startWebSpeech}
                style={{
                  padding: '10px 20px',
                  background: '#ffb86c',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  fontSize: '0.95rem'
                }}
              >
                Use Web Speech (Google)
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
