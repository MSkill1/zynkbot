import { useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

export function useVoiceInput() {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const audioBufferRef = useRef([]);

  const isAndroid = () => !!window.VoskBridge;
  // 'vosk' (offline, no punctuation) or 'openai' (cloud Whisper). Persists in localStorage.
  const getSource = () => localStorage.getItem('zynkbot_voice_input_source') || 'vosk';

  // ── Android / Vosk path ──────────────────────────────────────────────────

  const startRecordingVosk = () => {
    window.__voskPartial = null; // clear any stale wake-word callback before dictation
    setIsRecording(true);
    window.__voskError = (msg) => {
      setIsRecording(false);
      console.error('[VoiceInput] Vosk start error:', msg);
      if (!msg.includes('not downloaded')) alert('Voice input error: ' + msg);
    };
    window.VoskBridge.startListening();
  };

  const stopRecordingVosk = () => {
    setIsRecording(false);
    setIsTranscribing(true);
    return new Promise((resolve) => {
      window.__voskResult = (text) => {
        window.__voskResult = null;
        window.__voskError = null;
        setIsTranscribing(false);
        resolve(text || '');
      };
      window.__voskError = (msg) => {
        window.__voskResult = null;
        window.__voskError = null;
        setIsTranscribing(false);
        console.error('[VoiceInput] Vosk transcription error:', msg);
        resolve('');
      };
      window.VoskBridge.stopListening();
    });
  };

  // ── Desktop / Web Audio path ─────────────────────────────────────────────

  const floatTo16BitPCM = (float32Array) => {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
  };

  const encodeWAV = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.byteLength);
    const view = new DataView(buffer);
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.byteLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.byteLength, true);
    const wavData = new Uint8Array(buffer);
    wavData.set(new Uint8Array(samples), 44);
    return wavData;
  };

  const startRecordingDesktop = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      streamRef.current = stream;
      audioBufferRef.current = [];
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      processor.onaudioprocess = (e) => {
        audioBufferRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      setIsRecording(true);
    } catch (error) {
      console.error('[VoiceInput] Failed to start recording:', error);
      const isLinux = navigator.platform.toLowerCase().includes('linux') && !window.AndroidPaths;
      alert(isLinux
        ? 'Microphone access is not available in the desktop app yet.\n\nOffline dictation via Vosk is coming in v0.9.5. For now, please type your message.'
        : 'Microphone access denied. Please grant microphone permission to Zynkbot in your system settings.');
    }
  };

  // WebAudio → WAV → OpenAI Whisper. Used by Android when source === 'openai'.
  // On desktop, both engines use the Rust mic path (this function is unused there).
  const stopRecordingDesktop = async () => {
    setIsRecording(false);
    setIsTranscribing(true);
    try {
      if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
      if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
      if (audioContextRef.current) { await audioContextRef.current.close(); audioContextRef.current = null; }

      const totalLength = audioBufferRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const audioData = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of audioBufferRef.current) { audioData.set(chunk, offset); offset += chunk.length; }

      const pcmData = floatTo16BitPCM(audioData);
      const wavData = encodeWAV(pcmData, 16000);
      const audioArray = Array.from(wavData);
      const text = await invoke('transcribe_audio', { audioData: audioArray });
      return text;
    } catch (error) {
      console.error('[VoiceInput] Transcription failed:', error);
      alert('Transcription failed: ' + error);
      return '';
    } finally {
      setIsTranscribing(false);
      audioBufferRef.current = [];
    }
  };

  // ── Public API ───────────────────────────────────────────────────────────

  // Desktop dictation: both engines capture mic via Rust (cpal), bypassing the
  // WebView getUserMedia problem on Linux. Only the stop command differs.
  const startRecordingDesktopRust = async (source) => {
    const cmd = source === 'openai' ? 'start_openai_recording' : 'start_vosk_recording';
    try {
      await invoke(cmd);
      setIsRecording(true);
    } catch (error) {
      console.error('[VoiceInput] Desktop start failed:', error);
      alert('Voice input failed to start: ' + error);
    }
  };

  const stopRecordingDesktopRust = async (source) => {
    const cmd = source === 'openai' ? 'stop_openai_recording' : 'stop_vosk_recording';
    setIsRecording(false);
    setIsTranscribing(true);
    try {
      return await invoke(cmd);
    } catch (error) {
      console.error('[VoiceInput] Desktop stop failed:', error);
      alert('Transcription failed: ' + error);
      return '';
    } finally {
      setIsTranscribing(false);
    }
  };

  const startRecording = async () => {
    const source = getSource();
    if (isAndroid()) {
      // Android WebView is Chromium — getUserMedia works, so OpenAI uses the
      // same WebAudio path as pre-Vosk. Vosk still goes through the Kotlin bridge.
      if (source === 'openai') return startRecordingDesktop();
      return startRecordingVosk();
    }
    // Desktop: both engines go through Rust mic capture (bypasses Linux WebView bug).
    return startRecordingDesktopRust(source);
  };

  const stopRecording = async () => {
    if (!isRecording) return '';
    const source = getSource();
    if (isAndroid()) {
      if (source === 'openai') return stopRecordingDesktop();
      return stopRecordingVosk();
    }
    return stopRecordingDesktopRust(source);
  };

  return { isRecording, isTranscribing, startRecording, stopRecording };
}
