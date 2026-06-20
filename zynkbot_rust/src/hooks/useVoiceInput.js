import { useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * Hook for local speech-to-text using whisper.cpp
 *
 * Privacy-first: All audio processing happens locally via Tauri backend.
 * No audio data leaves the device.
 *
 * Usage:
 * const { isRecording, isTranscribing, startRecording, stopRecording } = useVoiceInput();
 *
 * startRecording(); // User speaks
 * const text = await stopRecording(); // Returns transcribed text
 */
export function useVoiceInput() {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);
  const audioBufferRef = useRef([]);

  // Helper: Convert Float32Array to 16-bit PCM
  const floatTo16BitPCM = (float32Array) => {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return buffer;
  };

  // Helper: Create WAV file with proper RIFF headers
  const encodeWAV = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.byteLength);
    const view = new DataView(buffer);

    // Write WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');                                      // ChunkID
    view.setUint32(4, 36 + samples.byteLength, true);          // ChunkSize
    writeString(8, 'WAVE');                                      // Format
    writeString(12, 'fmt ');                                     // Subchunk1ID
    view.setUint32(16, 16, true);                               // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);                                // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true);                                // NumChannels (1 for mono)
    view.setUint32(24, sampleRate, true);                       // SampleRate
    view.setUint32(28, sampleRate * 2, true);                   // ByteRate
    view.setUint16(32, 2, true);                                // BlockAlign
    view.setUint16(34, 16, true);                               // BitsPerSample
    writeString(36, 'data');                                     // Subchunk2ID
    view.setUint32(40, samples.byteLength, true);               // Subchunk2Size

    // Copy audio data
    const wavData = new Uint8Array(buffer);
    wavData.set(new Uint8Array(samples), 44);

    return wavData;
  };

  const startRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,     // Mono
          sampleRate: 16000,   // 16kHz (whisper expects this)
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      streamRef.current = stream;
      audioBufferRef.current = [];

      // Create Web Audio API context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);

      // Use ScriptProcessorNode to capture raw audio
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioBufferRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setIsRecording(true);
      console.log('[VoiceInput] Recording started (Web Audio API, 16kHz mono)');
    } catch (error) {
      console.error('[VoiceInput] Failed to start recording:', error);
      console.error('[VoiceInput] Error details:', error.name, error.message);

      // More helpful error message for Linux/WebKitGTK users
      const isLinux = navigator.platform.toLowerCase().includes('linux');
      const errorMsg = isLinux
        ? 'Microphone access not available in Tauri on Linux.\n\nThis is a WebKitGTK limitation. You can:\n1. Use the type input instead\n2. Grant microphone permissions to WebKitGTK at OS level\n3. Use the Windows version of Zynkbot for full voice support'
        : 'Microphone access denied. Please check your browser/system settings and grant microphone permissions.';

      alert(errorMsg);
    }
  };

  const stopRecording = async () => {
    if (!isRecording || !audioContextRef.current) {
      return '';
    }

    setIsRecording(false);
    setIsTranscribing(true);

    try {
      // Stop audio processing
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }

      // Stop media stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      // Close audio context
      if (audioContextRef.current) {
        await audioContextRef.current.close();
        audioContextRef.current = null;
      }

      console.log('[VoiceInput] Recording stopped, processing audio...');

      // Concatenate all audio chunks
      const totalLength = audioBufferRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
      const audioData = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of audioBufferRef.current) {
        audioData.set(chunk, offset);
        offset += chunk.length;
      }

      console.log('[VoiceInput] Collected', totalLength, 'samples');

      // Convert to 16-bit PCM
      const pcmData = floatTo16BitPCM(audioData);

      // Encode as WAV with proper headers
      const wavData = encodeWAV(pcmData, 16000);

      console.log('[VoiceInput] Created WAV file, size:', wavData.length, 'bytes');

      // Send to Tauri backend for transcription
      const audioArray = Array.from(wavData);
      const text = await invoke('transcribe_audio', { audioData: audioArray });

      console.log('[VoiceInput] Transcription complete:', text);

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

  return {
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording
  };
}
