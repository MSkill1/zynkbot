import React, { useState, useEffect, useLayoutEffect, useRef } from "react";
import { v4 as uuidv4 } from 'uuid';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open as openFileDialog } from '@tauri-apps/plugin-dialog';
import "./styles/App.css";
// LiveInsightsPanel removed - simplified to single-column layout
import ContainmentModeSelector from "./components/ContainmentModeSelector";
import ChatMessage from "./components/ChatMessage";
import MemoryManager from "./components/MemoryManager";
import AboutModal from "./components/AboutModal";
import GettingStartedModal from "./components/GettingStartedModal";
import WhyZynkbotModal from "./components/WhyZynkbotModal";
import APIKeyModal from "./components/APIKeyModal";
import ZynkSyncPanel from "./components/ZynkSyncPanel";
import ZynkLinkPanel from "./components/ZynkLinkPanel";
import CollapsibleSidebar from "./components/CollapsibleSidebar";
import UserIdentityModal from "./components/UserIdentityModal";
import EnsembleModal from "./components/EnsembleModal";
import ConflictResolutionModal from "./components/ConflictResolutionModal";
import KnowledgeBasePanel from "./components/KnowledgeBasePanel";
import KnowledgeBaseManager from "./components/KnowledgeBaseManager";
import VoiceButton from "./components/VoiceButton";
import ZChatModal from "./components/ZChatModal";
import ZynkClusterModal from "./components/ZynkClusterModal";
import OnboardingModal from "./components/OnboardingModal";
import SetupWizard from "./components/SetupWizard";
import SnapInModal from "./components/SnapInModal";
import ConversationHistoryPanel from "./components/ConversationHistoryPanel";
import VoiceModal from "./components/VoiceModal";

// Parse a wake-word transcript into a structured voice command.
// Returns { type, ...params } or null if no command matched.
// Vosk produces lowercase, no-punctuation text — regexes are written accordingly.
function normalizeNumbers(text) {
  const words = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15',
    'sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19',
    'twenty':'20','thirty':'30','forty':'40','fifty':'50','sixty':'60',
    'ninety':'90','hundred':'100',
  };
  return text.replace(/\b(twenty|thirty|forty|fifty|sixty|ninety)-?(one|two|three|four|five|six|seven|eight|nine)\b/gi, (_, tens, ones) => {
    return String(parseInt(words[tens.toLowerCase()]) + parseInt(words[ones.toLowerCase()]));
  }).replace(/\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|ninety|hundred)\b/gi, m => words[m.toLowerCase()] || m);
}

function parseVoiceCommand(text) {
  const t = normalizeNumbers(text.toLowerCase().trim());

  // Timer: "set a timer for 5 minutes" / "5 minute timer" / "timer 30 seconds"
  const timerRes = [
    /(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)/,
    /(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+timer/,
  ];
  for (const re of timerRes) {
    const m = t.match(re);
    if (m) {
      const val = parseFloat(m[1]);
      const unit = m[2];
      let seconds;
      if (unit.startsWith('hour') || unit.startsWith('hr')) seconds = Math.round(val * 3600);
      else if (unit.startsWith('min')) seconds = Math.round(val * 60);
      else seconds = Math.round(val);
      return { type: 'timer', seconds };
    }
  }

  // Alarm: "set an alarm for 7:30" / "wake me up at 7" / "alarm at 7:30 pm"
  const alarmM = t.match(
    /(?:set\s+(?:an?\s+)?alarm\s+(?:for|at)|alarm\s+(?:for|at)|wake\s+me\s+up\s+at)\s+(\d{1,2})(?:[:\s](\d{1,2}))?\s*(am|pm)?/
  );
  if (alarmM) {
    let hour = parseInt(alarmM[1]);
    const minute = parseInt(alarmM[2] || '0');
    const ampm = (alarmM[3] || '').toLowerCase();
    if (ampm === 'pm' && hour !== 12) hour += 12;
    if (ampm === 'am' && hour === 12) hour = 0;
    return { type: 'alarm', hour, minute };
  }

  // Stopwatch: "start stopwatch" / "stopwatch"
  if (/(?:start|begin)\s+(?:the\s+|a\s+)?stopwatch|^stopwatch$/.test(t)) {
    return { type: 'stopwatch' };
  }

  // Stop TTS: "zynkbot stop" / "stop talking" / "stop dictating"
  if (/^(zynkbot\s+)?stop(\s+(talking|dictating|reading))?$/.test(t)) {
    return { type: 'stop_tts' };
  }

  return null;
}

// API Base URL - DEPRECATED: All API calls now use Tauri commands
// Keeping this for legacy components that haven't been migrated yet (ZynkSync, ZynkLink, etc.)
const API_BASE_URL = process.env.REACT_APP_API_URL || (window.__TAURI__ ? 'http://localhost:5000' : '');

// Access gate component
function AccessGate() {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{
        maxWidth: '600px',
        width: '100%',
        background: 'rgba(255, 255, 255, 0.05)',
        backdropFilter: 'blur(10px)',
        borderRadius: '20px',
        padding: '50px 40px',
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        border: '1px solid rgba(255, 255, 255, 0.18)',
        color: '#ffffff'
      }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '700',
          marginBottom: '10px',
          background: 'linear-gradient(45deg, #50fa7b, #8be9fd)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text'
        }}>Zynkbot</h1>
        
        <div style={{
          fontSize: '1.2rem',
          color: '#8be9fd',
          marginBottom: '30px',
          fontWeight: '300'
        }}>Privacy-First AI Assistant</div>
        
        <p style={{
          fontSize: '1rem',
          lineHeight: '1.8',
          color: 'rgba(255, 255, 255, 0.9)',
          marginBottom: '30px'
        }}>
          An AI assistant designed for mobile devices with transparent, user-controlled memory. 
          Zynkbot puts you in control of your data with local model support and advanced privacy features.
        </p>
          
        <p style={{
          fontSize: '1rem',
          color: 'rgba(255, 255, 255, 0.85)'
        }}>
          For access, please email:{' '}
          <a href="mailto:matt@containai.ai" style={{
            color: '#8be9fd',
            textDecoration: 'none',
            fontWeight: '600'
          }}>matt@containai.ai</a>
        </p>

        <div style={{
          textAlign: 'center',
          color: 'rgba(255, 255, 255, 0.5)',
          fontSize: '0.9rem',
          marginTop: '30px'
        }}>
          Currently in private development • Early 2026 launch
        </div>
      </div>
    </div>
  );
}

export default function App() {
  // Persistent user ID from backend (shared across all YOUR devices)
  const [userId, setUserId] = useState(null);
  const [isLoadingIdentity, setIsLoadingIdentity] = useState(true);

  // Session ID persists until tab closes (mutable so resume can switch sessions)
  const [sessionId, setSessionId] = useState(() => {
    try {
      const stored = sessionStorage.getItem('zynkbot_session_id');
      if (stored) return stored;
      const newId = uuidv4();
      sessionStorage.setItem('zynkbot_session_id', newId);
      return newId;
    } catch (error) {
      console.error('sessionStorage error:', error);
      return uuidv4();
    }
  });

  // Messages persist in session
  const [messages, setMessages] = useState(() => {
    try {
      const stored = sessionStorage.getItem('zynkbot_messages');
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('sessionStorage error:', error);
      return [];
    }
  });

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const TIPS = [
    <>Tip: start with "<strong>Remember:</strong>" to force save directly into your memories — use it when you want exact details saved precisely.</>,
    <>Tap <strong>Memory Manager</strong> to see everything Zynkbot knows about you. You can edit, delete, or add memories at any time.</>,
    <>Open <strong>History</strong> to browse past conversations and copy any response.</>,
    <><strong>ZynkSync</strong> keeps your memories in sync across all your devices automatically — no account required.</>,
    <><strong>ZynkLink</strong> lets you share files directly with paired devices over your local network. No cloud, no upload.</>,
    <><strong>Ensemble mode</strong> queries multiple AI models at once and synthesizes their answers — great for questions where accuracy matters.</>,
    <>Add documents to the <strong>Knowledge Base</strong> — Zynkbot can answer questions about them using RAG search.</>,
    <>Everything stays on your device. Your memories and conversations never leave unless you choose to back them up.</>,
    <>Include "<strong>Zynkbot</strong>" in your question to ask about its own features — e.g. "Zynkbot, how does Ensemble mode work?"</>,
    <>The <strong>mic button</strong> lets you dictate your message — transcribed locally on-device, no internet needed.</>,
  ];
  const [tipIdx, setTipIdx] = useState(() => Math.floor(Math.random() * TIPS.length));
  const [editingMessageId, setEditingMessageId] = useState(null);
  const isSendingRef = useRef(false); // synchronous guard against concurrent sends
  const [modelType, setModelType] = useState(() => {
    return localStorage.getItem('zynkbot_preferred_model') || 'local';
  });
  const [availableModels, setAvailableModels] = useState([]);
  const [containmentMode, setContainmentMode] = useState("guardian");
  const [showAbout, setShowAbout] = useState(false);
  const [showDemoGuide, setShowDemoGuide] = useState(false);
  const [showWhyZynkbot, setShowWhyZynkbot] = useState(false);
  const [showAPIKeys, setShowAPIKeys] = useState(false);
  const [showUserIdentity, setShowUserIdentity] = useState(false);
  const [showEnsemble, setShowEnsemble] = useState(false);
  const [showConflictResolution, setShowConflictResolution] = useState(false);
  const [showZynkCluster, setShowZynkCluster] = useState(false);
  const [showConversationHistory, setShowConversationHistory] = useState(false);
  const [currentConflict, setCurrentConflict] = useState(null);
  const [chatDevice, setChatDevice] = useState(null);
  const [currentDeviceId, setCurrentDeviceId] = useState(null);
  const [showKBManager, setShowKBManager] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [searchKBEnabled, setSearchKBEnabled] = useState(false);
  const [kbLocked, setKbLocked] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState([]); // [{ name, content, size, isImage, ... }]
  // Collapsible section states - all start collapsed
  const [showGettingStarted, setShowGettingStarted] = useState(false);
  const [showZynkSyncSection, setShowZynkSyncSection] = useState(false);
  const [showZynkLinkSection, setShowZynkLinkSection] = useState(false);
  const [showKnowledgeBaseSection, setShowKnowledgeBaseSection] = useState(false);
  const [showZynkClusterSection, setShowZynkClusterSection] = useState(false);
  const [showSnapInsSection, setShowSnapInsSection] = useState(false);
  const [showSnapInModal, setShowSnapInModal] = useState(false);
  const [isLoadingEinstein, setIsLoadingEinstein] = useState(false);
  const [heyZynkEnabled, setHeyZynkEnabled] = useState(() =>
    localStorage.getItem('zynkbot_hey_zynk_enabled') !== 'false'
  );
  const [ttsEnabled, setTtsEnabled] = useState(() =>
    localStorage.getItem('zynkbot_tts_enabled') !== 'false'
  );
  const [webSearchAutoExecute, setWebSearchAutoExecute] = useState(() =>
    localStorage.getItem('zynkbot_web_search_auto') === 'true'
  );
  const [conversationModeEnabled, setConversationModeEnabled] = useState(() =>
    localStorage.getItem('zynkbot_conversation_mode') === 'true'
  );
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const [isTtsSpeaking, setIsTtsSpeaking] = useState(false);
  // Set from VoiceButton via the 'zynkbot:dictation' event. The dictation button and
  // the wake word detector share one microphone, and this component owns the detector.
  const [isDictating, setIsDictating] = useState(false);
  const [voiceInputSource, setVoiceInputSource] = useState(() => {
    // 'vosk' = offline (no punctuation), 'openai' = cloud Whisper (has punctuation)
    return localStorage.getItem('zynkbot_voice_input_source') || 'vosk';
  });
  const [wakeWordModelReady, setWakeWordModelReady] = useState(() =>
    window.WakeWordBridge ? window.WakeWordBridge.isModelReady() : false
  );
  const [wakeWordDownloadProgress, setWakeWordDownloadProgress] = useState(0);
  const [wakeWordDownloadError, setWakeWordDownloadError] = useState('');
  const [wakeWordFlash, setWakeWordFlash] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 768);
  const memoryManagerRef = useRef(null);
  const conversationEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const userScrolledUpRef = useRef(false);
  const inputTextareaRef = useRef(null);
  const voiceButtonRef = useRef(null);

  // Wake word recording — managed directly, NOT through useVoiceInput hook.
  // VoiceButton owns useVoiceInput exclusively; wake word owns its own VoskBridge calls.
  const [isWakeRecording, setIsWakeRecording] = useState(false);
  const silenceTimerRef = useRef(null);
  const wakeTriggeredRef = useRef(false);
  const ttsSourceRef = useRef(null);
  const ttsAudioCtxRef = useRef(null);
  // Stable ref to stopWakeRecording so the waveform overlay can call it without stale closure
  const stopWakeRecordingRef = useRef(null);
  // Stable ref to handleSendMessage so wake word effect never captures a stale closure
  const handleSendMessageRef = useRef(null);
  // Conversation loop — true while Hey Zynk is in a multi-turn session
  const conversationLoopActiveRef = useRef(false);
  const startListeningLoopRef = useRef(null);

  // Splice transcribed text into the input at the cursor position, padding with
  // spaces when adjacent to non-whitespace so words don't run together.
  const insertTranscriptAtCursor = (text) => {
    if (!text) return;
    const el = inputTextareaRef.current;
    setInput((current) => {
      if (!el || typeof el.selectionStart !== 'number') {
        // No ref or no selection info — append with a space if there's existing text.
        return current ? `${current.replace(/\s*$/, '')} ${text}` : text;
      }
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const before = current.slice(0, start);
      const after = current.slice(end);
      const leadPad = before && !/\s$/.test(before) ? ' ' : '';
      const trailPad = after && !/^\s/.test(after) ? ' ' : '';
      const insertion = `${leadPad}${text}${trailPad}`;
      // Restore cursor position after React re-renders.
      const nextCursor = before.length + insertion.length - trailPad.length;
      requestAnimationFrame(() => {
        if (inputTextareaRef.current) {
          inputTextareaRef.current.focus();
          inputTextareaRef.current.setSelectionRange(nextCursor, nextCursor);
        }
      });
      return `${before}${insertion}${after}`;
    });
  };

  const stopTts = () => {
    try { ttsSourceRef.current?.stop(); } catch (_) {}
    try { ttsAudioCtxRef.current?.close(); } catch (_) {}
    ttsSourceRef.current = null;
    ttsAudioCtxRef.current = null;
    setIsTtsSpeaking(false);
  };

  const speakResponse = async (text) => {
    if (!text?.trim()) return;
    stopTts();
    try {
      const keys = await invoke('get_api_keys');
      const apiKey = keys['OPENAI_API_KEY'];
      if (!apiKey) return;
      const res = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'tts-1', input: text.slice(0, 4096), voice: 'alloy' }),
      });
      if (!res.ok) return;
      const audioData = await res.arrayBuffer();
      const audioCtx = new AudioContext();
      ttsAudioCtxRef.current = audioCtx;
      const buffer = await audioCtx.decodeAudioData(audioData);
      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      ttsSourceRef.current = source;
      setIsTtsSpeaking(true);
      source.start();
      source.onended = () => {
        setIsTtsSpeaking(false);
        ttsSourceRef.current = null;
        ttsAudioCtxRef.current = null;
        if (conversationLoopActiveRef.current) {
          startListeningLoopRef.current?.();
        }
      };
    } catch (e) {
      console.error('[TTS]', e);
      setIsTtsSpeaking(false);
    }
  };

  const executeVoiceCommand = (cmd, originalText) => {
    const now = new Date().toISOString();
    let confirmation = '';
    if (cmd.type === 'timer') {
      window.VoiceCommandBridge.setTimer(cmd.seconds);
      const h = Math.floor(cmd.seconds / 3600);
      const m = Math.floor((cmd.seconds % 3600) / 60);
      const s = cmd.seconds % 60;
      const parts = [];
      if (h) parts.push(`${h} hour${h > 1 ? 's' : ''}`);
      if (m) parts.push(`${m} minute${m > 1 ? 's' : ''}`);
      if (s && !h) parts.push(`${s} second${s > 1 ? 's' : ''}`);
      confirmation = `⏱️ Timer set for ${parts.join(' and ')}.`;
    } else if (cmd.type === 'alarm') {
      window.VoiceCommandBridge.setAlarm(cmd.hour, cmd.minute, 'Zynkbot');
      const h12 = cmd.hour % 12 || 12;
      const ampm = cmd.hour < 12 ? 'AM' : 'PM';
      const min = cmd.minute.toString().padStart(2, '0');
      confirmation = `⏰ Alarm set for ${h12}:${min} ${ampm}.`;
    } else if (cmd.type === 'stopwatch') {
      window.VoiceCommandBridge.startStopwatch();
      confirmation = '⏱️ Stopwatch started.';
    }
    setMessages(prev => [
      ...prev,
      { id: Date.now(), role: 'user', content: originalText, timestamp: now, recalled_memories: [] },
      { id: Date.now() + 1, role: 'assistant', content: confirmation, timestamp: now, recalled_memories: [], metadata: { model_backend: 'voice command' } },
    ]);
  };

  const handleTranscript = (text) => {
    if (text) insertTranscriptAtCursor(text);
  };

  const [showScrollButton, setShowScrollButton] = useState(false);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);


  // Lock body scroll when any modal is open to prevent scrollbar ghost artifact
  const anyModalOpen = showAbout || showDemoGuide || showWhyZynkbot || showAPIKeys ||
    showUserIdentity || showEnsemble || showConflictResolution || showZynkCluster ||
    showConversationHistory || showKBManager || showOnboarding ||
    showSnapInModal;
  useLayoutEffect(() => {
    if (anyModalOpen) {
      document.body.style.overflow = 'hidden';
      document.body.classList.add('modal-open');
    } else {
      document.body.style.overflow = '';
      document.body.classList.remove('modal-open');
    }
    return () => {
      document.body.style.overflow = '';
      document.body.classList.remove('modal-open');
    };
  }, [anyModalOpen]);

  // Check for access token in URL
  const [hasAccess] = useState(true); // Disabled for local development
  const [needsSetup, setNeedsSetup] = useState(null); // null = checking, true/false = result

  // Check if first-run setup is needed
  useEffect(() => {
    invoke('check_needs_setup')
      .then(needs => setNeedsSetup(needs))
      .catch(() => setNeedsSetup(false));
  }, []);

  // Fetch persistent user identity from backend on mount
  useEffect(() => {
    const fetchIdentity = async () => {
      try {
        console.log('[App] Fetching user identity from backend...');
        const identity = await invoke('get_user_identity');
        console.log('[App] Loaded persistent identity:', identity);
        setUserId(identity.user_id);
        setIsLoadingIdentity(false);
      } catch (error) {
        console.error('[App] Failed to fetch identity:', error);
        // Fallback to random UUID for this session only
        const fallbackId = uuidv4();
        console.warn('[App] Using fallback session-only user_id:', fallbackId);
        setUserId(fallbackId);
        setIsLoadingIdentity(false);
      }
    };
    fetchIdentity();
  }, []);

  // Function to fetch models (can be called on demand)
  const fetchModels = async () => {
    try {
      // Use Rust backend (Tauri command)
      const models = await invoke('get_models');
      console.log('✅ Models loaded from RUST:', models);

      setAvailableModels(models);

      // On first run (nothing stored), default to first local GGUF model.
      // After that: always honour whatever the user last selected — never override it.
      // Migration: if localStorage somehow holds a raw GGUF path that doesn't match any
      // model in the list (e.g. path changed after reinstall), treat it as unset.
      let stored = localStorage.getItem('zynkbot_preferred_model');
      if (stored && stored.endsWith('.gguf') && !models.find(m => m.id === stored)) {
        console.log('[Models] Stale GGUF path in localStorage, clearing:', stored);
        localStorage.removeItem('zynkbot_preferred_model');
        stored = null;
      }
      if (!stored) {
        // Priority: local (privacy) → custom/Ollama → first available API model.
        // Android has no local models, so it naturally falls through to custom or API.
        const firstLocal = models.find(m => m.model_type === 'local');
        const custom = models.find(m => m.id === 'custom');
        const firstApi = models.find(m => m.model_type === 'api' && m.id !== 'custom');
        const defaultModel = firstLocal || custom || firstApi;
        if (defaultModel) {
          setModelType(defaultModel.id);
          localStorage.setItem('zynkbot_preferred_model', defaultModel.id);
          console.log('[Models] First run — defaulting to:', defaultModel.id);
        }
      } else if (stored === 'local' && !models.find(m => m.id === 'local' || m.model_type === 'local')) {
        // Stale 'local' stored value but no local models available (e.g. Android after a
        // desktop-only install migrated over). Fall through to custom/API instead of leaving
        // the user stuck on a backend that will error out.
        const custom = models.find(m => m.id === 'custom');
        const firstApi = models.find(m => m.model_type === 'api' && m.id !== 'custom');
        const fallback = custom || firstApi;
        if (fallback) {
          setModelType(fallback.id);
          localStorage.setItem('zynkbot_preferred_model', fallback.id);
          console.log('[Models] Stale "local" preference with no local models — switching to:', fallback.id);
        }
      } else if (stored && stored !== modelType) {
        // localStorage changed out-of-band (e.g. APIKeyModal switched to 'custom' after
        // connecting to a desktop). Sync the in-memory state so the picker reflects it.
        setModelType(stored);
        console.log('[Models] Syncing modelType to stored preference:', stored);
      }
      // If the user already has a preference stored, leave it alone regardless of
      // whether the model appears in the current list (e.g. Ollama might be offline).

      return models;
    } catch (error) {
      console.error('Failed to fetch models from Rust:', error);
      const fallbackModels = [{ id: "local", name: "Local Model", type: "local" }];
      setAvailableModels(fallbackModels);
      console.warn('Using local fallback model list');
      return fallbackModels;
    }
  };

  // Fetch available models on mount - NOW USING RUST!
  useEffect(() => {
    fetchModels();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save messages to sessionStorage whenever they change
  useEffect(() => {
    try {
      sessionStorage.setItem('zynkbot_messages', JSON.stringify(messages));
    } catch (error) {
      console.error('Failed to save messages to sessionStorage:', error);
    }
  }, [messages]);

  useEffect(() => {
    console.log('Messages state updated. Total:', messages.length);
    if (messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      console.log('Last message:', lastMsg);
      if (lastMsg.role === 'assistant') {
        console.log('Last assistant message recalled memories:', lastMsg.recalled_memories);
      }
    }
  }, [messages]);

  // Persist selected model across sessions
  useEffect(() => {
    localStorage.setItem('zynkbot_preferred_model', modelType);
  }, [modelType]);

  // Listen for contradiction detection events from backend
  useEffect(() => {
    let unlisten;

    const setupListener = async () => {
      unlisten = await listen('contradiction-detected', (event) => {
        console.log('[App] ⚠️ CONTRADICTION DETECTED from backend:', event.payload);

        // Save the full payload data (needed for resolution)
        setCurrentConflict(event.payload);
        setShowConflictResolution(true);

        console.log('[App] Conflict modal opened');
      });
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  useEffect(() => {
    if (!conversationModeEnabled) return;
    let wakeLock = null;
    const acquire = async () => {
      try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (_) {}
    };
    acquire();
    const onVisible = () => { if (document.visibilityState === 'visible') acquire(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      document.removeEventListener('visibilitychange', onVisible);
      try { wakeLock?.release(); } catch (_) {}
    };
  }, [conversationModeEnabled]);

  // Wake word service lifecycle — starts/stops with the Hey Zynk toggle on Android.
  // Wake recording uses VoskBridge directly — completely separate from useVoiceInput/VoiceButton.
  useEffect(() => {
    if (!window.WakeWordBridge) return;

    const SILENCE_MS = 1000;

    const CLOSING_PHRASES = /\b(thank(?:\s+you)?\s+zynk|goodbye\s+zynk|zynk\s+stop|stop\s+listening|that'?s?\s+all|close\s+session)\b/i;

    // Stop wake recording and return the transcript. Does NOT go through useVoiceInput.
    const stopWakeRecording = () => {
      clearTimeout(silenceTimerRef.current);
      window.__voskPartial = null;
      setIsWakeRecording(false);
      if (!window.VoskBridge) return Promise.resolve('');
      return new Promise((resolve) => {
        window.__voskResult = (text) => {
          window.__voskResult = null;
          window.__voskError = null;
          resolve(text || '');
        };
        window.__voskError = () => {
          window.__voskResult = null;
          window.__voskError = null;
          resolve('');
        };
        window.VoskBridge.stopListening();
      });
    };
    stopWakeRecordingRef.current = stopWakeRecording;

    // End the conversation loop: play the closing chime and re-arm the ONNX detector.
    const endConversationLoop = () => {
      conversationLoopActiveRef.current = false;
      try {
        const audio = new Audio('/wake_chime_close.wav');
        audio.play().catch(() => {});
      } catch (_) {}
      if (window.WakeWordBridge.isModelReady()) {
        setTimeout(() => window.WakeWordBridge.start(0.72), 1500);
      }
    };

    const autoSendWake = async () => {
      console.log('[WakeWord] autoSendWake firing');
      const text = await stopWakeRecording();
      console.log('[WakeWord] transcript:', text);
      if (!text) {
        // No speech heard — end the loop so ONNX can re-arm
        conversationLoopActiveRef.current = false;
        if (window.WakeWordBridge.isModelReady()) {
          setTimeout(() => window.WakeWordBridge.start(0.72), 1000);
        }
        return;
      }
      if (conversationLoopActiveRef.current && CLOSING_PHRASES.test(text)) {
        console.log('[WakeWord] closing phrase detected — ending conversation loop');
        endConversationLoop();
        return;
      }
      wakeTriggeredRef.current = true;
      handleSendMessageRef.current?.(text);
    };

    // Start the chime→Vosk→silence cycle. Used both for the initial wake and for
    // each subsequent turn of the conversation loop.
    const startListeningLoop = () => {
      if (!window.VoskBridge) return;
      if (window.__dictationActive) return;
      const startRecording = () => {
        window.__voskPartial = (partial) => {
          if (partial.trim()) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = setTimeout(autoSendWake, SILENCE_MS);
          }
        };
        window.VoskBridge.startListening();
        setIsWakeRecording(true);
        silenceTimerRef.current = setTimeout(autoSendWake, 8000);
      };
      try {
        const audio = new Audio('/wake_chime.wav');
        audio.onended = startRecording;
        audio.onerror = startRecording;
        audio.play().catch(startRecording);
      } catch (_) {
        startRecording();
      }
    };
    startListeningLoopRef.current = startListeningLoop;

    window.__wakeWordDetected = () => {
      // Don't hijack VoskBridge while the dictation button is active
      if (window.__dictationActive) {
        console.log('[WakeWord] ignored — dictation in progress');
        return;
      }
      console.log('[WakeWord] detected — playing chime then recording');
      window.WakeWordBridge.stop();
      setWakeWordFlash(true);
      setTimeout(() => setWakeWordFlash(false), 2500);
      if (conversationModeEnabled) {
        conversationLoopActiveRef.current = true;
      }
      startListeningLoop();
    };
    window.__wakeWordModelReady = () => setWakeWordModelReady(true);
    window.__wakeWordDownloadProgress = (n) => setWakeWordDownloadProgress(n);
    window.__wakeWordDownloadError = (msg) => {
      setWakeWordDownloadError(msg);
      setWakeWordDownloadProgress(0);
    };
    // Screen-off path: WakeWordService recorded via Kotlin, delivers transcript here.
    // Always enters conversation loop mode since the app just unlocked for this session.
    window.__handleScreenOffTranscript = (transcript) => {
      if (!transcript?.trim()) return;
      console.log('[WakeWord] screen-off transcript received:', transcript);
      conversationLoopActiveRef.current = true;
      wakeTriggeredRef.current = true;
      handleSendMessageRef.current?.(transcript.trim());
    };

    if (heyZynkEnabled) {
      if (window.WakeWordBridge.isModelReady()) {
        window.WakeWordBridge.start(0.72);
      }
    } else {
      window.WakeWordBridge.stop();
    }

    return () => {
      window.__wakeWordDetected = null;
      window.__wakeWordModelReady = null;
      window.__wakeWordDownloadProgress = null;
      window.__wakeWordDownloadError = null;
      window.__handleScreenOffTranscript = null;
      conversationLoopActiveRef.current = false;
      if (window.WakeWordBridge) window.WakeWordBridge.stop();
    };
  }, [heyZynkEnabled, conversationModeEnabled]);

  // VoiceButton announces when the dictation button is in use. Tracked here because
  // the wake detector is owned by this component, and VoiceButton's own useVoiceInput
  // state cannot be read from outside it.
  useEffect(() => {
    const onDictation = (e) => setIsDictating(!!e.detail?.active);
    window.addEventListener('zynkbot:dictation', onDictation);
    return () => window.removeEventListener('zynkbot:dictation', onDictation);
  }, []);

  // Pause wake word detection while dictating, wake-recording, or TTS is playing, and
  // resume only once all three are clear. Dictation has to be included: the detector
  // and the dictation button share one microphone, so leaving it running meant the
  // user's own dictation speech could trigger a wake recording the moment they hit
  // send. The delay before resuming also lets the tail of that speech age out of the
  // detector's rolling audio buffer.
  useEffect(() => {
    if (!window.WakeWordBridge || !heyZynkEnabled) return;
    if (isWakeRecording || isTtsSpeaking || isDictating) {
      window.WakeWordBridge.stop();
    } else if (!conversationLoopActiveRef.current) {
      const t = setTimeout(() => {
        if (window.WakeWordBridge && window.WakeWordBridge.isModelReady()) {
          window.WakeWordBridge.start(0.72);
        }
      }, 5000);
      return () => clearTimeout(t);
    }
  }, [isWakeRecording, heyZynkEnabled, isTtsSpeaking, isDictating]);

  // Restart wake word when app returns to foreground (screen unlock, app switch back).
  // The service may have stopped while the screen was off; visibilitychange is the
  // reliable hook for this on Android WebView.
  useEffect(() => {
    if (!window.WakeWordBridge) return;
    const handleVisible = () => {
      if (!heyZynkEnabled || isWakeRecording || isTtsSpeaking || isDictating) return;
      if (window.WakeWordBridge.isModelReady()) {
        window.WakeWordBridge.start(0.72);
      }
    };
    document.addEventListener('visibilitychange', handleVisible);
    return () => document.removeEventListener('visibilitychange', handleVisible);
  }, [heyZynkEnabled, isWakeRecording, isTtsSpeaking, isDictating]);

  // Auto-scroll: only follow new content if the user hasn't scrolled up intentionally.
  // userScrolledUpRef is set by real scroll events only — never by programmatic scrolls —
  // so it reflects genuine user intent and can't be cleared by our own auto-scroll.
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;
    if (userScrolledUpRef.current) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isLoading]);

  // Mount-once scroll listener: tracks user intent and scroll button visibility.
  // Empty deps — never re-runs — so we never artificially read distanceFromBottom
  // right after our own programmatic scroll (which was clearing the ref in prior attempts).
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;
    const onScroll = () => {
      const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
      const scrolledUp = distanceFromBottom > 80;
      userScrolledUpRef.current = scrolledUp;
      setShowScrollButton(scrolledUp);
    };
    container.addEventListener('scroll', onScroll);
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  // Voice input now handled by VoiceButton component (using whisper.cpp)

  const handleClearConversation = () => {
    if (window.confirm('Clear conversation history? This will remove all messages from the current chat.')) {
      setMessages([]);
    }
  };

  const handleResumeSession = ({ sessionId: resumedId, messages: pastMessages }) => {
    setSessionId(resumedId);
    try { sessionStorage.setItem('zynkbot_session_id', resumedId); } catch (_) {}
    setMessages(pastMessages);
    // Scroll to bottom after loading
    setTimeout(() => {
      const el = document.getElementById('messages-end');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // Conflict Resolution Handlers
  const handleConflictResolve = async (resolution) => {
    console.log('Conflict resolved:', resolution);

    try {
      // Map frontend options to backend decisions
      // Frontend: memoryA = OLD, memoryB = NEW
      // Backend: keep_old, keep_new, keep_both
      let decision;
      if (resolution.option === 'memoryA') {
        decision = 'keep_old';
      } else if (resolution.option === 'memoryB') {
        decision = 'keep_new';
      } else if (resolution.option === 'not_a_contradiction') {
        decision = 'not_a_contradiction';
      } else if (resolution.option === 'keep_both') {
        decision = 'keep_both';
      } else if (resolution.option === 'both_with_explanation') {
        decision = 'both_with_explanation';
      } else {
        throw new Error(`Unknown resolution option: ${resolution.option}`);
      }

      console.log(`Calling resolve_memory_conflict_v2 with decision: ${decision}`);

      // Call the NEW resolve command with pending memory data
      const result = await invoke('resolve_memory_conflict_v2', {
        pendingMemoryJson: JSON.stringify(currentConflict.pending_memory),
        conflictingMemoryId: currentConflict.memoryA.id,
        decision: decision,
        explanation: resolution.explanation || null,
        relationshipsJson: JSON.stringify(currentConflict.relationships || []),
        userId: currentConflict.user_id || userId,
        sessionId: currentConflict.session_id || sessionId,
      });

      console.log('✅ Conflict resolved successfully:', result);

      // Show success message
      if (decision === 'keep_old') {
        alert('Kept existing memory, discarded new statement');
      } else if (decision === 'keep_new') {
        alert('Kept new memory, deleted old memory');
      } else if (decision === 'not_a_contradiction') {
        alert('Both memories kept — contradiction edge removed');
      } else if (decision === 'keep_both') {
        alert('Both memories kept with contradiction edge. You can resolve this later in the Memory Manager.');
      } else if (decision === 'both_with_explanation') {
        alert('Both memories kept — explanation stored as a resolving memory');
      }

      // Refresh the memory manager to show updated memories
      if (memoryManagerRef.current && memoryManagerRef.current.refreshMemories) {
        memoryManagerRef.current.refreshMemories();
      }
    } catch (error) {
      console.error('Error resolving conflict:', error);
      alert(`Failed to resolve conflict: ${error}`);
    }

    setShowConflictResolution(false);
    setCurrentConflict(null);
  };

  const IMAGE_EXTENSIONS = ['jpg','jpeg','png','gif','webp','bmp'];
  const MIME_TYPES = { jpg:'image/jpeg', jpeg:'image/jpeg', png:'image/png', gif:'image/gif', webp:'image/webp', bmp:'image/bmp' };

  const handleAttachFile = async () => {
    const result = await openFileDialog({
      multiple: true,
      filters: [
        { name: 'Files', extensions: [...IMAGE_EXTENSIONS, 'txt','md','rs','js','jsx','ts','tsx','py','json','toml','yaml','yml','sh','css','html','c','cpp','h','go','java','rb','php','swift','kt'] }
      ]
    });
    if (!result) return;
    const paths = Array.isArray(result) ? result : [result];
    const newFiles = [];
    for (const path of paths) {
      try {
        // On Android, openFileDialog returns a content:// URI that Rust's fs::read can't open.
        // Use the AndroidPaths bridge which reads via ContentResolver instead.
        const name = window.AndroidPaths ? window.AndroidPaths.getFileName(path) : path.split('/').pop();
        const ext = name.split('.').pop().toLowerCase();
        if (IMAGE_EXTENSIONS.includes(ext)) {
          const base64 = window.AndroidPaths
            ? window.AndroidPaths.readFileBase64(path)
            : await invoke('read_file_base64', { path });
          if (!base64) throw new Error('Could not read image');
          const mimeType = MIME_TYPES[ext] || 'image/jpeg';
          newFiles.push({ name, base64, mimeType, size: base64.length, isImage: true });
        } else {
          const content = window.AndroidPaths
            ? window.AndroidPaths.readFileText(path)
            : await invoke('read_text_file', { path });
          newFiles.push({ name, content, size: content.length, isImage: false });
        }
      } catch (e) {
        alert(`Could not read file: ${e}`);
      }
    }
    if (newFiles.length > 0) setAttachedFiles(prev => [...prev, ...newFiles]);
  };

  const handleCameraCapture = async () => {
    if (!window.AndroidCamera) return;
    try {
      const path = await new Promise((resolve, reject) => {
        window.__camResolve = resolve;
        window.__camReject = reject;
        window.AndroidCamera.takePicture();
      });
      const name = path.split('/').pop();
      const base64 = await invoke('read_file_base64', { path });
      setAttachedFiles(prev => [...prev, { name, base64, mimeType: 'image/jpeg', size: base64.length, isImage: true }]);
    } catch (e) {
      if (e !== 'cancelled') alert(`Camera error: ${e}`);
    }
  };

  const handleSendMessage = async (message, options = {}) => {
    if (!message.trim()) return;

    // Capture and reset wake flag before any early returns
    const triggeredByWake = wakeTriggeredRef.current;
    wakeTriggeredRef.current = false;

    // stop_tts works without VoiceCommandBridge
    const cmd = parseVoiceCommand(message);
    if (cmd?.type === 'stop_tts') { stopTts(); setInput(''); return; }

    // Other voice commands require VoiceCommandBridge (Android)
    if (window.VoiceCommandBridge && cmd) {
      executeVoiceCommand(cmd, message); setInput(''); return;
    }

    if (isSendingRef.current) return; // prevent re-entrant calls before React re-renders
    isSendingRef.current = true;
    const skipUserMessageAdd = options.skipUserMessageAdd === true;

    // Scroll to bottom when the user sends — they want to see the response.
    // Reset the "user scrolled up" flag so auto-scroll follows the new response.
    userScrolledUpRef.current = false;
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }

    // Disable send button immediately to prevent double-clicks
    setIsLoading(true);
    setInput('');

    // Check if Knowledge Base search is enabled for this message
    const kbEnabled = searchKBEnabled;
    if (kbEnabled) {
      console.log('[KB RAG] Knowledge Base search enabled for this message');
      // Only reset if not locked — locked keeps KB on across messages
      if (!kbLocked) {
        setSearchKBEnabled(false);
      }
    }

    // Inject attached file content into the message sent to the backend.
    // userQuery carries the clean question so safety checks, memory search, and
    // conversation history see intent — not the file dump.
    let messageToSend = message;
    let userQuery = undefined;
    let imageData = undefined;
    if (attachedFiles.length > 0) {
      const textFiles = attachedFiles.filter(f => !f.isImage);
      const imageFiles = attachedFiles.filter(f => f.isImage);
      if (textFiles.length > 0) {
        const textBlock = textFiles.map(f => `[Attached file: ${f.name}]\n\`\`\`\n${f.content}\n\`\`\``).join('\n\n');
        messageToSend = `${textBlock}\n\nUser question: ${message}`;
        userQuery = message;
      }
      if (imageFiles.length > 0) {
        imageData = imageFiles.map(f => ({ base64: f.base64, mimeType: f.mimeType }));
        userQuery = userQuery || message;
      }
      setAttachedFiles([]);
    }

    // NOTE: Contradiction checking has been moved to async background task in Rust backend
    // This allows the conversation to flow naturally without blocking on contradiction checks
    // If a contradiction is detected after memory storage, user will be notified via UI

    // STEP 1: Send message to backend
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: message,  // Show clean message in chat UI (no raw file dump)
      timestamp: new Date().toISOString()
    };

    if (!skipUserMessageAdd) {
      setMessages(prev => [...prev, userMessage]);
    }

    try {
      console.log('=== SENDING TO RUST BACKEND ===');
      console.log('Message:', message);
      console.log('Session ID:', sessionId);
      console.log('User ID:', userId);
      console.log('Containment Mode:', containmentMode);
      console.log('Backend:', modelType);

      // Build conversation history from recent messages
      // Send up to last 50 messages - backend will apply adaptive limits based on model type
      // (Local models: 8 messages, API models: 40 messages)
      // Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
      const conversationHistory = messages.slice(-50).map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      console.log('Conversation History:', conversationHistory.length, 'messages (backend will apply adaptive limits)');
      console.log('KB Enabled:', kbEnabled);

      // Add a streaming placeholder message immediately so the user sees tokens as they arrive
      const streamId = Date.now() + 1;
      setMessages(prev => [...prev, {
        id: streamId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        isStreaming: true,
      }]);

      // Listen for token events from the Rust streaming backend
      const unlisten = await listen('stream-token', (event) => {
        setMessages(prev => prev.map(msg =>
          msg.id === streamId
            ? { ...msg, content: msg.content + event.payload }
            : msg
        ));
      });

      // Call Rust backend via Tauri invoke (still awaited for metadata + memory processing)
      // Knowledge Base RAG search is handled automatically in the backend when kbEnabled=true
      const response = await invoke('send_message_with_memory', {
        message: messageToSend,
        userQuery,
        userId,
        sessionId,
        backend: modelType,
        containmentMode,
        conversationHistory,
        kbEnabled,
        imageData
      });

      unlisten(); // Stop listening for tokens — stream is complete

      console.log('=== RUST BACKEND RESPONSE ===');
      console.log('Full response:', response);

      // Check if LLM detected need for web search
      if (response.web_search_needed) {
        console.log('[WebSearch] LLM requested web search');
        console.log('[WebSearch] Query:', response.web_search_query);
        console.log('[WebSearch] Original query:', response.original_query);

        if (webSearchAutoExecute && triggeredByWake) {
          await handleExecuteWebSearch(streamId, response.web_search_query, response.original_query);
          return;
        }

        // Replace streaming placeholder with the web search confirmation message
        const assistantMessage = {
          id: streamId,
          role: 'assistant',
          content: response.reply_text || '',
          timestamp: new Date().toISOString(),
          web_search_needed: true,
          web_search_query: response.web_search_query,
          original_query: response.original_query,
          isStreaming: false,
          metadata: {
            model_backend: response.model_backend,
            containment_mode: response.containment_mode,
          }
        };

        setMessages(prev => prev.map(msg => msg.id === streamId ? assistantMessage : msg));
        if (triggeredByWake && ttsEnabled) speakResponse(assistantMessage.content);
        return; // Don't continue with normal processing
      }

      // Replace streaming placeholder with the final message (same content + metadata)
      const assistantMessage = {
        id: streamId,
        role: 'assistant',
        content: response.reply_text || '',
        timestamp: new Date().toISOString(),
        isStreaming: false,
        recalled_memories: response.recalled_memories || [],
        metadata: {
          schema: response.schema,
          model_backend: response.model_backend,
          containment_mode: response.containment_mode,
          recalled_memories: response.recalled_memories || []
        }
      };

      console.log('Assistant message object:', assistantMessage);
      console.log('Recalled memories:', assistantMessage.recalled_memories);

      setMessages(prev => prev.map(msg => msg.id === streamId ? assistantMessage : msg));
      if (triggeredByWake && ttsEnabled) speakResponse(assistantMessage.content);

      console.log('[App] Metadata set:', {
        model_backend: response.model_backend,
        containment_mode: response.containment_mode,
        recalled_memories_count: response.recalled_memories?.length || 0
      });

      // Refresh Recent Memories to show new memories created during conversation
      if (memoryManagerRef.current) {
        console.log('[App] Triggering Recent Memories refresh after message');
        memoryManagerRef.current.refresh();
      }

    } catch (error) {
      console.error('Error sending message to Rust:', error);
      // Replace any in-progress streaming placeholder with the error, or add a new error message
      setMessages(prev => {
        const hasStreaming = prev.some(msg => msg.isStreaming);
        if (hasStreaming) {
          return prev.map(msg => msg.isStreaming
            ? { ...msg, isStreaming: false, content: `Error: ${error}. Please check your connection and try again.` }
            : msg
          );
        }
        return [...prev, {
          id: Date.now() + 1,
          role: 'assistant',
          content: `Error: ${error}. Please check your connection and try again.`,
          timestamp: new Date().toISOString()
        }];
      });
    } finally {
      isSendingRef.current = false;
      setIsLoading(false);
    }
  };

  // Keep ref current on every render so wake word effect never holds a stale closure
  handleSendMessageRef.current = handleSendMessage;

  // Stop the currently streaming generation.
  // Optimistically flip UI out of loading state immediately — the backend's
  // post-stream work (memory extraction, sync) continues in the background but
  // the user shouldn't have to wait for it. The re-entrant send guard is also
  // cleared so a fresh Send can be sent right away.
  const handleStopGeneration = async () => {
    setIsLoading(false);
    isSendingRef.current = false;
    try {
      await invoke('cancel_generation');
    } catch (e) {
      console.error('Failed to cancel generation:', e);
    }
  };

  // Edit the last user message: turn it into an inline editable textarea in place.
  // The ChatMessage component renders a textarea + Save/Cancel when its id matches
  // editingMessageId.
  const handleEditLastUser = () => {
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') { lastUserIdx = i; break; }
    }
    if (lastUserIdx === -1) return;
    setEditingMessageId(messages[lastUserIdx].id);
  };

  // Called from ChatMessage's Save button: update the message content in place,
  // drop any assistant response that followed, then re-run inference with the
  // edited content (no duplicate user message in the conversation log).
  const handleSaveEdit = async (messageId, newContent) => {
    setEditingMessageId(null);
    const idx = messages.findIndex(m => m.id === messageId);
    if (idx === -1) return;
    // Update content and drop anything after (usually a stale assistant reply)
    const truncated = messages.slice(0, idx + 1).map((m, i) =>
      i === idx ? { ...m, content: newContent } : m
    );
    setMessages(truncated);
    // Regenerate response — skip re-adding the user message since it's already in the list
    await handleSendMessage(newContent, { skipUserMessageAdd: true });
  };

  const handleCancelEdit = () => setEditingMessageId(null);

  // Regenerate the last assistant response: strip from last user onwards, then re-send
  const handleRegenerateLast = async () => {
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') { lastUserIdx = i; break; }
    }
    if (lastUserIdx === -1) return;
    const prompt = messages[lastUserIdx].content;
    setMessages(messages.slice(0, lastUserIdx));
    await handleSendMessage(prompt);
  };

  // Execute web search when user confirms
  const handleExecuteWebSearch = async (messageId, searchQuery, originalQuery) => {
    console.log('[WebSearch] User confirmed web search');
    console.log('[WebSearch] Message ID:', messageId);
    console.log('[WebSearch] Search query:', searchQuery);
    console.log('[WebSearch] Original query:', originalQuery);

    try {
      setIsLoading(true);

      // Update the message to show that search is in progress
      setMessages(prev => prev.map(msg =>
        msg.id === messageId
          ? { ...msg, content: `Searching the web for: "${searchQuery}"...`, web_search_needed: false }
          : msg
      ));

      // Execute the web search
      const searchResults = await invoke('execute_web_search', {
        query: searchQuery,
        maxResults: 5,
        fetchTopN: 3
      });

      console.log('[WebSearch] Search results:', searchResults);

      // Build context from search results to send to LLM
      let searchContext = `Here are the web search results for "${searchQuery}":\n\n`;

      if (searchResults.results && searchResults.results.length > 0) {
        searchResults.results.forEach((result, index) => {
          searchContext += `Source ${index + 1}: ${result.title}\n`;
          searchContext += `URL: ${result.url}\n`;
          if (result.snippet) {
            searchContext += `Summary: ${result.snippet}\n`;
          }
          if (result.content) {
            // Include more content for LLM to use
            searchContext += `Content: ${result.content.substring(0, 800)}\n`;
          }
          searchContext += `\n`;
        });
      } else {
        searchContext += `No results found.\n`;
      }

      // Send search results to LLM to generate natural answer, streaming tokens into the message
      console.log('[WebSearch] Sending search results to LLM for synthesis...');

      const llmPrompt = `Answer the user's question using the web search results below. Be direct and concise — do not explain your reasoning process, do not narrate what you are doing, just answer.\n\nQuestion: "${originalQuery}"\n\n${searchContext}\n\nAnswer directly based on the search results. If the results don't contain enough information, say so briefly.`;

      // Clear the "Searching..." placeholder and start streaming tokens into the message
      setMessages(prev => prev.map(msg =>
        msg.id === messageId ? { ...msg, content: '', isStreaming: true } : msg
      ));

      const unlisten = await listen('stream-token', (event) => {
        setMessages(prev => prev.map(msg =>
          msg.id === messageId
            ? { ...msg, content: msg.content + event.payload }
            : msg
        ));
      });

      const llmResponse = await invoke('send_message_with_memory', {
        message: llmPrompt,
        userId,
        sessionId,
        backend: modelType,
        containmentMode,
        conversationHistory: [], // Don't include history for this synthesis
        skipContainment: true, // Skip safety checks for internal web search synthesis
        skipMemoryStorage: true // Don't store web search results as memories
      });

      unlisten();
      console.log('[WebSearch] LLM synthesized answer:', llmResponse.reply_text);

      // Finalize message with full metadata and search results for transparency
      setMessages(prev => prev.map(msg =>
        msg.id === messageId
          ? {
              ...msg,
              content: llmResponse.reply_text,
              isStreaming: false,
              web_search_results: searchResults,
              web_search_query: searchQuery
            }
          : msg
      ));

      if (wakeTriggeredRef.current && ttsEnabled) speakResponse(llmResponse.reply_text);

      console.log('[WebSearch] Search complete and answer synthesized');

    } catch (error) {
      console.error('[WebSearch] Error executing search:', error);

      // Update message with error
      setMessages(prev => prev.map(msg =>
        msg.id === messageId
          ? { ...msg, content: `Web search failed: ${error}. Please try again.` }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  // Show access gate if user doesn't have access
  if (!hasAccess) {
    return <AccessGate />;
  }

  // Still checking whether first-run setup is needed — show spinner to avoid blank flash
  if (needsSetup === null) {
    return (
      <div style={{
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        height: '100vh', background: '#181a20', color: '#f8f8f2'
      }}>
        <div style={{textAlign: 'center'}}>
          <h2 style={{color: '#8be9fd'}}>Loading Zynkbot...</h2>
          <p style={{color: '#9aa5c4'}}>Memory without surveillance, intelligence without manipulation.</p>
        </div>
      </div>
    );
  }

  // Show setup wizard on first run
  if (needsSetup === true) {
    return <SetupWizard onComplete={async () => {
      setNeedsSetup(false);
      for (let i = 0; i < 8; i++) {
        const models = await fetchModels();
        const gotReal = models?.length > 0 && !(models.length === 1 && models[0].id === 'local');
        if (gotReal) break;
        await new Promise(r => setTimeout(r, 750));
      }
    }} />;
  }

  // Wait for user identity to load before rendering
  if (isLoadingIdentity || !userId) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: '#282a36',
        color: '#f8f8f2'
      }}>
        <div style={{textAlign: 'center'}}>
          <h2 style={{color: '#8be9fd'}}>Loading Zynkbot...</h2>
          <p style={{color: '#9aa5c4'}}>Memory without surveillance, intelligence without manipulation.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app dark-theme">
      <header className="app-header">
        <div>
          <h1>Zynkbot</h1>
          <p style={{margin: '5px 0', fontSize: '0.95rem', color: '#8be9fd'}}>
            Open Beta v0.9: Privacy-first AI companion with transparent editable memory, cross-device sync, filesharing and chat, RAG knowledge base, ensemble response, experimental snap-in development platform & distributed compute. Available on Windows, Linux, and Android.
          </p>
        </div>
        <div className="header-buttons">
          <button onClick={() => setShowWhyZynkbot(true)} className="header-btn">Why Zynkbot?</button>
          <button onClick={() => setShowDemoGuide(true)} className="header-btn">Getting Started</button>
          <button onClick={() => setShowAbout(true)} className="header-btn">About</button>
        </div>
      </header>

      {/* Collapsible Settings Sidebar */}
      <CollapsibleSidebar
        icon="⚙️"
        title="System Controls"
        onInfoClick={() => setShowUserIdentity(true)}
        onVoiceClick={() => setShowVoiceModal(true)}
        hideToggle={showConversationHistory}
        onOpen={() => {
          setShowAbout(false);
          setShowDemoGuide(false);
          setShowWhyZynkbot(false);
          setShowAPIKeys(false);
          setShowUserIdentity(false);
          setShowEnsemble(false);
          setShowConflictResolution(false);
          setShowZynkCluster(false);
          setShowKBManager(false);
          setShowOnboarding(false);
          setShowSnapInModal(false);
          memoryManagerRef.current?.close();
        }}
      >
        <ContainmentModeSelector
          currentMode={containmentMode}
          onModeChange={setContainmentMode}
        />

        <div className="model-selector">
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px'}}>
            <h3 style={{color: '#f8f8f2', fontSize: '1.1rem', margin: 0}}>Current Model ↓</h3>
            <div style={{display: 'flex', gap: '8px'}}>
              {!isMobile && (
              <button
                onClick={async () => {
                  try {
                    await invoke('open_models_folder');
                  } catch (error) {
                    console.error('Failed to open models folder:', error);
                    alert('Failed to open models folder: ' + error);
                  }
                }}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: '#50fa7b',
                  color: '#282a36',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  transition: 'background 0.2s'
                }}
                onMouseOver={(e) => e.target.style.background = '#70ffab'}
                onMouseOut={(e) => e.target.style.background = '#50fa7b'}
                title="Open models folder to add/manage local .gguf model files"
              >
                📁 Add Models
              </button>
              )}
              <button
                onClick={() => setShowAPIKeys(true)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  border: 'none',
                  background: '#bd93f9',
                  color: '#282a36',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  transition: 'background 0.2s'
                }}
                onMouseOver={(e) => e.target.style.background = '#cda3f9'}
                onMouseOut={(e) => e.target.style.background = '#bd93f9'}
                title="Configure API keys for Anthropic Claude and other cloud services"
              >
                🔑 API Keys
              </button>
            </div>
          </div>
          <select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            style={{
              width: '100%',
              padding: '8px',
              marginBottom: '10px',
              borderRadius: '4px',
              border: '1px solid #44475a',
              background: '#ffffff',
              color: '#000000',
              fontSize: '0.95rem'
            }}
          >
            {availableModels.length === 0 ? (
              <option>Loading models...</option>
            ) : (
              availableModels.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))
            )}
          </select>
          <p style={{fontSize: '0.85em', color: '#9aa5c4', marginTop: '10px'}}>
            {modelType && modelType.endsWith('.gguf') ? (
              <>
                <strong>Local Model:</strong> Runs on your device for complete privacy.
                Performance depends on hardware - add .gguf model files to the models/ directory.
              </>
            ) : (
              <>
                <strong>API Model:</strong> Fast responses via cloud API.
                Local models offer complete privacy and run on your device.
              </>
            )}
          </p>
        </div>

        {/* Getting Started */}
        <div style={{marginTop: '20px', padding: '15px', background: '#1e1f29', borderRadius: '8px', border: '1px solid #44475a'}}>
          <div
            onClick={() => setShowGettingStarted(!showGettingStarted)}
            style={{
              color: '#8be9fd',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showGettingStarted ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🚀 Getting Started</span>
            <span style={{ fontSize: '0.8rem' }}>{showGettingStarted ? '▼' : '▶'}</span>
          </div>
          {showGettingStarted && (
          <div>
          <p style={{fontSize: '0.9rem', color: '#8be9fd', lineHeight: '1.6', marginBottom: '15px'}}>
            New to Zynkbot? Load the Einstein demo below to see how memory and relationships work.
            After loading, open the <strong>Memory Manager</strong> to explore the memories and their connections.
          </p>

          {/* Einstein Demo */}
          <div style={{marginBottom: '20px', padding: '15px', background: '#282a36', borderRadius: '6px', border: '1px solid #44475a'}}>
          <h4 style={{color: '#bd93f9', fontSize: '0.95rem', marginTop: 0, marginBottom: '10px'}}>👨‍🔬 Try Einstein Demo</h4>
          <button
            onClick={async (e) => {
              const button = e.target;
              const originalText = button.textContent;

              // Prevent double-clicks or concurrent executions
              if (button.disabled) {
                console.log('[UI] Button already processing, ignoring click');
                return;
              }

              try {
                const confirmed = window.confirm(
                  'Load Einstein Demo\n\n' +
                  'This will load Einstein\'s 59 pre-built memories and relationships into your memory database.\n\n' +
                  'Make sure you have already cleared your memories in Memory Manager before continuing.\n\n' +
                  'This takes only a few seconds.\n\n' +
                  'Continue?'
                );

                if (!confirmed) {
                  return;
                }

                button.textContent = '⏳ Loading Einstein Demo...';
                button.disabled = true;
                button.style.opacity = '0.7';
                button.style.cursor = 'wait';

                const result = await invoke('apply_einstein_seed', {
                  userId: userId
                });

                console.log('Einstein seed applied:', result);

                button.textContent = '✅ Success! Reloading...';
                alert(
                  `✅ Einstein Demo Loaded!\n\n` +
                  `📚 Memories: ${result.loaded_count}\n` +
                  `🔗 Relationships: ${result.relationships_created || 0}\n\n` +
                  `Check Memory Manager to explore the memory graph!`
                );
                // Reload immediately after function completes (not after 1 second)
                window.location.reload();
              } catch (error) {
                console.error('Failed to load Einstein demo:', error);

                // Clear loading modal
                setIsLoadingEinstein(false);

                // Reset button
                button.textContent = originalText;
                button.disabled = false;
                button.style.opacity = '1';
                button.style.cursor = 'pointer';

                alert(`❌ Error: ${error}`);
              }
            }}
            style={{
              width: '100%',
              padding: '12px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              transition: 'transform 0.2s',
              marginBottom: '10px'
            }}
            onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            Load Einstein Demo
          </button>
          <p style={{fontSize: '0.85rem', color: '#9aa5c4', lineHeight: '1.4'}}>
            Loads Einstein's <strong>59 pre-built memories</strong> and relationships into your memory database in seconds.
            <br />Clear your memories in Memory Manager first if needed, then click to explore the Einstein demo.
          </p>
          </div>

          {/* Start Using Zynkbot */}
          <div style={{padding: '15px', background: '#282a36', borderRadius: '6px', border: '1px solid #44475a'}}>
          <h4 style={{color: '#4CAF50', fontSize: '0.95rem', marginTop: 0, marginBottom: '10px'}}>🎯 Begin Your Journey</h4>
          <p style={{fontSize: '0.85rem', color: '#9aa5c4', lineHeight: '1.6', marginBottom: '15px'}}>
            Once you've explored the Einstein demo, complete this quick 5-minute onboarding to let Zynkbot start getting to know you.
            Your conversations stay private on your device, and you can edit or delete anything at any time.
          </p>
          <button
            onClick={() => setShowOnboarding(true)}
            style={{
              width: '100%',
              padding: '12px',
              background: 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              transition: 'transform 0.2s',
              marginBottom: '10px'
            }}
            onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            🎯 Start 5-Minute Onboarding
          </button>
          <p style={{fontSize: '0.85rem', color: '#8be9fd', lineHeight: '1.6', padding: '10px', background: '#1e1f29', borderRadius: '4px', border: '1px solid #44475a'}}>
            💡 <strong>Tip:</strong> To ask about Zynkbot's features, include "Zynkbot" in your question — e.g., <em>"Zynkbot, how does Ensemble Mode work?"</em>
          </p>
          <p style={{fontSize: '0.85rem', color: '#ffb86c', lineHeight: '1.6', padding: '10px', background: '#1e1f29', borderRadius: '4px', border: '1px solid #44475a', marginTop: '8px'}}>
            ⚠️ <strong>Before you start:</strong> If you've been exploring with the Albert Einstein demo user, open the Memory Manager and click "Clear All Memories" first — otherwise his memories will conflict with yours during onboarding.
          </p>
          </div>
          </div>
          )}
        </div>

        <div style={{marginTop: '20px', padding: '15px', background: '#282a36', borderRadius: '8px', border: '1px solid #44475a'}}>
          <div
            onClick={() => setShowZynkSyncSection(!showZynkSyncSection)}
            style={{
              color: '#50fa7b',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showZynkSyncSection ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🔄 ZynkSync - Memory Sync</span>
            <span style={{ fontSize: '0.8rem' }}>{showZynkSyncSection ? '▼' : '▶'}</span>
          </div>
          {showZynkSyncSection && (
        <ZynkSyncPanel
          apiBaseUrl={API_BASE_URL}
          userId={userId}
          onOpenUserIdentity={() => setShowUserIdentity(true)}
          onOpenChat={(device, deviceId) => {
            setChatDevice(device);
            setCurrentDeviceId(deviceId);
          }}
          onIdentityAdopted={(newUserId) => setUserId(newUserId)}
          onMemoriesSynced={() => memoryManagerRef.current?.refresh()}
        />
          )}
        </div>

        <div style={{marginTop: '20px', padding: '15px', background: '#282a36', borderRadius: '8px', border: '1px solid #44475a'}}>
          <div
            onClick={() => setShowZynkLinkSection(!showZynkLinkSection)}
            style={{
              color: '#f1fa8c',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showZynkLinkSection ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🔗 ZynkLink - File Sharing & Chat</span>
            <span style={{ fontSize: '0.8rem' }}>{showZynkLinkSection ? '▼' : '▶'}</span>
          </div>
          {showZynkLinkSection && (
        <ZynkLinkPanel
          apiBaseUrl={API_BASE_URL}
          onOpenUserIdentity={() => setShowUserIdentity(true)}
          userId={userId}
        />
          )}
        </div>

        {/* Knowledge Base Panel - External Reference Documents */}
        <div style={{marginTop: '20px', padding: '15px', background: '#282a36', borderRadius: '8px', border: '1px solid #44475a'}}>
          <div
            onClick={() => setShowKnowledgeBaseSection(!showKnowledgeBaseSection)}
            style={{
              color: '#ffb86c',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showKnowledgeBaseSection ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>📚 Knowledge Base RAG</span>
            <span style={{ fontSize: '0.8rem' }}>{showKnowledgeBaseSection ? '▼' : '▶'}</span>
          </div>
          {showKnowledgeBaseSection && (
        <KnowledgeBasePanel
          userId={userId}
          onManageDocuments={() => setShowKBManager(true)}
        />
          )}
        </div>

        {/* === EXPERIMENTAL FEATURES === */}

        {/* Snap-ins - Professional & Personal AI Workspaces (Experimental) */}
        <div style={{marginTop: '40px', marginBottom: '15px', padding: '15px', background: '#282a36', borderRadius: '8px', border: '1px solid #ff79c6'}}>
          <div
            onClick={() => setShowSnapInsSection(!showSnapInsSection)}
            style={{
              color: '#ff79c6',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showSnapInsSection ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🧩 Snap-ins (Experimental)</span>
            <span style={{ fontSize: '0.8rem' }}>{showSnapInsSection ? '▼' : '▶'}</span>
          </div>
          {showSnapInsSection && (
          <div>
          <p style={{fontSize: '0.9rem', color: '#ff79c6', lineHeight: '1.6', marginBottom: '15px'}}>
            <strong>Snap-ins</strong> are domain-specific workspaces for professional and personal use.
            They integrate with Zynkbot's memory system while maintaining strict privacy boundaries.
          </p>
          <button
            onClick={() => setShowSnapInModal(true)}
            style={{
              width: '100%',
              padding: '12px',
              background: 'linear-gradient(135deg, #ff79c6 0%, #e869b3 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              transition: 'transform 0.2s',
              marginBottom: '10px'
            }}
            onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            📝 Try Sample: Therapist Journal
          </button>
          <p style={{fontSize: '0.85rem', color: '#ff79c6', lineHeight: '1.4', margin: '0'}}>
            <strong>Concept demo:</strong> See how a professional workspace could organize notes, maintain privacy, and integrate with RAG search.
            <br /><em style={{fontSize: '0.8rem', color: '#9aa5c4'}}>Proof-of-concept | Architecture foundation for professional snap-ins</em>
          </p>
          </div>
          )}
        </div>

        {/* ZynkCluster - hidden on mobile (desktop-only feature) */}
        {!isMobile && <div style={{marginTop: '15px', marginBottom: '150px', padding: '15px', background: '#282a36', borderRadius: '8px', border: '1px solid #bd93f9'}}>
          <div
            onClick={() => setShowZynkClusterSection(!showZynkClusterSection)}
            style={{
              color: '#bd93f9',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              marginBottom: showZynkClusterSection ? '10px' : '0',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>🔬 ZynkCluster (Upcoming)</span>
            <span style={{ fontSize: '0.8rem' }}>{showZynkClusterSection ? '▼' : '▶'}</span>
          </div>
          {showZynkClusterSection && (
          <div>
          <button
            onClick={() => setShowZynkCluster(true)}
            style={{
              width: '100%',
              padding: '12px',
              background: 'linear-gradient(135deg, #bd93f9 0%, #9b72e6 100%)',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              transition: 'transform 0.2s',
              marginBottom: '10px'
            }}
            onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
          >
            🔬 ZynkCluster (Upcoming)
          </button>
          <p style={{fontSize: '0.85rem', color: '#bd93f9', lineHeight: '1.4', margin: '0'}}>
            Research on <strong>distributed MoE inference</strong> - a novel approach to running large models across local networks.
            <br /><em style={{fontSize: '0.8rem', color: '#9aa5c4'}}>Design phase | Architecture docs & code examples available</em>
          </p>
          </div>
          )}
        </div>}

      </CollapsibleSidebar>

      {/* Main Content - Chat + Insights (2 columns) */}
      <div className={isMobile ? 'mobile-main-content' : undefined} style={{
        display: 'flex',
        gap: isMobile ? 0 : '20px',
        padding: isMobile ? undefined : '0 20px 20px 20px',
        maxWidth: isMobile ? undefined : '1800px',
        margin: isMobile ? undefined : '0 auto',
        alignItems: isMobile ? undefined : 'stretch',
        minHeight: isMobile ? undefined : 'calc(100vh - 200px)'
      }}>
        {/* Left: Conversation + Recent Memories */}
        <div className={isMobile ? 'mobile-left-col' : undefined} style={{flex: '1 1 60%', minWidth: '0', display: 'flex', flexDirection: 'column', gap: isMobile ? 0 : '20px', height: isMobile ? undefined : '100%'}}>
          <div className={isMobile ? 'mobile-conv-section' : undefined}>
            <div style={{marginBottom: '15px', display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
              <h2 style={{margin: 0, color: '#8be9fd'}}>Conversation</h2>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={handleClearConversation}
                  title="Start a new conversation"
                  style={{
                    padding: '5px 14px',
                    background: 'rgba(98,114,164,0.25)',
                    color: '#8be9fd',
                    border: '1px solid #6272a4',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    transition: 'all 0.2s',
                  }}
                  onMouseOver={(e) => { e.target.style.background = 'rgba(98,114,164,0.45)'; }}
                  onMouseOut={(e) => { e.target.style.background = 'rgba(98,114,164,0.25)'; }}
                >
                  ✏️ New
                </button>
                <button
                  onClick={() => setShowConversationHistory(true)}
                  disabled={containmentMode === 'hipaa'}
                  title={containmentMode === 'hipaa' ? 'Conversation history is disabled in HIPAA mode' : 'Browse past conversations'}
                  style={{
                    padding: '5px 14px',
                    background: containmentMode === 'hipaa' ? '#44475a' : 'rgba(98,114,164,0.25)',
                    color: containmentMode === 'hipaa' ? '#6272a4' : '#8be9fd',
                    border: '1px solid ' + (containmentMode === 'hipaa' ? '#44475a' : '#6272a4'),
                    borderRadius: '6px',
                    cursor: containmentMode === 'hipaa' ? 'not-allowed' : 'pointer',
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    opacity: containmentMode === 'hipaa' ? 0.5 : 1,
                    transition: 'all 0.2s',
                  }}
                  onMouseOver={(e) => { if (containmentMode !== 'hipaa') { e.target.style.background = 'rgba(98,114,164,0.45)'; } }}
                  onMouseOut={(e) => { if (containmentMode !== 'hipaa') { e.target.style.background = 'rgba(98,114,164,0.25)'; } }}
                >
                  History
                </button>
              </div>
            </div>
            {availableModels.length === 0 && (
              <div style={{
                margin: '8px 12px',
                padding: '12px 16px',
                background: '#2a1f00',
                border: '1px solid #f1c40f',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '10px',
                flexWrap: 'wrap',
              }}>
                <span style={{ color: '#f1c40f', fontSize: '0.9rem' }}>
                  ⚠️ No AI model configured — add an API key to start chatting.
                </span>
                <button
                  onClick={() => setShowAPIKeys(true)}
                  style={{
                    padding: '6px 14px',
                    background: '#f1c40f',
                    color: '#1a1a00',
                    border: 'none',
                    borderRadius: '6px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                    whiteSpace: 'nowrap',
                  }}
                >
                  🔑 Set Up API Keys
                </button>
              </div>
            )}
            <div className={isMobile ? 'mobile-conv-history-wrap' : undefined} style={{ position: 'relative' }}>
              <div className="conversation-history" ref={chatContainerRef} style={{minHeight: 'unset', marginBottom: 0}}>
                {messages.length === 0 ? (
                  <p style={{color: '#9aa5c4'}}>Start a conversation...</p>
                ) : (
                  (() => {
                    // Find indices of the last user message and last assistant message
                    // so ChatMessage can render Edit / Regenerate icons on those only.
                    let lastUserIdx = -1, lastAssistantIdx = -1;
                    for (let i = messages.length - 1; i >= 0; i--) {
                      if (lastUserIdx === -1 && messages[i].role === 'user') lastUserIdx = i;
                      if (lastAssistantIdx === -1 && messages[i].role === 'assistant') lastAssistantIdx = i;
                      if (lastUserIdx !== -1 && lastAssistantIdx !== -1) break;
                    }
                    return messages.map((msg, idx) => (
                      <ChatMessage
                        key={msg.id}
                        message={msg}
                        metadata={msg.role === 'assistant' ? msg.metadata : null}
                        onExecuteWebSearch={handleExecuteWebSearch}
                        sessionId={sessionId}
                        userId={userId}
                        onEdit={!isLoading && idx === lastUserIdx ? handleEditLastUser : undefined}
                        onRegenerate={!isLoading && idx === lastAssistantIdx ? handleRegenerateLast : undefined}
                        isEditing={editingMessageId === msg.id}
                        onSaveEdit={(newContent) => handleSaveEdit(msg.id, newContent)}
                        onCancelEdit={handleCancelEdit}
                      />
                    ));
                  })()
                )}
                {isLoading && (
                  <div className="message bot-message">
                    <div className="message-content">
                      {modelType && modelType.endsWith('.gguf')
                        ? '⏳ Processing with local model (this may take a minute or two on CPU)...'
                        : 'Thinking...'}
                    </div>
                  </div>
                )}
                <div ref={conversationEndRef} />
              </div>
              {showScrollButton && (
                <button
                  onClick={() => { userScrolledUpRef.current = false; conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }}
                  style={{
                    position: 'absolute',
                    bottom: '12px',
                    right: '16px',
                    width: '44px',
                    height: '44px',
                    borderRadius: '50%',
                    background: 'rgba(99, 102, 241, 0.9)',
                    border: 'none',
                    color: '#fff',
                    fontSize: '22px',
                    lineHeight: 1,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.45)',
                    zIndex: 10,
                  }}
                  aria-label="Scroll to bottom"
                >
                  ↓
                </button>
              )}
            </div>

            {/* Rotating tip — only shown before first message and while input is empty */}
            {messages.length === 0 && !input && (
              <div style={{ textAlign: 'center', marginBottom: '6px', padding: '0 12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
                <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.65)', fontStyle: 'italic', lineHeight: '1.6' }}>
                  {TIPS[tipIdx % TIPS.length]}
                </span>
                <button
                  onClick={() => setTipIdx(i => (i + 1) % TIPS.length)}
                  title="Next tip"
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.45)', fontSize: '13px', padding: '0 2px', lineHeight: 1, flexShrink: 0 }}
                >↻</button>
              </div>
            )}

            {/* Attached file chips */}
            {attachedFiles.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
                {attachedFiles.map((file, idx) => (
                  <div key={idx} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '6px 10px',
                    background: '#21222c',
                    border: `1px solid ${file.isImage ? '#bd93f9' : file.size > 50000 ? '#ffb86c' : '#50fa7b'}`,
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    color: '#f8f8f2',
                    minWidth: 0,
                    maxWidth: '100%',
                  }}>
                    {file.isImage ? (
                      <img
                        src={`data:${file.mimeType};base64,${file.base64}`}
                        alt="preview"
                        style={{ width: '32px', height: '32px', objectFit: 'cover', borderRadius: '4px', flexShrink: 0 }}
                      />
                    ) : <span style={{ flexShrink: 0 }}>📎</span>}
                    <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>{file.name}</span>
                    {file.isImage && (
                      <span style={{ color: '#bd93f9', fontSize: '0.75rem', flexShrink: 0, whiteSpace: 'nowrap' }}>🖼️ vision</span>
                    )}
                    {!file.isImage && file.size > 50000 && (
                      <span style={{ color: '#ffb86c', fontSize: '0.75rem', flexShrink: 0, whiteSpace: 'nowrap' }}>⚠️ large</span>
                    )}
                    <button
                      onClick={() => setAttachedFiles(prev => prev.filter((_, i) => i !== idx))}
                      style={{ flexShrink: 0, marginLeft: 'auto', background: 'none', border: 'none', color: '#ff5555', cursor: 'pointer', fontSize: '1rem', lineHeight: 1, padding: '0 2px' }}
                      title="Remove attachment"
                    >×</button>
                  </div>
                ))}
              </div>
            )}

            {/* INPUT LAYOUT: Textarea + 2x2 Button Grid */}
            <div className="chat-input-wrapper" style={{ display: 'flex', gap: '10px', alignItems: 'stretch', marginTop: '8px' }}>
              {/* Left: Textarea with KB button inside */}
              <div style={{ position: 'relative', flex: 1, minWidth: 0 }}>
                <textarea
                  ref={inputTextareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && !isLoading && !window.AndroidPaths) {
                      e.preventDefault();
                      handleSendMessage(input);
                    }
                  }}
                  placeholder={window.AndroidPaths ? "Type your message..." : "Type your message... (Shift+Enter for new line)"}
                  className="query-input"
                  disabled={isLoading}
                  rows={isMobile ? 3 : 4}
                  style={{
                    width: '100%',
                    resize: 'vertical',
                    minHeight: isMobile ? '60px' : '90px',
                    maxHeight: isMobile ? '135px' : '200px',
                    padding: '12px',
                    paddingBottom: '45px',
                    paddingRight: isMobile ? '50px' : '12px',
                    background: '#282a36',
                    border: '1px solid #44475a',
                    borderRadius: '8px',
                    color: '#f8f8f2',
                    fontSize: '1rem',
                    fontFamily: 'inherit',
                    boxSizing: 'border-box'
                  }}
                />

                {/* Bottom-left button row — flex container so buttons always stay adjacent */}
                <div style={{ position: 'absolute', bottom: '8px', left: '8px', display: 'flex', gap: '4px', zIndex: 10 }}>
                  {/* KB Search Button — three states: off / on-next / locked */}
                  <button
                    onClick={() => {
                      if (kbLocked) {
                        // Locked → off: release lock and disable
                        setKbLocked(false);
                        setSearchKBEnabled(false);
                      } else if (searchKBEnabled) {
                        // On-next → locked: keep enabled and lock it
                        setKbLocked(true);
                      } else {
                        // Off → on-next
                        setSearchKBEnabled(true);
                      }
                    }}
                    disabled={isLoading}
                    title={
                      kbLocked
                        ? "KB locked on — click to turn off"
                        : searchKBEnabled
                          ? "KB on for next message — click to lock on"
                          : "Click to search Knowledge Base"
                    }
                    style={{
                      height: '28px',
                      padding: '0 10px',
                      background: kbLocked
                        ? 'linear-gradient(135deg, #ffb86c 0%, #ff79c6 100%)'
                        : searchKBEnabled
                          ? 'linear-gradient(135deg, #8be9fd 0%, #50fa7b 100%)'
                          : 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                      color: (kbLocked || searchKBEnabled) ? '#282a36' : '#f8f8f2',
                      border: kbLocked
                        ? '2px solid #ffb86c'
                        : searchKBEnabled
                          ? '2px solid #50fa7b'
                          : 'none',
                      borderRadius: '6px',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '0.7rem',
                      transition: 'all 0.2s',
                      opacity: isLoading ? 0.5 : 1,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      boxShadow: kbLocked
                        ? '0 0 8px rgba(255, 184, 108, 0.6)'
                        : searchKBEnabled
                          ? '0 0 8px rgba(139, 233, 253, 0.5)'
                          : '0 2px 4px rgba(0,0,0,0.2)',
                    }}
                    onMouseOver={(e) => !isLoading && (e.currentTarget.style.transform = 'translateY(-1px)')}
                    onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                  >
                    {kbLocked ? '📚 KB LOCK' : searchKBEnabled ? '📚 KB ON' : '📚 KB'}
                  </button>

                  {/* Attach File Button */}
                  <button
                    onClick={handleAttachFile}
                    disabled={isLoading}
                    title={attachedFiles.length > 0 ? `${attachedFiles.length} file${attachedFiles.length > 1 ? 's' : ''} attached` : "Attach files or images"}
                    style={{
                      height: '28px',
                      padding: '0 10px',
                      background: attachedFiles.length > 0
                        ? 'linear-gradient(135deg, #ffb86c 0%, #ff79c6 100%)'
                        : 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                      color: attachedFiles.length > 0 ? '#282a36' : '#f8f8f2',
                      border: attachedFiles.length > 0 ? '2px solid #ffb86c' : 'none',
                      borderRadius: '6px',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '0.7rem',
                      transition: 'all 0.2s',
                      opacity: isLoading ? 0.5 : 1,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                    onMouseOver={(e) => !isLoading && (e.currentTarget.style.transform = 'translateY(-1px)')}
                    onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                  >
                    {attachedFiles.length > 0 ? `📎 ${attachedFiles.length} file${attachedFiles.length > 1 ? 's' : ''}` : '📎'}
                  </button>

                  {/* Camera button — Android only */}
                  {window.AndroidCamera && (
                    <button
                      onClick={handleCameraCapture}
                      disabled={isLoading}
                      title="Take a photo"
                      style={{
                        height: '28px',
                        padding: '0 10px',
                        background: 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                        color: '#f8f8f2',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        transition: 'all 0.2s',
                        opacity: isLoading ? 0.5 : 1,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      onMouseOver={(e) => !isLoading && (e.currentTarget.style.transform = 'translateY(-1px)')}
                      onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                    >
                      📷
                    </button>
                  )}
                </div>

                {/* Mobile: circular mic button, bottom-right of textarea */}
                {isMobile && (
                  <VoiceButton
                    ref={voiceButtonRef}
                    onTranscript={handleTranscript}
                    disabled={isLoading || isWakeRecording}
                    style={{
                      position: 'absolute',
                      bottom: '8px',
                      right: '8px',
                      width: '36px',
                      height: '36px',
                      minWidth: '36px',
                      minHeight: '36px',
                      borderRadius: '50%',
                      padding: '0',
                      zIndex: 10,
                      fontSize: '0.9rem',
                    }}
                  />
                )}
              </div>

              {/* Right: 2×2 Button Grid.
                  Desktop: Voice | Ensemble / Send | Clear
                  Mobile:  Ensemble | Clear / MemoryManager | Send */}
              <div className="chat-button-grid" style={{
                display: 'grid',
                gridTemplateColumns: '85px 85px',
                gridTemplateRows: '42px 42px',
                gap: '10px'
              }}>
                {/* Desktop-only: Voice in grid (mobile has circular button on textarea) */}
                {!isMobile && (
                  <VoiceButton
                    ref={voiceButtonRef}
                    onTranscript={handleTranscript}
                    disabled={isLoading || isWakeRecording}
                    style={{
                      width: '85px',
                      height: '42px',
                      minWidth: '85px',
                      maxWidth: '85px',
                      minHeight: '42px',
                      maxHeight: '42px',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                      gridColumn: 1,
                      gridRow: 1,
                    }}
                  />
                )}

                {/* Ensemble — mobile: TL (1,1); desktop: TR (1,2) */}
                <button
                  onClick={() => setShowEnsemble(true)}
                  disabled={isLoading || containmentMode === 'child'}
                  title={containmentMode === 'child' ? 'Ensemble mode is not available in Child safety mode' : 'Multi-model collaboration'}
                  style={{
                    width: '85px',
                    height: '42px',
                    minWidth: '85px',
                    maxWidth: '85px',
                    minHeight: '42px',
                    maxHeight: '42px',
                    padding: '0',
                    background: containmentMode === 'child' ? '#6272a4' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: (isLoading || containmentMode === 'child') ? 'not-allowed' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.75rem',
                    transition: 'all 0.2s',
                    opacity: (isLoading || containmentMode === 'child') ? 0.5 : 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    flex: 'none',
                    gridColumn: isMobile ? 1 : 2,
                    gridRow: 1,
                  }}
                  onMouseOver={(e) => !(isLoading || containmentMode === 'child') && (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  Ensemble
                </button>

                {/* Send/Stop — same grid slot. Turns into red Stop button while streaming. */}
                <button
                  onClick={isLoading ? handleStopGeneration : () => handleSendMessage(input)}
                  title={isLoading ? 'Stop generation' : 'Send message'}
                  style={{
                    width: '85px',
                    height: '42px',
                    minWidth: '85px',
                    maxWidth: '85px',
                    minHeight: '42px',
                    maxHeight: '42px',
                    padding: '0',
                    background: isLoading
                      ? 'linear-gradient(135deg, #ff5555 0%, #ff6b6b 100%)'
                      : 'linear-gradient(135deg, #50fa7b 0%, #3dd66b 100%)',
                    color: isLoading ? '#fff' : '#282a36',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.75rem',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    flex: 'none',
                    gridColumn: isMobile ? 2 : 1,
                    gridRow: 2,
                  }}
                  onMouseOver={(e) => (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  {isLoading ? '■ Stop' : 'Send'}
                </button>

                {/* Clear — mobile: TR (1,2); desktop: BR (2,2) */}
                <button
                  onClick={handleClearConversation}
                  disabled={isLoading}
                  title="Clear conversation"
                  style={{
                    width: '85px',
                    height: '42px',
                    minWidth: '85px',
                    maxWidth: '85px',
                    minHeight: '42px',
                    maxHeight: '42px',
                    padding: '0',
                    background: 'linear-gradient(135deg, #ff5555 0%, #ff6b6b 100%)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.75rem',
                    transition: 'all 0.2s',
                    opacity: isLoading ? 0.5 : 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    flex: 'none',
                    gridColumn: 2,
                    gridRow: isMobile ? 1 : 2,
                  }}
                  onMouseOver={(e) => !isLoading && (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  Clear
                </button>

                {/* Mobile-only: Memory Manager — BL (2,1); opens the modal via ref */}
                {isMobile && (
                  <button
                    onClick={() => memoryManagerRef.current?.open()}
                    disabled={isLoading}
                    title="Open Memory Manager"
                    style={{
                      width: '85px',
                      height: '42px',
                      minWidth: '85px',
                      maxWidth: '85px',
                      minHeight: '42px',
                      maxHeight: '42px',
                      padding: '0',
                      background: 'linear-gradient(135deg, #bd93f9 0%, #9575e0 100%)',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      fontWeight: 'bold',
                      fontSize: '0.72rem',
                      transition: 'all 0.2s',
                      opacity: isLoading ? 0.5 : 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                      flex: 'none',
                      gridColumn: 1,
                      gridRow: 2,
                    }}
                  >
                    📚 Memories
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* MemoryManager: renders inline widget + modal as siblings. On mobile,
              the .memory-manager widget div is hidden via CSS so only the modal is
              available — opened via memoryManagerRef.current.open() from the mobile
              button grid. */}
          <div>
            <MemoryManager
              ref={memoryManagerRef}
              user_id={userId}
              apiBaseUrl={API_BASE_URL}
              containmentMode={containmentMode}
            />
          </div>
        </div>

        {/* Live Insights panel removed - simplified to single-column layout */}
      </div>

      <WhyZynkbotModal isOpen={showWhyZynkbot} onClose={() => setShowWhyZynkbot(false)} />
      <AboutModal isOpen={showAbout} onClose={() => setShowAbout(false)} />
      <GettingStartedModal isOpen={showDemoGuide} onClose={() => setShowDemoGuide(false)} onOpenAPIKeys={() => setShowAPIKeys(true)} />
      <APIKeyModal
        isOpen={showAPIKeys}
        onClose={() => setShowAPIKeys(false)}
        onKeysChanged={async () => {
          console.log('🔄 API keys changed, refreshing models...');
          await fetchModels();
          console.log('✅ Models refreshed');
        }}
      />
      <UserIdentityModal isOpen={showUserIdentity} onClose={() => setShowUserIdentity(false)} apiBaseUrl={API_BASE_URL} sessionId={sessionId} />
      <ConversationHistoryPanel
        isOpen={showConversationHistory}
        onClose={() => setShowConversationHistory(false)}
        userId={userId}
        containmentMode={containmentMode}
        onResume={handleResumeSession}
      />
      <KnowledgeBaseManager
        isOpen={showKBManager}
        onClose={() => setShowKBManager(false)}
        userId={userId}
      />
      <ZChatModal
        isOpen={chatDevice !== null}
        onClose={() => {
          setChatDevice(null);
          setCurrentDeviceId(null);
        }}
        apiBaseUrl={API_BASE_URL}
        device={chatDevice}
        currentDeviceId={currentDeviceId}
      />
      <ConflictResolutionModal
        isOpen={showConflictResolution}
        conflict={currentConflict}
        onResolve={handleConflictResolve}
        onClose={() => {
          setShowConflictResolution(false);
          setCurrentConflict(null);
        }}
      />
      <EnsembleModal
        isOpen={showEnsemble}
        onClose={() => setShowEnsemble(false)}
        availableModels={availableModels}
        userId={userId}
        sessionId={sessionId}
        containmentMode={containmentMode}
        onEnsembleComplete={(result) => {
          const now = new Date().toISOString();
          const userMessage = {
            id: Date.now(),
            role: 'user',
            content: result.question,
            timestamp: now,
          };
          const ensembleMessage = {
            id: Date.now() + 1,
            role: 'assistant',
            content: result.synthesized_response,
            timestamp: now,
            recalled_memories: [],
            metadata: {
              model_backend: 'Ensemble: ' + result.individual_responses.map(r => r.model).join(', '),
              containment_mode: containmentMode,
              recalled_memories: [],
              ensemble: true,
              individual_responses: result.individual_responses
            }
          };
          setMessages(prev => [...prev, userMessage, ensembleMessage]);
        }}
      />
      <ZynkClusterModal
        isOpen={showZynkCluster}
        onClose={() => setShowZynkCluster(false)}
      />
      <OnboardingModal
        isOpen={showOnboarding}
        onClose={() => setShowOnboarding(false)}
        userId={userId}
      />
      <SnapInModal
        isOpen={showSnapInModal}
        onClose={() => setShowSnapInModal(false)}
        userId={userId}
      />
      <VoiceModal
        isOpen={showVoiceModal}
        onClose={() => setShowVoiceModal(false)}
        voiceInputSource={voiceInputSource}
        onVoiceSourceChange={(src) => { setVoiceInputSource(src); localStorage.setItem('zynkbot_voice_input_source', src); }}
        heyZynkEnabled={heyZynkEnabled}
        onHeyZynkChange={(v) => { setHeyZynkEnabled(v); localStorage.setItem('zynkbot_hey_zynk_enabled', v); }}
        wakeWordModelReady={wakeWordModelReady}
        wakeWordDownloadProgress={wakeWordDownloadProgress}
        wakeWordDownloadError={wakeWordDownloadError}
        onDownloadModels={() => { setWakeWordDownloadError(''); window.WakeWordBridge?.downloadModels(); }}
        ttsEnabled={ttsEnabled}
        onTtsEnabledChange={(v) => { setTtsEnabled(v); localStorage.setItem('zynkbot_tts_enabled', v); }}
        webSearchAutoExecute={webSearchAutoExecute}
        onWebSearchAutoExecuteChange={(v) => { setWebSearchAutoExecute(v); localStorage.setItem('zynkbot_web_search_auto', v.toString()); }}
        conversationModeEnabled={conversationModeEnabled}
        onConversationModeChange={(v) => { setConversationModeEnabled(v); localStorage.setItem('zynkbot_conversation_mode', v.toString()); }}
      />

      {/* Einstein Demo Loading Modal (blocking, no dismiss) */}
      {isLoadingEinstein && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.9)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div style={{
            background: '#282a36',
            borderRadius: '12px',
            padding: '40px',
            maxWidth: '600px',
            width: '100%',
            border: '2px solid #8be9fd',
            boxShadow: '0 8px 32px rgba(139, 233, 253, 0.3)'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '30px' }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>🧠</div>
              <h2 style={{ color: '#8be9fd', margin: '0 0 15px 0' }}>Restoring Einstein</h2>
              <p style={{ color: '#9aa5c4', fontSize: '0.9rem', margin: 0 }}>
                Applying pre-computed seed data. This should only take a moment...
              </p>
            </div>

            {/* Animated Progress Bar */}
            <div style={{
              width: '100%',
              height: '8px',
              background: '#44475a',
              borderRadius: '4px',
              overflow: 'hidden',
              position: 'relative'
            }}>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                height: '100%',
                width: '100%',
                background: 'linear-gradient(90deg, transparent, #8be9fd, transparent)',
                animation: 'shimmer 2s infinite',
                backgroundSize: '200% 100%'
              }}></div>
            </div>

            <p style={{
              textAlign: 'center',
              color: '#8be9fd',
              fontSize: '0.9rem',
              marginTop: '20px',
              fontStyle: 'italic'
            }}>
              ⏳ Creating 59 memories with embeddings and relationships...
            </p>
          </div>
        </div>
      )}

      {/* Waveform recording overlay — shown on mobile when mic is active */}
      {isWakeRecording && isMobile && (
        <div
          onClick={async () => { const text = await stopWakeRecordingRef.current?.(); if (text) { wakeTriggeredRef.current = true; handleSendMessageRef.current?.(text); } else { conversationLoopActiveRef.current = false; } }}
          style={{
            position: 'fixed',
            bottom: 0, left: 0, right: 0,
            background: 'rgba(30, 31, 41, 0.97)',
            borderTop: '1px solid #44475a',
            padding: '16px 20px 20px',
            zIndex: 9997,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '12px',
            cursor: 'pointer',
            userSelect: 'none',
          }}
        >
          <style>{`
            @keyframes wv { 0%,100%{height:4px} 50%{height:var(--h)} }
          `}</style>
          <div style={{ display: 'flex', alignItems: 'center', gap: '3px', height: '40px', pointerEvents: 'none' }}>
            {Array.from({ length: 32 }, (_, i) => {
              const peak = 8 + Math.floor(Math.abs(Math.sin(i * 0.7 + 1.2)) * 32);
              const delay = (i * 0.06).toFixed(2);
              const dur = (0.5 + (i % 5) * 0.12).toFixed(2);
              return (
                <div key={i} style={{
                  width: '3px',
                  borderRadius: '2px',
                  background: '#8be9fd',
                  '--h': `${peak}px`,
                  animation: `wv ${dur}s ${delay}s ease-in-out infinite`,
                  height: '4px',
                  alignSelf: 'center',
                }} />
              );
            })}
          </div>
          <p style={{ color: '#9aa5c4', fontSize: '0.85rem', margin: 0, pointerEvents: 'none' }}>
            Listening… tap anywhere to stop and send
          </p>
        </div>
      )}

      {/* Wake word detection flash banner */}
      {wakeWordFlash && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0,
          padding: '14px',
          background: 'rgba(139, 233, 253, 0.93)',
          color: '#282a36',
          textAlign: 'center',
          fontWeight: 'bold',
          fontSize: '1rem',
          zIndex: 9998,
          pointerEvents: 'none',
        }}>
          🎙️ Hey Zynk — speak now, then tap 🎤 to send
        </div>
      )}

      {/* Add shimmer animation */}
      <style>{`
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>

    </div>
  );
}
