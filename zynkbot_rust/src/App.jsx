import React, { useState, useEffect, useLayoutEffect, useRef, useCallback } from "react";
import { v4 as uuidv4 } from 'uuid';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open as openFileDialog, save as saveFileDialog } from '@tauri-apps/plugin-dialog';
import "./styles/App.css";
// LiveInsightsPanel removed - simplified to single-column layout
import ContainmentModeSelector from "./components/ContainmentModeSelector";
import ChatMessage from "./components/ChatMessage";
import MemoryManager from "./components/MemoryManager";
import AboutModal from "./components/AboutModal";
import ReportProblemModal from "./components/ReportProblemModal";
import GettingStartedModal from "./components/GettingStartedModal";
import WhyZynkbotModal from "./components/WhyZynkbotModal";
import APIKeyModal from "./components/APIKeyModal";
import ZynkSyncPanel from "./components/ZynkSyncPanel";
import ZynkLinkPanel from "./components/ZynkLinkPanel";
import CollapsibleSidebar from "./components/CollapsibleSidebar";
import ControlsPage from "./components/ControlsPage";
import ControlsMenuButton from "./components/ControlsMenuButton";
import { openUrl } from '@tauri-apps/plugin-opener';
import PageIntro, { introSeen } from "./components/PageIntro";
import SnapInPlans from "./components/SnapInPlans";
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
import CheckInCard from "./components/CheckInCard";
import ConversationHistoryPanel from "./components/ConversationHistoryPanel";
import VoiceModal from "./components/VoiceModal";
import { useVoiceSession, parseVoiceCommand, shouldSpeakReply, nativeTurnsToMessages } from './hooks/useVoiceSession';
import { confirmDialog, messageDialog } from './utils/confirmDialog';

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


// Shown under "Containment Mode" in the System Controls list (same names as the selector).
// How many tokens the selected model can take in one request. Published windows
// as of 2026-10-25, conservative where a provider's models differ; a model file on
// the phone runs in a 4,096-token context (local_models.rs), 8,192 on the desktop.
function modelWindowTokens(modelId, models) {
  const id = String(modelId || '').toLowerCase();
  const kind = String(models?.find(m => m.id === modelId)?.type || models?.find(m => m.id === modelId)?.model_type || '').toLowerCase();
  if (kind === 'local' || id.endsWith('.gguf') || id.startsWith('local')) return window.AndroidPaths ? 4096 : 8192;
  if (id.includes('anthropic') || id.includes('claude')) return 200000;
  if (id.includes('gpt-4.1')) return 1000000;
  if (id.includes('gpt-5')) return 400000;
  if (id.includes('openai') || id.includes('gpt') || id.startsWith('o')) return 128000;
  if (id.includes('grok-4')) return 256000;
  if (id.includes('xai') || id.includes('grok')) return 131072;
  if (id.includes('mistral')) return 128000;
  return 8192; // custom / Ollama: unknown, assume small
}

// Sources saved with a searched or page-read answer (sources_json on the row, 0028), back in
// the shape ChatMessage renders under the answer. Anything unreadable is simply no list.
const parseSources = (text) => {
  if (!text) return undefined;
  try { const v = JSON.parse(text); return v && Array.isArray(v.results) && v.results.length ? v : undefined; } catch (_) { return undefined; }
};

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
  // A restart (update, crash, swipe-away) forgets the page's session, but Rust still
  // knows the thread hands-free turns were joining. Put it back on screen instead of
  // opening an empty one; a thread "disappeared" that way on 2026-09-08.
  const restoredRef = useRef(false);
  // Android: a file arrived in the Zynkbot storage location through the Files app or
  // the Share button while the app was in the background (ZynkShareProvider). Re-index
  // the share even when the Link panel is not open — a linked phone lists from that index.
  useEffect(() => {
    if (!window.AndroidPaths) return;
    // Also the one place the share record is kept right: exactly one share, at the
    // current folder. Runs at startup, not only when the Link tab opens, so a peer
    // browsing this phone sees the right folder after the location changed (2026-09-19).
    window.__zynkShareChanged = async () => {
      try {
        const dir = window.AndroidPaths.getShareDir();
        if (!dir) return;
        const data = await invoke('list_my_shared_directories');
        const dirs = data.shared_directories || [];
        for (const d of dirs.filter(d => d.local_path !== dir)) {
          await invoke('unshare_directory', { shareId: d.id }).catch(() => {});
        }
        let share = dirs.find(d => d.local_path === dir);
        if (!share) {
          const res = await invoke('share_directory', { localPath: dir, shareName: 'ZynkbotShare', isReadable: true, isWritable: false });
          if (res?.success && res.share_id) share = { id: res.share_id };
        }
        if (share) await invoke('scan_shared_directory', { shareId: share.id, maxFiles: 1000 });
      } catch (e) {
        console.warn('[ZynkLink] Share registration / re-index failed:', e);
      }
    };
    window.__zynkShareChanged();
    return () => { window.__zynkShareChanged = null; };
  }, []);

  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    let fresh = false;
    try { fresh = !sessionStorage.getItem('zynkbot_restored'); sessionStorage.setItem('zynkbot_restored', '1'); } catch (_) { fresh = true; }
    if (!fresh) return;
    (async () => {
      try {
        const current = await invoke('get_current_session');
        if (!current || current === sessionId) return;
        const rows = await invoke('get_conversation_messages', { sessionId: current });
        if (!rows || rows.length === 0) return;
        setSessionId(current);
        try { sessionStorage.setItem('zynkbot_session_id', current); } catch (_) {}
        setMessages(rows.map((m) => ({ id: m.id, role: m.role, content: m.content, timestamp: m.created_at, recalled_memories: [], web_search_results: parseSources(m.sources_json) })));
        console.log('[Session] restored thread', current.slice(0, 8), 'with', rows.length, 'messages');
      } catch (e) {
        console.warn('[Session] could not restore the current thread:', e);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Startup Ollama check — runs once every launch. Ollama is local to the desktop
  // (a phone reaches it through the desktop relay), so this is only meaningful there.
  // It talks to Ollama over HTTP, so it needs no knowledge of where models are stored.
  useEffect(() => {
    if (window.AndroidPaths) return;
    (async () => {
      try {
        const st = await invoke('ollama_status');
        const extra = st.models && st.models.length ? ` — ${st.models.length} model(s): ${st.models.join(', ')}` : '';
        console.log(`[Ollama] startup check: ${st.state}${extra}`);
      } catch (e) {
        console.warn('[Ollama] startup check failed:', e);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
  const [inputCopied, setInputCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  // The local model's meter: tokens of the prompt read so far (engine event), cleared when
  // the first word streams in or the request ends (Matt, 2026-10-24).
  const [localProgress, setLocalProgress] = useState(null);
  // What the app is doing while it waits for the first word ("Searching the web…"); set by the backend.
  const [chatStatus, setChatStatus] = useState(null);
  useEffect(() => {
    let unlisten;
    listen('chat-status', (e) => { setChatStatus(typeof e.payload === 'string' ? e.payload : null); }).then(fn => { unlisten = fn; });
    return () => { if (unlisten) unlisten(); };
  }, []);
  useEffect(() => { if (!isLoading) setChatStatus(null); }, [isLoading]);

  useEffect(() => {
    let unlisten;
    listen('local-prompt-progress', (e) => { setLocalProgress(e.payload || null); }).then(fn => { unlisten = fn; });
    return () => { if (unlisten) unlisten(); };
  }, []);
  useEffect(() => { if (!isLoading) setLocalProgress(null); }, [isLoading]);
  const TIPS = [
    <>Tip: start with "<strong>Remember:</strong>" to save exactly what follows into your memories, typed or spoken. Any case works, and by voice just say "remember colon" before the fact.</>,
    <>Tap <strong>Memory Manager</strong> to see everything Zynkbot knows about you. You can edit, delete, or add memories at any time.</>,
    <>Open <strong>History</strong> to browse past conversations and copy any response.</>,
    <><strong>ZynkSync</strong> keeps your memories in sync across all your devices automatically — no account required.</>,
    <><strong>ZynkLink</strong> lets you share files directly with paired devices over your local network. No cloud, no upload.</>,
    <><strong>Ensemble mode</strong> queries multiple AI models at once and synthesizes their answers — great for questions where accuracy matters.</>,
    <>Add documents to the <strong>Knowledge Base</strong> — Zynkbot can answer questions about them using RAG search.</>,
    <>Everything stays on your device. Your memories and conversations never leave unless you choose to back them up.</>,
    <>Include "<strong>Zynkbot</strong>" in your question to ask about its own features — e.g. "Zynkbot, how does Ensemble mode work?"</>,
    <>The <strong>mic button</strong> lets you dictate your message — transcribed locally on-device, no internet needed.</>,
    <>Dictate anywhere you used to type: tap the <strong>mic</strong>, then <strong>Copy</strong>, then <strong>Clear</strong>.</>,
  ];
  const [tipIdx, setTipIdx] = useState(() => Math.floor(Math.random() * TIPS.length));
  const [editingMessageId, setEditingMessageId] = useState(null);
  const isSendingRef = useRef(false); // synchronous guard against concurrent sends
  const [modelType, setModelType] = useState(() => {
    return localStorage.getItem('zynkbot_preferred_model') || 'local';
  });
  const [availableModels, setAvailableModels] = useState([]);
  // The model files on this device, for the Current Model page (2026-10-23).
  const [localModelFiles, setLocalModelFiles] = useState([]);
  // The Zynkbot model download from the Current Model page (2026-10-27): the "Later" of
  // setup, and the way onto a device that was set up before the model was offered.
  const [zbDownload, setZbDownload] = useState({ active: false, percent: 0, downloaded: 0, total: 0, verifying: false, error: null });
  // The first time the local model is chosen, say once what to expect (Matt, 2026-10-23:
  // the limit is speed, not intelligence, so set the time expectation rather than hide
  // features). Never shown again on this device.
  useEffect(() => {
    if (!modelType || !modelType.endsWith('.gguf')) return;
    let seen = false;
    try { seen = !!localStorage.getItem('zynkbot_local_model_intro_seen'); } catch (_) {}
    if (seen) return;
    try { localStorage.setItem('zynkbot_local_model_intro_seen', '1'); } catch (_) {}
    messageDialog(
      'You are now using the local model. It runs entirely on this device and nothing leaves it.\n\n' +
      'What to expect: replies take half a minute to a minute on a phone, so keep questions short. ' +
      'Memories still save in the background. Web search works here too, just slower. Image questions need a cloud model, and so does Ensemble on a phone.\n\n' +
      'Switch back any time under Current Model.'
    );
  }, [modelType]);

  // The conversation's name, shown in one line above the messages with a pencil to rename
  // (Matt, 2026-10-23: "I can't believe it took this long"). The backend fills it from the
  // first message; History renames it too, and tells us through a window event.
  const [threadTitle, setThreadTitle] = useState('');
  // Conversation mode (2026-10-28): a short instruction block in front of the prompt, for this
  // page session; 'normal' sends nothing. The names and blocks live in chat_modes.rs.
  const [chatMode, setChatMode] = useState('normal');
  // The modes, in menu order: key, name, one line on what it does. The menu is drawn by the
  // page (2026-10-29, Matt): a standard drop-down's open list is a native GTK menu on Linux,
  // which ignores the page colours and shows no description per entry.
  const MODES = [
    ['normal', 'Normal', 'The usual Zynkbot: warm, direct and honest. Nothing extra is sent.'],
    ['tool', 'Tool', 'Does what is asked and nothing more. No commentary, no suggestions.'],
    ['debate', 'Debate', 'Challenges weak reasoning and names the weakest point; argues the other side when asked.'],
    ['premise', 'Premise', 'Takes what you state as true for this conversation and reasons from it, no fact-checking.'],
    ['editing', 'Editing', 'Fixes grammar, spelling and clarity only; keeps your wording, tone and meaning.'],
  ];
  const MODE_HELP = Object.fromEntries(MODES.map(([k, name, blurb]) => [k, k === 'normal' ? '' : `${name}: ${blurb}`]));
  // Check-in question (labs/check_in, 2026-10-29): asked once a day at the start of a
  // conversation, about something the user said they would do; experimental in beta 5.
  const [pendingCheckin, setPendingCheckin] = useState(null);
  const [checkinEnabled, setCheckinEnabled] = useState(true);
  const containmentModeRef = useRef('sovereign');
  const checkinAskedThisOpenRef = useRef(false);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const modeMenuRef = useRef(null);
  useEffect(() => {
    if (!modeMenuOpen) return undefined;
    const away = (e) => { if (modeMenuRef.current && !modeMenuRef.current.contains(e.target)) setModeMenuOpen(false); };
    const esc = (e) => { if (e.key === 'Escape') setModeMenuOpen(false); };
    document.addEventListener('mousedown', away);
    document.addEventListener('touchstart', away);
    document.addEventListener('keydown', esc);
    return () => {
      document.removeEventListener('mousedown', away);
      document.removeEventListener('touchstart', away);
      document.removeEventListener('keydown', esc);
    };
  }, [modeMenuOpen]);
  // At the app's open, or on a return after ten quiet minutes, ask the backend for the one
  // check-in question of the day. Never while something is typed or a card is open.
  const maybeAskCheckin = async (onOpen, force = false) => {
    if (!userId || pendingCheckin) return false;
    const el = inputTextareaRef.current;
    if (!force && el && el.value && el.value.trim()) return false;
    if (!force && !onOpen && Date.now() - lastTouchRef.current < 10 * 60 * 1000) return false;
    try {
      const q = await invoke('next_memory_checkin', { userId, containmentMode: containmentModeRef.current, force });
      if (q) { setPendingCheckin(q); return true; }
    } catch (e) { console.warn('[App] check-in:', e); }
    return false;
  };
  // Bring the card into view when it appears, even if the user had scrolled up.
  useEffect(() => {
    if (!pendingCheckin) return;
    userScrolledUpRef.current = false;
    const t = setTimeout(() => conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 80);
    return () => clearTimeout(t);
  }, [pendingCheckin]);
  const answerCheckin = async (answer) => {
    const q = pendingCheckin;
    setPendingCheckin(null);
    if (!q) return;
    try {
      const note = await invoke('answer_memory_checkin', { memoryId: q.memory_id, answer });
      if (note) notify(note, 3500);
    } catch (e) { notify(`Could not save that: ${e}`, 6000); }
  };
  const [editingThreadTitle, setEditingThreadTitle] = useState(null); // null, or the draft text
  const fetchThreadTitle = async () => {
    if (!userId || !sessionId) return;
    try {
      const sessions = await invoke('list_conversation_sessions', { userId, limit: 200, offset: 0 });
      const mine = (sessions || []).find(s => s.session_id === sessionId);
      setThreadTitle(mine?.title || '');
    } catch (_) {}
  };
  useEffect(() => {
    fetchThreadTitle();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, userId, messages.length]);
  useEffect(() => {
    const onRenamed = (e) => { if (e.detail?.sessionId === sessionId) setThreadTitle(e.detail.title || ''); };
    window.addEventListener('zynk-session-renamed', onRenamed);
    return () => window.removeEventListener('zynk-session-renamed', onRenamed);
  }, [sessionId]);
  const commitThreadTitle = async () => {
    const value = (editingThreadTitle || '').trim();
    setEditingThreadTitle(null);
    if (!value || value === threadTitle || !userId) return;
    try {
      await invoke('set_session_title', { sessionId, userId, title: value });
      setThreadTitle(value);
    } catch (e) { console.error('[Thread title] rename failed:', e); }
  };
  // One quiet line above the input when a memory replaced an older one (Matt, 2026-10-23:
  // "so you know something changed without a card"). Fades on its own.
  const [memoryUpdateNotice, setMemoryUpdateNotice] = useState('');
  // The same line carries other one-sentence notices (Ensemble on the local model,
  // 2026-10-24); showNotice sets it and clears it after 12 s.
  const noticeTimer = useRef(null);
  const showNotice = (text) => {
    setMemoryUpdateNotice(text);
    clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setMemoryUpdateNotice(''), 12000);
  };
  useEffect(() => {
    let unlisten; let timer;
    listen('memory-superseded', async (e) => {
      const { new_memory_id, old_memory_id } = e.payload || {};
      try {
        const [fresh, old] = await Promise.all([
          invoke('get_memory', { memoryId: new_memory_id }),
          invoke('get_memory', { memoryId: old_memory_id }),
        ]);
        const name = (m) => (m?.title || m?.content || '').replace(/\s+/g, ' ').slice(0, 60);
        setMemoryUpdateNotice(`Updated: "${name(fresh)}" replaces "${name(old)}". The old one is kept as history in Memory Manager.`);
      } catch (_) {
        setMemoryUpdateNotice('Updated: a memory replaced an older one. The old one is kept as history in Memory Manager.');
      }
      clearTimeout(timer);
      timer = setTimeout(() => setMemoryUpdateNotice(''), 12000);
    }).then(fn => { unlisten = fn; });
    return () => { if (unlisten) unlisten(); clearTimeout(timer); };
  }, []);
  // The sending phone holds its screen on while it serves a ZynkLink download
  // (MainActivity.keepScreenOn; events from zynklink.rs handle_zynklink_download).
  useEffect(() => {
    if (!window.AndroidPaths?.keepScreenOn) return;
    let active = 0; let unStart, unEnd;
    listen('zynklink:serving:start', () => { active += 1; try { window.AndroidPaths.keepScreenOn(true); } catch (_) {} }).then(fn => { unStart = fn; });
    listen('zynklink:serving:end', () => { active = Math.max(0, active - 1); try { window.AndroidPaths.keepScreenOn(active > 0); } catch (_) {} }).then(fn => { unEnd = fn; });
    return () => { if (unStart) unStart(); if (unEnd) unEnd(); };
  }, []);
  // Sovereign, not Guardian: a new user's first real answer used to be an
  // occasional hard block from a demo-grade classifier, which is a bad first
  // impression for a safety layer that exists to be shown off, not to actually
  // gate a real conversation. Sovereign warns instead of blocking (Matt, 2026-09-20).
  const [containmentMode, setContainmentMode] = useState("sovereign");
  useEffect(() => { containmentModeRef.current = containmentMode; }, [containmentMode]);
  // "Report a problem" — opened from the sidebar or a reply's ⚑ button.
  const [showReport, setShowReport] = useState(false);
  const [reportContext, setReportContext] = useState('');
  const openReport = (context = '') => { setReportContext(context || ''); setShowReport(true); };
  const [showAbout, setShowAbout] = useState(false);
  const [showDemoGuide, setShowDemoGuide] = useState(false);
  const [showWhyZynkbot, setShowWhyZynkbot] = useState(false);
  const [showUserIdentity, setShowUserIdentity] = useState(false);
  const [showEnsemble, setShowEnsemble] = useState(false);
  const [showConflictResolution, setShowConflictResolution] = useState(false);
  const [showZynkCluster, setShowZynkCluster] = useState(false);
  const [showConversationHistory, setShowConversationHistory] = useState(false);
  const [currentConflict, setCurrentConflict] = useState(null);
  const [chatDevice, setChatDevice] = useState(null);
  const [currentDeviceId, setCurrentDeviceId] = useState(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [searchKBEnabled, setSearchKBEnabled] = useState(false);
  const [kbLocked, setKbLocked] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState([]); // [{ name, content, size, isImage, ... }]
  // Collapsible section states - all start collapsed
  // System Controls: which section's page is open (null = the list). One at a time.
  const [controlsPage, setControlsPage] = useState(null);
  // Getting Started leads the System Controls list until onboarding has been completed
  // on this device (or the user unticks it on the Getting Started page); then it sits last.
  const [gettingStartedOnTop, setGettingStartedOnTopState] = useState(
    () => localStorage.getItem('zynkbot_getting_started_on_top') !== '0'
  );
  const setGettingStartedOnTop = (onTop) => {
    localStorage.setItem('zynkbot_getting_started_on_top', onTop ? '1' : '0');
    setGettingStartedOnTopState(onTop);
  };
  const gettingStartedButton = (
    <ControlsMenuButton
      icon="🚀"
      title="Getting Started"
      subtitle="Kitchen tip, Einstein demo, guided setup"
      onClick={() => setControlsPage('start')}
    />
  );
  const [showSnapInModal, setShowSnapInModal] = useState(false);
  const [isLoadingEinstein, setIsLoadingEinstein] = useState(false);
  const [copyAllDone, setCopyAllDone] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 768);
  const memoryManagerRef = useRef(null);
  const conversationEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const userScrolledUpRef = useRef(false);
  // True while a finger is on the chat (touchstart..touchend). Auto-scroll never
  // fires during a touch, so a drag is never fought token by token.
  const touchActiveRef = useRef(false);
  // Set right before we move scrollTop ourselves, so the scroll event that follows
  // is not mistaken for the user "returning to the bottom".
  const programmaticScrollRef = useRef(false);
  const inputTextareaRef = useRef(null);
  const voiceButtonRef = useRef(null);

  const voice = useVoiceSession({ setMessages });


  // Splice transcribed text into the input at the cursor position, padding with
  // spaces when adjacent to non-whitespace so words don't run together.
  // Live dictation (2026-10-20): while the mic is on, the words recognised so far are
  // shown in the box at the spot where the cursor was when recording started; the
  // finished transcript then takes their place. `before`/`after` are the text around
  // that spot, fixed for the whole recording.
  const liveDictationRef = useRef(null);
  const padJoin = (before, text, after) => {
    const leadPad = before && text && !/\s$/.test(before) ? ' ' : '';
    const trailPad = after && text && !/^\s/.test(after) ? ' ' : '';
    const body = `${leadPad}${text}${trailPad}`;
    return { value: `${before}${body}${after}`, cursor: before.length + body.length - trailPad.length };
  };
  const handleLiveDictation = (text) => {
    if (text === null) {
      // Recording over. If no final transcript replaced the live words, drop them.
      const live = liveDictationRef.current;
      liveDictationRef.current = null;
      if (live && live.showing) setInput(`${live.before}${live.after}`);
      return;
    }
    if (!liveDictationRef.current) {
      const el = inputTextareaRef.current;
      const current = input;
      const start = el && typeof el.selectionStart === 'number' ? el.selectionStart : current.length;
      const end = el && typeof el.selectionEnd === 'number' ? el.selectionEnd : current.length;
      liveDictationRef.current = { before: current.slice(0, start), after: current.slice(end), showing: false };
    }
    const live = liveDictationRef.current;
    if (!text) return;
    live.showing = true;
    const { value, cursor } = padJoin(live.before, text, live.after);
    setInput(value);
    requestAnimationFrame(() => {
      const el = inputTextareaRef.current;
      if (el) { el.setSelectionRange(cursor, cursor); el.scrollTop = el.scrollHeight; }
    });
  };
  const handleDictationTranscript = (text) => {
    if (!text) return;
    const live = liveDictationRef.current;
    if (!live) return insertTranscriptAtCursor(text);
    liveDictationRef.current = null;
    const { value, cursor } = padJoin(live.before, text, live.after);
    setInput(value);
    requestAnimationFrame(() => {
      const el = inputTextareaRef.current;
      if (el) { el.focus(); el.setSelectionRange(cursor, cursor); }
    });
  };

  // Copy everything in the box (2026-10-20): Zynkbot is the dictation tool on Linux and
  // on GrapheneOS, where the keyboard has none, and the text then goes somewhere else.
  const copyInputText = async () => {
    const text = input;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const el = inputTextareaRef.current;
      if (el) { el.focus(); el.select(); try { document.execCommand('copy'); } catch (__) {} }
    }
    setInputCopied(true);
    setTimeout(() => setInputCopied(false), 1500);
  };

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

  const [showScrollButton, setShowScrollButton] = useState(false);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);


  // Lock body scroll when any modal is open to prevent scrollbar ghost artifact
  const anyModalOpen = showAbout || showDemoGuide || showWhyZynkbot ||
    showUserIdentity || showEnsemble || showConflictResolution || showZynkCluster ||
    showConversationHistory || showOnboarding ||
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

  const fetchLocalModelFiles = async () => {
    try { setLocalModelFiles(await invoke('list_local_model_files')); } catch (_) { setLocalModelFiles([]); }
  };
  const formatModelSize = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
  const handleDownloadZbModel = async () => {
    if (zbDownload.active) return;
    setZbDownload({ active: true, percent: 0, downloaded: 0, total: 0, verifying: false, error: null });
    const unlisten = await listen('setup:llm_progress', (e) => {
      setZbDownload(d => ({ ...d, percent: e.payload.percent, downloaded: e.payload.downloaded, total: e.payload.total, verifying: !!e.payload.verifying }));
    });
    try {
      await invoke('download_user_model', { modelId: 'zb-qwen3-4b' });
      await fetchModels();
      await fetchLocalModelFiles();
      setZbDownload({ active: false, percent: 100, downloaded: 0, total: 0, verifying: false, error: null });
    } catch (e) {
      setZbDownload(d => ({ ...d, active: false, error: String(e) }));
    } finally {
      unlisten();
    }
  };
  // After a model file left the device: refresh both lists and, if it was the one in
  // use, fall back to the first model that remains.
  const afterModelFileGone = async (path) => {
    let models = [];
    try { models = await invoke('get_models'); setAvailableModels(models); } catch (_) {}
    await fetchLocalModelFiles();
    if (modelType === path) setModelType(models.find(m => m.id !== path)?.id || 'local');
  };
  const handleRemoveModelFile = async (file) => {
    const ok = await confirmDialog(`Remove ${file.filename} (${formatModelSize(file.size_bytes)}) from this device? This cannot be undone.`);
    if (!ok) return;
    try {
      await invoke('delete_user_model', { filename: file.filename });
      await afterModelFileGone(file.path);
    } catch (e) { await messageDialog(`Could not remove the model: ${e}`); }
  };
  // Desktop: the save dialog names the destination and Rust moves the file. Android: the
  // system save-as picker (MainActivity.moveFileTo) streams it where the user chooses.
  const handleMoveModelFile = async (file) => {
    try {
      if (window.AndroidPaths?.moveFileTo) {
        setModelMoveBusy(file.path);
        await new Promise((resolve, reject) => {
          window.__zmvResolve = resolve; window.__zmvReject = reject;
          if (!window.AndroidPaths.moveFileTo(file.path)) { window.__zmvResolve = null; window.__zmvReject = null; reject('file not found'); }
        });
      } else {
        const dest = await saveFileDialog({ defaultPath: file.filename, title: 'Move model to' });
        if (!dest) return;
        setModelMoveBusy(file.path);
        await invoke('move_user_model', { filename: file.filename, destinationPath: dest });
      }
      await afterModelFileGone(file.path);
    } catch (e) {
      if (e !== 'cancelled') await messageDialog(`Could not move the model: ${e}`);
    } finally { setModelMoveBusy(null); }
  };
  const [modelMoveBusy, setModelMoveBusy] = useState(null);
  useEffect(() => {
    if (controlsPage === 'model') fetchLocalModelFiles();
    if (controlsPage === 'experimental') invoke('get_checkin_enabled').then((on) => setCheckinEnabled(on !== false)).catch(() => {});
  }, [controlsPage]);
  useEffect(() => {
    const onChanged = () => { fetchModels(); fetchLocalModelFiles(); };
    window.addEventListener('zynk-models-changed', onChanged);
    return () => window.removeEventListener('zynk-models-changed', onChanged);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        const firstLocal = models.find(m => (m.type || m.model_type) === 'local');
        const custom = models.find(m => m.id === 'custom');
        const firstApi = models.find(m => (m.type || m.model_type) === 'api' && m.id !== 'custom');
        const defaultModel = firstLocal || custom || firstApi;
        if (defaultModel) {
          setModelType(defaultModel.id);
          localStorage.setItem('zynkbot_preferred_model', defaultModel.id);
          console.log('[Models] First run — defaulting to:', defaultModel.id);
        }
      } else if (stored === 'local' && !models.find(m => m.id === 'local' || (m.type || m.model_type) === 'local')) {
        // Stale 'local' stored value but no local models available (e.g. Android after a
        // desktop-only install migrated over). Fall through to custom/API instead of leaving
        // the user stuck on a backend that will error out.
        const custom = models.find(m => m.id === 'custom');
        const firstApi = models.find(m => (m.type || m.model_type) === 'api' && m.id !== 'custom');
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

  // A thread deleted on another device closes here if it is the one on screen, so
  // nothing is typed into a ghost (the backend also refuses to save into it). The
  // sidebar list refreshes on the same event (ConversationHistoryPanel).
  useEffect(() => {
    let unlisten = null;
    (async () => {
      unlisten = await listen('zynksync-threads-changed', (event) => {
        const deleted = event?.payload?.deleted || [];
        if (deleted.includes(sessionIdRef.current)) {
          const fresh = uuidv4();
          setSessionId(fresh);
          try { sessionStorage.setItem('zynkbot_session_id', fresh); } catch (_) {}
          setMessages([]);
          notify('This conversation was deleted on another device, so it was closed here.', 6000);
        }
      });
    })();
    return () => { if (unlisten) unlisten(); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-fetch when API keys arrive from a paired device (the backend applies them
  // and emits this per key). The model list is derived from which providers have
  // keys, so without this the dropdown stayed empty until an app restart even
  // though the push had succeeded (OnePlus fresh install, 2026-09-11).
  useEffect(() => {
    let unlisten = null;
    let timer = null;
    (async () => {
      unlisten = await listen('api-keys-updated', () => {
        // A push sends one event per key; coalesce the burst into one refetch.
        clearTimeout(timer);
        timer = setTimeout(() => { fetchModels(); }, 300);
      });
    })();
    return () => { clearTimeout(timer); if (unlisten) unlisten(); };
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

  // Persist selected model across sessions — in localStorage for the UI, and in the
  // backend's .env (ZYNK_MODEL_BACKEND) so the native Android voice path, which
  // cannot read localStorage, answers with the same model you picked here.
  useEffect(() => {
    // 'local' is the placeholder the state starts with, not a choice. On a phone there are
    // no local models, and persisting it here on the first render made fetchModels treat
    // it as a stored preference and leave the typed path on a backend that cannot answer
    // (tester, Pixel 7a, 2026-09-22). Leave nothing stored, so fetchModels picks the first
    // provider with a key the moment there is one.
    if (modelType === 'local' && window.AndroidPaths) return;
    localStorage.setItem('zynkbot_preferred_model', modelType);
    invoke('set_preferred_backend', { backend: modelType }).catch((e) =>
      console.warn('[Backend] could not persist preferred backend:', e)
    );
  }, [modelType]);

  // The hands-free (native) path shares this thread. Tell the backend which thread is
  // on screen so "Hey Zynk" questions continue it with its history, and pull finished
  // hands-free exchanges into the chat when they complete or when the app comes back
  // to the front (a push to a paused WebView is lost, so resume drains again).
  // One-time enrichment of memories stored before the extractor returned event
  // dates, tags, tone and entities (2026-09-08). Runs in the background in Rust,
  // once per app start, and does nothing once every memory is marked. No button:
  // there is nothing for a user to decide here.
  const enrichKickedRef = useRef(false);
  useEffect(() => {
    if (!userId || !modelType || enrichKickedRef.current) return;
    enrichKickedRef.current = true;
    const t = setTimeout(() => {
      invoke('enrich_memory_backlog', { userId, backend: modelType })
        .then((r) => { if (r?.queued) console.log('[Enrich] annotating', r.queued, 'older memories'); })
        .catch((e) => console.warn('[Enrich] not started:', e));
    }, 15000); // let startup, sync and the first exchange settle first
    return () => clearTimeout(t);
  }, [userId, modelType]);

  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    sessionIdRef.current = sessionId;
    invoke('set_current_session', { sessionId }).catch((e) =>
      console.warn('[Session] could not record current session:', e)
    );
  }, [sessionId]);

  useEffect(() => {
    const drain = () => {
      let turns = [];
      try { turns = JSON.parse(window.WakeWordBridge?.drainNativeTurns?.() || '[]'); } catch (_) { return; }
      const additions = nativeTurnsToMessages(turns, sessionIdRef.current);
      if (additions.length === 0) return;
      setMessages((prev) => {
        const seen = new Set(prev.map((m) => m.id));
        return [...prev, ...additions.filter((m) => !seen.has(m.id))];
      });
    };
    window.__nativeTurns = drain;
    const onVisible = () => { if (document.visibilityState === 'visible') drain(); };
    document.addEventListener('visibilitychange', onVisible);
    drain();
    return () => {
      window.__nativeTurns = null;
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, []);

  // Listen for contradiction detection events from backend
  useEffect(() => {
    let unlisten;

    // Cards that keep (2026-10-26): a card raised while the window was away is a row in
    // memory_cards; show the oldest unanswered one on open and on every return.
    const checkPendingCards = async () => {
      if (!userId) return;
      try {
        const cards = await invoke('list_pending_memory_cards', { userId });
        if (cards && cards.length > 0) {
          setCurrentConflict(prev => prev || cards[0]);
          setShowConflictResolution(true);
          return;
        }
      } catch (e) { console.warn('[App] pending cards:', e); }
      // No card waiting: the check-in question gets the slot (once at open, then on quiet returns).
      const first = !checkinAskedThisOpenRef.current;
      checkinAskedThisOpenRef.current = true;
      maybeAskCheckin(first);
    };
    checkPendingCards();
    window.__zynkResumed = checkPendingCards;
    const onVisible = () => { if (document.visibilityState === 'visible') checkPendingCards(); };
    document.addEventListener('visibilitychange', onVisible);
    // On the desktop, switching to another window and back does not change the page's
    // visibility (WebKitGTK reports hidden only when minimised), so the return is the
    // window regaining focus (2026-10-29: a card raised while Matt was in the window
    // showed only after a page refresh).
    window.addEventListener('focus', checkPendingCards);

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
      document.removeEventListener('visibilitychange', onVisible);
      window.removeEventListener('focus', checkPendingCards);
      if (window.__zynkResumed === checkPendingCards) window.__zynkResumed = null;
      if (unlisten) {
        unlisten();
      }
    };
    // maybeAskCheckin reads refs and current state; re-running this effect per render would re-register the listener.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  // (Wake lock, wake word lifecycle, dictation listener, pause/visibility effects
  //  all live in useVoiceSession — see src/hooks/useVoiceSession.js)

  // Auto-scroll: follow new content unless the user has scrolled up, and never while
  // a finger is on the list.
  //
  // Why intent comes from input events, not position (Android, 2026-09-04, third
  // attempt): during streaming a token lands every few tens of milliseconds. The old
  // position-only rule needed the user to get more than 80px from the bottom between
  // two tokens; a slow drag never managed it, each token snapped the list back down,
  // and the scroll event from OUR snap then read "at bottom" and cleared the flag.
  // Result: dragged down to the bottom over and over. Now a touch suspends
  // auto-scroll outright, any upward drag or wheel-up marks "scrolled up" at once, and
  // scroll events caused by our own snap are ignored.
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;
    if (userScrolledUpRef.current || touchActiveRef.current) return;
    const target = container.scrollHeight - container.clientHeight;
    if (container.scrollTop >= target) return;   // already there: no event will follow
    programmaticScrollRef.current = true;
    container.scrollTop = container.scrollHeight;
  }, [messages, isLoading]);

  // Listeners follow the chat container element itself, attached through a callback ref
  // (see attachChatContainer on the element). The previous mount-once effect attached to
  // whatever element existed at App's first render and returned silently when there was
  // none; on the phone the chat renders after initialisation, so no listener ever ran,
  // userScrolledUpRef never became true, and every streamed token snapped the list to the
  // bottom (build22, 2026-09-04). The console lines are breadcrumbs: on Android they land
  // in logcat as CONSOLE lines, which is the only trace this bug leaves.
  const chatListenersCleanupRef = useRef(null);
  const trace = (m) => { try { window.WakeWordBridge?.log?.(m); } catch (_) {} console.log(m); };
  const attachChatContainer = useCallback((node) => {
    if (chatListenersCleanupRef.current) {
      chatListenersCleanupRef.current();
      chatListenersCleanupRef.current = null;
    }
    chatContainerRef.current = node;
    if (!node) return;
    const container = node;
    const distanceFromBottom = () =>
      container.scrollHeight - container.scrollTop - container.clientHeight;
    const setScrolledUp = (up) => {
      if (userScrolledUpRef.current !== up) trace(`[scroll] reading older = ${up}`);
      userScrolledUpRef.current = up;
      setShowScrollButton(up);
    };
    let touchStartY = 0;
    const onScroll = () => {
      if (programmaticScrollRef.current) { programmaticScrollRef.current = false; return; }
      const up = distanceFromBottom() > 80;
      // While a finger is down, a scroll event may only ever mark "reading older";
      // returning to following happens on touchend (at the very bottom) or on the
      // momentum scroll after it, never mid-drag.
      if (touchActiveRef.current) { if (up) setScrolledUp(true); return; }
      setScrolledUp(up);
    };
    const onTouchStart = (e) => {
      touchActiveRef.current = true;
      touchStartY = e.touches?.[0]?.clientY ?? 0;
    };
    const onTouchMove = (e) => {
      const y = e.touches?.[0]?.clientY ?? touchStartY;
      if (y - touchStartY > 10) setScrolledUp(true);   // finger moving down = reading older
    };
    const onTouchEnd = () => {
      touchActiveRef.current = false;
      if (distanceFromBottom() <= 8) setScrolledUp(false);
    };
    const onWheel = (e) => { if (e.deltaY < 0) setScrolledUp(true); };
    const opts = { passive: true };
    container.addEventListener('scroll', onScroll, opts);
    container.addEventListener('touchstart', onTouchStart, opts);
    container.addEventListener('touchmove', onTouchMove, opts);
    container.addEventListener('touchend', onTouchEnd, opts);
    container.addEventListener('touchcancel', onTouchEnd, opts);
    container.addEventListener('wheel', onWheel, opts);
    trace('[scroll] listeners attached to the chat container');
    chatListenersCleanupRef.current = () => {
      container.removeEventListener('scroll', onScroll, opts);
      container.removeEventListener('touchstart', onTouchStart, opts);
      container.removeEventListener('touchmove', onTouchMove, opts);
      container.removeEventListener('touchend', onTouchEnd, opts);
      container.removeEventListener('touchcancel', onTouchEnd, opts);
      container.removeEventListener('wheel', onWheel, opts);
      trace('[scroll] listeners detached');
    };
  }, []);

  // Voice input now handled by VoiceButton component (using whisper.cpp)

  // "New" starts a new thread. Until 0.9.6-beta2 it only emptied the screen and
  // kept the same session id, so every later message — and every hands-free turn —
  // was appended to the thread the user thought they had left; History then showed
  // one thread holding several conversations and no "previous" one to go back to
  // (tester report, 2026-09-14). The old thread stays in History untouched.
  const handleNewConversation = async () => {
    if (messages.length > 0 && !(await confirmDialog('Start a new conversation? The current one stays in Conversation History.'))) return;
    const fresh = uuidv4();
    setSessionId(fresh);
    try { sessionStorage.setItem('zynkbot_session_id', fresh); } catch (_) {}
    setMessages([]);
    myAnchorRef.current = Date.now(); // a deliberate new conversation: stay until another device writes after this
  };

  const handleCopyAll = () => {
    const text = messages
      .map(msg => {
        const label = msg.role === 'user' ? 'You' : 'Zynkbot';
        return `${label}: ${msg.content}`;
      })
      .join('\n\n');
    navigator.clipboard.writeText(text).then(() => {
      setCopyAllDone(true);
      setTimeout(() => setCopyAllDone(false), 1500);
    });
  };

  const handleResumeSession = ({ sessionId: resumedId, messages: pastMessages }) => {
    setSessionId(resumedId);
    try { sessionStorage.setItem('zynkbot_session_id', resumedId); } catch (_) {}
    // Rows straight from the database carry their saved sources as text; the page wants the list.
    setMessages((pastMessages || []).map((m) => (m.web_search_results || !m.sources_json) ? m : { ...m, web_search_results: parseSources(m.sources_json) }));
    // Scroll to bottom after loading
    setTimeout(() => {
      const el = document.getElementById('messages-end');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  // Open the conversation a memory came from, at the exchange that produced it
  // (Memory Manager → "Open conversation", 2026-10-16).
  const handleOpenMemoryConversation = async (memory) => {
    try {
      const src = await invoke('find_memory_source', { memoryId: memory.id });
      if (!src.session_id) { notify('This memory was not made in a conversation'); return; }
      if (!src.session_exists) { notify('That conversation has been deleted'); return; }
      const pastMessages = await invoke('get_conversation_messages', { sessionId: src.session_id });
      if (!pastMessages || pastMessages.length === 0) { notify('That conversation is empty'); return; }
      memoryManagerRef.current?.close();
      handleResumeSession({ sessionId: src.session_id, messages: pastMessages || [] });
      if (src.message_id != null) {
        setTimeout(() => {
          const el = document.getElementById(`message-${src.message_id}`);
          if (!el) return;
          el.scrollIntoView({ behavior: 'smooth', block: 'center' });
          el.classList.add('message-highlight');
          setTimeout(() => el.classList.add('message-highlight-fade'), 2500);
          setTimeout(() => el.classList.remove('message-highlight', 'message-highlight-fade'), 4500);
        }, 300);
      }
    } catch (e) {
      notify(`Could not open the conversation: ${e}`, 6000);
    }
  };

  // A short non-blocking notice at the bottom of the window (2026-09-20). alert()
  // blocks the page and needs a tap; this fades on its own.
  const [notice, setNotice] = useState('');
  const noticeTimerRef = useRef(null);
  const notify = (text, ms = 4000) => {
    setNotice(text);
    clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = setTimeout(() => setNotice(''), ms);
  };

  // ONE SHARED CONVERSATION ACROSS DEVICES (2026-10-28, Matt): whichever device wrote to a
  // thread last, that thread is the current one everywhere. A device lands in it when it is
  // brought back to the front (or has been idle for a while and a sync brings news); it
  // never moves while the user is typing, waiting for a reply or has a file attached.
  // "New" is respected: a device that started a new conversation stays there until another
  // device writes after that. Old threads stay in History to reopen at any time.
  const liveRef = useRef({});
  liveRef.current = { sessionId, isLoading, hasDraft: input.trim().length > 0 || attachedFiles.length > 0, editing: editingMessageId !== null, count: messages.length, userId };
  const myAnchorRef = useRef(0);      // when this device last wrote, or asked for a new conversation
  const lastTouchRef = useRef(Date.now());
  const adoptingRef = useRef(false);
  const adoptLatestThread = async (why, allowSwitch = true) => {
    const st = liveRef.current;
    if (!st.userId || adoptingRef.current || st.isLoading || st.hasDraft || st.editing) return;
    adoptingRef.current = true;
    try {
      const latest = await invoke('get_latest_thread', { userId: st.userId });
      if (!latest) return;
      const theirs = new Date(latest.last_active).getTime();
      const rows = await invoke('get_conversation_messages', { sessionId: latest.session_id });
      if (!rows || rows.length === 0) return;
      const asMessage = (m) => ({ id: m.id, role: m.role, content: m.content, timestamp: m.created_at, recalled_memories: [], web_search_results: parseSources(m.sources_json) });
      if (latest.session_id === liveRef.current.sessionId) {
        // Same thread, written on another device meanwhile: append what this device lacks.
        // The thread is replaced by the stored copy, never appended to: the page can hold
        // fewer rows than the database (a search turn, a pre-search answer), so counting
        // and appending could double a message.
        const have = liveRef.current.count;
        if (rows.length > have && theirs > myAnchorRef.current + 2000) {
          setMessages(rows.map(asMessage));
          myAnchorRef.current = theirs;
          console.log('[Session] this thread was continued on another device; refreshed (' + why + ')');
        }
        return;
      }
      // A different thread: only when another device wrote after this one last did, and
      // (for a sync event) only once this device has been quiet for a while.
      if (!allowSwitch || theirs <= myAnchorRef.current + 2000) return;
      handleResumeSession({ sessionId: latest.session_id, messages: rows.map(asMessage) });
      myAnchorRef.current = theirs;
      notify('Continuing the conversation from your other device.', 3500);
      console.log('[Session] switched to the thread another device was using (' + why + ')');
    } catch (e) {
      console.warn('[Session] could not check for the shared thread:', e);
    } finally {
      adoptingRef.current = false;
    }
  };
  // When this device's own reply finishes, its write lands a moment later: that write is ours.
  const wasLoadingRef = useRef(false);
  useEffect(() => {
    if (wasLoadingRef.current && !isLoading) myAnchorRef.current = Math.max(myAnchorRef.current, Date.now() + 5000);
    wasLoadingRef.current = isLoading;
  }, [isLoading]);
  useEffect(() => {
    // News that arrived while this device was busy is checked again once it has been
    // quiet for 45 s; every touch pushes that check back (2026-10-29: the first check
    // happened only at the moment the rows arrived, so a phone touched just before
    // never switched at all).
    let quietTimer = null;
    const checkWhenQuiet = () => {
      if (quietTimer) clearTimeout(quietTimer);
      quietTimer = setTimeout(() => { quietTimer = null; adoptLatestThread('quiet after sync', true); }, 45000);
    };
    const touch = () => { lastTouchRef.current = Date.now(); if (quietTimer) checkWhenQuiet(); };
    const onVisible = () => { if (document.visibilityState === 'visible') adoptLatestThread('returned to this device'); };
    const onFocus = () => adoptLatestThread('returned to this device');
    window.addEventListener('keydown', touch); window.addEventListener('pointerdown', touch);
    document.addEventListener('visibilitychange', onVisible); window.addEventListener('focus', onFocus);
    let unlisten = null;
    (async () => {
      // A sync round brought news: the open thread refreshes at once (nothing jumps); moving
      // to another thread waits until this device has been quiet for 45 s.
      unlisten = await listen('zynksync-threads-changed', () => {
        const quiet = Date.now() - lastTouchRef.current > 45000;
        adoptLatestThread('sync', quiet);
        if (!quiet) checkWhenQuiet();
      });
    })();
    const startup = setTimeout(() => adoptLatestThread('app started'), 2500);
    return () => {
      window.removeEventListener('keydown', touch); window.removeEventListener('pointerdown', touch);
      document.removeEventListener('visibilitychange', onVisible); window.removeEventListener('focus', onFocus);
      clearTimeout(startup); if (quietTimer) clearTimeout(quietTimer); if (unlisten) unlisten();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  // 🔊 on a reply: read it aloud, or stop if it is the one being read (2026-09-20).
  const [speakingMessageId, setSpeakingMessageId] = useState(null);
  const handleSpeakMessage = async (msg) => {
    if (voice.isTtsSpeaking && speakingMessageId === msg.id) { voice.stopTts(); setSpeakingMessageId(null); return; }
    setSpeakingMessageId(msg.id);
    const ok = await voice.speakResponse(msg.content);
    if (ok === false) { setSpeakingMessageId(null); notify('Read aloud needs an OpenAI key (System Controls → Current Model)', 6000); }
  };

  // Conflict Resolution Handlers. The dialog closes the moment the user confirms;
  // the resolution runs behind it (it used to hold the dialog open for the whole
  // thing, including a 10 s timeout per unreachable sync peer — 2026-09-20).
  const handleConflictResolve = async (resolution) => {
    console.log('Conflict resolved:', resolution);
    const conflict = currentConflict;
    setShowConflictResolution(false);
    setCurrentConflict(null);
    if (!conflict) return;

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

      let result;
      if (conflict.stored_memory_id) {
        // The on-device model's card: both memories are already stored (store first, ask
        // after, 2026-10-23). "Keep both" on the "Is this an update?" form links them as related.
        const isUpdateQuestion = typeof conflict.reason === 'string' && conflict.reason.startsWith('Possibly an update');
        const storedDecision = (decision === 'keep_both' && isUpdateQuestion) ? 'keep_both_related' : decision;
        console.log(`Calling resolve_stored_conflict with decision: ${storedDecision}`);
        result = await invoke('resolve_stored_conflict', {
          newMemoryId: conflict.stored_memory_id,
          oldMemoryId: conflict.memoryA.id,
          decision: storedDecision,
          explanation: resolution.explanation || null,
          userId: conflict.user_id || userId,
          sessionId: conflict.session_id || sessionId,
        });
      } else {
        console.log(`Calling resolve_memory_conflict_v2 with decision: ${decision}`);
        result = await invoke('resolve_memory_conflict_v2', {
          pendingMemoryJson: JSON.stringify(conflict.pending_memory),
          conflictingMemoryId: conflict.memoryA.id,
          decision: decision,
          explanation: resolution.explanation || null,
          relationshipsJson: JSON.stringify(conflict.relationships || []),
          userId: conflict.user_id || userId,
          sessionId: conflict.session_id || sessionId,
        });
      }

      console.log('✅ Conflict resolved successfully:', result);
      if (conflict.card_id) {
        try { await invoke('answer_memory_card', { cardId: conflict.card_id, decision }); } catch (e) { console.warn('[App] card answer:', e); }
        if (window.__zynkResumed) window.__zynkResumed();
      }

      // Show success message
      if (decision === 'keep_old') {
        notify('Kept the existing memory; the new statement was discarded');
      } else if (decision === 'keep_new') {
        notify('Kept the new memory; the old one is kept as history');
      } else if (decision === 'not_a_contradiction') {
        notify('Both memories kept; no longer marked as contradicting');
      } else if (decision === 'keep_both') {
        notify('Both memories kept, still marked as contradicting. You can resolve it later in the Memory Manager.', 6000);
      } else if (decision === 'both_with_explanation') {
        notify('Both memories kept; your explanation was stored as a memory');
      }

      // Refresh the memory manager to show updated memories
      if (memoryManagerRef.current && memoryManagerRef.current.refreshMemories) {
        memoryManagerRef.current.refreshMemories();
      }
    } catch (error) {
      console.error('Error resolving conflict:', error);
      notify(`Could not resolve the conflict: ${error}`, 8000);
    }
  };

  const IMAGE_EXTENSIONS = ['jpg','jpeg','png','gif','webp','bmp'];
  const MIME_TYPES = { jpg:'image/jpeg', jpeg:'image/jpeg', png:'image/png', gif:'image/gif', webp:'image/webp', bmp:'image/bmp' };

  // Android: the dialog plugin's generic picker hands back one item from most
  // galleries. Use the phone's photo picker (checkbox multi-select) for photos and
  // the documents picker in multi mode for files. Both resolve an array of URIs.
  const pickOnAndroid = (kind) => new Promise((resolve, reject) => {
    window.__pickFilesResolve = resolve;
    window.__pickFilesReject = reject;
    if (kind === 'images') window.AndroidPaths.pickImages();
    else window.AndroidPaths.pickDocuments();
  });

  // Read files (content:// URIs on Android, paths on desktop) into composer attachments.
  // A file's first bytes say what it is regardless of its name or what Android
  // could guess — the same way any image viewer identifies a file. Only called
  // when both the extension and Android's own MIME lookup came back empty.
  const sniffImageMimeFromBase64 = (b64) => {
    try {
      const head = atob(b64.slice(0, 64)); // ~48 bytes: plenty for every signature below
      const b = (i) => head.charCodeAt(i);
      if (b(0) === 0xFF && b(1) === 0xD8 && b(2) === 0xFF) return 'image/jpeg';
      if (b(0) === 0x89 && b(1) === 0x50 && b(2) === 0x4E && b(3) === 0x47) return 'image/png';
      if (b(0) === 0x47 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x38) return 'image/gif';
      if (b(0) === 0x42 && b(1) === 0x4D) return 'image/bmp';
      if (b(0) === 0x52 && b(1) === 0x49 && b(2) === 0x46 && b(3) === 0x46 &&
          b(8) === 0x57 && b(9) === 0x45 && b(10) === 0x42 && b(11) === 0x50) return 'image/webp';
      return null;
    } catch (_) { return null; }
  };

  const attachFromPaths = async (paths) => {
    const newFiles = [];
    for (const path of paths) {
      try {
        // On Android, openFileDialog returns a content:// URI that Rust's fs::read can't open.
        // Use the AndroidPaths bridge which reads via ContentResolver instead.
        const name = window.AndroidPaths ? window.AndroidPaths.getFileName(path) : path.split('/').pop();
        const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : '';
        // Android: trust the provider's type over the name — a photo shared into the
        // Zynkbot location can carry a bare id for a name (2026-09-20).
        let providerType = '';
        try { providerType = window.AndroidPaths?.getMimeType?.(path) || ''; } catch (_) {}
        // Android looks at the file's own first bytes (2026-10-25): a gallery photo with
        // no usable name or type was read as text and sent as megabytes of garbage.
        let kind = '';
        if (window.AndroidPaths?.probeFile) { try { kind = JSON.parse(window.AndroidPaths.probeFile(path)).kind || ''; } catch (_) {} }
        console.log('[Attach] picked', path, 'name:', name, 'type:', providerType, 'kind:', kind);
        if (IMAGE_EXTENSIONS.includes(ext) || providerType.startsWith('image/') || kind === 'image') {
          // Android decodes and scales the picture itself (any format the phone opens,
          // HEIC included) to a JPEG no wider than 1568 px; the raw bytes are the fallback.
          let base64 = '';
          let mimeType = '';
          if (window.AndroidPaths?.readImageJpegBase64) {
            base64 = window.AndroidPaths.readImageJpegBase64(path, 1568);
            if (base64) mimeType = 'image/jpeg';
          }
          if (!base64) {
            base64 = window.AndroidPaths ? window.AndroidPaths.readFileBase64(path) : await invoke('read_file_base64', { path });
            mimeType = MIME_TYPES[ext] || (providerType.startsWith('image/') ? providerType : 'image/jpeg');
          }
          if (!base64) throw new Error(`Could not read image ${name}`);
          newFiles.push({ name, base64, mimeType, size: base64.length, isImage: true });
        } else if (['pdf', 'docx', 'doc'].includes(kind) || ['pdf', 'docx', 'doc'].includes(ext)) {
          // A PDF or a Word document goes through the same text extractor the knowledge base uses.
          const filePath = window.AndroidPaths?.copyToCache ? window.AndroidPaths.copyToCache(path) : path;
          if (!filePath) throw new Error(`Could not read ${name}`);
          const content = await invoke('read_attachment_text', { path: filePath });
          newFiles.push({ name, content, size: content.length, isImage: false });
        } else if (kind === 'binary') {
          alert(`${name} can't be attached: it is not a picture, a PDF, a Word document or a text file.`);
        } else {
          // Neither the name nor Android's own answer said what this is — Android
          // reports nothing for a file with no extension, and has no other way to
          // check. Read it and look at its own first bytes before giving up and
          // treating it as text; a renamed screenshot was sent as raw EXIF/JPEG
          // text instead of a picture until this (Matt, 2026-09-20).
          let sniffed = null;
          if (window.AndroidPaths) {
            try {
              const probe = window.AndroidPaths.readFileBase64(path);
              const mime = probe ? sniffImageMimeFromBase64(probe) : null;
              if (mime) sniffed = { base64: probe, mimeType: mime };
            } catch (_) { /* not readable as bytes either; fall through to text */ }
          }
          if (sniffed) {
            newFiles.push({ name, base64: sniffed.base64, mimeType: sniffed.mimeType, size: sniffed.base64.length, isImage: true });
          } else {
            const content = window.AndroidPaths
              ? window.AndroidPaths.readFileText(path)
              : await invoke('read_text_file', { path });
            newFiles.push({ name, content, size: content.length, isImage: false });
          }
        }
      } catch (e) {
        alert(`Could not read file: ${e}`);
      }
    }
    if (newFiles.length > 0) setAttachedFiles(prev => [...prev, ...newFiles]);
  };

  const handleAttachFile = async (kind) => {
    let result;
    if (window.AndroidPaths?.pickDocuments) {
      try {
        result = await pickOnAndroid(kind === 'images' ? 'images' : 'documents');
      } catch (e) {
        if (e !== 'cancelled') alert(`Could not open the picker: ${e}`);
        return;
      }
    } else {
      result = await openFileDialog({
        multiple: true,
        filters: [
          { name: 'Files', extensions: [...IMAGE_EXTENSIONS, 'txt','md','rs','js','jsx','ts','tsx','py','json','toml','yaml','yml','sh','css','html','c','cpp','h','go','java','rb','php','swift','kt'] }
        ]
      });
    }
    if (!result) return;
    await attachFromPaths(Array.isArray(result) ? result : [result]);
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
    myAnchorRef.current = Date.now() + 1000; // this device is writing now: no other device's older news moves it

    // Capture and reset wake flags before any early returns
    const triggeredByWake = voice.wakeTriggeredRef.current;
    voice.wakeTriggeredRef.current = false;
    const handsFree = voice.handsFreeRef.current;
    voice.handsFreeRef.current = false;
    const speak = shouldSpeakReply({ handsFree, speakInApp: voice.ttsEnabled });

    // stop_tts works without VoiceCommandBridge
    const cmd = parseVoiceCommand(message);
    if (cmd?.type === 'stop_tts') { voice.stopTts(); setInput(''); return; }

    // Clock commands run here, never through the model: ZynkTimer on the phone, the
    // in-app timer on the desktop (2026-10-25).
    if (cmd) {
      voice.executeVoiceCommand(cmd, message); setInput(''); return;
    }

    if (isSendingRef.current) return; // prevent re-entrant calls before React re-renders
    isSendingRef.current = true;
    const skipUserMessageAdd = options.skipUserMessageAdd === true;
    // Regenerate and edit-and-resend pass the history explicitly. They trim the old
    // reply out of the list with setMessages() and then call this function, but
    // `messages` here is still the previous render's list, old reply included — so
    // the model was handed its own answer as context and replied "as I said above,
    // here it is again" (2026-09-06).
    const historyBase = Array.isArray(options.history) ? options.history : messages;

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
        // The cap is the model's window, not a character count (Matt, 2026-10-25):
        // roughly four characters to a token, a quarter of the window kept for the
        // conversation and the reply. Over it, the user decides; nothing is cut silently.
        const windowTokens = modelWindowTokens(modelType, availableModels);
        const fileTokens = Math.ceil(textBlock.length / 4);
        const roomTokens = windowTokens - Math.max(2000, Math.floor(windowTokens / 4));
        if (fileTokens > roomTokens) {
          const modelName = (availableModels.find(m => m.id === modelType)?.name || 'this model').replace(/\.gguf$/i, '');
          const k = (n) => `${Math.round(n / 1000)},000`;
          const ok = await confirmDialog(
            `These files are about ${k(fileTokens)} tokens of text, and ${modelName} can read about ${k(roomTokens)} in one message with room left to answer.\n\n` +
            `The reply may fail or ignore the end of the file. For something this long, add it to the Knowledge Base instead: it is indexed there and searched when you ask.\n\n` +
            `Send anyway?`
          );
          if (!ok) { isSendingRef.current = false; return; }
        }
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
      timestamp: new Date().toISOString(),
      // The pictures stay with the message while the thread is open (2026-10-28), so the
      // next turns can send them again and the model can compare or refer back.
      images: imageData,
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
      // The newest three pictures in the thread travel with their turns (2026-10-28).
      const recent = historyBase.slice(-50);
      let picturesLeft = 3;
      const withPictures = new Set();
      for (let i = recent.length - 1; i >= 0 && picturesLeft > 0; i--) {
        const n = Array.isArray(recent[i].images) ? recent[i].images.length : 0;
        if (n > 0 && n <= picturesLeft) { withPictures.add(i); picturesLeft -= n; }
      }
      const conversationHistory = recent.map((msg, i) => ({
        role: msg.role,
        content: msg.content,
        ...(withPictures.has(i) ? { images: msg.images } : {}),
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
        imageData,
        mode: chatMode === 'normal' ? undefined : chatMode,
      });

      unlisten(); // Stop listening for tokens — stream is complete
      // The model's one-line record of the picture goes under the message that carried it,
      // the same way the thread stores it (2026-10-28).
      if (response?.picture_note) {
        setMessages(prev => prev.map(msg =>
          msg.id === userMessage.id ? { ...msg, content: `${msg.content}\n[Picture: ${response.picture_note}]` } : msg
        ));
      }
      // A "Hey Zynk" question the page answered counts as a real wake, the same as
      // one the phone answers itself with the screen off. Without this the "Send my
      // wake-word clips" count only grew from screen-off use (Matt, 2026-09-20).
      if (triggeredByWake && response?.reply_text && !response.reply_text.trim().startsWith('NO_QUERY')) {
        try { window.WakeWordBridge?.markAnswered?.(); } catch (_) {}
      }

      console.log('=== RUST BACKEND RESPONSE ===');
      console.log('Full response:', response);

      // Check if LLM detected need for web search
      if (response.web_search_needed) {
        console.log('[WebSearch] LLM requested web search');
        console.log('[WebSearch] Query:', response.web_search_query);
        console.log('[WebSearch] Original query:', response.original_query);

        // Hands-free always searches, never asks first (Matt, 2026-09-20, GitHub #26):
        // there is no way to answer "want me to search?" by voice, so it always
        // ran the search from here on; the `webSearchAutoExecute` toggle used to
        // gate this and no longer does — see the Rust side of the same change.
        if (triggeredByWake) {
          await handleExecuteWebSearch(streamId, response.web_search_query, response.original_query, speak, response.reply_text || '');
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
        if (speak) voice.speakResponse(assistantMessage.content);
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
        // A page the user pointed at, read for this reply (2026-10-27): shown like search sources.
        web_search_results: response.web_search_results || undefined,
        metadata: {
          schema: response.schema,
          model_backend: response.model_backend,
          containment_mode: response.containment_mode,
          kb_note: response.kb_note,
          recalled_memories: response.recalled_memories || []
        }
      };

      console.log('Assistant message object:', assistantMessage);
      console.log('Recalled memories:', assistantMessage.recalled_memories);

      setMessages(prev => prev.map(msg => msg.id === streamId ? assistantMessage : msg));
      if (speak) voice.speakResponse(assistantMessage.content);

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

  // Keep voice session ref current on every render so wake callbacks never hold a stale closure
  voice.handleSendMessageRef.current = handleSendMessage;

  // True whenever there is something to stop. isLoading covers generation; TTS
  // playback starts only AFTER isLoading goes false, which is why a Stop tied to
  // isLoading alone vanished exactly when Zynkbot began speaking.
  const canStop = isLoading || voice.isTtsSpeaking || voice.isNativeSpeaking;

  // Stop the currently streaming generation and/or spoken playback.
  // Optimistically flip UI out of loading state immediately — the backend's
  // post-stream work (memory extraction, sync) continues in the background but
  // the user shouldn't have to wait for it. The re-entrant send guard is also
  // cleared so a fresh Send can be sent right away.
  const handleStopGeneration = async () => {
    voice.stopTts();
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
    // Regenerate response — skip re-adding the user message since it's already in the
    // list, and send the history up to (not including) the edited message: the
    // message itself travels separately, and the stale reply must not.
    await handleSendMessage(newContent, { skipUserMessageAdd: true, history: truncated.slice(0, idx) });
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
    const history = messages.slice(0, lastUserIdx);
    setMessages(history);
    await handleSendMessage(prompt, { history });
  };

  // Execute web search when user confirms
  // `speak` is decided by the request that triggered the search; the confirm
  // button in ChatMessage omits it, so a tapped search follows the in-app setting.
  // firstAnswer: what the model said before deciding to search. When there is one it
  // stays as its own message and the searched answer goes in a new message beneath
  // it, instead of replacing it (Matt, 2026-09-20: "the original message was useful,
  // and it disappeared").
  const handleExecuteWebSearch = async (messageId, searchQuery, originalQuery, speak = voice.ttsEnabled, firstAnswer = '') => {
    console.log('[WebSearch] User confirmed web search');
    console.log('[WebSearch] Message ID:', messageId);
    console.log('[WebSearch] Search query:', searchQuery);
    console.log('[WebSearch] Original query:', originalQuery);

    const keepFirst = !!(firstAnswer && firstAnswer.trim());
    const targetId = keepFirst ? `${messageId}-search` : messageId;
    try {
      setIsLoading(true);

      // Show that the search is in progress: in a new message under the first answer,
      // or in place of the confirmation prompt when there was no answer to keep.
      setMessages(prev => {
        // The first answer lives in `messageId`; the search reply goes into a new
        // message right after it. Looking the first answer up by the new id (which does
        // not exist yet) put the search reply at the top of the conversation (2026-09-20).
        const kept = prev.map(msg =>
          msg.id === messageId ? { ...msg, web_search_needed: false, isStreaming: false, content: keepFirst ? firstAnswer : msg.content } : msg
        );
        if (!keepFirst) return kept.map(msg => msg.id === targetId ? { ...msg, content: `Searching the web for: "${searchQuery}"...` } : msg);
        if (kept.some(msg => msg.id === targetId)) return kept;
        const idx = kept.findIndex(msg => msg.id === messageId);
        const first = idx >= 0 ? kept[idx] : null;
        const searching = { id: targetId, role: 'assistant', content: `Searching the web for: "${searchQuery}"...`, timestamp: new Date().toISOString(), isStreaming: true, metadata: first?.metadata };
        const at = idx < 0 ? kept.length : idx + 1;
        return [...kept.slice(0, at), searching, ...kept.slice(at)];
      });

      // Execute the web search
      // The engine is chosen in Rust by the backend (2026-10-27): a cloud model searches
      // through its own provider and comes back with the answer written; a local model
      // gets results from Brave, a SearXNG server or DuckDuckGo for the synthesis below.
      // A cloud model's own search streams its answer as it is written: the first word
      // replaces the "Searching…" line, the rest follows it.
      let streamedWords = false;
      const unlistenSearch = await listen('stream-token', (event) => {
        const first = !streamedWords;
        streamedWords = true;
        setMessages(prev => prev.map(msg =>
          msg.id === targetId
            ? { ...msg, content: (first ? '' : msg.content) + event.payload }
            : msg
        ));
      });
      let searchResults;
      try {
        searchResults = await invoke('execute_web_search', {
          query: searchQuery,
          maxResults: 5,
          fetchTopN: 3,
          backend: modelType
        });
      } finally {
        unlistenSearch();
      }

      console.log('[WebSearch] Search results:', searchResults);
      // What goes into the thread with the answer (0028): titles, links and snippets only,
      // not the page bodies the synthesis prompt gets.
      const threadSources = {
        query: searchQuery,
        results: (searchResults.results || []).map((r) => ({ title: r.title, url: r.url, snippet: r.snippet })),
        num_results: (searchResults.results || []).length,
      };

      if (searchResults.answer) {
        setMessages(prev => prev.map(msg =>
          msg.id === targetId
            ? { ...msg, content: searchResults.answer, isStreaming: false, web_search_results: searchResults, web_search_query: searchQuery }
            : msg
        ));
        if (speak) voice.speakResponse(searchResults.answer);
        console.log('[WebSearch] Answer came from the provider\'s own search (' + searchResults.engine + ')');
        // Into the thread, as the synthesis call used to do on its own.
        try {
          await invoke('log_web_search_answer', { sessionId, userId, query: searchQuery, answer: searchResults.answer, backend: modelType, containmentMode, sources: threadSources });
        } catch (e) { console.warn('[WebSearch] could not log the answer:', e); }
        return;
      }

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
        msg.id === targetId ? { ...msg, content: '', isStreaming: true } : msg
      ));

      const unlisten = await listen('stream-token', (event) => {
        setMessages(prev => prev.map(msg =>
          msg.id === targetId
            ? { ...msg, content: msg.content + event.payload }
            : msg
        ));
      });

      const llmResponse = await invoke('send_message_with_memory', {
        message: llmPrompt,
        // History stores the clean query, not the 5 KB prompt with the search results
        // in it, which showed up in the thread as if the user had typed it (2026-09-20).
        userQuery: `Web search: "${searchQuery}"`,
        userId,
        sessionId,
        backend: modelType,
        containmentMode,
        conversationHistory: [], // Don't include history for this synthesis
        skipContainment: true, // Skip safety checks for internal web search synthesis
        skipMemoryStorage: true, // Don't store web search results as memories
        webSources: threadSources // saved with the answer, so the list survives a restart
      });

      unlisten();
      console.log('[WebSearch] LLM synthesized answer:', llmResponse.reply_text);

      // Finalize message with full metadata and search results for transparency
      setMessages(prev => prev.map(msg =>
        msg.id === targetId
          ? {
              ...msg,
              content: llmResponse.reply_text,
              isStreaming: false,
              web_search_results: searchResults,
              web_search_query: searchQuery
            }
          : msg
      ));

      if (speak) voice.speakResponse(llmResponse.reply_text);

      console.log('[WebSearch] Search complete and answer synthesized');

    } catch (error) {
      console.error('[WebSearch] Error executing search:', error);

      // Update message with error
      setMessages(prev => prev.map(msg =>
        msg.id === targetId
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
        onReportClick={() => openReport('')}
        // Hide the sidebar's floating ⚙️/✕ whenever ANY overlay is open, so only one
        // close button exists at that corner. Otherwise the two overlap and the first
        // tap toggles the sidebar instead of closing the modal (reported on API Keys,
        // then on the User Identity modal opened from the sidebar's info button).
        hideToggle={showConversationHistory || showEnsemble || voice.showVoiceModal
          || showUserIdentity || showAbout || showDemoGuide || showWhyZynkbot || showConflictResolution || showZynkCluster || showReport
          || controlsPage !== null}
        onOpen={() => {
          setShowAbout(false);
          setShowDemoGuide(false);
          setShowWhyZynkbot(false);
          setShowUserIdentity(false);
          setShowEnsemble(false);
          setShowConflictResolution(false);
          setShowZynkCluster(false);
          setShowOnboarding(false);
          setShowSnapInModal(false);
          setControlsPage(null);
          memoryManagerRef.current?.close();
        }}
        // Opening the sidebar hides an open memory card (above). When the sidebar closes,
        // any card still unanswered comes back (2026-10-29: Matt opened the sidebar to
        // switch models seconds after a card appeared and it was gone until a refresh).
        onClose={() => { if (window.__zynkResumed) window.__zynkResumed(); }}
      >
        {/* One button per section; each opens a page of its own (ControlsPage below).
            Getting Started leads the list until onboarding has been completed on this
            device or the user unticks "keep at top" on its page; then it sits last. */}
        {gettingStartedOnTop && gettingStartedButton}
        <ContainmentModeSelector
          currentMode={containmentMode}
          onModeChange={setContainmentMode}
        />
        <ControlsMenuButton
          icon="🤖"
          title="Current Model"
          subtitle={availableModels.find(m => m.id === modelType)?.name || 'Loading models…'}
          accent="#bd93f9"
          onClick={() => setControlsPage('model')}
        />
        <ControlsMenuButton
          icon="🎙"
          title="Voice"
          subtitle="Dictation, wake word, spoken replies"
          accent="#8be9fd"
          onClick={() => voice.setShowVoiceModal(true)}
        />
        <ControlsMenuButton
          icon="🔄"
          title="ZynkSync"
          subtitle="Your own devices: shared memory, notes between them"
          accent="#50fa7b"
          onClick={() => setControlsPage('sync')}
        />
        <ControlsMenuButton
          icon="🔗"
          title="ZynkLink"
          subtitle="Other people's Zynkbots: files and chat"
          accent="#f1fa8c"
          onClick={() => setControlsPage('link')}
        />
        <ControlsMenuButton
          icon="📚"
          title="Knowledge Base"
          subtitle="Your documents, searched when you ask"
          accent="#ffb86c"
          onClick={() => setControlsPage('kb')}
        />
        <ControlsMenuButton
          icon="🧩"
          title="Snap-ins"
          subtitle="Planned: therapist, legal, journalist"
          accent="#ff79c6"
          dim
          onClick={() => setControlsPage('snap')}
        />
        {!isMobile && <ControlsMenuButton
          icon="🔬"
          title="ZynkCluster"
          subtitle="Upcoming"
          dim
          onClick={() => setControlsPage('cluster')}
        />}
        <ControlsMenuButton
          icon="🧪"
          title="Experimental"
          subtitle="Features under test in this beta"
          onClick={() => setControlsPage('experimental')}
        />
        {!gettingStartedOnTop && gettingStartedButton}
        <ControlsMenuButton
          icon="📖"
          title="Help and documentation"
          subtitle="The Zynkbot guides, on the web"
          onClick={() => openUrl('https://github.com/MSkill1/zynkbot#readme')}
        />
        <div style={{ height: '60px' }} />

      </CollapsibleSidebar>

      {/* System Controls pages: one at a time, on top of the sidebar. */}
      {controlsPage === 'model' && (
        <ControlsPage icon="🤖" title="Current Model" accent="#bd93f9" introKey="model" intro={<>Zynkbot answers with the model marked here. A cloud model needs a key from that company, entered below, and your messages go to it; under its API terms they are not used to train on or to profile you, which is why Zynkbot uses keys and not a subscription. A model file on this device works offline and nothing leaves the device. <b>Try this:</b> switch models and ask the same question twice.</>}
          subtitle={(availableModels.find(m => m.id === modelType)?.name || 'No model selected').replace(/\.gguf$/i, '')}
          onClose={() => setControlsPage(null)}>
          {/* ui-redo (2026-10-24): one page for the choice and the keys. Rows instead of a
              drop-down; the keys and servers section that was its own window sits below. */}
          <div className="model-rows">
            {availableModels.length === 0 && <div style={{ color: '#9aa5c4', marginBottom: '10px' }}>Loading models…</div>}
            {availableModels.map(m => {
              const isLocal = (m.type || m.model_type) === 'local' || m.id.endsWith('.gguf');
              const file = isLocal ? localModelFiles.find(f => f.path === m.id || m.id.endsWith(f.filename)) : null;
              const selected = modelType === m.id;
              return (
                <div key={m.id} className={`model-row${selected ? ' selected' : ''}`} role="button" tabIndex={0}
                  onClick={() => setModelType(m.id)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setModelType(m.id); }}>
                  <span className="controls-menu-icon">{isLocal ? '📱' : '☁️'}</span>
                  <div className="controls-menu-text">
                    <b>{isLocal ? m.name.replace(/\.gguf$/i, '') : m.name}</b>
                    <small>{isLocal ? `On this device, offline, slower${file ? ' · ' + formatModelSize(file.size_bytes) : ''}` : 'Cloud model, uses its key below'}</small>
                  </div>
                  {file && (
                    <span className="model-row-actions" onClick={(e) => e.stopPropagation()}>
                      <button onClick={() => handleMoveModelFile(file)} disabled={modelMoveBusy === file.path} title="Move this model file somewhere else on this device">{modelMoveBusy === file.path ? 'Moving…' : 'Move to…'}</button>
                      <button className="danger" onClick={() => handleRemoveModelFile(file)} title="Remove this model file from this device">Remove</button>
                    </span>
                  )}
                  <span className="model-row-check">{selected ? '✓' : ''}</span>
                </div>
              );
            })}
          </div>
          {localModelFiles.length === 0 && (
            <div style={{ margin: '6px 0 10px' }}>
              <button className="model-add-btn" disabled={zbDownload.active} onClick={handleDownloadZbModel}
                title="Qwen3-4B-Instruct-2507 Q4_0 from Hugging Face, checked against its checksum">
                {zbDownload.active ? (zbDownload.verifying ? 'Checking the file…' : `Downloading… ${zbDownload.percent}%${zbDownload.total ? ' (' + formatModelSize(zbDownload.downloaded) + ' of ' + formatModelSize(zbDownload.total) + ')' : ''}`) : '⬇ Download the Zynkbot model (2.4 GB)'}
              </button>
              <div style={{ fontSize: '0.8rem', color: '#9aa5c4', marginTop: '4px' }}>
                Runs on this device with no key. Wi-Fi recommended; keep the app open. An interrupted download continues where it stopped.
              </div>
              {zbDownload.error && <div style={{ fontSize: '0.8rem', color: '#ff5555', marginTop: '4px' }}>{zbDownload.error}</div>}
            </div>
          )}
          {!isMobile ? (
            <button className="model-add-btn" title="Open the models folder to add .gguf files"
              onClick={async () => { try { await invoke('open_models_folder'); } catch (error) { alert('Failed to open models folder: ' + error); } }}>
              📁 Add a model file
            </button>
          ) : (
            <div style={{ fontSize: '0.85rem', color: '#9aa5c4', margin: '6px 0 14px' }}>To add a model file on this phone: the ZynkLink menu, Browse files, pick the file, then "→ Models".</div>
          )}
          <APIKeyModal embedded isOpen={false} onClose={() => {}} onKeysChanged={async () => { await fetchModels(); }} />
        </ControlsPage>
      )}

      {controlsPage === 'start' && (
        <ControlsPage icon="🚀" title="Getting Started" subtitle="Kitchen tip, Einstein demo, guided setup" onClose={() => setControlsPage(null)}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '14px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={gettingStartedOnTop}
              onChange={(e) => setGettingStartedOnTop(e.target.checked)}
              style={{ width: 'auto', margin: 0 }}
            />
            Keep Getting Started at the top of System Controls
          </label>
          <p style={{fontSize: '0.9rem', color: '#8be9fd', lineHeight: '1.6', marginBottom: '10px'}}>
            <strong>Try it in the kitchen.</strong> Tell Zynkbot what you cooked tonight and how it went.
            Tomorrow ask "what can I make tonight?" It remembers what you liked, what you refused, and how
            you are trying to eat. Fell off the diet? Tell it to forget those weeks and start again. The memory is yours to edit.
          </p>
          <p style={{fontSize: '0.9rem', color: '#8be9fd', lineHeight: '1.6', marginBottom: '15px'}}>
            Or load the Einstein demo below to see how memory and relationships work.
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
                const confirmed = await confirmDialog(
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
                // Awaited native dialog: a plain alert() does not block on Windows, so the
                // reload below fired with the dialog still up and it appeared twice (KI-047).
                await messageDialog(
                  `✅ Einstein Demo Loaded!\n\n` +
                  `📚 Memories: ${result.loaded_count}\n` +
                  `🔗 Relationships: ${result.relationships_created || 0}\n\n` +
                  `Check Memory Manager to explore the memory graph!`
                );
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
        </ControlsPage>
      )}

      {controlsPage === 'sync' && (
        <ControlsPage icon="🔄" title="ZynkSync" subtitle="Your own devices: shared memory, notes between them" accent="#50fa7b" introKey="sync" intro={<>ZynkSync joins your own phones and computers into one memory: what you tell one, the others know. Devices pair once with a code and sync over your network. Need a password, a note or a message on your other device? The Chat button here sends it from your phone to your computer; for a file, link the two in ZynkLink. <b>Try this:</b> generate a code on the device that has your memories and enter it on the other.</>} onClose={() => setControlsPage(null)}>
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
        </ControlsPage>
      )}

      {controlsPage === 'link' && (
        <ControlsPage icon="🔗" title="ZynkLink" subtitle="Other people's Zynkbots: files and chat" accent="#f1fa8c" introKey="link" intro={<>ZynkLink connects your Zynkbot with someone else's: files through the shared drive and a private chat, device to device. You exchange a code once. Either of you can unlink at any time; the other device is told and drops the link too. You can link your own devices as well, to move files easily between your phone and your computer or tablet. <b>Try this:</b> share a photo to the Zynkbot drive; the linked device sees it under Shared With Me.</>} onClose={() => setControlsPage(null)}>
        <ZynkLinkPanel
          apiBaseUrl={API_BASE_URL}
          onOpenUserIdentity={() => setShowUserIdentity(true)}
          userId={userId}
        />
        </ControlsPage>
      )}

      {controlsPage === 'kb' && (
        <ControlsPage icon="📚" title="Knowledge Base" subtitle="Your documents, searched when you ask" accent="#ffb86c" introKey="kb" intro={<>The knowledge base is where you put files that matter to you or to a project: notes, manuals, papers, contracts. Index them and Zynkbot knows what is in them. Turn KB on in chat and it reads the parts that match your question before it answers, so you can discuss your own documents in plain conversation. <b>Try this:</b> add a text file, index it, turn on KB and ask about it.</>} onClose={() => setControlsPage(null)}>
        <KnowledgeBasePanel userId={userId} />
        <KnowledgeBaseManager embedded isOpen={false} onClose={() => {}} userId={userId} />
        </ControlsPage>
      )}

      {controlsPage === 'snap' && (
        <ControlsPage icon="🧩" title="Snap-ins" subtitle="Planned: therapist, legal, journalist" accent="#ff79c6" onClose={() => setControlsPage(null)}>
          <SnapInPlans onTrySample={() => setShowSnapInModal(true)} />
        </ControlsPage>
      )}

      {controlsPage === 'experimental' && (
        <ControlsPage icon="🧪" title="Experimental" subtitle="Features under test in this beta" onClose={() => setControlsPage(null)}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '0.85rem', color: '#f8f8f2', marginBottom: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={checkinEnabled}
              onChange={async (e) => {
                const on = e.target.checked;
                setCheckinEnabled(on);
                try { await invoke('set_checkin_enabled', { enabled: on }); } catch (err) { notify(`Could not save: ${err}`, 5000); }
                if (!on) setPendingCheckin(null);
              }}
              style={{ width: 'auto', margin: '3px 0 0' }}
            />
            <span>
              <strong>Ask about old plans</strong>
              <div style={{ color: '#9aa5c4', marginTop: '2px', lineHeight: 1.45 }}>
                When something you said you would do (an errand, a purchase, a chore) is about to leave Zynkbot's memory after two weeks,
                Zynkbot asks once, at the start of a conversation and at most once a day, whether it happened. Done keeps it as history,
                Not yet keeps it two more weeks, Never mind deletes it. Plans that are not short-lived are never asked about.
              </div>
            </span>
          </label>
          <button
            type="button"
            disabled={!checkinEnabled}
            onClick={async () => {
              const asked = await maybeAskCheckin(true, true);
              setControlsPage(null);
              if (!asked) notify('Nothing to ask about yet: no short-lived plan of yours is waiting.', 5000);
            }}
            title="Ask the next question now, ignoring the once-a-day rule and the two-week timing (for testing)"
            style={{ marginTop: '8px', background: '#44475a', color: '#f8f8f2', border: 'none', borderRadius: '6px', padding: '8px 14px', cursor: checkinEnabled ? 'pointer' : 'default', opacity: checkinEnabled ? 1 : 0.5, fontSize: '0.85rem' }}
          >
            Try it now
          </button>
        </ControlsPage>
      )}
      {controlsPage === 'cluster' && (
        <ControlsPage icon="🔬" title="ZynkCluster" subtitle="Upcoming" onClose={() => setControlsPage(null)}>
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
        </ControlsPage>
      )}

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
              <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                <button
                  onClick={handleNewConversation}
                  title="Start a new conversation"
                  style={{
                    padding: '5px 12px',
                    whiteSpace: 'nowrap',
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
                {messages.length > 0 && (
                  <button
                    onClick={handleCopyAll}
                    title="Copy the whole conversation"
                    style={{
                      padding: '5px 12px',
                    whiteSpace: 'nowrap',
                      background: copyAllDone ? 'rgba(80,250,123,0.2)' : 'rgba(98,114,164,0.25)',
                      color: copyAllDone ? '#50fa7b' : '#8be9fd',
                      border: '1px solid ' + (copyAllDone ? '#50fa7b' : '#6272a4'),
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: '600',
                      transition: 'all 0.2s',
                    }}
                    onMouseOver={(e) => { if (!copyAllDone) { e.currentTarget.style.background = 'rgba(98,114,164,0.45)'; } }}
                    onMouseOut={(e) => { if (!copyAllDone) { e.currentTarget.style.background = 'rgba(98,114,164,0.25)'; } }}
                  >
                    {copyAllDone ? 'Copied!' : 'Copy'}
                  </button>
                )}
                <button
                  onClick={() => setShowConversationHistory(true)}
                  disabled={containmentMode === 'hipaa'}
                  title={containmentMode === 'hipaa' ? 'Conversation history is disabled in HIPAA mode' : 'Browse past conversations'}
                  style={{
                    padding: '5px 12px',
                    whiteSpace: 'nowrap',
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
                  onClick={() => setControlsPage('model')}
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
                  🔑 Set up a model
                </button>
              </div>
            )}
            <div className={isMobile ? 'mobile-conv-history-wrap' : undefined} style={{ position: 'relative' }}>
<div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '2px 4px 8px', borderBottom: '1px solid #44475a', marginBottom: '4px', minHeight: '28px' }}>
                {editingThreadTitle !== null ? (
                  <input
                    autoFocus
                    value={editingThreadTitle}
                    onChange={(e) => setEditingThreadTitle(e.target.value)}
                    onBlur={commitThreadTitle}
                    onKeyDown={(e) => { if (e.key === 'Enter') commitThreadTitle(); if (e.key === 'Escape') setEditingThreadTitle(null); }}
                    maxLength={120}
                    style={{ flex: 1, fontSize: '0.9rem', padding: '2px 6px', borderRadius: '4px', border: '1px solid #6272a4', background: '#282a36', color: '#f8f8f2' }}
                  />
                ) : (
                  <span
                    onClick={() => setEditingThreadTitle(threadTitle)}
                    title="Tap to rename this conversation"
                    style={{ flex: '0 1 auto', minWidth: 0, fontSize: '0.95rem', fontWeight: threadTitle ? 700 : 400, color: threadTitle ? '#8be9fd' : '#6272a4', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', cursor: 'text' }}
                  >
                    {threadTitle || 'New conversation'}
                  </span>
                )}
                {editingThreadTitle === null && (
                  <button
                    type="button"
                    onClick={() => setEditingThreadTitle(threadTitle)}
                    title="Rename this conversation"
                    style={{ background: 'transparent', border: 'none', color: '#9aa5c4', cursor: 'pointer', fontSize: '0.95rem', padding: '0 4px' }}
                  >✏️</button>
                )}
                {/* The pencil follows the title text; this spacer pushes the Mode control to the right (2026-10-29). */}
                <span style={{ flex: 1 }} />
                <div ref={modeMenuRef} style={{ position: 'relative', flexShrink: 0 }}>
                  <button
                    type="button"
                    onClick={() => setModeMenuOpen(o => !o)}
                    title="Conversation mode: how the model behaves in this conversation"
                    aria-haspopup="menu"
                    aria-expanded={modeMenuOpen}
                    style={{ fontSize: '0.75rem', background: '#282a36', color: chatMode === 'normal' ? '#c8d0e0' : '#f1fa8c', border: '1px solid #44475a', borderRadius: '4px', padding: '2px 8px', cursor: 'pointer', whiteSpace: 'nowrap' }}
                  >
                    Mode: {(MODES.find(([k]) => k === chatMode) || MODES[0])[1]} ▾
                  </button>
                  {modeMenuOpen && (
                    <div role="menu" style={{ position: 'absolute', right: 0, top: 'calc(100% + 4px)', zIndex: 50, width: '320px', maxWidth: '88vw', background: '#282a36', border: '1px solid #44475a', borderRadius: '6px', boxShadow: '0 6px 18px rgba(0,0,0,0.45)', padding: '4px' }}>
                      {MODES.map(([k, name, blurb]) => (
                        <button
                          key={k}
                          type="button"
                          role="menuitemradio"
                          aria-checked={k === chatMode}
                          onClick={() => { setChatMode(k); setModeMenuOpen(false); }}
                          style={{ display: 'block', width: '100%', textAlign: 'left', background: k === chatMode ? 'rgba(139, 233, 253, 0.12)' : 'transparent', border: 'none', borderRadius: '4px', padding: '6px 8px', cursor: 'pointer', color: '#f8f8f2' }}
                          onMouseEnter={(e) => { if (k !== chatMode) e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
                          onMouseLeave={(e) => { if (k !== chatMode) e.currentTarget.style.background = 'transparent'; }}
                        >
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: k === chatMode ? '#8be9fd' : '#f8f8f2' }}>{name}{k === chatMode ? ' ✓' : ''}</div>
                          <div style={{ fontSize: '0.75rem', color: '#9aa5c4', lineHeight: 1.35 }}>{blurb}</div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {chatMode !== 'normal' && (
                <div style={{ fontSize: '0.75rem', color: '#f1fa8c', padding: '0 4px 6px', opacity: 0.9 }}>
                  {MODE_HELP[chatMode]} <span style={{ color: '#9aa5c4' }}>Set the mode back to Normal to switch it off.</span>
                </div>
              )}
              <div className="conversation-history" ref={attachChatContainer} style={{minHeight: 'unset', marginBottom: 0}}>
                {messages.length === 0 ? (
                  introSeen('chat')
                    ? <p style={{color: '#9aa5c4'}}>Start a conversation...</p>
                    : <PageIntro introKey="chat" accent="#8be9fd" noLink>Type or dictate a message. Under the box, <b>KB</b> searches your documents and the paper clip attaches files and photos. Zynkbot remembers what you tell it about yourself, here on this device; Memory Manager shows what it kept, and you can edit or delete any of it. With your own API key, the model company answers each question and bills you for it; under its API terms it does not use your messages to train on or to build a profile of you, the way a chat subscription can. <b>Try this:</b> tell it what you cooked tonight and how it went.</PageIntro>
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
                        onReport={msg.role === 'assistant' ? openReport : undefined}
                        onSpeak={msg.role === 'assistant' && !msg.isStreaming ? () => handleSpeakMessage(msg) : undefined}
                        isSpeaking={voice.isTtsSpeaking && speakingMessageId === msg.id}
                        isEditing={editingMessageId === msg.id}
                        onSaveEdit={(newContent) => handleSaveEdit(msg.id, newContent)}
                        onCancelEdit={handleCancelEdit}
                      />
                    ));
                  })()
                )}
                {isLoading && (modelType?.endsWith('.gguf') || !messages.some(m => m.isStreaming && m.content)) && (
                  <div className="message bot-message">
                    <div className="message-content">
                      {modelType && modelType.endsWith('.gguf') ? (() => {
                        const total = localProgress?.total || 0;
                        const done = Math.min(localProgress?.done || 0, total);
                        const reading = total > 0 && done < total;
                        const pct = total > 0 ? Math.round(done / total * 100) : 0;
                        return (
                          <div>
                            <div>{reading ? `⏳ Reading your conversation and memories… ${pct}%` : (total > 0 ? '✍️ Writing the answer…' : '⏳ Local model is starting…')}</div>
                            <div style={{ marginTop: '6px', height: '6px', borderRadius: '3px', background: '#44475a', overflow: 'hidden' }}>
                              <div className={reading ? '' : 'zb-pulse'} style={{ height: '100%', width: reading ? `${pct}%` : '100%', background: '#8be9fd', transition: 'width 0.3s' }} />
                            </div>
                          </div>
                        );
                      })() : (chatStatus || 'Thinking...')}
                    </div>
                  </div>
                )}
                {/* The question card sits at the end of the thread, like a new message (2026-10-30:
                    drawn above the thread it went unseen on a phone with a long conversation). */}
                {pendingCheckin && <CheckInCard checkin={pendingCheckin} onAnswer={answerCheckin} onDismiss={() => answerCheckin('other')} />}
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
            {/* Timer / alarm / stopwatch chips: one each, only while something is set, running or ringing. */}
            {voice.timerState && (() => {
              const ts = voice.timerState;
              const timers = ts.timers || [];
              const alarms = ts.alarms || [];
              if (!(ts.ringing || timers.length || alarms.length || ts.sw_running || ts.sw_elapsed_ms > 0)) return null;
              const fmt = (secs) => { const s = Math.max(0, Math.floor(secs)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60; return (h ? `${h}:${String(m).padStart(2, '0')}` : `${m}`) + ':' + String(r).padStart(2, '0'); };
              const chip = { display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '4px 10px', borderRadius: '14px', background: '#1e1f29', border: '1px solid #44475a', fontSize: '0.8rem', color: '#f8f8f2', whiteSpace: 'nowrap' };
              const btn = { background: 'none', border: 'none', color: '#8be9fd', cursor: 'pointer', font: 'inherit', padding: 0 };
              return (
                <div style={{ marginBottom: '6px', padding: '0 4px' }}>
                  {ts.ringing && (
                    <button onClick={voice.stopRinging}
                      style={{ width: '100%', padding: '18px', borderRadius: '12px', border: 'none', background: '#ffb86c', color: '#282a36', fontSize: '1.15rem', fontWeight: 'bold', cursor: 'pointer', marginBottom: '6px' }}>
                      🔔 Stop
                    </button>
                  )}
                  <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '2px' }}>
                    {timers.map((t) => <span key={`t${t.id}`} style={chip}>⏱ {fmt((t.end - Date.now()) / 1000)} <button style={btn} onClick={() => voice.cancelTimerById(t.id)}>Cancel</button></span>)}
                    {alarms.map((al) => <span key={`a${al.id}`} style={chip}>⏰ {new Date(al.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })} <button style={btn} onClick={() => voice.cancelAlarmById(al.id)}>Cancel</button></span>)}
                    {ts.sw_running && <span style={chip}>⏱ Stopwatch {fmt(ts.sw_elapsed_ms / 1000)} <button style={btn} onClick={voice.stopStopwatch}>Stop</button></span>}
                    {!ts.sw_running && ts.sw_elapsed_ms > 0 && <span style={chip}>⏱ Stopped at {fmt(ts.sw_elapsed_ms / 1000)} <button style={btn} onClick={voice.resetStopwatch}>Reset</button></span>}
                  </div>
                </div>
              );
            })()}
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

            {memoryUpdateNotice && (
              <div
                onClick={() => setMemoryUpdateNotice('')}
                title="Tap to dismiss"
                style={{ fontSize: '0.8rem', color: '#9aa5c4', padding: '4px 8px', marginTop: '6px', borderLeft: '3px solid #50fa7b', background: '#282a36', borderRadius: '4px', cursor: 'pointer' }}
              >
                🔁 {memoryUpdateNotice}
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

                  {/* Attach File Button (documents; on Android the photo picker is the next button) */}
                  <button
                    onClick={() => handleAttachFile('documents')}
                    disabled={isLoading}
                    title={attachedFiles.length > 0 ? `${attachedFiles.length} file${attachedFiles.length > 1 ? 's' : ''} attached` : (window.AndroidPaths?.pickImages ? "Attach files" : "Attach files or images")}
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

                  {/* Photos button — Android only: the phone's photo picker, several at once */}
                  {window.AndroidPaths?.pickImages && (
                    <button
                      onClick={() => handleAttachFile('images')}
                      disabled={isLoading}
                      title="Choose photos"
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
                      🖼️
                    </button>
                  )}

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

                  {/* Copy button — only while there is text in the box */}
                  {input.trim() && (
                    <button
                      onClick={copyInputText}
                      title="Copy the text in the box"
                      style={{
                        height: '28px',
                        padding: '0 10px',
                        background: inputCopied ? '#50fa7b' : 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                        color: inputCopied ? '#282a36' : '#f8f8f2',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        transition: 'all 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {inputCopied ? '✓ Copied' : '⧉ Copy'}
                    </button>
                  )}
                  {/* Clear button (2026-10-22): same rule as Copy — only while there is text.
                      Dictated text that went elsewhere, or a false start, goes in one tap. */}
                  {input.trim() && (
                    <button
                      onClick={() => { setInput(''); inputTextareaRef.current?.focus(); }}
                      title="Clear the text in the box"
                      style={{
                        height: '28px',
                        padding: '0 10px',
                        background: 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                        color: '#f8f8f2',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      ✕ Clear
                    </button>
                  )}
                </div>

                {/* Mobile: circular mic button, bottom-right of textarea */}
                {isMobile && (
                  <VoiceButton
                    ref={voiceButtonRef}
                    onTranscript={handleDictationTranscript}
                    onLive={handleLiveDictation}
                    disabled={isLoading}
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
                    onTranscript={handleDictationTranscript}
                    onLive={handleLiveDictation}
                    disabled={isLoading}
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
                {/* On the local model the button stays tappable but grey: a disabled button
                    on a phone shows no tooltip and looked live while doing nothing (Matt,
                    2026-10-24). The tap explains in the notice line instead. */}
                <button
                  onClick={() => {
                    // Phones only (2026-10-29, Matt): the desktop opens Ensemble whatever the current model
                    // is; its window lists the local model next to the cloud ones.
                    if (window.AndroidPaths && modelType && modelType.endsWith('.gguf')) { showNotice('Ensemble needs cloud models. Pick one in the model menu to use it.'); return; }
                    setShowEnsemble(true);
                  }}
                  disabled={isLoading || containmentMode === 'child'}
                  title={containmentMode === 'child' ? 'Ensemble mode is not available in Child safety mode'
                    : (window.AndroidPaths && modelType && modelType.endsWith('.gguf')) ? 'Ensemble needs cloud models; switch off the local model to use it'
                    : 'Multi-model collaboration'}
                  style={{
                    width: '85px',
                    height: '42px',
                    minWidth: '85px',
                    maxWidth: '85px',
                    minHeight: '42px',
                    maxHeight: '42px',
                    padding: '0',
                    background: (containmentMode === 'child' || (window.AndroidPaths && modelType && modelType.endsWith('.gguf'))) ? '#6272a4' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: (isLoading || containmentMode === 'child') ? 'not-allowed' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.75rem',
                    transition: 'all 0.2s',
                    opacity: (isLoading || containmentMode === 'child' || (window.AndroidPaths && modelType && modelType.endsWith('.gguf'))) ? 0.5 : 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    flex: 'none',
                    gridColumn: isMobile ? 1 : 2,
                    gridRow: 1,
                  }}
                  onMouseOver={(e) => !(isLoading || containmentMode === 'child' || (window.AndroidPaths && modelType && modelType.endsWith('.gguf'))) && (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  Ensemble
                </button>

                {/* Send. Stopping lives in its own button now, so this no longer
                    doubles as Stop — a Send that changes meaning mid-request was
                    unavailable for the case that actually mattered (stopping speech,
                    which begins after generation finishes). */}
                <button
                  onClick={() => handleSendMessage(input)}
                  disabled={isLoading}
                  title={isLoading ? 'Waiting for the current response' : 'Send message'}
                  style={{
                    width: '85px',
                    height: '42px',
                    minWidth: '85px',
                    maxWidth: '85px',
                    minHeight: '42px',
                    maxHeight: '42px',
                    padding: '0',
                    background: 'linear-gradient(135deg, #50fa7b 0%, #3dd66b 100%)',
                    color: '#282a36',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: isLoading ? 'not-allowed' : 'pointer',
                    opacity: isLoading ? 0.5 : 1,
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
                  Send
                </button>

                {/* Stop — mobile: TR (1,2); desktop: BR (2,2).
                    Replaces the old Clear button, which duplicated the "New"
                    button in the Conversation header. Live during generation AND
                    during spoken playback. */}
                <button
                  onClick={handleStopGeneration}
                  disabled={!canStop}
                  title={canStop ? 'Stop speaking and stop generating' : 'Nothing to stop'}
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
                    cursor: canStop ? 'pointer' : 'not-allowed',
                    fontWeight: 'bold',
                    fontSize: '0.75rem',
                    transition: 'all 0.2s',
                    opacity: canStop ? 1 : 0.5,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                    flex: 'none',
                    gridColumn: 2,
                    gridRow: isMobile ? 1 : 2,
                  }}
                  onMouseOver={(e) => canStop && (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  ■ Stop
                </button>

                {/* Mobile-only: Memory Manager — BL (2,1); opens the modal via ref. Never disabled
                    while a reply is in progress (Matt, 2026-10-22): with a local model a reply
                    can take a minute, and the memories are there to be read meanwhile. */}
                {isMobile && (
                  <button
                    onClick={() => memoryManagerRef.current?.open()}
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
                      cursor: 'pointer',
                      fontWeight: 'bold',
                      fontSize: '0.72rem',
                      transition: 'all 0.2s',
                      opacity: 1,
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
              onOpenConversation={handleOpenMemoryConversation}
            />
          </div>
        </div>

        {/* Live Insights panel removed - simplified to single-column layout */}
      </div>

      <WhyZynkbotModal isOpen={showWhyZynkbot} onClose={() => setShowWhyZynkbot(false)} />
      <AboutModal isOpen={showAbout} onClose={() => setShowAbout(false)} />
      <ReportProblemModal
        isOpen={showReport}
        onClose={() => setShowReport(false)}
        context={reportContext}
        backend={modelType}
        threadText={messages.map(m => `${m.role === 'user' ? 'You' : 'Zynkbot'}: ${m.content}`).join('\n\n')}
      />
      <GettingStartedModal isOpen={showDemoGuide} onClose={() => setShowDemoGuide(false)} onOpenAPIKeys={() => setControlsPage('model')} />
      <UserIdentityModal isOpen={showUserIdentity} onClose={() => setShowUserIdentity(false)} apiBaseUrl={API_BASE_URL} sessionId={sessionId} />
      <ConversationHistoryPanel
        currentSessionId={sessionId}
        isOpen={showConversationHistory}
        onClose={() => setShowConversationHistory(false)}
        userId={userId}
        containmentMode={containmentMode}
        onResume={handleResumeSession}
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
      {notice && (
        <div role="status" style={{ position: 'fixed', left: '50%', bottom: '24px', transform: 'translateX(-50%)', background: '#282a36', color: '#f8f8f2', border: '1px solid #6272a4', borderRadius: '8px', padding: '10px 16px', fontSize: '0.9rem', zIndex: 10000, maxWidth: '90vw', boxShadow: '0 4px 16px rgba(0,0,0,0.4)' }}>
          {notice}
        </div>
      )}
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
        onCompleted={() => setGettingStartedOnTop(false)}
        userId={userId}
      />
      <SnapInModal
        isOpen={showSnapInModal}
        onClose={() => setShowSnapInModal(false)}
        userId={userId}
      />
      <VoiceModal
        isOpen={voice.showVoiceModal}
        onClose={() => voice.setShowVoiceModal(false)}
        voiceInputSource={voice.voiceInputSource}
        onVoiceSourceChange={voice.setVoiceInputSource}
        heyZynkEnabled={voice.heyZynkEnabled}
        onHeyZynkChange={voice.setHeyZynkEnabled}
        fruitlessTone={voice.fruitlessTone}
        onFruitlessToneChange={voice.setFruitlessTone}
        wakeWordModelReady={voice.wakeWordModelReady}
        wakeState={voice.wakeState}
        wakeWordDownloadProgress={voice.wakeWordDownloadProgress}
        wakeWordDownloadError={voice.wakeWordDownloadError}
        onDownloadModels={() => window.WakeWordBridge?.downloadModels()}
        ttsEnabled={voice.ttsEnabled}
        onTtsEnabledChange={voice.setTtsEnabled}
        keepScreenAwake={voice.keepScreenAwake}
        onKeepScreenAwakeChange={voice.setKeepScreenAwake}
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
