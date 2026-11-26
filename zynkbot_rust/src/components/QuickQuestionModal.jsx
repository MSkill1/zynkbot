import React, { useState, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import VoiceButton from './VoiceButton';
import ChatMessage from './ChatMessage';
import ControlsPage from './ControlsPage';

// Quick question (Matt, 2026-11-12). A one-off question asked in its own window and
// answered without the open conversation's history, so the thread being worked on is
// not interrupted and nothing typed in its box is lost. Memories are recalled as for any
// message. Every exchange is filed in one standing "Quick questions" thread in History
// (the resume logic skips that thread, see get_latest_thread). The question can be
// dictated and the answer copied. Earlier questions are read inside this window
// ("Earlier"), and the thread is kept out of History (Matt, 2026-11-12: History is full
// enough). The window is the shared page shell: full screen on a phone, a card on the desktop.
export const QUICK_SESSION_PREFIX = 'quick-questions-';

export default function QuickQuestionModal({ isOpen, onClose, userId, backend, containmentMode, isMobile }) {
  const [question, setQuestion] = useState('');
  const [exchange, setExchange] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [earlier, setEarlier] = useState(null); // null: hidden; a list: shown
  const [loadingEarlier, setLoadingEarlier] = useState(false);
  const textareaRef = useRef(null);
  const sessionId = QUICK_SESSION_PREFIX + userId;

  // The latest onClose and loading flag, read through refs so the effect below runs
  // once per open: with them in its dependency list it re-ran on every App render and
  // re-focused the box, which on the phone popped the keyboard back up while the user
  // was reading an earlier answer (code review, 2026-11-26).
  const onCloseRef = useRef(onClose); onCloseRef.current = onClose;
  const isLoadingRef = useRef(isLoading); isLoadingRef.current = isLoading;
  useEffect(() => {
    if (!isOpen) {
      // Closed: the next open is a fresh box (Matt, 2026-11-12, Pixel). The exchange
      // just answered is in the thread, under Earlier.
      setQuestion(''); setExchange(null); setError(null); setEarlier(null);
      return;
    }
    const t = setTimeout(() => textareaRef.current?.focus(), 50);
    const onKey = (e) => { if (e.key === 'Escape' && !isLoadingRef.current) onCloseRef.current(); };
    window.addEventListener('keydown', onKey);
    return () => { clearTimeout(t); window.removeEventListener('keydown', onKey); };
  }, [isOpen]);

  if (!isOpen) return null;

  const insertTranscript = (text) => {
    if (!text) return;
    setQuestion(q => (q && !q.endsWith(' ') ? q + ' ' : q) + text);
  };

  const ask = async () => {
    const q = question.trim();
    if (!q || isLoading) return;
    setIsLoading(true);
    setError(null);
    const now = new Date().toISOString();
    setExchange({ id: Date.now(), question: q, askedAt: now, answer: '', streaming: true });
    setQuestion('');
    let unlisten = null;
    try {
      unlisten = await listen('stream-token', (event) => {
        setExchange(x => x ? { ...x, answer: x.answer + event.payload } : x);
      });
      const response = await invoke('send_message_with_memory', {
        message: q,
        userId,
        sessionId,
        backend,
        containmentMode,
        conversationHistory: [],
        kbEnabled: false,
      });
      const metadata = { model_backend: response?.model_backend, containment_mode: response?.containment_mode, blocked: response?.blocked };
      if (response?.web_search_needed && response?.web_search_query) {
        // The model asked for a web search. There is no "search?" button in this window,
        // so it searches at once, as hands-free does (Matt, 2026-11-18: the request
        // itself landed in the box as the answer). The searched answer replaces the
        // streamed request; the thread keeps it as "Web search: <query>" plus the answer.
        await searchAndAnswer(response.web_search_query, q, metadata);
        return;
      }
      setExchange(x => x ? { ...x, answer: response?.reply_text || '', streaming: false, metadata } : x);
      // Named once; the title is kept when it is already set.
      invoke('set_session_title', { sessionId, userId, title: 'Quick questions' }).catch(() => {});
    } catch (e) {
      setError(String(e));
      setExchange(x => x ? { ...x, streaming: false } : x);
    } finally {
      if (unlisten) unlisten();
      setIsLoading(false);
    }
  };

  // The same search the conversation runs from its "search the web" button (App.jsx
  // handleExecuteWebSearch): a cloud model's own search comes back with the answer
  // written; a local model gets results and a second, memory-free turn writes the answer.
  const searchAndAnswer = async (searchQuery, originalQuery, metadata) => {
    setExchange(x => x ? { ...x, answer: `Searching the web for: "${searchQuery}"…`, streaming: true, metadata } : x);
    let streamedWords = false;
    const unlistenSearch = await listen('stream-token', (event) => {
      const first = !streamedWords;
      streamedWords = true;
      setExchange(x => x ? { ...x, answer: (first ? '' : x.answer) + event.payload } : x);
    });
    let searchResults;
    try {
      searchResults = await invoke('execute_web_search', { query: searchQuery, maxResults: 5, fetchTopN: 3, backend });
    } finally {
      unlistenSearch();
    }
    const threadSources = {
      query: searchQuery,
      results: (searchResults.results || []).map((r) => ({ title: r.title, url: r.url, snippet: r.snippet })),
      num_results: (searchResults.results || []).length,
    };
    if (searchResults.answer) {
      setExchange(x => x ? { ...x, answer: searchResults.answer, streaming: false, metadata, webSearchQuery: searchQuery, webSearchResults: searchResults } : x);
      invoke('log_web_search_answer', { sessionId, userId, query: searchQuery, answer: searchResults.answer, backend, containmentMode, sources: threadSources })
        .catch(e => console.warn('[QuickQuestion] could not log the searched answer:', e));
      return;
    }
    let searchContext = `Here are the web search results for "${searchQuery}":\n\n`;
    if (searchResults.results && searchResults.results.length > 0) {
      searchResults.results.forEach((result, index) => {
        searchContext += `Source ${index + 1}: ${result.title}\nURL: ${result.url}\n`;
        if (result.snippet) searchContext += `Summary: ${result.snippet}\n`;
        if (result.content) searchContext += `Content: ${result.content.substring(0, 800)}\n`;
        searchContext += '\n';
      });
    } else {
      searchContext += 'No results found.\n';
    }
    const llmPrompt = `Answer the user's question using the web search results below. Be direct and concise — do not explain your reasoning process, do not narrate what you are doing, just answer.\n\nQuestion: "${originalQuery}"\n\n${searchContext}\n\nAnswer directly based on the search results. If the results don't contain enough information, say so briefly.`;
    setExchange(x => x ? { ...x, answer: '', streaming: true } : x);
    const unlisten = await listen('stream-token', (event) => {
      setExchange(x => x ? { ...x, answer: x.answer + event.payload } : x);
    });
    try {
      const llmResponse = await invoke('send_message_with_memory', {
        message: llmPrompt,
        userQuery: `Web search: "${searchQuery}"`,
        userId,
        sessionId,
        backend,
        containmentMode,
        conversationHistory: [],
        skipContainment: true,
        skipMemoryStorage: true,
        webSources: threadSources,
      });
      setExchange(x => x ? { ...x, answer: llmResponse?.reply_text || '', streaming: false, metadata, webSearchQuery: searchQuery, webSearchResults: searchResults } : x);
    } finally {
      unlisten();
    }
  };

  // Earlier exchanges from the standing thread, newest first, as question-and-answer pairs.
  const toggleEarlier = async () => {
    if (earlier) { setEarlier(null); return; }
    setLoadingEarlier(true);
    try {
      const rows = await invoke('get_conversation_messages', { sessionId });
      const pairs = [];
      for (const r of rows || []) {
        if (r.role === 'user') pairs.push({ q: r, a: null });
        else if (r.role === 'assistant' && pairs.length && !pairs[pairs.length - 1].a) pairs[pairs.length - 1].a = r;
      }
      setEarlier(pairs.reverse().slice(0, 30));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingEarlier(false);
    }
  };

  const stop = async () => {
    try { await invoke('cancel_generation'); } catch (_) {}
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isMobile) { e.preventDefault(); ask(); }
  };

  // The contents are a column so the answer area scrolls on its own under the question
  // box: filling the shell's full-screen card on a phone, capped at the card's height on
  // the desktop (the shell's head and padding take about 110px).
  const columnStyle = isMobile ? {
    display: 'flex', flexDirection: 'column', height: 'calc(100dvh - 110px)',
  } : {
    display: 'flex', flexDirection: 'column', maxHeight: 'calc(92vh - 110px)',
  };

  return (
    <ControlsPage
      icon="⚡"
      title="Quick question"
      accent="#8be9fd"
      onClose={() => { if (!isLoading) onClose(); }}
      closeLabel="Close"
      zIndex={2000}
      dismissOnOutside={!isLoading}
    >
      {/* The shared page shell (ui-redo, 2026-11-18). ✕ and a tap outside wait for an answer in progress. */}
      <div style={columnStyle}>
        <div style={{ color: '#9aa5c4', fontSize: '0.82rem', lineHeight: 1.5, marginBottom: '10px' }}>
          Answered on its own, without the conversation you have open. Earlier questions stay in this window.
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'stretch', marginBottom: '10px' }}>
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type or dictate a question…"
            rows={isMobile ? 3 : 2}
            disabled={isLoading}
            style={{
              flex: 1, padding: '10px', borderRadius: '8px', border: '1px solid #44475a',
              background: '#282a36', color: '#f8f8f2', fontSize: '1rem', resize: 'none', fontFamily: 'inherit',
            }}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <VoiceButton onTranscript={insertTranscript} disabled={isLoading}
              style={{ width: '64px', height: '42px', minWidth: '64px', minHeight: '42px' }} />
            <button onClick={ask} disabled={isLoading || !question.trim()} title="Ask"
              style={{
                width: '64px', height: '42px', border: 'none', borderRadius: '8px', fontWeight: 'bold',
                background: 'linear-gradient(135deg, #50fa7b 0%, #3dd66b 100%)', color: '#282a36',
                cursor: (isLoading || !question.trim()) ? 'not-allowed' : 'pointer',
                opacity: (isLoading || !question.trim()) ? 0.5 : 1,
              }}>
              Ask
            </button>
          </div>
        </div>

        <div style={{ flex: 1, overflow: 'auto', minHeight: isMobile ? 0 : '120px', borderTop: '1px solid #44475a', paddingTop: '8px' }}>
          {!exchange && !error && !earlier && (
            <div style={{ color: '#6272a4', fontSize: '0.9rem', padding: '8px 4px' }}>The answer appears here.</div>
          )}
          {exchange && (
            <>
              <ChatMessage
                message={{ id: exchange.id, role: 'user', content: exchange.question, timestamp: exchange.askedAt }}
                sessionId={sessionId}
                userId={userId}
              />
              <ChatMessage
                message={{
                  id: exchange.id + 1, role: 'assistant',
                  content: exchange.answer || (exchange.streaming ? 'Waiting…' : ''),
                  timestamp: exchange.askedAt, isStreaming: exchange.streaming,
                  web_search_query: exchange.webSearchQuery, web_search_results: exchange.webSearchResults,
                }}
                metadata={exchange.metadata || null}
                sessionId={sessionId}
                userId={userId}
                onStop={exchange.streaming && isLoading ? stop : undefined}
              />
            </>
          )}
          {error && (
            <div style={{ color: '#ff5555', fontSize: '0.9rem', padding: '8px 4px' }}>{error}</div>
          )}
          {earlier && (
            <div style={{ marginTop: exchange ? '10px' : 0 }}>
              <div style={{ color: '#9aa5c4', fontSize: '0.8rem', borderTop: '1px solid #44475a', padding: '8px 4px 4px' }}>
                Earlier questions{earlier.length === 0 ? ': none yet' : ''}
              </div>
              {earlier
                .filter(p => !(exchange && p.q.content === exchange.question && (p.a?.content || '') === exchange.answer))
                .map(p => (
                  <React.Fragment key={p.q.id}>
                    <ChatMessage message={{ id: p.q.id, role: 'user', content: p.q.content, timestamp: p.q.created_at }} sessionId={sessionId} userId={userId} />
                    {p.a && <ChatMessage message={{ id: p.a.id, role: 'assistant', content: p.a.content, timestamp: p.a.created_at }} metadata={null} sessionId={sessionId} userId={userId} />}
                  </React.Fragment>
                ))}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px', paddingTop: '10px' }}>
          <button onClick={toggleEarlier} disabled={isLoading || loadingEarlier} title="Questions asked here before"
            style={{
              padding: '10px 16px', background: 'none', color: '#8be9fd', border: '1px solid #44475a', borderRadius: '8px',
              cursor: (isLoading || loadingEarlier) ? 'not-allowed' : 'pointer', fontWeight: 'bold', minHeight: '42px',
            }}>
            {loadingEarlier ? 'Loading…' : earlier ? 'Hide earlier' : 'Earlier'}
          </button>
          <button onClick={onClose} disabled={isLoading}
            style={{
              padding: '10px 22px', background: '#44475a', color: '#f8f8f2', border: 'none', borderRadius: '8px',
              cursor: isLoading ? 'not-allowed' : 'pointer', fontWeight: 'bold', opacity: isLoading ? 0.5 : 1, minHeight: '42px',
            }}>
            Done
          </button>
        </div>
      </div>
    </ControlsPage>
  );
}
