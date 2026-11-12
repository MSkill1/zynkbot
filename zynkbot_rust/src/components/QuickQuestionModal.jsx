import React, { useState, useRef, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import VoiceButton from './VoiceButton';
import ChatMessage from './ChatMessage';

// Quick question (Matt, 2026-11-12). A one-off question asked in its own window and
// answered without the open conversation's history, so the thread being worked on is
// not interrupted and nothing typed in its box is lost. Memories are recalled as for any
// message. Every exchange is filed in one standing "Quick questions" thread in History
// (the resume logic skips that thread, see get_latest_thread). The question can be
// dictated and the answer copied. Earlier questions are read inside this window
// ("Earlier"), and the thread is kept out of History (Matt, 2026-11-12: History is full
// enough). The window fills the screen on a phone and is a dialog on the desktop.
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

  useEffect(() => {
    if (!isOpen) return;
    const t = setTimeout(() => textareaRef.current?.focus(), 50);
    const onKey = (e) => { if (e.key === 'Escape' && !isLoading) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => { clearTimeout(t); window.removeEventListener('keydown', onKey); };
  }, [isOpen, isLoading, onClose]);

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
      setExchange(x => x ? {
        ...x,
        answer: response?.reply_text || '',
        streaming: false,
        metadata: { model_backend: response?.model_backend, containment_mode: response?.containment_mode, blocked: response?.blocked },
      } : x);
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

  const panelStyle = isMobile ? {
    background: '#1e1f29', width: '100%', height: '100%', display: 'flex', flexDirection: 'column',
    padding: '14px 12px', boxSizing: 'border-box',
  } : {
    background: '#1e1f29', borderRadius: '12px', padding: '22px 24px', maxWidth: '680px', width: '100%',
    maxHeight: '90vh', display: 'flex', flexDirection: 'column', border: '1px solid #44475a',
    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)', boxSizing: 'border-box',
  };

  return (
    <div data-modal=""
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0, 0, 0, 0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 1000, padding: isMobile ? 0 : '20px',
      }}
      onClick={(e) => { if (e.target === e.currentTarget && !isLoading) onClose(); }}
    >
      <div style={panelStyle}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <h2 style={{ color: '#8be9fd', margin: 0, fontSize: isMobile ? '1.2rem' : '1.4rem' }}>⚡ Quick question</h2>
          <button onClick={onClose} disabled={isLoading} title="Close"
            style={{ background: 'none', border: 'none', color: '#f8f8f2', fontSize: '1.4rem', cursor: isLoading ? 'not-allowed' : 'pointer', padding: '4px 8px', opacity: isLoading ? 0.5 : 1 }}>
            ✕
          </button>
        </div>
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
    </div>
  );
}
