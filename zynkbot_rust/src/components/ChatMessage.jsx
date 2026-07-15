import React, { useState, useEffect } from "react";
import { openUrl } from '@tauri-apps/plugin-opener';
import "../styles/ChatMessage.css";

export default function ChatMessage({ message, metadata, onExecuteWebSearch, sessionId, userId }) {
  // Handle both old format {user, bot} and new format {role, content}
  const isUserMessage = message.role === 'user';
  const content = message.content;
  const recalled_memories = message.recalled_memories || metadata?.recalled_memories || [];
  const schema = metadata?.schema;
  const webSearchNeeded = message.web_search_needed;
  const webSearchQuery = message.web_search_query;
  const originalQuery = message.original_query;
  const webSearchResults = message.web_search_results;

  // State for editable search query
  const [editedQuery, setEditedQuery] = useState(webSearchQuery || "");
  const [showSearchPrompt, setShowSearchPrompt] = useState(webSearchNeeded);

  // Sync when the message object is replaced (e.g. streaming placeholder → final web-search message)
  useEffect(() => {
    setShowSearchPrompt(webSearchNeeded);
    if (webSearchQuery) setEditedQuery(webSearchQuery);
  }, [webSearchNeeded, webSearchQuery]);

  if (isUserMessage) {
    return (
      <div className="chat-message">
        <div className="message-user">
          <div className="message-header">
            <strong>You:</strong>
            {schema && (
              <span className={`schema-badge badge-${schema}`}>
                {schema}
              </span>
            )}
          </div>
          <div className="message-content">{content}</div>
        </div>
      </div>
    );
  }

  // Bot message
  return (
    <div className="chat-message">
      <div className="message-bot">
        <div className="message-header">
          <strong>Zynkbot:</strong>
          {recalled_memories.length > 0 && (
            <span className="memory-indicator">
              📚 {recalled_memories.length} {recalled_memories.length === 1 ? 'memory' : 'memories'}
            </span>
          )}
        </div>
        <div className="message-content">{content}</div>

        {/* Show web search confirmation with editable query */}
        {showSearchPrompt && onExecuteWebSearch && (
          <div style={{
            marginTop: '15px',
            padding: '15px',
            background: 'rgba(139, 233, 253, 0.1)',
            border: '1px solid rgba(139, 233, 253, 0.3)',
            borderRadius: '8px'
          }}>
            <p style={{ margin: '0 0 10px 0', color: '#f8f8f2', fontSize: '0.9rem' }}>
              🌐 <strong>Web Search Required</strong>
            </p>
            <p style={{ margin: '0 0 10px 0', color: '#9aa5c4', fontSize: '0.85rem' }}>
              To answer this question, I need to search the web for current information.
            </p>

            {/* Editable search query input */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{
                display: 'block',
                marginBottom: '6px',
                color: '#8be9fd',
                fontSize: '0.85rem',
                fontWeight: '500'
              }}>
                Search query:
              </label>
              <input
                type="text"
                value={editedQuery}
                onChange={(e) => setEditedQuery(e.target.value)}
                placeholder="Edit search query..."
                style={{
                  width: '100%',
                  padding: '10px',
                  background: '#282a36',
                  border: '1px solid #44475a',
                  borderRadius: '6px',
                  color: '#f8f8f2',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={() => {
                  if (editedQuery.trim()) {
                    onExecuteWebSearch(message.id, editedQuery.trim(), originalQuery);
                    setShowSearchPrompt(false);
                  }
                }}
                disabled={!editedQuery.trim()}
                style={{
                  flex: 1,
                  padding: '10px 20px',
                  background: editedQuery.trim()
                    ? 'linear-gradient(135deg, #8be9fd 0%, #50fa7b 100%)'
                    : '#44475a',
                  border: 'none',
                  borderRadius: '6px',
                  color: editedQuery.trim() ? '#282a36' : '#6272a4',
                  fontWeight: '600',
                  cursor: editedQuery.trim() ? 'pointer' : 'not-allowed',
                  fontSize: '0.9rem',
                  transition: 'all 0.2s',
                  opacity: editedQuery.trim() ? 1 : 0.6
                }}
                onMouseEnter={(e) => {
                  if (editedQuery.trim()) {
                    e.target.style.transform = 'scale(1.05)';
                    e.target.style.boxShadow = '0 4px 12px rgba(139, 233, 253, 0.4)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.target.style.transform = 'scale(1)';
                  e.target.style.boxShadow = 'none';
                }}
              >
                🔍 Search the Web
              </button>

              <button
                onClick={() => setShowSearchPrompt(false)}
                style={{
                  padding: '10px 20px',
                  background: 'rgba(255, 85, 85, 0.2)',
                  border: '1px solid rgba(255, 85, 85, 0.4)',
                  borderRadius: '6px',
                  color: '#ff5555',
                  fontWeight: '600',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = 'rgba(255, 85, 85, 0.3)';
                  e.target.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = 'rgba(255, 85, 85, 0.2)';
                  e.target.style.transform = 'scale(1)';
                }}
              >
                ✕ Cancel
              </button>
            </div>
          </div>
        )}

        {/* Show web search sources below the answer */}
        {webSearchResults && webSearchResults.results && webSearchResults.results.length > 0 && (
          <div className="recalled-memories-detail" style={{
            marginTop: '10px',
            padding: '10px',
            background: 'rgba(139, 233, 253, 0.05)',
            borderRadius: '6px',
            border: '1px solid rgba(139, 233, 253, 0.2)'
          }}>
            <details>
              <summary style={{ cursor: 'pointer', color: '#8be9fd', fontWeight: '500' }}>
                🌐 View {webSearchResults.results.length} search source{webSearchResults.results.length !== 1 ? 's' : ''}
              </summary>
              <div style={{ marginTop: '10px' }}>
                {webSearchResults.results.map((result, idx) => (
                  <div key={idx} style={{
                    marginBottom: '15px',
                    paddingBottom: '15px',
                    borderBottom: idx < webSearchResults.results.length - 1 ? '1px solid rgba(139, 233, 253, 0.1)' : 'none'
                  }}>
                    <div style={{ fontWeight: '600', color: '#f8f8f2', marginBottom: '5px' }}>
                      {idx + 1}. {result.title}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: '#8be9fd', marginBottom: '5px', wordBreak: 'break-all' }}>
                      <span
                        onClick={() => openUrl(result.url)}
                        style={{ color: '#8be9fd', textDecoration: 'underline', cursor: 'pointer' }}
                      >
                        {result.url}
                      </span>
                    </div>
                    {result.snippet && (
                      <div style={{ fontSize: '0.85rem', color: '#9aa5c4', fontStyle: 'italic' }}>
                        {result.snippet}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}

        {/* Show recalled memories below the message */}
        {recalled_memories.length > 0 && (
          <div className="recalled-memories-detail">
            <details>
              <summary>View recalled memories</summary>
              <ul>
                {recalled_memories.map((mem, idx) => (
                  <li key={idx}>
                    {mem.content || mem.fact || JSON.stringify(mem)}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
