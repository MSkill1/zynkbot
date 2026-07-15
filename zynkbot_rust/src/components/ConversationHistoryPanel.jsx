import React, { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";

// Groups sessions by relative date for display
function dateLabel(dateStr) {
  const d = new Date(dateStr);
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterdayStart = new Date(todayStart - 86400000);
  const weekStart = new Date(todayStart - 6 * 86400000);

  if (d >= todayStart) return "Today";
  if (d >= yesterdayStart) return "Yesterday";
  if (d >= weekStart) return "This Week";
  return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function groupSessions(sessions) {
  const groups = {};
  for (const s of sessions) {
    const label = dateLabel(s.last_active);
    if (!groups[label]) groups[label] = [];
    groups[label].push(s);
  }
  return groups;
}

export default function ConversationHistoryPanel({ isOpen, onClose, userId, containmentMode, onResume }) {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const searchTimeout = useRef(null);

  const isHipaaMode = containmentMode === "hipaa";

  const loadSessions = useCallback(async () => {
    if (!userId || isHipaaMode) return;
    setIsLoading(true);
    try {
      const result = await invoke("list_conversation_sessions", {
        userId,
        limit: 100,
        offset: 0,
      });
      setSessions(result || []);
    } catch (e) {
      console.error("[ConvHistory] Failed to load sessions:", e);
    } finally {
      setIsLoading(false);
    }
  }, [userId, isHipaaMode]);

  const runSearch = useCallback(async (query, from, to) => {
    if (!userId) return;
    if (!query.trim() && !from && !to) {
      loadSessions();
      return;
    }
    setIsLoading(true);
    try {
      const result = await invoke("search_conversations", {
        userId,
        query: query.trim(),
        dateFrom: from || null,
        dateTo: to || null,
      });
      setSessions(result || []);
    } catch (e) {
      console.error("[ConvHistory] Search failed:", e);
    } finally {
      setIsLoading(false);
    }
  }, [userId, loadSessions]);

  // Debounced search
  useEffect(() => {
    if (!isOpen) return;
    clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => {
      runSearch(searchQuery, dateFrom, dateTo);
    }, 300);
    return () => clearTimeout(searchTimeout.current);
  }, [searchQuery, dateFrom, dateTo, isOpen, runSearch]);

  // Load on open
  useEffect(() => {
    if (isOpen && !isHipaaMode) {
      loadSessions();
      setSelectedSession(null);
      setMessages([]);
    }
  }, [isOpen, isHipaaMode, loadSessions]);

  const openSession = async (session) => {
    setSelectedSession(session);
    setIsLoadingMessages(true);
    try {
      const result = await invoke("get_conversation_messages", {
        sessionId: session.session_id,
      });
      setMessages(result || []);
    } catch (e) {
      console.error("[ConvHistory] Failed to load messages:", e);
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const deleteSession = async (e, sessionId) => {
    e.stopPropagation();
    if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
    try {
      await invoke("delete_conversation_session", { sessionId, userId });
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId));
      if (selectedSession?.session_id === sessionId) {
        setSelectedSession(null);
        setMessages([]);
      }
    } catch (e) {
      console.error("[ConvHistory] Delete failed:", e);
    }
  };

  const clearAllHistory = async () => {
    if (!window.confirm("Delete ALL conversation history? This cannot be undone.")) return;
    try {
      await invoke("clear_conversation_history", { userId });
      setSessions([]);
      setSelectedSession(null);
      setMessages([]);
    } catch (e) {
      console.error("[ConvHistory] Clear all failed:", e);
    }
  };

  if (!isOpen) return null;

  const panelStyle = {
    position: "fixed",
    top: 0,
    right: 0,
    width: "460px",
    height: "100vh",
    background: "#282a36",
    borderLeft: "1px solid #44475a",
    display: "flex",
    flexDirection: "column",
    zIndex: 1002,
    boxShadow: "-4px 0 20px rgba(0,0,0,0.5)",
  };

  // HIPAA mode — show disabled state
  if (isHipaaMode) {
    return (
      <div className="conv-history-panel" style={panelStyle}>
        <div style={{ padding: "20px", borderBottom: "1px solid #44475a", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ margin: 0, color: "#f8f8f2", fontSize: "1rem" }}>🕐 Conversation History</h2>
          <button onClick={onClose} className="conv-history-close" style={{ background: "none", border: "none", color: "#6272a4", fontSize: "1.2rem", cursor: "pointer" }}>×</button>
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "40px", textAlign: "center", color: "#6272a4" }}>
          <div>
            <div style={{ fontSize: "2rem", marginBottom: "12px" }}>🔒</div>
            <p>Conversation history is disabled in HIPAA mode.</p>
            <p style={{ fontSize: "0.85rem" }}>No conversation records are stored when HIPAA containment is active.</p>
          </div>
        </div>
      </div>
    );
  }

  const groups = groupSessions(sessions);

  return (
    <div className="conv-history-panel" style={panelStyle}>
      {/* Header */}
      <div style={{ padding: "16px 20px", borderBottom: "1px solid #44475a", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#1e1f2e" }}>
        <h2 style={{ margin: 0, color: "#f8f8f2", fontSize: "1rem", fontWeight: "bold", flex: 1, minWidth: 0 }}>
          {selectedSession ? (
            <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <button
                onClick={() => { setSelectedSession(null); setMessages([]); }}
                style={{ background: "none", border: "none", color: "#8be9fd", cursor: "pointer", fontSize: "0.9rem", padding: 0, flexShrink: 0 }}
              >← Back</button>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {selectedSession.title || "Conversation"}
              </span>
            </span>
          ) : "🕐 Conversation History"}
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
          {selectedSession && onResume && (
            <button
              onClick={() => {
                onResume({
                  sessionId: selectedSession.session_id,
                  messages: messages.map((m) => ({
                    id: m.id,
                    role: m.role,
                    content: m.content,
                    timestamp: m.created_at,
                  })),
                });
                onClose();
              }}
              style={{
                background: "#50fa7b",
                border: "none",
                color: "#282a36",
                fontSize: "0.8rem",
                fontWeight: "bold",
                padding: "5px 12px",
                borderRadius: "5px",
                cursor: "pointer",
              }}
              title="Continue this conversation"
            >
              Resume
            </button>
          )}
          {!selectedSession && sessions.length > 0 && (
            <button
              onClick={clearAllHistory}
              style={{ background: "none", border: "1px solid #ff5555", color: "#ff5555", fontSize: "0.75rem", padding: "3px 8px", borderRadius: "4px", cursor: "pointer" }}
              title="Delete all conversation history"
            >
              Clear All
            </button>
          )}
          <button onClick={onClose} className="conv-history-close" style={{ background: "none", border: "none", color: "#6272a4", fontSize: "1.4rem", cursor: "pointer", lineHeight: 1 }}>×</button>
        </div>
      </div>

      {/* Search (session list view only) */}
      {!selectedSession && (
        <div style={{ padding: "12px 16px", borderBottom: "1px solid #44475a", background: "#21222c" }}>
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "#282a36",
              border: "1px solid #44475a",
              borderRadius: "6px",
              color: "#f8f8f2",
              fontSize: "0.9rem",
              boxSizing: "border-box",
              outline: "none",
            }}
          />
          <div style={{ display: "flex", gap: "8px", marginTop: "8px", alignItems: "center" }}>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              style={{ flex: 1, padding: "5px 8px", background: "#282a36", border: "1px solid #44475a", borderRadius: "4px", color: "#6272a4", fontSize: "0.8rem" }}
            />
            <span style={{ color: "#6272a4", fontSize: "0.8rem" }}>to</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              style={{ flex: 1, padding: "5px 8px", background: "#282a36", border: "1px solid #44475a", borderRadius: "4px", color: "#6272a4", fontSize: "0.8rem" }}
            />
            {(dateFrom || dateTo) && (
              <button
                onClick={() => { setDateFrom(""); setDateTo(""); }}
                title="Clear date filter"
                style={{ background: "none", border: "none", color: "#ff5555", cursor: "pointer", fontSize: "1.1rem", padding: "0 2px", flexShrink: 0 }}
                onMouseOver={(e) => e.target.style.opacity = "0.7"}
                onMouseOut={(e) => e.target.style.opacity = "1"}
              >×</button>
            )}
          </div>
        </div>
      )}

      {/* Content — direction:rtl moves scrollbar to left edge, away from main app scrollbar */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 0", direction: "rtl" }}>

        {/* Session list */}
        {!selectedSession && (
          <div style={{ direction: "ltr" }}>
            {isLoading && (
              <div style={{ padding: "20px", textAlign: "center", color: "#6272a4" }}>Loading...</div>
            )}
            {!isLoading && sessions.length === 0 && (
              <div style={{ padding: "40px 20px", textAlign: "center", color: "#6272a4" }}>
                <p>{searchQuery ? "No conversations match your search." : "No conversation history yet."}</p>
                <p style={{ fontSize: "0.85rem" }}>Conversations are saved automatically as you chat.</p>
              </div>
            )}
            {!isLoading && Object.entries(groups).map(([label, group]) => (
              <div key={label}>
                <div style={{ padding: "6px 16px 4px", fontSize: "0.75rem", color: "#6272a4", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  {label}
                </div>
                {group.map((session) => (
                  <div
                    key={session.session_id}
                    onClick={() => openSession(session)}
                    style={{
                      padding: "10px 20px 10px 16px",
                      cursor: "pointer",
                      borderBottom: "1px solid rgba(68,71,90,0.4)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      transition: "background 0.15s",
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = "rgba(68,71,90,0.3)"}
                    onMouseOut={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ color: "#f8f8f2", fontSize: "0.9rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {session.title || "Untitled conversation"}
                      </div>
                      <div style={{ color: "#6272a4", fontSize: "0.75rem", marginTop: "3px" }}>
                        {session.message_count} messages
                        {session.model_backend && ` · ${session.model_backend}`}
                        {" · "}{new Date(session.last_active).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </div>
                    </div>
                    <button
                      onClick={(e) => deleteSession(e, session.session_id)}
                      title="Delete conversation"
                      style={{ background: "none", border: "none", color: "#6272a4", cursor: "pointer", fontSize: "1rem", padding: "0 0 0 8px", flexShrink: 0 }}
                      onMouseOver={(e) => e.target.style.color = "#ff5555"}
                      onMouseOut={(e) => e.target.style.color = "#6272a4"}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* Message view */}
        {selectedSession && (
          <div style={{ padding: "12px 16px", direction: "ltr" }}>
            {isLoadingMessages && (
              <div style={{ textAlign: "center", color: "#6272a4", padding: "20px" }}>Loading...</div>
            )}
            {!isLoadingMessages && messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  marginBottom: "12px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <div style={{
                  maxWidth: "85%",
                  padding: "10px 14px",
                  borderRadius: msg.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                  background: msg.role === "user" ? "#6272a4" : "#282a36",
                  border: msg.role === "assistant" ? "1px solid #44475a" : "none",
                  color: "#f8f8f2",
                  fontSize: "0.875rem",
                  lineHeight: "1.5",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}>
                  {msg.content}
                </div>
                <div style={{ fontSize: "0.7rem", color: "#44475a", marginTop: "3px", padding: "0 4px" }}>
                  {new Date(msg.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
