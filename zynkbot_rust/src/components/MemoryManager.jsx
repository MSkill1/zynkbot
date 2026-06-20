import React, { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import { invoke } from '@tauri-apps/api/core';
import MemoryManagerModal from "./MemoryManagerModal";
import "../styles/MemoryManager.css";

const MemoryManager = forwardRef(({ user_id, apiBaseUrl, containmentMode }, ref) => {
  const isHipaaMode = containmentMode === 'hipaa';
  const [showModal, setShowModal] = useState(false);
  const [memories, setMemories] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchMemories = useCallback(async () => {
    setIsLoading(true);
    try {
      console.log('[MemoryManager] Fetching memories for user_id:', user_id);

      // Use Rust backend via Tauri invoke
      const result = await invoke('list_memories', {
        userId: user_id,
        sessionId: null,
        namespace: null  // Get all namespaces
      });

      console.log('[MemoryManager] Raw result from Tauri:', result);
      console.log('[MemoryManager] Result length:', result?.length);

      // Filter out system memories and get only recent 8 user memories
      const userMemories = (result || [])
        .filter(mem => (mem.user_id || '').toLowerCase() !== 'system')
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .slice(0, 8);

      console.log('[MemoryManager] Filtered user memories:', userMemories.length);
      console.log('[MemoryManager] First memory:', userMemories[0]);

      setMemories(userMemories);
    } catch (error) {
      console.error("[MemoryManager] Failed to fetch memories:", error);
    } finally {
      setIsLoading(false);
    }
  }, [user_id]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  // Expose fetchMemories to parent component via ref
  useImperativeHandle(ref, () => ({
    refresh: fetchMemories
  }));

  if (isLoading) {
    return (
      <div className="memory-manager">
        <h3>Recent Memories</h3>
        <div className="loading-state">Loading...</div>
      </div>
    );
  }

  return (
    <>
      <div className="memory-manager">
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '12px',
          cursor: 'pointer',
          padding: '8px',
          background: 'rgba(139, 233, 253, 0.05)',
          borderRadius: '6px',
          border: '1px solid rgba(139, 233, 253, 0.2)'
        }}
        onClick={() => setIsExpanded(!isExpanded)}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.2rem', transition: 'transform 0.2s', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}>
              ▶
            </span>
            <h3 style={{ margin: 0 }}>Recent Memories {!isExpanded && `(${memories.length})`}</h3>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (!isHipaaMode) setShowModal(true);
              }}
              disabled={isHipaaMode}
              title={isHipaaMode ? 'Memory storage is disabled in HIPAA mode' : 'Open Memory Manager'}
              style={{
                padding: '6px 12px',
                borderRadius: '4px',
                border: 'none',
                background: isHipaaMode ? '#44475a' : '#bd93f9',
                color: isHipaaMode ? '#6272a4' : '#fff',
                cursor: isHipaaMode ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem',
                fontWeight: 'bold',
                opacity: isHipaaMode ? 0.6 : 1
              }}
            >
              📚 Memory Manager
            </button>
          </div>
        </div>

      {isExpanded && (
        <>
          {isHipaaMode ? (
            <div style={{
              fontSize: '0.85rem',
              color: '#6272a4',
              padding: '12px',
              background: 'rgba(98, 114, 164, 0.1)',
              borderRadius: '6px',
              border: '1px solid rgba(98, 114, 164, 0.3)',
              textAlign: 'center'
            }}>
              🔒 Personal memory storage is disabled in HIPAA mode
            </div>
          ) : (
          <>
          <div style={{
            fontSize: '0.85rem',
            color: '#8be9fd',
            marginBottom: '15px',
            padding: '10px',
            background: 'rgba(139, 233, 253, 0.1)',
            borderRadius: '6px',
            border: '1px solid rgba(139, 233, 253, 0.2)'
          }}>
            📝 Showing your 8 most recent memories. <br />
            Click "📚 Memory Manager" above for advanced search, editing, relationship viewing, and interactive graph visualization.
          </div>

          {memories.length === 0 ? (
            <div className="empty-state">
              <p>No memories yet. Start a conversation and Zynkbot will remember important details.</p>
            </div>
          ) : (
            <div className="memory-list">
              {memories.map((mem) => (
                <div key={mem.id} className="memory-item" style={{
                  background: '#282a36',
                  padding: '12px',
                  borderRadius: '6px',
                  marginBottom: '10px',
                  border: '1px solid #44475a'
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px'
                  }}>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <span style={{
                        background: '#8b5cf6',
                        color: '#fff',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: 'bold'
                      }}>
                        {(mem.namespace || 'personal').toUpperCase()}
                      </span>
                      {mem.title && (
                        <span style={{
                          color: '#f1fa8c',
                          fontSize: '0.85rem',
                          fontWeight: 'bold'
                        }}>
                          {mem.title}
                        </span>
                      )}
                    </div>
                    <span style={{
                      color: '#9aa5c4',
                      fontSize: '0.75rem'
                    }}>
                      {new Date(mem.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <div style={{
                    color: '#f8f8f2',
                    fontSize: '0.9rem',
                    lineHeight: '1.5',
                    maxHeight: '60px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {mem.content}
                  </div>
                </div>
              ))}
            </div>
          )}
          </>
          )}
        </>
      )}
      </div>

      {/* Full Memory Manager Modal */}
      <MemoryManagerModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        userId={user_id}
        onMemoriesChanged={fetchMemories}
      />
    </>
  );
});

MemoryManager.displayName = 'MemoryManager';

export default MemoryManager;
