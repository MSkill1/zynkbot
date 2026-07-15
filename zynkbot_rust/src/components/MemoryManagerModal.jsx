import React, { useState, useEffect, useCallback } from "react";
import { invoke } from '@tauri-apps/api/core';
import MemoryGraphModal from "./MemoryGraphModal";
import "../styles/MemoryManagerModal.css";

// Canonical namespace list mirrors nlp_enhancer.rs detect_namespace() patterns
const CANONICAL_NAMESPACES = [
  'personal', 'work', 'career', 'health', 'family', 'education',
  'technology', 'science', 'philosophy', 'politics', 'travel',
  'achievements', 'biography'
];

export default function MemoryManagerModal({ isOpen, onClose, userId, onMemoriesChanged }) {
  const [memories, setMemories] = useState([]);
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [relationships, setRelationships] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterNamespace, setFilterNamespace] = useState("all");
  const [namespaces, setNamespaces] = useState(CANONICAL_NAMESPACES);
  const [showGraphModal, setShowGraphModal] = useState(false);
  const [filterEventType, setFilterEventType] = useState("all");
  const [filterDateFrom, setFilterDateFrom] = useState("");
  const [filterDateTo, setFilterDateTo] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [editFormData, setEditFormData] = useState({
    title: '',
    content: '',
    namespace: 'personal'
  });
  const [selectedMemoryIds, setSelectedMemoryIds] = useState([]);
  const [selectAll, setSelectAll] = useState(false);
  const [isMobile, setIsMobile] = useState(() => typeof window !== 'undefined' && window.innerWidth <= 768);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth <= 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Fetch all memories for this user
  const fetchMemories = useCallback(async () => {
    setIsLoading(true);
    try {
      const result = await invoke('list_memories', {
        userId: userId,
        sessionId: null,
        namespace: filterNamespace === "all" ? null : filterNamespace,
        eventType: filterEventType === "all" ? null : filterEventType,
        dateFrom: filterDateFrom || null,
        dateTo: filterDateTo || null
      });
      console.log('Fetched memories:', result);
      setMemories(result);
    } catch (error) {
      console.error('Failed to fetch memories:', error);
      alert(`Error fetching memories: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [userId, filterNamespace, filterEventType, filterDateFrom, filterDateTo]);

  // Fetch all distinct namespaces
  const fetchNamespaces = useCallback(async () => {
    try {
      const result = await invoke('get_namespaces', {
        userId: userId
      });
      console.log('Fetched namespaces:', result.namespaces);
      // Extract namespace strings from objects [{namespace: "personal", count: 5}]
      const namespaceStrings = (result.namespaces || []).map(ns =>
        typeof ns === 'string' ? ns : ns.namespace
      );
      // Merge DB namespaces with canonical list, preserving order, adding any novel ones at end
      const merged = [
        ...CANONICAL_NAMESPACES,
        ...namespaceStrings.filter(ns => !CANONICAL_NAMESPACES.includes(ns))
      ];
      setNamespaces(merged);
    } catch (error) {
      console.error('Failed to fetch namespaces:', error);
      // Keep default namespaces on error
    }
  }, [userId]);

  // Fetch relationships for selected memory
  const fetchRelationships = async (memoryId) => {
    console.log('[MemoryManagerModal] fetchRelationships called for memory:', memoryId);
    try {
      console.log('[MemoryManagerModal] Invoking get_memory_links...');
      const result = await invoke('get_memory_links', {
        memoryId: memoryId
      });
      console.log('[MemoryManagerModal] Raw result from get_memory_links:', result);

      // Extract links array - now includes full memory details from backend
      const links = result.links || [];
      console.log('[MemoryManagerModal] Extracted links with embedded memory details:', links);
      console.log('[MemoryManagerModal] Number of links:', links.length);

      // Add direction field based on whether this memory is source or target
      const transformedLinks = links.map((link) => {
        const isOutgoing = link.source_memory_id === memoryId;

        return {
          ...link,
          direction: isOutgoing ? 'outgoing' : 'incoming',
          // related_memory_id and related_memory are already included from backend
        };
      });

      console.log('[MemoryManagerModal] Transformed links with direction:', transformedLinks);
      setRelationships(transformedLinks);
    } catch (error) {
      console.error('[MemoryManagerModal] ERROR fetching relationships:', error);
      setRelationships([]);
    }
  };

  // Load memories and namespaces when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchMemories();
      fetchNamespaces();
    }
  }, [isOpen, fetchMemories, fetchNamespaces]);

  // Load relationships when memory is selected
  useEffect(() => {
    if (selectedMemory) {
      fetchRelationships(selectedMemory.id);
    } else {
      setRelationships([]);
    }
  }, [selectedMemory]);

  // Exit edit mode when a new memory is selected
  useEffect(() => {
    if (isEditing) {
      setIsEditing(false);
      setEditFormData({
        title: '',
        content: '',
        namespace: 'personal'
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedMemory?.id]); // Only trigger when memory ID changes, not when isEditing changes

  // Delete memory
  const handleDelete = async (memoryId) => {
    if (!window.confirm('Are you sure you want to delete this memory?')) {
      return;
    }

    try {
      await invoke('delete_memory', { memoryId });
      // Refresh list
      await fetchMemories();
      // Clear selection if deleted memory was selected
      if (selectedMemory && selectedMemory.id === memoryId) {
        setSelectedMemory(null);
      }
      // No success alert - the memory disappearing from the list is confirmation enough

      // Trigger refresh in parent component (Recent Memories)
      if (onMemoriesChanged) {
        onMemoriesChanged();
      }
    } catch (error) {
      console.error('Failed to delete memory:', error);
      alert(`Error deleting memory: ${error}`);
    }
  };

  // Delete relationship
  const handleDeleteRelationship = async (linkId) => {
    if (!window.confirm('Are you sure you want to delete this relationship?')) {
      return;
    }

    try {
      await invoke('delete_memory_link', { linkId });
      // Refresh relationships
      if (selectedMemory) {
        await fetchRelationships(selectedMemory.id);
      }
      // No success alert - the relationship disappearing from the list is confirmation enough
    } catch (error) {
      console.error('Failed to delete relationship:', error);
      alert(`Error deleting relationship: ${error}`);
    }
  };

  // Toggle individual memory selection
  const toggleMemorySelection = (memoryId) => {
    setSelectedMemoryIds(prev => {
      if (prev.includes(memoryId)) {
        return prev.filter(id => id !== memoryId);
      } else {
        return [...prev, memoryId];
      }
    });
  };

  // Toggle select all
  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedMemoryIds([]);
      setSelectAll(false);
    } else {
      const allIds = filteredMemories.map(m => m.id);
      setSelectedMemoryIds(allIds);
      setSelectAll(true);
    }
  };

  // Bulk delete selected memories
  const handleBulkDelete = async () => {
    if (selectedMemoryIds.length === 0) {
      return;
    }

    const count = selectedMemoryIds.length;
    if (!window.confirm(`Are you sure you want to delete ${count} ${count === 1 ? 'memory' : 'memories'}?`)) {
      return;
    }

    try {
      // Delete all selected memories
      for (const memoryId of selectedMemoryIds) {
        await invoke('delete_memory', { memoryId });
      }

      // Refresh list
      await fetchMemories();

      // Clear selections
      setSelectedMemoryIds([]);
      setSelectAll(false);

      // Clear detail view if selected memory was deleted
      if (selectedMemory && selectedMemoryIds.includes(selectedMemory.id)) {
        setSelectedMemory(null);
      }

      // Trigger refresh in parent component
      if (onMemoriesChanged) {
        onMemoriesChanged();
      }
    } catch (error) {
      console.error('Failed to delete memories:', error);
      alert(`Error deleting memories: ${error}`);
    }
  };

  const handleUpdateRelationType = async (linkId, newRelationType) => {
    try {
      await invoke('update_memory_link', {
        linkId,
        relationType: newRelationType,
        strength: null,
        notes: null
      });

      // Refresh relationships
      if (selectedMemory) {
        await fetchRelationships(selectedMemory.id);
      }
    } catch (error) {
      console.error('Failed to update relationship:', error);
      alert(`Error updating relationship: ${error}`);
    }
  };

  // Start editing a memory
  const handleStartEdit = (memory) => {
    console.log('[MemoryManager] Starting edit mode for memory:', memory.id);
    setIsEditing(true);
    setEditFormData({
      title: memory.title || '',
      content: memory.content || '',
      namespace: memory.namespace || 'personal'
    });
    console.log('[MemoryManager] Edit mode enabled, form data set');
  };

  // Cancel editing
  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditFormData({
      title: '',
      content: '',
      namespace: 'personal'
    });
  };

  // Save edited memory
  const handleSaveEdit = async () => {
    if (!selectedMemory) {
      console.error('[Save] No memory selected');
      alert('No memory selected to save');
      return;
    }

    console.log('[Save] Saving memory:', selectedMemory.id);
    console.log('[Save] Edit form data:', editFormData);

    try {
      console.log('[Save] Sending update request with data:', editFormData);

      await invoke('update_memory', {
        memoryId: selectedMemory.id,
        title: editFormData.title || null,
        content: editFormData.content,
        namespace: editFormData.namespace
      });

      console.log('[Save] Update successful');

      // Update the selected memory locally with the new data
      setSelectedMemory({
        ...selectedMemory,
        title: editFormData.title || null,
        content: editFormData.content,
        namespace: editFormData.namespace
      });

      // Exit edit mode
      setIsEditing(false);

      // Refresh memories list in background
      fetchMemories();

      // Trigger refresh in parent component
      if (onMemoriesChanged) {
        onMemoriesChanged();
      }
    } catch (error) {
      console.error('[Save] Failed to update memory:', error);
      const errorMessage = typeof error === 'string' ? error : (error.message || 'Unknown error');
      alert(`Error updating memory: ${errorMessage}`);
    }
  };

  // Clear all memories for this user
  const handleClearAllMemories = async () => {
    // Double confirmation with strong warning
    const firstConfirm = window.confirm(
      '⚠️ WARNING: This will DELETE ALL memories for this user!\n\n' +
      'This includes:\n' +
      '- All memory content\n' +
      '- All relationships/links\n' +
      '- All embeddings\n\n' +
      'This action CANNOT be undone!\n\n' +
      'Are you sure you want to continue?'
    );

    if (!firstConfirm) {
      return;
    }

    // Second confirmation - make them type to confirm
    const secondConfirm = window.confirm(
      '⚠️ FINAL CONFIRMATION ⚠️\n\n' +
      'You are about to permanently delete ALL memories.\n\n' +
      'Click OK to proceed with deletion, or Cancel to abort.'
    );

    if (!secondConfirm) {
      return;
    }

    try {
      const result = await invoke('clear_all_memories', {
        userId: userId
      });

      const clearHistory = window.confirm(
        `Cleared ${result.deleted_count} memories.\n\nWould you also like to clear your conversation history for a complete fresh start?`
      );
      if (clearHistory) {
        await invoke('clear_conversation_history', { userId });
      }

      // Refresh the memory list
      await fetchMemories();
      setSelectedMemory(null);
      setRelationships([]);

      // Trigger refresh in parent component (Recent Memories)
      if (onMemoriesChanged) {
        onMemoriesChanged();
      }
    } catch (error) {
      console.error('Failed to clear memories:', error);
      alert(`Error clearing memories: ${error}`);
    }
  };

  // Filter memories by search query
  const filteredMemories = memories.filter(mem =>
    mem.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (mem.title && mem.title.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  if (!isOpen) return null;

  // ── Mobile layout ─────────────────────────────────────────────────────────
  if (isMobile) {
    return (
      <div style={{ position: 'fixed', inset: 0, background: '#282a36', zIndex: 1000, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <div style={{ padding: '12px 16px', paddingTop: 'calc(env(safe-area-inset-top, 28px) + 12px)', borderBottom: '1px solid #44475a', background: '#1e1f2e', display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
          {selectedMemory ? (
            <>
              <button
                onClick={() => { setSelectedMemory(null); setIsEditing(false); }}
                style={{ background: 'none', border: 'none', color: '#8be9fd', fontSize: '0.9rem', cursor: 'pointer', padding: '8px 4px', fontWeight: 'bold', flexShrink: 0 }}
              >← Back</button>
              <span style={{ color: '#f8f8f2', fontWeight: 'bold', fontSize: '0.9rem', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {selectedMemory.title || 'Memory Detail'}
              </span>
            </>
          ) : (
            <>
              <h2 style={{ margin: 0, color: '#50fa7b', fontSize: '1.1rem', flex: 1 }}>Memory Manager</h2>
              <button onClick={handleClearAllMemories} style={{ background: 'none', border: '1px solid #ff5555', color: '#ff5555', fontSize: '0.75rem', padding: '5px 10px', borderRadius: '4px', cursor: 'pointer', flexShrink: 0 }}>Clear All</button>
            </>
          )}
        </div>

        {/* Floating close button - bottom right, like settings panel */}
        <button
          onClick={onClose}
          style={{
            position: 'fixed', bottom: '20px', right: '20px',
            width: '56px', height: '56px', borderRadius: '50%',
            background: '#44475a', color: '#f8f8f2', border: 'none',
            fontSize: '1.5rem', cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
            zIndex: 1010, display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}
        >✕</button>

        {/* LIST VIEW */}
        {!selectedMemory && (
          <>
            {/* Filter controls */}
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #44475a', background: '#21222c', flexShrink: 0 }}>
              <input
                type="text"
                placeholder="Search memories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-input"
                style={{ width: '100%', boxSizing: 'border-box', marginBottom: '8px' }}
              />
              <div style={{ display: 'flex', gap: '8px' }}>
                <select value={filterNamespace} onChange={(e) => setFilterNamespace(e.target.value)} className="filter-select" style={{ flex: 1 }}>
                  <option value="all">All Namespaces</option>
                  {namespaces.map((ns) => (
                    <option key={ns} value={ns}>{ns.charAt(0).toUpperCase() + ns.slice(1)}</option>
                  ))}
                </select>
                <select value={filterEventType} onChange={(e) => setFilterEventType(e.target.value)} className="filter-select" style={{ flex: 1 }}>
                  <option value="all">All Events</option>
                  <option value="marriage">Marriage</option>
                  <option value="birth">Birth</option>
                  <option value="death">Death</option>
                  <option value="job_change">Job Change</option>
                  <option value="job_loss">Job Loss</option>
                  <option value="moving">Moving</option>
                  <option value="purchase">Purchase</option>
                  <option value="graduation">Graduation</option>
                  <option value="travel">Travel</option>
                  <option value="illness">Illness</option>
                  <option value="achievement">Achievement</option>
                </select>
                <button onClick={fetchMemories} style={{ padding: '8px 14px', background: '#50fa7b', color: '#282a36', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', flexShrink: 0, fontSize: '1rem' }}>🔄</button>
              </div>
            </div>

            {/* Count */}
            <div style={{ padding: '8px 16px', flexShrink: 0 }}>
              <span style={{ color: '#8be9fd', fontWeight: 'bold', fontSize: '0.9rem' }}>Memories ({filteredMemories.length})</span>
            </div>

            {/* Scrollable list */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '0 16px 16px', WebkitOverflowScrolling: 'touch' }}>
              {isLoading ? (
                <div className="loading">Loading...</div>
              ) : filteredMemories.length === 0 ? (
                <div className="empty-state">
                  {searchQuery ? 'No memories match your search' : 'No memories found'}
                </div>
              ) : (
                filteredMemories.map((mem) => (
                  <div key={mem.id} style={{ background: '#1e1f29', border: '1px solid #44475a', borderRadius: '8px', padding: '12px', marginBottom: '10px' }}>
                    {/* Badge + date */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                      <span style={{ background: '#bd93f9', color: '#fff', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 'bold' }}>
                        {mem.namespace.toUpperCase()}
                      </span>
                      <span style={{ color: '#6272a4', fontSize: '0.75rem' }}>
                        {new Date(mem.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    {/* Title + preview — tap to open detail */}
                    <div onClick={() => setSelectedMemory(mem)} style={{ cursor: 'pointer', marginBottom: '10px' }}>
                      {mem.title && (
                        <div style={{ color: '#8be9fd', fontWeight: '600', fontSize: '0.95rem', marginBottom: '4px' }}>
                          {mem.title}
                        </div>
                      )}
                      <div style={{ color: '#a0a0a0', fontSize: '0.85rem', lineHeight: '1.4' }}>
                        {mem.content.substring(0, 100)}{mem.content.length > 100 ? '…' : ''}
                      </div>
                    </div>
                    {/* Action buttons */}
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button
                        onClick={() => { setSelectedMemory(mem); handleStartEdit(mem); }}
                        style={{ flex: 1, height: '42px', background: '#8be9fd', color: '#282a36', border: 'none', borderRadius: '6px', fontWeight: 'bold', fontSize: '0.85rem', cursor: 'pointer' }}
                      >✏️ Edit</button>
                      <button
                        onClick={() => { setSelectedMemory(mem); setShowGraphModal(true); }}
                        style={{ flex: 1, height: '42px', background: '#bd93f9', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 'bold', fontSize: '0.85rem', cursor: 'pointer' }}
                      >🔗 Graph</button>
                      <button
                        onClick={() => handleDelete(mem.id)}
                        style={{ width: '42px', height: '42px', background: '#ff5555', color: '#fff', border: 'none', borderRadius: '6px', fontSize: '1.1rem', cursor: 'pointer', flexShrink: 0 }}
                      >🗑️</button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        )}

        {/* DETAIL SHEET */}
        {selectedMemory && (
          <>
            {/* Edit / Delete action bar */}
            <div style={{ padding: '10px 16px', background: '#21222c', borderBottom: '1px solid #44475a', display: 'flex', gap: '8px', flexShrink: 0 }}>
              {isEditing ? (
                <>
                  <button onClick={handleSaveEdit} className="save-button" style={{ flex: 1, height: '42px' }}>💾 Save</button>
                  <button onClick={handleCancelEdit} className="cancel-button" style={{ flex: 1, height: '42px' }}>✕ Cancel</button>
                </>
              ) : (
                <>
                  <button onClick={() => handleStartEdit(selectedMemory)} className="edit-button" style={{ flex: 1, height: '42px' }}>✏️ Edit</button>
                  <button onClick={() => handleDelete(selectedMemory.id)} className="delete-button" style={{ flex: 1, height: '42px' }}>🗑️ Delete</button>
                </>
              )}
            </div>

            {/* Scrollable detail content */}
            <div style={{ flex: 1, overflowY: 'auto', WebkitOverflowScrolling: 'touch' }}>
              <div style={{ padding: '16px' }}>

                {/* Fields: edit or view */}
                {isEditing ? (
                  <div className="memory-details-content">
                    <div className="detail-row">
                      <strong>Title:</strong>
                      <input type="text" className="edit-input" value={editFormData.title}
                        onChange={(e) => setEditFormData({ ...editFormData, title: e.target.value })}
                        placeholder="Enter title (optional)" />
                    </div>
                    <div className="detail-row">
                      <strong>Namespace:</strong>
                      <select className="edit-select" value={editFormData.namespace}
                        onChange={(e) => setEditFormData({ ...editFormData, namespace: e.target.value })}>
                        {namespaces.map((ns) => (
                          <option key={ns} value={ns}>{ns.charAt(0).toUpperCase() + ns.slice(1)}</option>
                        ))}
                      </select>
                    </div>
                    <div className="detail-section">
                      <strong>Content:</strong>
                      <textarea className="edit-textarea" value={editFormData.content}
                        onChange={(e) => setEditFormData({ ...editFormData, content: e.target.value })}
                        rows={8} />
                    </div>
                  </div>
                ) : (
                  <div className="memory-details-content">
                    <div className="detail-row">
                      <strong>Namespace:</strong> {selectedMemory.namespace}
                    </div>
                    <div className="detail-row">
                      <strong>Created:</strong> {new Date(selectedMemory.created_at).toLocaleString()}
                    </div>
                    <div className="detail-section">
                      <strong>Content:</strong>
                      <div className="content-box">{selectedMemory.content}</div>
                    </div>
                    {selectedMemory.entities_detected && selectedMemory.entities_detected.length > 0 && (
                      <div className="detail-row">
                        <strong>Entities:</strong>
                        <div className="tags-container">
                          {selectedMemory.entities_detected.map((entity, i) => (
                            <span key={i} className="tag"
                              style={{ background: '#8be9fd', color: '#282a36', fontFamily: 'monospace', fontSize: '0.85rem' }}
                              title={`${entity.label || 'ENTITY'} (${(entity.score * 100).toFixed(0)}%)`}>
                              {entity.word}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {selectedMemory.event_type && (
                      <div className="detail-row">
                        <strong>Event:</strong> {selectedMemory.event_type}
                      </div>
                    )}
                  </div>
                )}

                {/* Relationships */}
                <div style={{ marginTop: '20px', borderTop: '1px solid #44475a', paddingTop: '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ color: '#8be9fd', fontWeight: 'bold', fontSize: '0.95rem' }}>
                      Relationships ({relationships.length})
                    </span>
                    <button onClick={() => setShowGraphModal(true)} className="graph-button" style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
                      🔗 View Graph
                    </button>
                  </div>
                  {relationships.length === 0 ? (
                    <div style={{ color: '#6272a4', fontSize: '0.9rem', textAlign: 'center', padding: '16px 0' }}>No relationships yet</div>
                  ) : (
                    relationships.map((rel, i) => (
                      <div key={i} style={{ background: '#1e1f29', border: '1px solid #44475a', borderRadius: '8px', padding: '10px', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                          <span style={{ background: '#50fa7b', color: '#282a36', padding: '2px 8px', borderRadius: '4px', fontSize: '0.78rem', fontWeight: 'bold' }}>
                            {rel.relation_type}
                          </span>
                          <span style={{ color: '#f1fa8c', fontWeight: 'bold' }}>{rel.direction === 'outgoing' ? '→' : '←'}</span>
                          <button
                            onClick={() => handleDeleteRelationship(rel.id)}
                            style={{ marginLeft: 'auto', background: '#ff5555', border: 'none', color: '#fff', borderRadius: '4px', width: '26px', height: '26px', cursor: 'pointer', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}
                          >✕</button>
                        </div>
                        {rel.related_memory ? (
                          <>
                            <div
                              style={{ color: '#8be9fd', fontWeight: '600', fontSize: '0.85rem', marginBottom: '4px', cursor: 'pointer' }}
                              onClick={() => setSelectedMemory(rel.related_memory)}
                            >{rel.related_memory.title || 'Untitled'}</div>
                            <div style={{ color: '#a0a0a0', fontSize: '0.8rem', lineHeight: '1.4' }}>
                              {rel.related_memory.content.substring(0, 80)}…
                            </div>
                          </>
                        ) : (
                          <div style={{ color: '#ff5555', fontSize: '0.85rem' }}>Memory ID: {rel.related_memory_id}</div>
                        )}
                        <div style={{ color: '#6272a4', fontSize: '0.75rem', marginTop: '4px' }}>
                          {(rel.confidence * 100).toFixed(0)}% confidence
                        </div>
                      </div>
                    ))
                  )}
                </div>

              </div>
            </div>
          </>
        )}

        {/* Graph modal */}
        {showGraphModal && selectedMemory && (
          <MemoryGraphModal
            isOpen={showGraphModal}
            onClose={() => setShowGraphModal(false)}
            memoryId={selectedMemory.id}
            userId={userId}
            onNavigate={async (newMemoryId) => {
              try {
                const memory = await invoke('get_memory', { memoryId: newMemoryId });
                if (!memory) throw new Error(`Failed to fetch memory ${newMemoryId}`);
                setSelectedMemory(memory);
                setShowGraphModal(false);
                fetchRelationships(newMemoryId);
              } catch (error) {
                console.error('Failed to navigate to memory:', error);
                alert(`Error loading memory: ${error.message}`);
              }
            }}
          />
        )}
      </div>
    );
  }
  // ── End mobile layout ──────────────────────────────────────────────────────

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="memory-manager-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <h2>Memory Manager</h2>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <button
              onClick={handleClearAllMemories}
              className="delete-button"
              style={{
                backgroundColor: '#ff5555',
                padding: '8px 16px',
                fontSize: '14px'
              }}
              title="Clear all memories for this user"
            >
              🗑️ Clear All Memories
            </button>
            <button onClick={onClose} className="close-button">✕</button>
          </div>
        </div>

        {/* Controls */}
        <div className="modal-controls">
          <input
            type="text"
            placeholder="Search memories..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          <select
            value={filterNamespace}
            onChange={(e) => setFilterNamespace(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Namespaces</option>
            {namespaces.map((ns) => (
              <option key={ns} value={ns}>
                {ns.charAt(0).toUpperCase() + ns.slice(1)}
              </option>
            ))}
          </select>
          <select
            value={filterEventType}
            onChange={(e) => setFilterEventType(e.target.value)}
            className="filter-select"
            title="Filter by event type"
          >
            <option value="all">All Events</option>
            <option value="marriage">Marriage</option>
            <option value="birth">Birth</option>
            <option value="death">Death</option>
            <option value="job_change">Job Change</option>
            <option value="job_loss">Job Loss</option>
            <option value="moving">Moving</option>
            <option value="purchase">Purchase</option>
            <option value="graduation">Graduation</option>
            <option value="travel">Travel</option>
            <option value="illness">Illness</option>
            <option value="achievement">Achievement</option>
          </select>
          <input
            type="date"
            value={filterDateFrom}
            onChange={(e) => setFilterDateFrom(e.target.value)}
            className="date-input"
            placeholder="From date"
            title="Filter from date"
          />
          <input
            type="date"
            value={filterDateTo}
            onChange={(e) => setFilterDateTo(e.target.value)}
            className="date-input"
            placeholder="To date"
            title="Filter to date"
          />
          <button onClick={fetchMemories} className="refresh-button">
            🔄 Refresh
          </button>
        </div>

        {/* Main Content - 3 Columns */}
        <div className="modal-content">
          {/* Left: Memory List */}
          <div className="memory-list-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3>Memories ({filteredMemories.length})</h3>
              {filteredMemories.length > 0 && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.9rem', cursor: 'pointer' }}>
                    <input
                      type="checkbox"
                      checked={selectAll}
                      onChange={handleSelectAll}
                      style={{ cursor: 'pointer' }}
                    />
                    Select All
                  </label>
                  {selectedMemoryIds.length > 0 && (
                    <button
                      onClick={handleBulkDelete}
                      className="delete-button"
                      style={{ padding: '4px 10px', fontSize: '0.85rem' }}
                    >
                      🗑️ Delete ({selectedMemoryIds.length})
                    </button>
                  )}
                </div>
              )}
            </div>
            {isLoading ? (
              <div className="loading">Loading...</div>
            ) : filteredMemories.length === 0 ? (
              <div className="empty-state">
                {searchQuery ? 'No memories match your search' : 'No memories found'}
              </div>
            ) : (
              <div className="memory-list-scroll">
                {filteredMemories.map((mem) => (
                  <div
                    key={mem.id}
                    className={`memory-list-item ${selectedMemory?.id === mem.id ? 'selected' : ''}`}
                  >
                    <div className="memory-list-item-header">
                      <label
                        style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginRight: '8px' }}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="checkbox"
                          checked={selectedMemoryIds.includes(mem.id)}
                          onChange={() => toggleMemorySelection(mem.id)}
                          style={{ cursor: 'pointer', marginRight: '8px' }}
                        />
                      </label>
                      <div style={{ flex: 1 }} onClick={() => setSelectedMemory(mem)}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                            <span className="memory-namespace-badge">
                              {mem.namespace.toUpperCase()}
                            </span>
                          </div>
                          <span className="memory-date">
                            {new Date(mem.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div onClick={() => setSelectedMemory(mem)} style={{ cursor: 'pointer' }}>
                      {mem.title && (
                        <div className="memory-list-item-title">
                          {mem.title.length > 60 ? mem.title.substring(0, 60) + '...' : mem.title}
                        </div>
                      )}
                      <div className="memory-list-item-content">
                        {mem.content.substring(0, mem.title ? 80 : 100)}
                        {mem.content.length > (mem.title ? 80 : 100) && '...'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Center: Memory Details */}
          <div className="memory-details-panel">
            {selectedMemory ? (
              <>
                <div className="memory-details-header">
                  <h3>{isEditing ? 'Edit Memory' : 'Memory Details'}</h3>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {isEditing ? (
                      <>
                        <button
                          onClick={handleSaveEdit}
                          className="save-button"
                        >
                          💾 Save
                        </button>
                        <button
                          onClick={handleCancelEdit}
                          className="cancel-button"
                        >
                          ✕ Cancel
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => handleStartEdit(selectedMemory)}
                          className="edit-button"
                        >
                          ✏️ Edit
                        </button>
                        <button
                          onClick={() => handleDelete(selectedMemory.id)}
                          className="delete-button"
                        >
                          🗑️ Delete
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {isEditing ? (
                  <div className="memory-details-content">
                    <div className="detail-row">
                      <strong>ID:</strong> {selectedMemory.id}
                    </div>
                    <div className="detail-row">
                      <strong>Title:</strong>
                      <input
                        type="text"
                        className="edit-input"
                        value={editFormData.title}
                        onChange={(e) => setEditFormData({ ...editFormData, title: e.target.value })}
                        placeholder="Enter title (optional)"
                      />
                    </div>
                    <div className="detail-row">
                      <strong>Namespace:</strong>
                      <select
                        className="edit-select"
                        value={editFormData.namespace}
                        onChange={(e) => setEditFormData({ ...editFormData, namespace: e.target.value })}
                      >
                        {namespaces.map((ns) => (
                          <option key={ns} value={ns}>
                            {ns.charAt(0).toUpperCase() + ns.slice(1)}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="detail-section">
                      <strong>Content:</strong>
                      <textarea
                        className="edit-textarea"
                        value={editFormData.content}
                        onChange={(e) => setEditFormData({ ...editFormData, content: e.target.value })}
                        rows={8}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="memory-details-content">
                    <div className="detail-row">
                      <strong>ID:</strong> {selectedMemory.id}
                    </div>
                    {selectedMemory.title && (
                      <div className="detail-row">
                        <strong>Title:</strong> {selectedMemory.title}
                      </div>
                    )}
                    <div className="detail-row">
                      <strong>Namespace:</strong> {selectedMemory.namespace}
                    </div>
                    <div className="detail-row">
                      <strong>Created:</strong>{' '}
                      {new Date(selectedMemory.created_at).toLocaleString()}
                    </div>
                    <div className="detail-row">
                      <strong>Source:</strong> {selectedMemory.source_type || 'N/A'}
                    </div>
                    {selectedMemory.session_id && (
                      <div className="detail-row">
                        <strong>Session:</strong>{' '}
                        {selectedMemory.session_id.substring(0, 8)}...
                      </div>
                    )}
                    <div className="detail-section">
                      <strong>Content:</strong>
                      <div className="content-box">
                        {selectedMemory.content}
                      </div>
                    </div>
                    {selectedMemory.original_text && (
                      <div className="detail-section">
                        <strong>Original:</strong>
                        <div className="content-box" style={{ color: '#9aa5c4', fontStyle: 'italic' }}>
                          {selectedMemory.original_text}
                        </div>
                      </div>
                    )}
                    {selectedMemory.entities_detected && selectedMemory.entities_detected.length > 0 && (
                      <div className="detail-row">
                        <strong>Entities:</strong>
                        <div className="tags-container">
                          {selectedMemory.entities_detected.map((entity, i) => (
                            <span
                              key={i}
                              className="tag"
                              style={{
                                background: '#8be9fd',
                                color: '#282a36',
                                fontFamily: 'monospace',
                                fontSize: '0.85rem'
                              }}
                              title={`${entity.label || 'ENTITY'} (confidence: ${(entity.score * 100).toFixed(0)}%)`}
                            >
                              {entity.word}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {selectedMemory.event_type && (
                      <div className="detail-row">
                        <strong>Event Type:</strong> {selectedMemory.event_type}
                        {selectedMemory.event_date && (() => {
                          const d = new Date(selectedMemory.event_date);
                          return !isNaN(d.getTime()) ? (
                            <span style={{ marginLeft: '8px', color: '#9aa5c4' }}>
                              ({d.toLocaleDateString()})
                            </span>
                          ) : null;
                        })()}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-state">
                Select a memory to view details
              </div>
            )}
          </div>

          {/* Right: Relationships */}
          <div className="relationships-panel">
            {selectedMemory ? (
              <>
                <div className="relationships-header">
                  <h3>Relationships ({relationships.length})</h3>
                  <button
                    onClick={() => setShowGraphModal(true)}
                    className="graph-button"
                    disabled={!selectedMemory}
                    title={selectedMemory ? "View interactive knowledge graph" : "Select a memory first"}
                  >
                    🔗 View Graph
                  </button>
                </div>
                {relationships.length === 0 ? (
                  <div className="empty-state">
                    No relationships yet
                  </div>
                ) : (
                  <div className="relationships-list">
                    {relationships.map((rel, i) => (
                      <div key={i} className="relationship-item">
                        <div className="relationship-header">
                          <select
                            className="relationship-type-select"
                            value={rel.relation_type}
                            onChange={(e) => handleUpdateRelationType(rel.id, e.target.value)}
                            title="Change relationship type"
                          >
                            <option value="supports">supports</option>
                            <option value="contradicts">contradicts</option>
                            <option value="elaborates">elaborates</option>
                            <option value="reminds_of">reminds_of</option>
                            <option value="caused_by">caused_by</option>
                            <option value="quotes">quotes</option>
                            <option value="resolves">resolves</option>
                          </select>
                          <span className="relationship-direction">
                            {rel.direction === 'outgoing' ? '→' : '←'}
                          </span>
                          <button
                            onClick={() => handleDeleteRelationship(rel.id)}
                            className="delete-rel-button"
                            title="Delete this relationship"
                          >
                            ✕
                          </button>
                        </div>
                        <div className="relationship-details">
                          {rel.related_memory ? (
                            <>
                              <div style={{ marginBottom: '8px', borderBottom: '1px solid #44475a', paddingBottom: '8px' }}>
                                <strong
                                  style={{
                                    color: '#8be9fd',
                                    cursor: 'pointer',
                                    textDecoration: 'underline'
                                  }}
                                  onClick={() => {
                                    // Navigate to the related memory
                                    setSelectedMemory(rel.related_memory);
                                  }}
                                  title="Click to view this memory"
                                >
                                  {rel.related_memory.title || 'Untitled'}
                                </strong>
                                <div style={{
                                  fontSize: '0.85rem',
                                  color: '#f8f8f2',
                                  marginTop: '4px',
                                  maxHeight: '60px',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis'
                                }}>
                                  {rel.related_memory.content.substring(0, 100)}...
                                </div>
                              </div>
                            </>
                          ) : (
                            <div style={{ color: '#ff5555', fontSize: '0.85rem' }}>
                              <strong>Failed to load memory</strong> (ID: {rel.related_memory_id})
                            </div>
                          )}
                          <div>
                            <strong>Confidence:</strong> {(rel.confidence * 100).toFixed(0)}%
                          </div>
                          <div>
                            <strong>Created:</strong>{' '}
                            {new Date(rel.created_at).toLocaleDateString()}
                          </div>
                          {rel.notes && (
                            <div>
                              <strong>Notes:</strong> {rel.notes}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="empty-state">
                Select a memory to view relationships
              </div>
            )}
          </div>
        </div>

        {/* Graph Modal */}
        {showGraphModal && selectedMemory && (
          <MemoryGraphModal
            isOpen={showGraphModal}
            onClose={() => setShowGraphModal(false)}
            memoryId={selectedMemory.id}
            userId={userId}
            onNavigate={async (newMemoryId) => {
              // Fetch the memory by ID from API (not just cached list)
              // This ensures we can navigate to ANY memory in the graph, not just the 8 recent ones
              try {
                const memory = await invoke('get_memory', { memoryId: newMemoryId });
                if (!memory) {
                  throw new Error(`Failed to fetch memory ${newMemoryId}`);
                }

                setSelectedMemory(memory);
                setShowGraphModal(false);
                // Fetch relationships for the new memory
                fetchRelationships(newMemoryId);
              } catch (error) {
                console.error('Failed to navigate to memory:', error);
                alert(`Error loading memory: ${error.message}`);
              }
            }}
          />
        )}
      </div>
    </div>
  );
}
