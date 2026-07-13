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
  // New NLP filter states
  const [filterEventType, setFilterEventType] = useState("all");
  const [filterDateFrom, setFilterDateFrom] = useState("");
  const [filterDateTo, setFilterDateTo] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [editFormData, setEditFormData] = useState({
    title: '',
    content: '',
    namespace: 'personal'
  });
  const [selectedMemoryIds, setSelectedMemoryIds] = useState([]); // For bulk delete
  const [selectAll, setSelectAll] = useState(false);

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
