import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import '../styles/KnowledgeBaseManager.css';

export default function KnowledgeBaseManager({ isOpen, onClose, userId }) {
  const [documents, setDocuments] = useState([]);
  const [availableFiles, setAvailableFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexingFile, setIndexingFile] = useState('');
  const [indexingProgress, setIndexingProgress] = useState(null);
  const [kbFolderPath, setKbFolderPath] = useState('');
  const [expandedFolders, setExpandedFolders] = useState(new Set(['_root']));

  // Build folder tree structure from flat document list
  const buildFolderTree = (docs) => {
    const tree = {};

    docs.forEach(doc => {
      // Strip KB folder prefix from absolute paths to show relative paths
      // e.g., /home/matt/.config/Zynkbot/KB/[uuid]/example.txt -> example.txt
      // Keep relative paths unchanged (e.g., snap_ins/therapist/john_doe/file.txt)
      let relativePath = doc.file_path;
      if (kbFolderPath && doc.file_path.startsWith(kbFolderPath)) {
        relativePath = doc.file_path.substring(kbFolderPath.length);
        // Remove leading slash if present
        if (relativePath.startsWith('/')) {
          relativePath = relativePath.substring(1);
        }
      }

      const pathParts = relativePath.split('/');
      const folderPath = pathParts.slice(0, -1);

      if (folderPath.length === 0) {
        // Root level file
        if (!tree._root) {
          tree._root = { type: 'folder', name: 'Documents', path: '_root', children: {}, files: [] };
        }
        tree._root.files.push(doc);
      } else {
        // Navigate/create folder structure
        let current = tree;
        let currentPath = '';

        folderPath.forEach((folder, index) => {
          currentPath = currentPath ? `${currentPath}/${folder}` : folder;

          if (!current[folder]) {
            current[folder] = {
              type: 'folder',
              name: folder,
              path: currentPath,
              children: {},
              files: []
            };
          }

          // For last folder, add the file
          if (index === folderPath.length - 1) {
            current[folder].files.push(doc);
          }

          current = current[folder].children;
        });
      }
    });

    return tree;
  };

  const toggleFolder = (folderPath) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderPath)) {
        newSet.delete(folderPath);
      } else {
        newSet.add(folderPath);
      }
      return newSet;
    });
  };

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, userId]);

  useEffect(() => {
    let unlisten;
    listen('kb:indexing_progress', (event) => {
      const { current, total } = event.payload;
      setIndexingProgress({ current, total });
    }).then(fn => { unlisten = fn; });
    return () => { if (unlisten) unlisten(); };
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    try {
      // Get KB folder path
      const path = await invoke('get_kb_folder_path', { userId });
      setKbFolderPath(path);

      // Load indexed documents
      const freshDocs = await loadIndexedDocuments();

      // Scan for available files (pass fresh docs to avoid stale state)
      await scanKBFolder(path, freshDocs);
    } catch (error) {
      console.error('[KB Manager] Error loading data:', error);
      alert(`Error loading data: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  const loadIndexedDocuments = async () => {
    try {
      const docs = await invoke('list_kb_documents', { userId });
      setDocuments(docs);
      console.log('[KB Manager] Loaded', docs.length, 'indexed documents');
      return docs; // Return fresh docs for immediate use
    } catch (error) {
      console.error('[KB Manager] Error loading documents:', error);
      throw error;
    }
  };

  const scanKBFolder = async (folderPath, indexedDocs = null) => {
    setIsScanning(true);
    try {
      const files = await invoke('scan_knowledge_base', { directory: folderPath });

      // Filter out already indexed files (use passed docs or state)
      const docsToUse = indexedDocs !== null ? indexedDocs : documents;
      const indexedPaths = new Set(docsToUse.map(doc => doc.file_path));
      const unindexed = files.filter(file => !indexedPaths.has(file.path));

      setAvailableFiles(unindexed);
      console.log('[KB Manager] Found', unindexed.length, 'unindexed files');
    } catch (error) {
      console.error('[KB Manager] Error scanning folder:', error);
    } finally {
      setIsScanning(false);
    }
  };

  const handleIndexFile = async (filePath, fileName) => {
    setIsIndexing(true);
    setIndexingFile(fileName);
    setIndexingProgress(null);

    try {
      console.log('[KB Manager] Indexing:', fileName);

      await invoke('index_kb_document', {
        userId,
        filePath
      });

      // Refresh lists
      const freshDocs = await loadIndexedDocuments();
      await scanKBFolder(kbFolderPath, freshDocs);

      alert(`Successfully indexed: ${fileName}`);
    } catch (error) {
      console.error('[KB Manager] Error indexing file:', error);
      alert(`Failed to index ${fileName}: ${error}`);
    } finally {
      setIsIndexing(false);
      setIndexingFile('');
      setIndexingProgress(null);
    }
  };

  const handleIndexAll = async () => {
    if (availableFiles.length === 0) {
      alert('No files to index');
      return;
    }

    const confirmed = window.confirm(
      `Index all ${availableFiles.length} files?\n\n` +
      `⏱️ Embedding runs on CPU without GPU acceleration.\n` +
      `Large files (100KB+) can take 5–15 minutes each to embed.\n` +
      `Multiple large files may take 30+ minutes total.\n\n` +
      `Do not close the app while indexing is in progress.`
    );

    if (!confirmed) return;

    setIsIndexing(true);

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < availableFiles.length; i++) {
      const file = availableFiles[i];
      setIndexingFile(file.name);
      setIndexingProgress({ current: i + 1, total: availableFiles.length });

      try {
        await invoke('index_kb_document', {
          userId,
          filePath: file.path
        });
        successCount++;
      } catch (error) {
        console.error(`[KB Manager] Error indexing ${file.name}:`, error);
        errorCount++;
      }
    }

    // Refresh lists
    const freshDocs = await loadIndexedDocuments();
    await scanKBFolder(kbFolderPath, freshDocs);

    setIsIndexing(false);
    setIndexingFile('');
    setIndexingProgress(null);

    alert(
      `Indexing complete!\n\n` +
      `Success: ${successCount}\n` +
      `Errors: ${errorCount}`
    );
  };

  const handleRemoveDocument = async (doc) => {
    const confirmed = window.confirm(
      `Remove "${doc.file_name}" from index?\n\n` +
      `This will delete all ${doc.chunk_count} chunks.\n` +
      `The file itself will not be deleted.`
    );

    if (!confirmed) return;

    try {
      await invoke('remove_kb_document', {
        userId,
        filePath: doc.file_path
      });

      // Refresh lists
      const freshDocs = await loadIndexedDocuments();
      await scanKBFolder(kbFolderPath, freshDocs);

      console.log('[KB Manager] Removed:', doc.file_name);
    } catch (error) {
      console.error('[KB Manager] Error removing document:', error);
      alert(`Failed to remove document: ${error}`);
    }
  };

  const handleReindexDocument = async (doc) => {
    const confirmed = window.confirm(
      `Re-index "${doc.file_name}"?\n\n` +
      `This will regenerate all embeddings for this document.`
    );

    if (!confirmed) return;

    await handleIndexFile(doc.file_path, doc.file_name);
  };

  // Render folder tree recursively
  const renderFolderTree = (treeNode, level = 0) => {
    if (!treeNode) return null;

    const isExpanded = expandedFolders.has(treeNode.path);
    const hasChildren = Object.keys(treeNode.children || {}).length > 0;
    const hasFiles = (treeNode.files || []).length > 0;

    return (
      <div key={treeNode.path} style={{ marginLeft: level > 0 ? '20px' : '0' }}>
        {/* Folder Header */}
        {treeNode.name && (
          <div
            onClick={() => toggleFolder(treeNode.path)}
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '8px 12px',
              cursor: 'pointer',
              background: '#44475a',
              borderRadius: '6px',
              marginBottom: '8px',
              fontWeight: 'bold',
              color: treeNode.name === 'snap_ins' ? '#ff79c6' : '#8be9fd'
            }}
          >
            <span style={{ marginRight: '8px' }}>
              {hasChildren || hasFiles ? (isExpanded ? '📂' : '📁') : '📄'}
            </span>
            <span>{treeNode.name}</span>
            <span style={{ marginLeft: '8px', fontSize: '0.9rem' }}>
              {(hasChildren || hasFiles) && (isExpanded ? '▼' : '▶')}
            </span>
          </div>
        )}

        {/* Folder Contents (when expanded) */}
        {isExpanded && (
          <div>
            {/* Child Folders */}
            {Object.values(treeNode.children || {}).map(child =>
              renderFolderTree(child, level + 1)
            )}

            {/* Files in this folder */}
            {(treeNode.files || []).map((doc) => (
              <div key={doc.id} className="kb-document-item" style={{ marginBottom: '8px' }}>
                <div className="kb-doc-main">
                  <div className="kb-doc-icon">📄</div>
                  <div className="kb-doc-info">
                    <div className="kb-doc-name">{doc.file_name}</div>
                    <div className="kb-doc-meta">
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span className="kb-separator">•</span>
                      <span>{doc.chunk_count} chunks</span>
                      <span className="kb-separator">•</span>
                      <span>{formatDate(doc.indexed_at)}</span>
                    </div>
                    {doc.status !== 'indexed' && (
                      <div className={`kb-doc-status status-${doc.status}`}>
                        {doc.status}
                      </div>
                    )}
                  </div>
                </div>
                <div className="kb-doc-actions">
                  <button
                    onClick={() => handleReindexDocument(doc)}
                    className="kb-action-btn kb-reindex-btn"
                    title="Re-index document"
                  >
                    🔄
                  </button>
                  <button
                    onClick={() => handleRemoveDocument(doc)}
                    className="kb-action-btn kb-delete-btn"
                    title="Remove from index"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  if (!isOpen) return null;

  return (
    <div className="kb-manager-overlay" onClick={onClose}>
      <div className="kb-manager-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="kb-manager-header">
          <h2>Knowledge Base Document Manager</h2>
          <button className="kb-close-button" onClick={onClose}>×</button>
        </div>

        {/* Loading State */}
        {isLoading ? (
          <div className="kb-loading">
            <div className="loading-spinner"></div>
            <p>Loading documents...</p>
          </div>
        ) : (
          <>
            {/* Indexing Progress */}
            {isIndexing && (
              <div className="kb-indexing-progress" style={{ flexShrink: 0 }}>
                <div className="progress-header">
                  <span>Indexing: {indexingFile}</span>
                  {indexingProgress
                    ? <span>{indexingProgress.current} / {indexingProgress.total} chunks</span>
                    : <span>Preparing…</span>
                  }
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{
                      width: indexingProgress && indexingProgress.total > 0
                        ? `${(indexingProgress.current / indexingProgress.total) * 100}%`
                        : '0%'
                    }}
                  ></div>
                </div>
                <div className="progress-note">
                  ⏱️ Embedding runs on CPU — large files (100KB+) can take 5–15 minutes each. Do not close the app.
                </div>
              </div>
            )}

            {/* Sections body — flex so both sections share available height */}
            <div className="kb-body">

            {/* Indexed Documents Section */}
            <div className="kb-section">
              <div className="kb-section-header">
                <h3>Indexed Documents ({documents.length})</h3>
              </div>

              {documents.length === 0 ? (
                <div className="kb-empty-state">
                  <p>No documents indexed yet.</p>
                  <p className="kb-hint">Add files to your KB folder and index them below.</p>
                </div>
              ) : (
                <div className="kb-document-list">
                  {Object.values(buildFolderTree(documents)).map(folder =>
                    renderFolderTree(folder)
                  )}
                </div>
              )}
            </div>

            {/* Available Files Section */}
            <div className="kb-section">
              <div className="kb-section-header">
                <h3>Available Files ({availableFiles.length})</h3>
                <div className="kb-section-actions">
                  <button
                    onClick={loadData}
                    disabled={isScanning || isIndexing}
                    className="kb-scan-button"
                  >
                    {isScanning ? 'Scanning...' : '🔄 Refresh'}
                  </button>
                  {availableFiles.length > 0 && (
                    <button
                      onClick={handleIndexAll}
                      disabled={isIndexing}
                      className="kb-index-all-button"
                    >
                      Index All ({availableFiles.length})
                    </button>
                  )}
                </div>
              </div>

              {availableFiles.length === 0 ? (
                <div className="kb-empty-state">
                  <p>{isScanning ? 'Scanning...' : 'No new files found.'}</p>
                  <p className="kb-hint">
                    {isScanning ? 'Please wait...' : 'All files in the KB folder are already indexed.'}
                  </p>
                </div>
              ) : (
                <div className="kb-file-list">
                  {availableFiles.map((file, index) => (
                    <div key={index} className="kb-file-item">
                      <div className="kb-file-main">
                        <div className="kb-file-icon">📄</div>
                        <div className="kb-file-info">
                          <div className="kb-file-name">{file.name}</div>
                          <div className="kb-file-meta">
                            <span>{formatFileSize(file.size)}</span>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => handleIndexFile(file.path, file.name)}
                        disabled={isIndexing}
                        className="kb-index-button"
                      >
                        Index
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            </div> {/* end kb-body */}

            {/* Footer Info */}
            <div className="kb-footer-info">
              <p className="kb-folder-path">
                <strong>KB Folder:</strong> {kbFolderPath}
              </p>
              <p className="kb-help-text">
                Place documents in the KB folder and click "Index" to make them searchable.
                Supported formats: TXT, MD, JSON, and code files. (PDF support coming soon)
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
