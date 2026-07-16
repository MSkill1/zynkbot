import React, { useState, useEffect, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { save } from '@tauri-apps/plugin-dialog';
import '../styles/ZynkFileBrowserModal.css';

/** Format seconds as "1h 23m 45s", "2m 15s", or "8s". */
function formatDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h}h ${mm}m ${s}s`;
}

/** Format a bytes/second number as "1.4 MB/s" or "850 KB/s". */
function formatRate(bytesPerSecond) {
  if (!isFinite(bytesPerSecond) || bytesPerSecond <= 0) return '—';
  if (bytesPerSecond < 1024) return `${bytesPerSecond.toFixed(0)} B/s`;
  if (bytesPerSecond < 1048576) return `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
  return `${(bytesPerSecond / 1048576).toFixed(1)} MB/s`;
}

function buildFolderTree(files) {
  const root = { children: {}, files: [] };
  files.forEach(file => {
    const parts = file.relative_path.replace(/\\/g, '/').split('/');
    const dirs = parts.slice(0, -1);
    let node = root;
    for (const dir of dirs) {
      if (!node.children[dir]) {
        node.children[dir] = { name: dir, children: {}, files: [] };
      }
      node = node.children[dir];
    }
    node.files.push(file);
  });
  return root;
}

function fileIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  if (ext === 'pdf') return '📑';
  if (['doc', 'docx', 'txt', 'md', 'rtf'].includes(ext)) return '📄';
  if (['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp'].includes(ext)) return '🖼️';
  if (['mp3', 'wav', 'ogg', 'flac'].includes(ext)) return '🎵';
  if (['mp4', 'avi', 'mov', 'mkv'].includes(ext)) return '🎬';
  if (['xls', 'xlsx', 'csv'].includes(ext)) return '📊';
  if (['zip', 'tar', 'gz', '7z'].includes(ext)) return '📦';
  if (['rs', 'js', 'jsx', 'ts', 'tsx', 'py', 'go', 'java', 'c', 'cpp'].includes(ext)) return '💻';
  if (['json', 'yaml', 'yml', 'toml', 'xml'].includes(ext)) return '⚙️';
  return '📄';
}

const KB_TYPES = new Set(['pdf', 'txt', 'md', 'csv', 'json', 'yaml', 'yml', 'xml', 'html', 'htm', 'toml', 'rs', 'js', 'jsx', 'ts', 'tsx', 'py', 'java', 'cpp', 'c', 'h', 'css', 'log']);
const isKBCompatible = (filename) => KB_TYPES.has(filename.split('.').pop().toLowerCase());

function formatBytes(bytes) {
  if (!bytes) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

function FolderNode({ node, name, level, expandedFolders, onToggle, onKB, onSave, shareId, deviceId }) {
  const isExpanded = expandedFolders.has(name);
  const childKeys = Object.keys(node.children);

  return (
    <div className="zfb-folder-node" style={{ paddingLeft: level > 0 ? '16px' : '0' }}>
      <div className="zfb-folder-header" onClick={() => onToggle(name)}>
        <span>{isExpanded ? '📂' : '📁'}</span>
        <span className="zfb-folder-name">{node.name}</span>
      </div>

      {isExpanded && (
        <div className="zfb-folder-contents">
          {childKeys.map(key => (
            <FolderNode
              key={key}
              node={node.children[key]}
              name={`${name}/${key}`}
              level={level + 1}
              expandedFolders={expandedFolders}
              onToggle={onToggle}
              onKB={onKB}
              onSave={onSave}
              shareId={shareId}
              deviceId={deviceId}
            />
          ))}
          {node.files.map((file, idx) => {
            const filename = file.relative_path.replace(/\\/g, '/').split('/').pop();
            return (
              <div key={idx} className="zfb-file-row">
                <span className="zfb-file-icon">{fileIcon(filename)}</span>
                <span className="zfb-file-name" title={file.relative_path}>{filename}</span>
                {file.file_size > 0 && (
                  <span className="zfb-file-size">{formatBytes(file.file_size)}</span>
                )}
                <div className="zfb-file-actions">
                  {isKBCompatible(filename) && (
                    <button
                      className="zfb-btn-kb"
                      onClick={() => onKB(shareId, file.relative_path, deviceId)}
                      title="Download to Knowledge Base"
                    >
                      → KB
                    </button>
                  )}
                  <button
                    className="zfb-btn-save"
                    onClick={() => onSave(shareId, file.relative_path, deviceId)}
                    title="Save to custom location"
                  >
                    Save…
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function ZynkFileBrowserModal({ isOpen, onClose, shareId, deviceId, shareName, userId }) {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [expandedFolders, setExpandedFolders] = useState(new Set());

  // Active downloads keyed by relative_path. Each entry holds:
  //   { bytesWritten, totalBytes, startedAt (ms), status: 'transferring'|'complete' }
  // Populated by listening for zynklink:download:{start,progress,complete} events
  // emitted by the Rust streaming download in src-tauri/src/lib.rs.
  const [downloads, setDownloads] = useState({});

  // A ticking "now" so the progress panel re-renders ~every second to update
  // speed/ETA without needing a new event from the backend.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (Object.keys(downloads).length === 0) return undefined;
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [downloads]);

  const loadFiles = useCallback(async () => {
    if (!shareId || !deviceId) return;
    setLoading(true);
    setMessage('');
    try {
      const data = await invoke('list_shared_files', { shareId, deviceId });
      setFiles(data.files || []);
    } catch (err) {
      setMessage('✗ Failed to load files: ' + err);
    } finally {
      setLoading(false);
    }
  }, [shareId, deviceId]);

  useEffect(() => {
    if (isOpen) {
      setFiles([]);
      setExpandedFolders(new Set());
      loadFiles();
    }
  }, [isOpen, loadFiles]);

  // Subscribe to Tauri download events from the streaming ZynkLink download
  // backend (lib.rs::download_to_custom_location / download_to_knowledge_base).
  // Keep listeners alive for the entire modal session so downloads kicked off
  // late in the session are still tracked.
  useEffect(() => {
    if (!isOpen) return undefined;

    const unlisteners = [];

    listen('zynklink:download:start', (event) => {
      const { relative_path, total_bytes } = event.payload || {};
      if (!relative_path) return;
      setDownloads(prev => ({
        ...prev,
        [relative_path]: {
          bytesWritten: 0,
          totalBytes: total_bytes ?? null,
          startedAt: Date.now(),
          status: 'transferring',
        },
      }));
    }).then(fn => unlisteners.push(fn));

    listen('zynklink:download:progress', (event) => {
      const { relative_path, bytes_written, total_bytes } = event.payload || {};
      if (!relative_path) return;
      setDownloads(prev => {
        const existing = prev[relative_path];
        if (!existing) {
          // Progress event arrived before start — synthesize an entry so the
          // user still sees progress.
          return {
            ...prev,
            [relative_path]: {
              bytesWritten: bytes_written ?? 0,
              totalBytes: total_bytes ?? null,
              startedAt: Date.now(),
              status: 'transferring',
            },
          };
        }
        return {
          ...prev,
          [relative_path]: {
            ...existing,
            bytesWritten: bytes_written ?? existing.bytesWritten,
            totalBytes: total_bytes ?? existing.totalBytes,
          },
        };
      });
    }).then(fn => unlisteners.push(fn));

    listen('zynklink:download:complete', (event) => {
      const { relative_path, total_bytes } = event.payload || {};
      if (!relative_path) return;
      setDownloads(prev => {
        const existing = prev[relative_path];
        if (!existing) return prev;
        return {
          ...prev,
          [relative_path]: {
            ...existing,
            bytesWritten: total_bytes ?? existing.bytesWritten,
            totalBytes: total_bytes ?? existing.totalBytes,
            status: 'complete',
          },
        };
      });
      // Auto-clear the entry 5s after completion so the panel doesn't clutter
      // up after long sessions with many downloads.
      setTimeout(() => {
        setDownloads(prev => {
          if (!prev[relative_path]) return prev;
          const next = { ...prev };
          delete next[relative_path];
          return next;
        });
      }, 5000);
    }).then(fn => unlisteners.push(fn));

    listen('zynklink:download:cancelled', (event) => {
      const { relative_path } = event.payload || {};
      if (!relative_path) return;
      // Remove the row immediately on cancellation — the user already knows
      // they clicked cancel; no need to leave a "cancelled" status hanging
      // around. The error toast from setMessage() already conveys what happened.
      setDownloads(prev => {
        if (!prev[relative_path]) return prev;
        const next = { ...prev };
        delete next[relative_path];
        return next;
      });
    }).then(fn => unlisteners.push(fn));

    return () => {
      unlisteners.forEach(fn => { try { fn(); } catch (e) { /* ignore */ } });
    };
  }, [isOpen]);

  // Trigger backend cancellation for a download in-flight. The backend's
  // streaming loop sees the flag on its next chunk-poll, deletes its .part
  // file, and emits zynklink:download:cancelled (which the listener above
  // uses to clear the row).
  const handleCancelDownload = async (relativePath) => {
    try {
      await invoke('cancel_zynklink_download', { relativePath });
    } catch (err) {
      // Most likely cause: the download finished or failed a moment before we
      // could cancel it. Either way, clean up the UI state optimistically.
      setDownloads(prev => {
        if (!prev[relativePath]) return prev;
        const next = { ...prev };
        delete next[relativePath];
        return next;
      });
    }
  };

  const toggleFolder = (path) => {
    setExpandedFolders(prev => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  };

  const handleKB = async (shareId, relativePath, deviceId) => {
    if (!userId) { setMessage('✗ User ID not available'); return; }
    const filename = relativePath.replace(/\\/g, '/').split('/').pop();
    setMessage(`Downloading ${filename} to Knowledge Base…`);
    try {
      const savedPath = await invoke('download_to_knowledge_base', { shareId, relativePath, deviceId, userId });
      setMessage(`✓ Downloaded to Knowledge Base: ${savedPath}`);
      setTimeout(() => setMessage(''), 5000);
    } catch (err) {
      setMessage(`✗ Failed: ${err}`);
      setTimeout(() => setMessage(''), 5000);
    }
  };

  const handleSave = async (shareId, relativePath, deviceId) => {
    const filename = relativePath.replace(/\\/g, '/').split('/').pop();
    try {
      let selectedPath;
      if (window.AndroidPaths) {
        // Android: save to the app's ZynkbotShare folder (no permissions needed)
        const shareDir = window.AndroidPaths.getShareDir();
        selectedPath = `${shareDir}/${filename}`;
      } else {
        selectedPath = await save({ defaultPath: filename, title: 'Save File' });
      }
      if (!selectedPath) { setMessage('Cancelled'); setTimeout(() => setMessage(''), 2000); return; }
      setMessage(`Downloading ${filename}…`);
      const savedPath = await invoke('download_to_custom_location', { shareId, relativePath, deviceId, destinationPath: selectedPath });
      setMessage(`✓ Saved to: ${savedPath}`);
      setTimeout(() => setMessage(''), 8000);
    } catch (err) {
      setMessage(`✗ Failed: ${err}`);
      setTimeout(() => setMessage(''), 5000);
    }
  };

  if (!isOpen) return null;

  const tree = buildFolderTree(files);
  const topFolderKeys = Object.keys(tree.children);

  return ReactDOM.createPortal(
    <div className="zfb-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="zfb-modal">
        <div className="zfb-header">
          <div className="zfb-header-left">
            <h2>📁 {shareName || 'Shared Files'}</h2>
            {!loading && (
              <span className="zfb-count">{files.length} file{files.length !== 1 ? 's' : ''}</span>
            )}
          </div>
          <div className="zfb-header-right">
            <button className="zfb-refresh-btn" onClick={loadFiles} disabled={loading} title="Refresh">
              🔄 Refresh
            </button>
            <button className="zfb-close-btn" onClick={onClose}>✕</button>
          </div>
        </div>

        <div className="zfb-body">
          {loading ? (
            <div className="zfb-loading">
              <div className="zfb-spinner" />
              <p>Loading files…</p>
            </div>
          ) : files.length === 0 ? (
            <div className="zfb-empty">
              <p>No files found in this share.</p>
              <p className="zfb-hint">Try refreshing or check that the remote device is online.</p>
            </div>
          ) : (
            <div className="zfb-tree">
              {/* Root-level files */}
              {tree.files.map((file, idx) => {
                const filename = file.relative_path.replace(/\\/g, '/').split('/').pop();
                return (
                  <div key={idx} className="zfb-file-row zfb-root-file">
                    <span className="zfb-file-icon">{fileIcon(filename)}</span>
                    <span className="zfb-file-name" title={file.relative_path}>{filename}</span>
                    {file.file_size > 0 && (
                      <span className="zfb-file-size">{formatBytes(file.file_size)}</span>
                    )}
                    <div className="zfb-file-actions">
                      {isKBCompatible(filename) && (
                        <button className="zfb-btn-kb" onClick={() => handleKB(shareId, file.relative_path, deviceId)} title="Download to Knowledge Base">→ KB</button>
                      )}
                      <button className="zfb-btn-save" onClick={() => handleSave(shareId, file.relative_path, deviceId)} title="Save to custom location">Save…</button>
                    </div>
                  </div>
                );
              })}
              {/* Top-level folders */}
              {topFolderKeys.map(key => (
                <FolderNode
                  key={key}
                  node={tree.children[key]}
                  name={key}
                  level={0}
                  expandedFolders={expandedFolders}
                  onToggle={toggleFolder}
                  onKB={handleKB}
                  onSave={handleSave}
                  shareId={shareId}
                  deviceId={deviceId}
                />
              ))}
            </div>
          )}
        </div>

        {Object.keys(downloads).length > 0 && (
          <div className="zfb-downloads">
            {Object.entries(downloads).map(([relPath, dl]) => {
              const filename = relPath.replace(/\\/g, '/').split('/').pop();
              const elapsedMs = Date.now() - dl.startedAt;
              const elapsedSec = Math.max(elapsedMs / 1000, 0.001);
              const speedBps = dl.bytesWritten / elapsedSec;
              const haveTotal = dl.totalBytes && dl.totalBytes > 0;
              const pct = haveTotal
                ? Math.min(100, (dl.bytesWritten / dl.totalBytes) * 100)
                : null;
              const remaining = haveTotal ? Math.max(0, dl.totalBytes - dl.bytesWritten) : null;
              const etaSec = (haveTotal && speedBps > 0) ? remaining / speedBps : null;

              if (dl.status === 'complete') {
                return (
                  <div key={relPath} className="zfb-download-row zfb-download-complete">
                    <div className="zfb-download-name">✓ {filename}</div>
                    <div className="zfb-download-meta">
                      {formatBytes(dl.bytesWritten)} · finished
                    </div>
                  </div>
                );
              }

              return (
                <div key={relPath} className="zfb-download-row">
                  <div className="zfb-download-header">
                    <div className="zfb-download-name">⬇ {filename}</div>
                    <button
                      className="zfb-download-cancel"
                      onClick={() => handleCancelDownload(relPath)}
                      title="Cancel this download"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="zfb-download-bar-wrapper">
                    <div
                      className={`zfb-download-bar ${pct === null ? 'zfb-download-bar-indeterminate' : ''}`}
                      style={pct !== null ? { width: `${pct}%` } : undefined}
                    />
                  </div>
                  <div className="zfb-download-meta">
                    {haveTotal
                      ? `${formatBytes(dl.bytesWritten)} / ${formatBytes(dl.totalBytes)} (${pct.toFixed(1)}%)`
                      : formatBytes(dl.bytesWritten)
                    }
                    {' · '}
                    {formatRate(speedBps)}
                    {etaSec !== null && ` · ${formatDuration(etaSec)} left`}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {message && (
          <div className={`zfb-message ${message.startsWith('✓') ? 'zfb-success' : 'zfb-error'}`}>
            {message}
          </div>
        )}
      </div>
    </div>,
    document.body
  );
}
