import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open as openFolderDialog } from '@tauri-apps/plugin-dialog';
import ZChatModal from './ZChatModal';
import ZynkFileBrowserModal from './ZynkFileBrowserModal';

export default function ZynkLinkPanel({ apiBaseUrl, onOpenUserIdentity, userId }) {
  const [sharedDirs, setSharedDirs] = useState([]);
  const [remoteDirs, setRemoteDirs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [newDirPath, setNewDirPath] = useState('');
  const [newShareName, setNewShareName] = useState('');
  const [browserShare, setBrowserShare] = useState(null);
  const [codeToAccept, setCodeToAccept] = useState('');
  const [codeIPPart, setCodeIPPart] = useState('');
  const [codeNumPart, setCodeNumPart] = useState('');
  const [linkedUsers, setLinkedUsers] = useState([]);
  const [generatedCode, setGeneratedCode] = useState(null);
  const [chatDevice, setChatDevice] = useState(null); // Track which device chat is open for
  const [localDeviceId, setLocalDeviceId] = useState('');
  const [unreadCounts, setUnreadCounts] = useState({}); // Map of device_id -> unread count
  const [isMobile] = useState(() => window.innerWidth <= 768);
  const [showEnterCodeInput, setShowEnterCodeInput] = useState(false); // Toggle for enter code input

  const fetchSharedDirectories = useCallback(async () => {
    try {
      const data = await invoke('list_my_shared_directories');
      setSharedDirs(data.shared_directories || []);
    } catch (error) {
      console.debug('[ZynkLink] Failed to fetch shared directories:', error);
    }
  }, []);

  const fetchRemoteDirectories = useCallback(async () => {
    try {
      const data = await invoke('list_remote_directories');
      setRemoteDirs(data.shared_directories || []);
    } catch (error) {
      console.error('[ZynkLink] Failed to fetch remote directories:', error);
    }
  }, []);

  const fetchLinkedUsers = useCallback(async () => {
    try {
      const data = await invoke('list_zynklink_pairings');
      setLinkedUsers(data.linked_users || []);
    } catch (error) {
      console.debug('[ZynkLink] Failed to fetch linked users:', error);
    }
  }, []);

  const fetchUnreadCounts = useCallback(async () => {
    try {
      const data = await invoke('list_zynklink_pairings');
      const users = data.linked_users || [];

      const counts = {};
      for (const user of users) {
        try {
          const count = await invoke('zchat_get_unread_count', {
            fromDeviceId: user.device_id
          });
          counts[user.device_id] = count;
        } catch (error) {
          console.debug(`[ZChat] Failed to get unread count for ${user.device_id}:`, error);
          counts[user.device_id] = 0;
        }
      }
      setUnreadCounts(counts);
    } catch (error) {
      console.debug('[ZChat] Failed to fetch unread counts:', error);
    }
  }, []);

  useEffect(() => {
    // Get local device ID for chat
    const getDeviceId = async () => {
      try {
        const identity = await invoke('get_user_identity');
        setLocalDeviceId(identity.device_id);
      } catch (error) {
        console.error('[ZynkLink] Failed to get device ID:', error);
      }
    };

    // Auto-scan all local shared directories on panel open
    const autoScanSharedDirectories = async () => {
      try {
        const data = await invoke('list_my_shared_directories');
        const dirs = data.shared_directories || [];

        if (dirs.length > 0) {
          console.log(`[ZynkLink] Auto-scanning ${dirs.length} shared directories...`);

          // Scan all directories in parallel (lightweight operation)
          const scanPromises = dirs.map(dir =>
            invoke('scan_shared_directory', { shareId: dir.id, maxFiles: 1000 })
              .catch(err => console.debug(`[ZynkLink] Auto-scan failed for ${dir.id}:`, err))
          );

          await Promise.all(scanPromises);
          console.log('[ZynkLink] Auto-scan complete');
        }
      } catch (error) {
        console.debug('[ZynkLink] Auto-scan error:', error);
      }
    };

    getDeviceId();
    fetchSharedDirectories();
    fetchRemoteDirectories();
    fetchLinkedUsers();
    fetchUnreadCounts();
    autoScanSharedDirectories(); // Auto-scan on mount

    const interval = setInterval(() => {
      fetchSharedDirectories();
      fetchRemoteDirectories();
      fetchLinkedUsers();
      fetchUnreadCounts();
    }, 5000);  // Refresh every 5 seconds for faster updates
    return () => clearInterval(interval);
  }, [fetchSharedDirectories, fetchRemoteDirectories, fetchLinkedUsers, fetchUnreadCounts]);

  // Listen for immediate pairing updates from backend
  useEffect(() => {
    let unlisten;

    const setupListener = async () => {
      try {
        unlisten = await listen('zynklink-pairing-updated', (event) => {
          console.log('[ZynkLink] Pairing updated event received:', event.payload);
          // Immediately refresh all lists to show the newly linked device
          fetchSharedDirectories();
          fetchRemoteDirectories();
          fetchLinkedUsers();
        });
      } catch (error) {
        console.error('[ZynkLink] Failed to setup event listener:', error);
      }
    };

    setupListener();

    return () => {
      if (typeof unlisten === 'function') {
        unlisten();
      }
    };
  }, [fetchSharedDirectories, fetchRemoteDirectories, fetchLinkedUsers]);

  const handleGenerateCode = async () => {
    console.log('[ZynkLink] Generating code...');
    setLoading(true);
    setMessage('');
    try {
      // Get local IP and generate code
      const localIp = await invoke('get_local_ip');
      const result = await invoke('generate_zynklink_code');
      console.log('[ZynkLink] Code generated:', result.code, 'Local IP:', localIp);

      const fullCode = `${localIp}:${result.code}`;
      setGeneratedCode(fullCode);
      setMessage(`✓ ZynkLink code generated! Share this with the other device.`);
    } catch (error) {
      console.error('[ZynkLink] Failed to generate code:', error);
      setMessage('✗ Failed to generate code: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptCode = async () => {
    if (!codeToAccept.trim()) return;

    // Parse IP:CODE format (e.g., "192.168.0.100:456789")
    const input = codeToAccept.trim();
    const parts = input.split(':');

    if (parts.length !== 2) {
      setMessage('✗ Invalid format. Use IP:CODE (e.g., 192.168.0.100:456789)');
      return;
    }

    const [deviceIp, code] = parts;

    setLoading(true);
    setMessage('');
    try {
      console.log('[ZynkLink] Linking with device:', deviceIp, 'code:', code);
      const result = await invoke('link_with_zynklink_code', {
        code: code.trim(),
        deviceIp: deviceIp.trim()
      });
      setMessage(`✓ ${result.message}`);
      setCodeToAccept('');
      fetchLinkedUsers();
      fetchRemoteDirectories();
    } catch (error) {
      console.error('[ZynkLink] Failed to link:', error);
      setMessage('✗ Failed to link: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleUnlinkUser = async (linkedUserId) => {
    setLoading(true);
    setMessage('');
    try {
      await invoke('revoke_zynklink_pairing', { linkedUserId });
      setMessage(`✓ Unlinked from user`);
      fetchLinkedUsers();
      fetchRemoteDirectories();
    } catch (error) {
      setMessage('✗ Failed to unlink: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePause = async (linkedDeviceId, currentlyPaused) => {
    setLoading(true);
    setMessage('');
    try {
      await invoke('toggle_zynklink_pause', { linkedDeviceId, paused: !currentlyPaused });
      setMessage(currentlyPaused ? '✓ Link resumed' : '✓ Link paused');
      fetchLinkedUsers();
    } catch (error) {
      setMessage('✗ Failed to toggle pause: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleShareDirectory = async () => {
    if (!newDirPath.trim() || !newShareName.trim()) return;

    setLoading(true);
    setMessage('');
    try {
      const data = await invoke('share_directory', {
        localPath: newDirPath.trim(),
        shareName: newShareName.trim(),
        isReadable: true,
        isWritable: false
      });

      if (data.success && data.share_id) {
        setMessage(`✓ Directory shared successfully`);
        setNewDirPath('');
        setNewShareName('');
        fetchSharedDirectories();

        // Automatically scan the new share
        await invoke('scan_shared_directory', { shareId: data.share_id, maxFiles: 1000 });
      } else {
        setMessage('✗ Failed to share directory: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      setMessage('✗ Error: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleUnshareDirectory = async (shareId) => {
    setLoading(true);
    setMessage('');
    try {
      await invoke('unshare_directory', { shareId });
      setMessage(`✓ Directory unshared`);
      fetchSharedDirectories();
    } catch (error) {
      setMessage('✗ Error: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleRescanDirectory = async (shareId, shareName) => {
    setLoading(true);
    setMessage(`Rescanning ${shareName || 'directory'}...`);
    try {
      const result = await invoke('scan_shared_directory', { shareId, maxFiles: 1000 });
      const filesIndexed = result?.files_indexed || 0;
      setMessage(`✓ Rescan complete - ${filesIndexed} file(s) indexed`);
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('[ZynkLink] Rescan error:', error);
      setMessage(`✗ Rescan failed: ${error}`);
      setTimeout(() => setMessage(''), 5000);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenChat = (user) => {
    // Open chat modal
    setChatDevice(user);
    // Clear unread count for this device
    setUnreadCounts(prev => ({ ...prev, [user.device_id]: 0 }));
    // Note: Messages will be marked as read in ZChatModal when it fetches them
  };


  return (
    <div style={{
      background: '#282a36',
      border: '1px solid #44475a',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px'
    }}>
      {/* What is ZynkLink — only shown before any devices are linked */}
      {linkedUsers.length === 0 && <div style={{
        background: '#1e1f29',
        padding: '12px 15px',
        borderRadius: '6px',
        marginBottom: '16px',
        border: '1px solid #44475a',
        fontSize: '0.83rem',
        color: '#9aa5c4',
        lineHeight: '1.6'
      }}>
        <strong style={{ color: '#f1fa8c' }}>ZynkLink</strong> lets you browse and download files from another Zynkbot device on your local network, and chat directly between devices via ZChat. Unlike ZynkSync, it does <em>not</em> merge memories — it only shares files and enables messaging.
        <div style={{ marginTop: '8px', color: '#9aa5c4' }}>
          <strong style={{ color: '#bd93f9' }}>To link two devices:</strong>
          <ol style={{ marginTop: '4px', marginBottom: '0', paddingLeft: '18px' }}>
            <li>On <em>this</em> device: click <strong>Generate Code</strong> and share the code with the other device</li>
            <li>On the <em>other</em> device: open ZynkLink, click <strong>Link to Device</strong>, and enter the code</li>
            <li>Both devices will appear in each other's linked device list and can browse shared directories</li>
          </ol>
        </div>
      </div>}

      {/* This Device's Link Info (matches ZynkSync layout) */}
      <div style={{
        background: '#1e1f29',
        padding: '15px',
        borderRadius: '6px',
        marginBottom: '20px',
        border: '1px solid #44475a'
      }}>
        <div style={{ color: '#ffb86c', fontWeight: 'bold', fontSize: '0.95rem', marginBottom: '10px' }}>
          🔑 Generate Code (Current Device)
        </div>
        <p style={{ fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '10px', lineHeight: '1.5' }}>
          Generate a pairing code to link this device with another for file sharing and chat (code expires in 5 minutes):
        </p>
        <button
          onClick={handleGenerateCode}
          disabled={loading}
          style={{
            padding: '8px 16px',
            background: '#50fa7b',
            color: '#282a36',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'wait' : 'pointer',
            fontWeight: 'bold',
            fontSize: '0.9rem',
            marginBottom: '10px',
            width: '100%'
          }}
        >
          🔗 Generate Code (For New Device)
        </button>
        {generatedCode && (
          <div style={{
            padding: '15px',
            background: '#282a36',
            borderRadius: '4px',
            fontFamily: 'monospace',
            fontSize: '1.3rem',
            color: '#50fa7b',
            textAlign: 'center',
            letterSpacing: '2px',
            border: '2px solid #50fa7b',
            whiteSpace: 'nowrap',
            overflowX: 'auto'
          }}>
            {generatedCode}
          </div>
        )}
      </div>

      {/* Link to Device (matches ZynkSync "Add Device") */}
      <div style={{
        background: '#1e1f29',
        padding: '15px',
        borderRadius: '6px',
        marginBottom: '20px',
        border: '1px solid #44475a'
      }}>
        <div style={{ color: '#ffb86c', fontWeight: 'bold', fontSize: '0.95rem', marginBottom: '10px' }}>
          ➕ Enter Code (From Other User's Device)
        </div>
        <p style={{ fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '10px', lineHeight: '1.5' }}>
          Enter the IP:code from another user's device to link for file sharing and chat:
        </p>
        {!showEnterCodeInput ? (
          <button
            onClick={() => setShowEnterCodeInput(true)}
            style={{
              padding: '8px 16px',
              background: '#ffb86c',
              color: '#282a36',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 'bold',
              width: '100%'
            }}
          >
            ➕ Enter Code From Other User's Device
          </button>
        ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <input
              type="text"
              inputMode="decimal"
              value={codeIPPart}
              onChange={(e) => { setCodeIPPart(e.target.value); setCodeToAccept(e.target.value + ':' + codeNumPart); }}
              placeholder="192.168.0.100"
              style={{
                flex: 3, padding: '10px', background: '#282a36',
                border: '1px solid #44475a', borderRadius: '4px',
                color: '#f8f8f2', fontSize: '0.9rem', fontFamily: 'monospace'
              }}
            />
            <span style={{ color: '#f8f8f2', fontSize: '1.2rem', fontWeight: 'bold', padding: '0 2px' }}>:</span>
            <input
              type="text"
              inputMode="numeric"
              value={codeNumPart}
              onChange={(e) => { setCodeNumPart(e.target.value); setCodeToAccept(codeIPPart + ':' + e.target.value); }}
              placeholder="456789"
              style={{
                flex: 2, padding: '10px', background: '#282a36',
                border: '1px solid #44475a', borderRadius: '4px',
                color: '#f8f8f2', fontSize: '0.9rem', fontFamily: 'monospace'
              }}
            />
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleAcceptCode}
              disabled={!codeToAccept.trim() || loading}
              style={{
                flex: 1,
                padding: '10px 20px',
                background: !codeToAccept.trim() || loading ? '#44475a' : '#50fa7b',
                color: '#282a36',
                border: 'none',
                borderRadius: '4px',
                cursor: (codeToAccept.trim() && !loading) ? 'pointer' : 'not-allowed',
                fontWeight: 'bold',
                fontSize: '0.9rem'
              }}
            >
              Link
            </button>
            <button
              onClick={() => {
                setShowEnterCodeInput(false);
                setCodeToAccept('');
                setCodeIPPart('');
                setCodeNumPart('');
              }}
              disabled={loading}
              style={{
                flex: 1,
                padding: '10px 20px',
                background: '#44475a',
                color: '#f8f8f2',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'wait' : 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9rem'
              }}
            >
              Cancel
            </button>
          </div>
        </div>
        )}
      </div>

      {/* Linked Devices (matches ZynkSync "Devices") */}
      {linkedUsers.length > 0 && (
        <div style={{
          background: '#1e1f29',
          padding: '15px',
          borderRadius: '6px',
          marginBottom: '20px',
          border: '1px solid #44475a'
        }}>
          <div style={{ color: '#ffb86c', fontWeight: 'bold', fontSize: '0.95rem', marginBottom: '10px' }}>
            🔗 Linked Devices ({linkedUsers.length})
          </div>
          {linkedUsers.map(user => (
            <div
              key={user.user_id}
              style={{
                padding: '12px',
                background: '#282a36',
                borderRadius: '4px',
                marginBottom: '8px',
                border: '1px solid #50fa7b'
              }}
            >
              <div style={{ marginBottom: '8px' }}>
                <div style={{ fontSize: '0.9rem', color: user.is_online ? '#50fa7b' : '#ff5555', fontWeight: 'bold', marginBottom: '4px' }}>
                  ● {user.is_online ? 'Online' : 'Offline'}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#9aa5c4', marginBottom: '2px' }}>
                  User: {user.user_id.slice(0, 16)}...
                </div>
                <div style={{ fontSize: '0.8rem', color: '#9aa5c4', marginBottom: '2px' }}>
                  Device: {user.device_id.slice(0, 12)}...
                </div>
                <div style={{ fontSize: '0.8rem', color: '#9aa5c4' }}>
                  Linked: {new Date(user.linked_at).toLocaleDateString()}
                </div>
                {!user.is_online && user.last_seen_at && (
                  <div style={{ fontSize: '0.75rem', color: '#ff5555', marginTop: '4px' }}>
                    Last seen: {new Date(user.last_seen_at).toLocaleString()}
                  </div>
                )}
              </div>
              {user.is_paused && (
                <div style={{ fontSize: '0.8rem', color: '#ffb86c', fontWeight: 'bold', marginBottom: '6px' }}>
                  ⏸ Link paused — chat and file sharing are suspended
                </div>
              )}
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => !user.is_paused && handleOpenChat(user)}
                  disabled={loading || user.is_paused}
                  style={{
                    padding: '6px 14px',
                    background: user.is_paused ? '#44475a' : '#bd93f9',
                    color: user.is_paused ? '#6272a4' : '#282a36',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: loading || user.is_paused ? 'not-allowed' : 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: 'bold',
                    flex: 1
                  }}
                >
                  💬 Chat{!user.is_paused && unreadCounts[user.device_id] > 0 && ` (${unreadCounts[user.device_id]})`}
                </button>
                <button
                  onClick={() => handleTogglePause(user.device_id, user.is_paused)}
                  disabled={loading}
                  style={{
                    padding: '6px 14px',
                    background: user.is_paused ? '#50fa7b' : '#ffb86c',
                    color: '#282a36',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: loading ? 'wait' : 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: 'bold',
                    flex: 1
                  }}
                >
                  {user.is_paused ? '▶ Resume' : '⏸ Pause'}
                </button>

                <button
                  onClick={() => handleUnlinkUser(user.user_id)}
                  disabled={loading}
                  style={{
                    padding: '6px 14px',
                    background: '#ff5555',
                    color: '#f8f8f2',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: loading ? 'wait' : 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: 'bold',
                    flex: 1
                  }}
                >
                  ✕ Unlink
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Share New Directory */}
      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', color: '#f8f8f2', fontSize: '0.9rem', marginBottom: '5px' }}>
          <strong>📁 Share a Directory:</strong> <span style={{ color: '#9aa5c4', fontWeight: 'normal' }}>name it, pick the folder, tap Share.</span>
        </label>
        <input
          type="text"
          value={newShareName}
          onChange={(e) => setNewShareName(e.target.value)}
          placeholder="Share name (e.g., MyDocuments)"
          style={{
            width: '100%',
            padding: '8px',
            marginBottom: '8px',
            background: '#1e1f29',
            border: '1px solid #44475a',
            borderRadius: '4px',
            color: '#f8f8f2',
            fontSize: '0.9rem'
          }}
        />
        {isMobile ? (
          <div>
            {newDirPath && (
              <div style={{ fontSize: '0.8rem', color: '#9aa5c4', marginBottom: '8px', wordBreak: 'break-all' }}>
                Selected: {newDirPath}
              </div>
            )}
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={async () => {
                  if (window.AndroidFolderPicker) {
                    // Native Android folder picker via JavascriptInterface
                    try {
                      const path = await new Promise((resolve, reject) => {
                        window.__fpResolve = resolve;
                        window.__fpReject = reject;
                        window.AndroidFolderPicker.pick();
                      });
                      setNewDirPath(path);
                    } catch (e) {
                      if (e !== 'cancelled') alert('Folder picker error: ' + e);
                    }
                  } else {
                    // Desktop: tauri-plugin-dialog
                    try {
                      const selected = await openFolderDialog({ directory: true, multiple: false });
                      if (selected) setNewDirPath(typeof selected === 'string' ? selected : selected.path || String(selected));
                    } catch (e) {
                      alert('Could not open folder picker: ' + e);
                    }
                  }
                }}
                style={{
                  flex: 1,
                  padding: '10px',
                  background: '#1e1f29',
                  border: '1px solid #44475a',
                  borderRadius: '4px',
                  color: '#f8f8f2',
                  fontSize: '0.9rem',
                  cursor: 'pointer'
                }}
              >
                📂 Choose Folder
              </button>
              <button
                onClick={handleShareDirectory}
                disabled={loading || !newDirPath.trim() || !newShareName.trim()}
                style={{
                  padding: '10px 16px',
                  background: '#8be9fd',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'wait' : 'pointer',
                  fontWeight: 'bold',
                  fontSize: '0.9rem',
                  opacity: (!newDirPath.trim() || !newShareName.trim() || loading) ? 0.5 : 1
                }}
              >
                Share
              </button>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              value={newDirPath}
              onChange={(e) => setNewDirPath(e.target.value)}
              placeholder="e.g., C:\MyFiles or /home/user/files"
              style={{
                flex: 1,
                padding: '8px',
                background: '#1e1f29',
                border: '1px solid #44475a',
                borderRadius: '4px',
                color: '#f8f8f2',
                fontSize: '0.9rem'
              }}
            />
            <button
              onClick={handleShareDirectory}
              disabled={loading || !newDirPath.trim() || !newShareName.trim()}
              style={{
                padding: '8px 16px',
                background: '#8be9fd',
                color: '#282a36',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'wait' : 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9rem',
                opacity: (!newDirPath.trim() || !newShareName.trim() || loading) ? 0.5 : 1
              }}
            >
              Share
            </button>
          </div>
        )}
      </div>

      {/* My Shared Directories */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ color: '#f8f8f2', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
          📂 My Shared Directories ({sharedDirs.length})
        </div>
        {sharedDirs.length === 0 ? (
          <div style={{ color: '#9aa5c4', fontSize: '0.85rem', padding: '10px' }}>
            No directories shared. Use a paired desktop to share directories — they'll appear here automatically.
          </div>
        ) : (
          <div>
            {sharedDirs.map((dir) => {
              return (
                <div
                  key={dir.id}
                  style={{
                    background: '#1e1f29',
                    padding: '10px',
                    borderRadius: '4px',
                    marginBottom: '8px',
                    fontSize: '0.85rem'
                  }}
                >
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ color: '#f8f8f2', fontWeight: 'bold', marginBottom: '2px' }}>
                      {dir.share_name || dir.local_path}
                    </div>
                    <div style={{ color: '#9aa5c4', fontSize: '0.8rem' }}>
                      {dir.local_path}
                    </div>
                    <div style={{ color: '#9aa5c4', fontSize: '0.8rem', marginBottom: '8px' }}>
                      Shared: {new Date(dir.created_at).toLocaleDateString()}
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button
                        onClick={() => setBrowserShare({ shareId: dir.id, deviceId: dir.device_id, shareName: dir.share_name || dir.local_path })}
                        style={{
                          padding: '4px 10px',
                          background: '#8be9fd',
                          color: '#282a36',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '0.8rem',
                          fontWeight: 'bold'
                        }}
                      >
                        📂 Browse Files
                      </button>
                      <button
                        onClick={() => handleRescanDirectory(dir.id, dir.share_name)}
                        disabled={loading}
                        style={{
                          padding: '4px 10px',
                          background: '#50fa7b',
                          color: '#282a36',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: loading ? 'wait' : 'pointer',
                          fontSize: '0.8rem',
                          fontWeight: 'bold',
                          opacity: loading ? 0.5 : 1
                        }}
                        title="Rescan directory for new files"
                      >
                        🔄 Rescan
                      </button>
                      <button
                        onClick={() => handleUnshareDirectory(dir.id)}
                        disabled={loading}
                        style={{
                          padding: '4px 10px',
                          background: '#ff5555',
                          color: '#f8f8f2',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '0.8rem'
                        }}
                      >
                        Unshare
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Shared With Me (Remote Directories) */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ color: '#f8f8f2', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
          📂 Shared With Me ({remoteDirs.length})
        </div>
        {remoteDirs.length === 0 ? (
          <div style={{ color: '#9aa5c4', fontSize: '0.85rem', padding: '10px' }}>
            No directories shared from other devices yet. Pair with other devices to see their shares.
          </div>
        ) : (
          <div>
            {remoteDirs.map((dir) => {
              return (
                <div
                  key={dir.id}
                  style={{
                    background: '#1e1f29',
                    padding: '10px',
                    borderRadius: '4px',
                    marginBottom: '8px',
                    fontSize: '0.85rem',
                    border: '1px solid #bd93f9'
                  }}
                >
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ color: '#bd93f9', fontWeight: 'bold', marginBottom: '2px' }}>
                      {dir.share_name || dir.local_path}
                    </div>
                    <div style={{ color: '#9aa5c4', fontSize: '0.8rem' }}>
                      From device: {dir.device_id.substring(0, 8)}...
                    </div>
                    <div style={{ color: '#9aa5c4', fontSize: '0.8rem', marginBottom: '8px' }}>
                      Shared: {new Date(dir.created_at).toLocaleDateString()}
                    </div>
                    <button
                      onClick={() => setBrowserShare({ shareId: dir.id, deviceId: dir.device_id, shareName: dir.share_name || dir.local_path })}
                      style={{
                        padding: '4px 12px',
                        background: '#8be9fd',
                        color: '#282a36',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        fontWeight: 'bold'
                      }}
                    >
                      📂 Browse Files
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Message */}
      {message && (
        <div style={{
          marginTop: '10px',
          padding: '10px',
          background: message.includes('✓') ? '#1e3a1e' : '#3a1e1e',
          border: `1px solid ${message.includes('✓') ? '#50fa7b' : '#ff5555'}`,
          borderRadius: '4px',
          color: message.includes('✓') ? '#50fa7b' : '#ff5555',
          fontSize: '0.9rem'
        }}>
          {message}
        </div>
      )}

      {/* ZChat Modal */}
      {chatDevice && (
        <ZChatModal
          isOpen={!!chatDevice}
          onClose={() => { setChatDevice(null); fetchUnreadCounts(); }}
          apiBaseUrl={apiBaseUrl}
          device={chatDevice}
          currentDeviceId={localDeviceId}
        />
      )}

      {/* File Browser Modal */}
      {browserShare && (
        <ZynkFileBrowserModal
          isOpen={!!browserShare}
          onClose={() => setBrowserShare(null)}
          shareId={browserShare.shareId}
          deviceId={browserShare.deviceId}
          shareName={browserShare.shareName}
          userId={userId}
        />
      )}
    </div>
  );
}
