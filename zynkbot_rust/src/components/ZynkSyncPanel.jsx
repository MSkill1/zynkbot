import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export default function ZynkSyncPanel({ userId, onOpenUserIdentity, onOpenChat }) {
  const [peers, setPeers] = useState([]);
  const [syncStatus, setSyncStatus] = useState('stopped');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const autoRefresh = true;
  const [pairingCode, setPairingCode] = useState('');
  const [localIp, setLocalIp] = useState('');
  const [pairingInput, setPairingInput] = useState('');
  const [pairingIPPart, setPairingIPPart] = useState('');
  const [pairingNumPart, setPairingNumPart] = useState('');
  const [showAddDevice, setShowAddDevice] = useState(false);

  // Fetch peers from Rust backend
  const fetchPeers = useCallback(async () => {
    try {
      const peers = await invoke('get_zynksync_peers');
      setPeers(peers || []);
    } catch (error) {
      console.error('[ZynkSync] Failed to fetch peers:', error);
      // Don't show error message on every poll - only on user actions
    }
  }, []);

  // Start ZynkSync service
  const startZynkSync = useCallback(async () => {
    if (loading) return; // Prevent concurrent calls
    setLoading(true);
    try {
      setMessage('Resuming ZynkSync...');
      const result = await invoke('start_zynksync', {
        syncIntervalSecs: 60  // Sync every 60 seconds
      });
      setSyncStatus('running');
      setMessage('✓ ZynkSync resumed — syncing with devices');
      console.log('[ZynkSync] Service started:', result);

      // Start polling for peers
      if (autoRefresh) {
        fetchPeers();
      }
    } catch (error) {
      console.error('[ZynkSync] Failed to start:', error);
      setMessage('✗ Failed to start ZynkSync: ' + error);
      setSyncStatus('stopped');
    } finally {
      setLoading(false);
    }
  }, [loading, autoRefresh, fetchPeers]);

  // Stop ZynkSync service
  const stopZynkSync = useCallback(async () => {
    if (loading) return; // Prevent concurrent calls
    setLoading(true);
    try {
      setMessage('Pausing ZynkSync...');
      await invoke('stop_zynksync');
      setSyncStatus('stopped');
      setMessage('✓ ZynkSync paused');
      setPeers([]);
      setPairingCode('');
      setLocalIp('');
      setShowAddDevice(false);
    } catch (error) {
      console.error('[ZynkSync] Failed to stop:', error);
      setMessage('✗ Failed to stop ZynkSync: ' + error);
    } finally {
      setLoading(false);
    }
  }, [loading]);

  // Manual refresh: update peer list and trigger immediate sync
  const handleRefresh = useCallback(async () => {
    setMessage('Refreshing...');
    await fetchPeers();
    try {
      await invoke('broadcast_sync_to_all_peers', { userId: userId || '' });
      setMessage('✓ Refreshed');
    } catch {
      setMessage('✓ Refreshed (sync skipped — no active peers)');
    }
    setTimeout(() => setMessage(''), 2500);
  }, [fetchPeers, userId]);

  // Sync memories bidirectionally with a specific peer
  const handleSyncToPeer = async (peer) => {
    setLoading(true);
    setMessage('');
    try {
      const results = await invoke('broadcast_sync_to_all_peers', {
        userId: userId || ''
      });

      const result = results.find(r => r.peer_device_id === peer.device_id) || results[0];
      if (result && !result.success) {
        setMessage(`✗ Sync failed: ${result.error || 'Unknown error'}`);
      } else if (result) {
        const memSent = result.memories_sent || 0;
        const memReceived = result.memories_received || 0;
        const convSent = result.conversations_sent || 0;
        if (memSent === 0 && memReceived === 0 && convSent === 0) {
          setMessage(`✓ Already in sync with ${peer.device_name}`);
        } else {
          const parts = [];
          if (memSent > 0 || memReceived > 0) parts.push(`sent ${memSent}, received ${memReceived} memories`);
          if (convSent > 0) parts.push(`sent ${convSent} conversation messages`);
          setMessage(`✓ Synced with ${peer.device_name}: ${parts.join('; ')}`);
        }
      } else {
        setMessage(`✓ Sync complete`);
      }
      console.log('[ZynkSync] Sync result:', results);
    } catch (error) {
      console.error('[ZynkSync] Sync failed:', error);
      setMessage(`✗ Sync failed: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Get pairing code and IP address for this device
  const handleGetPairingCode = async () => {
    try {
      const [code, ip] = await Promise.all([
        invoke('get_zynksync_pairing_code'),
        invoke('get_local_ip')
      ]);
      setPairingCode(code);
      setLocalIp(ip);
      setMessage('✓ Pairing info ready - share the IP and code with another device');
    } catch (error) {
      console.error('[ZynkSync] Failed to get pairing info:', error);
      setMessage(`✗ Failed to get pairing info: ${error}`);
    }
  };

  // Copy pairing info to clipboard
  const handleCopyPairingInfo = () => {
    if (pairingCode && localIp) {
      const pairingInfo = `${localIp}:${pairingCode}`;
      navigator.clipboard.writeText(pairingInfo);
      setMessage('✓ IP:Code copied to clipboard!');
    }
  };

  // Add a device manually
  const handleAddDevice = async () => {
    const input = pairingInput.trim();
    if (!input) {
      setMessage('✗ Please enter the pairing code in IP:code format');
      return;
    }

    // Parse IP:code format
    const parts = input.split(':');
    if (parts.length !== 2) {
      setMessage('✗ Invalid format. Use IP:code (e.g., 192.168.0.100:456789)');
      return;
    }

    const [deviceIp, code] = parts;
    if (!deviceIp.trim() || !code.trim()) {
      setMessage('✗ Both IP and code are required');
      return;
    }
    if (code.trim().length !== 6 || !/^\d+$/.test(code.trim())) {
      setMessage('✗ Pairing code must be exactly 6 digits');
      return;
    }

    setLoading(true);
    setMessage('');
    try {
      const peer = await invoke('add_zynksync_device', {
        hostIp: deviceIp.trim(),
        pairingCode: code.trim()
      });

      // Check if host returned a user_id (for identity sync)
      if (peer.user_id && peer.user_id !== userId) {
        // Host has a different user_id - show warning dialog
        const warningMessage =
          `⚠️  IDENTITY SYNC WARNING ⚠️\n\n` +
          `You are about to join this device to a ZynkSync network.\n\n` +
          `This will:\n` +
          `• Clear ALL memories on THIS device\n` +
          `• Change your User ID to match the host device\n` +
          `• Start syncing with the host's memories\n\n` +
          `Host Device: ${peer.device_name}\n` +
          `Host User ID: ${peer.user_id}\n\n` +
          `This action CANNOT be undone!\n\n` +
          `Do you want to continue?`;

        if (!window.confirm(warningMessage)) {
          setMessage('✗ Pairing cancelled - identity not changed');
          setPairingInput('');
          setShowAddDevice(false);
          setLoading(false);
          return;
        }

        // User confirmed - migrate memories to new identity
        setMessage('⏳ Migrating memories to host identity...');

        try {
          // Step 1: Migrate all local memories to the new user_id
          // This preserves memories so they can sync with the new identity
          const migratedCount = await invoke('migrate_user_memories', {
            oldUserId: userId,
            newUserId: peer.user_id
          });
          console.log('[ZynkSync] Migrated', migratedCount, 'memories to new user_id');

          // Step 2: Update local identity to match host
          await invoke('set_user_identity', { userId: peer.user_id });
          console.log('[ZynkSync] Adopted host user_id:', peer.user_id);

          setMessage(`✓ Identity synced! Added device: ${peer.device_name}\n✓ This device is now part of the ZynkSync network\n✓ Memories will sync automatically`);

          // Notify parent component to refresh UI with new user_id
          if (onOpenUserIdentity) {
            setTimeout(() => {
              window.location.reload(); // Reload to refresh with new user_id
            }, 2000);
          }
        } catch (identityError) {
          console.error('[ZynkSync] Failed to sync identity:', identityError);
          setMessage(`✗ Failed to sync identity: ${identityError}\nPairing succeeded but identity not changed.`);
        }
      } else {
        // No identity sync needed - just normal pairing
        setMessage(`✓ Added device: ${peer.device_name}`);
      }

      setPairingInput('');
      setPairingIPPart('');
      setPairingNumPart('');
      setShowAddDevice(false);
      fetchPeers();
    } catch (error) {
      console.error('[ZynkSync] Failed to add device:', error);
      setMessage(`✗ Failed to add device: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Remove a device
  const handleRemoveDevice = async (peer) => {
    if (!window.confirm(`Remove device ${peer.device_name}?`)) {
      return;
    }

    setLoading(true);
    setMessage('');
    try {
      await invoke('remove_zynksync_device', {
        deviceId: peer.device_id
      });

      setMessage(`✓ Removed device: ${peer.device_name}`);
      fetchPeers();
    } catch (error) {
      console.error('[ZynkSync] Failed to remove device:', error);
      setMessage(`✗ Failed to remove device: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Check ZynkSync status on component mount — auto-start if peers exist
  useEffect(() => {
    const checkServiceStatus = async () => {
      try {
        const isRunning = await invoke('get_zynksync_status');
        if (isRunning) {
          setSyncStatus('running');
          setMessage('✓ ZynkSync is running');
          fetchPeers();
        } else {
          // Auto-start if we have known peers from a previous session
          const existingPeers = await invoke('get_zynksync_peers');
          if (existingPeers && existingPeers.length > 0) {
            await invoke('start_zynksync', { syncIntervalSecs: 60 });
            setSyncStatus('running');
            setMessage('✓ ZynkSync resumed automatically');
            fetchPeers();
          } else {
            setSyncStatus('stopped');
            setMessage('ZynkSync paused. Click "Resume Syncing" to start.');
          }
        }
      } catch (error) {
        console.error('[ZynkSync] Failed to check status:', error);
        setSyncStatus('stopped');
        setMessage('Click "Resume Syncing" to begin syncing with your devices.');
      }
    };

    checkServiceStatus();

    // Cleanup on unmount
    return () => {
      // Don't stop the service on unmount - let it keep running
      // The user can explicitly stop it via the button
    };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for pairing warnings from backend
  useEffect(() => {
    let unlisten;

    const setupListener = async () => {
      unlisten = await listen('zynksync://warning', (event) => {
        const warning = event.payload;
        console.log('[ZynkSync] Received warning event:', warning);

        if (warning && warning.message) {
          // Display warning as a message in the UI
          const emoji = warning.severity === 'high' ? '⚠️' : 'ℹ️';
          setMessage(`${emoji} ${warning.message}`);
        }
      });
    };

    setupListener();

    return () => {
      if (typeof unlisten === 'function') {
        unlisten();
      }
    };
  }, []);

  // Listen for remote-initiated unsync and refresh the peer list immediately
  useEffect(() => {
    let unlisten;
    const setup = async () => {
      unlisten = await listen('zynksync-device-removed', () => {
        fetchPeers();
        setMessage('✓ A device unsynced remotely — peer list updated.');
      });
    };
    setup();
    return () => { if (typeof unlisten === 'function') unlisten(); };
  }, [fetchPeers]);

  useEffect(() => {
    let unlisten;
    const setup = async () => {
      unlisten = await listen('zynksync-status-changed', (event) => {
        const { status } = event.payload;
        if (status === 'paused') {
          setSyncStatus('stopped');
          setMessage('ZynkSync paused by another device. Click "Resume Syncing" to start.');
        } else if (status === 'running') {
          setSyncStatus('running');
          setMessage('✓ ZynkSync resumed by another device.');
          fetchPeers();
        }
      });
    };
    setup();
    return () => { if (typeof unlisten === 'function') unlisten(); };
  }, [fetchPeers]);

  // Auto-refresh peers every 30 seconds when service is running
  useEffect(() => {
    if (syncStatus === 'running' && autoRefresh) {
      const interval = setInterval(() => {
        fetchPeers();
      }, 30000);  // 30 seconds

      return () => clearInterval(interval);
    }
  }, [syncStatus, autoRefresh, fetchPeers]);

  return (
    <div style={{
      background: '#282a36',
      border: '1px solid #44475a',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px'
    }}>
      {/* Control Buttons */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
        <button
          onClick={syncStatus === 'running' ? stopZynkSync : startZynkSync}
          disabled={loading}
          style={{
            flex: 1,
            padding: '8px 16px',
            background: syncStatus === 'running' ? '#ffb86c' : '#50fa7b',
            color: '#282a36',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'wait' : 'pointer',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            transition: 'all 0.2s',
            opacity: loading ? 0.5 : 1
          }}
        >
          {syncStatus === 'running' ? '⏸ Pause Syncing' : '▶ Resume Syncing'}
        </button>
        <button
          onClick={handleRefresh}
          disabled={loading}
          style={{
            flex: 0.7,
            padding: '8px 12px',
            background: '#6272a4',
            color: '#f8f8f2',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'wait' : 'pointer',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            transition: 'all 0.2s',
            opacity: loading ? 0.5 : 1
          }}
          title="Refresh device list and sync now"
        >
          🔄 Refresh
        </button>
        <button
          onClick={onOpenUserIdentity}
          style={{
            flex: 0.7,
            padding: '8px 12px',
            background: '#ff5555',
            color: '#f8f8f2',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '0.85rem',
            fontWeight: 'bold',
            transition: 'all 0.2s'
          }}
          onMouseOver={(e) => e.target.style.background = '#ff6b6b'}
          onMouseOut={(e) => e.target.style.background = '#ff5555'}
          title="Manage your identity and sync codes"
        >
          👤 Identity
        </button>
      </div>

      {/* Status Badge */}
      <div style={{
        display: 'inline-block',
        padding: '6px 12px',
        background: syncStatus === 'running' ? '#1e3a1e' : '#3a1e1e',
        border: `1px solid ${syncStatus === 'running' ? '#50fa7b' : '#ff5555'}`,
        borderRadius: '4px',
        fontSize: '0.8rem',
        fontWeight: 'bold',
        color: syncStatus === 'running' ? '#50fa7b' : '#ff5555',
        marginBottom: '15px'
      }}>
        {syncStatus === 'running' ? '● Service Running' : '○ Service Stopped'}
      </div>

      {/* Pairing Info Section */}
      {syncStatus === 'running' && (
        <div style={{
          marginBottom: '15px',
          padding: '12px',
          background: '#1e1f29',
          borderRadius: '6px',
          border: '1px solid #44475a'
        }}>
          <div style={{ color: '#ffb86c', fontWeight: 'bold', fontSize: '0.95rem', marginBottom: '10px' }}>
            🔑 Generate Code (Current Device)
          </div>
          <p style={{ fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '8px', lineHeight: '1.5' }}>
            Generate a pairing code to sync this device with another for memory synchronization (code expires in 10 minutes):
          </p>
          <p style={{ fontSize: '0.82rem', color: '#ff5555', marginBottom: '10px', lineHeight: '1.5' }}>
            ⚠️ <strong>Important:</strong> The device that <em>enters</em> this code will adopt this device's user identity. Its existing memories will be merged, with this device's data taking precedence where conflicts exist. Use the Memory Manager to clear memories first if you want a clean sync.
          </p>
          {pairingCode && localIp ? (
            <div>
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
                marginBottom: '10px',
                wordBreak: 'break-all',
                overflowWrap: 'break-word'
              }}>
                {localIp}:{pairingCode}
              </div>
              <button
                onClick={handleCopyPairingInfo}
                style={{
                  width: '100%',
                  padding: '10px 16px',
                  background: '#50fa7b',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: 'bold'
                }}
              >
                📋 Copy IP:Code
              </button>
            </div>
          ) : (
            <button
              onClick={handleGetPairingCode}
              style={{
                padding: '8px 16px',
                background: '#50fa7b',
                color: '#282a36',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9rem',
                width: '100%'
              }}
            >
              🔑 Generate Code (For New Device)
            </button>
          )}
        </div>
      )}

      {/* Add Device Section */}
      {syncStatus === 'running' && (
        <div style={{
          marginBottom: '15px',
          padding: '12px',
          background: '#1e1f29',
          borderRadius: '6px',
          border: '1px solid #44475a'
        }}>
          <div style={{ color: '#ffb86c', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
            ➕ Add Device
          </div>
          {!showAddDevice ? (
            <button
              onClick={() => setShowAddDevice(true)}
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
              ➕ Enter Code From Other Device
            </button>
          ) : (
            <div>
              <div style={{ fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '8px' }}>
                Enter the IP:code from the other device to sync with it:
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <input
                    type="text"
                    inputMode="decimal"
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck="false"
                    placeholder="192.168.0.100"
                    value={pairingIPPart}
                    onChange={(e) => { setPairingIPPart(e.target.value); setPairingInput(e.target.value + ':' + pairingNumPart); }}
                    disabled={loading}
                    style={{
                      width: '100%', boxSizing: 'border-box',
                      padding: '12px', background: '#282a36',
                      border: '1px solid #44475a', borderRadius: '4px',
                      color: '#f8f8f2', fontSize: '1rem', fontFamily: 'monospace'
                    }}
                  />
                  <input
                    type="text"
                    inputMode="numeric"
                    autoComplete="off"
                    placeholder="Code (456789)"
                    value={pairingNumPart}
                    onChange={(e) => { setPairingNumPart(e.target.value); setPairingInput(pairingIPPart + ':' + e.target.value); }}
                    onKeyPress={(e) => { if (e.key === 'Enter' && !loading) handleAddDevice(); }}
                    disabled={loading}
                    style={{
                      width: '100%', boxSizing: 'border-box',
                      padding: '12px', background: '#282a36',
                      border: '1px solid #44475a', borderRadius: '4px',
                      color: '#f8f8f2', fontSize: '1rem', fontFamily: 'monospace'
                    }}
                  />
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button
                    onClick={handleAddDevice}
                    disabled={loading}
                    style={{
                      flex: 1,
                      padding: '8px 16px',
                      background: '#50fa7b',
                      color: '#282a36',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      opacity: loading ? 0.5 : 1
                    }}
                  >
                    ➕ Add Device
                  </button>
                  <button
                    onClick={() => {
                      setShowAddDevice(false);
                      setPairingInput('');
                    }}
                    disabled={loading}
                    style={{
                      flex: 1,
                      padding: '8px 16px',
                      background: '#44475a',
                      color: '#f8f8f2',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      opacity: loading ? 0.5 : 1
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Devices */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ color: '#ffb86c', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
          📡 Devices ({peers.length})
        </div>

        {syncStatus === 'stopped' ? (
          <div style={{
            background: '#1e1f29',
            padding: '15px',
            borderRadius: '6px',
            fontSize: '0.85rem',
            color: '#9aa5c4',
            lineHeight: '1.6'
          }}>
            Start ZynkSync to begin syncing with your other devices.
          </div>
        ) : peers.length === 0 ? (
          <div style={{
            background: '#1e1f29',
            padding: '15px',
            borderRadius: '6px',
            fontSize: '0.85rem',
            lineHeight: '1.6'
          }}>
            <div style={{ color: '#8be9fd', marginBottom: '8px' }}>
              No devices added yet.
            </div>
            <div style={{ color: '#9aa5c4', marginBottom: '5px' }}>
              <strong>To sync devices:</strong>
            </div>
            <ol style={{ color: '#9aa5c4', marginLeft: '20px', marginTop: '0', marginBottom: '10px' }}>
              <li>Start ZynkSync on both devices</li>
              <li>On the device with existing data: Generate a 6-digit code</li>
              <li>On the other device: Enter the IP:code from step 2</li>
              <li>Devices will sync automatically every 60 seconds</li>
            </ol>
            <div style={{ color: '#9aa5c4', fontSize: '0.8rem', marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #44475a' }}>
              <strong style={{ color: '#ff5555' }}>To break sync:</strong> Use the Identity button to generate a new user ID. The de-synced device becomes a new user with its own Zynkbot (retaining memories from when it was linked). Use Memory Manager for a completely fresh start.
            </div>
          </div>
        ) : (
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {peers.map((peer, idx) => (
              <div
                key={idx}
                style={{
                  background: '#1e1f29',
                  padding: '12px',
                  borderRadius: '4px',
                  marginBottom: '10px',
                  fontSize: '0.85rem',
                  border: '2px solid #50fa7b'
                }}
              >
                {/* Device Header */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                  <div style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    background: '#50fa7b',
                    flexShrink: 0
                  }} />
                  <div style={{ color: '#f8f8f2', fontWeight: 'bold', flex: 1 }}>
                    {peer.device_name}
                  </div>
                </div>

                {/* Device Info */}
                <div style={{ color: '#9aa5c4', fontSize: '0.75rem', marginBottom: '10px' }}>
                  <div>ID: {peer.device_id?.slice(0, 16)}...</div>
                  <div>Host: {peer.host}</div>
                  <div>Port: {peer.port}</div>
                  <div>URL: {peer.url}</div>
                </div>

                {/* Action Buttons */}
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => handleSyncToPeer(peer)}
                    disabled={loading}
                    style={{
                      flex: 1,
                      padding: '8px',
                      background: '#50fa7b',
                      color: '#282a36',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      opacity: loading ? 0.5 : 1
                    }}
                    title="Sync memories to this device now"
                  >
                    🔄 Sync Now
                  </button>
                  <button
                    onClick={() => handleRemoveDevice(peer)}
                    disabled={loading}
                    style={{
                      flex: 1,
                      padding: '8px',
                      background: '#ff5555',
                      color: '#f8f8f2',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      opacity: loading ? 0.5 : 1
                    }}
                    title="Remove this device"
                  >
                    ❌ Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Message Display */}
      {message && (
        <div style={{
          marginTop: '10px',
          padding: '10px',
          background: message.includes('✓') ? '#1e3a1e' : '#3a1e1e',
          border: `1px solid ${message.includes('✓') ? '#50fa7b' : '#ff5555'}`,
          borderRadius: '4px',
          color: message.includes('✓') ? '#50fa7b' : '#ff5555',
          fontSize: '0.85rem',
          whiteSpace: 'pre-line'
        }}>
          {message}
        </div>
      )}

    </div>
  );
}
