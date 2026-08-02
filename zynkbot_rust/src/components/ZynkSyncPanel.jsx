import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export default function ZynkSyncPanel({ userId, onOpenUserIdentity, onOpenChat, onIdentityAdopted, onMemoriesSynced }) {
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

  const fetchPeers = useCallback(async () => {
    try {
      const peers = await invoke('get_zynksync_peers');
      setPeers(peers || []);
    } catch (error) {
      console.error('[ZynkSync] Failed to fetch peers:', error);
    }
  }, []);

  // Pause auto-sync
  const handlePause = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      setMessage('Pausing ZynkSync...');
      await invoke('stop_zynksync');
      setSyncStatus('stopped');
      setMessage('ZynkSync paused. Press Sync Now to sync immediately and resume.');
      setPairingCode('');
      setLocalIp('');
      setShowAddDevice(false);
    } catch (error) {
      setMessage('✗ Failed to pause: ' + error);
    } finally {
      setLoading(false);
    }
  }, [loading]);

  // Sync now: trigger immediate sync and resume auto-sync if paused
  const handleSyncNow = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      setMessage('Syncing...');
      // Start/resume the service if it was paused
      if (syncStatus !== 'running') {
        await invoke('start_zynksync', { syncIntervalSecs: 60 });
        setSyncStatus('running');
      }
      await fetchPeers();
      try {
        const results = await invoke('broadcast_sync_to_all_peers', { userId: userId || '' });
        const total = results.reduce((s, r) => s + (r.memories_sent || 0) + (r.memories_received || 0), 0);
        setMessage(total > 0 ? `✓ Synced — ${total} memories exchanged` : '✓ All devices already in sync');
        if (onMemoriesSynced) onMemoriesSynced();
      } catch {
        setMessage('✓ Resumed — no active peers to sync with yet');
      }
    } catch (error) {
      setMessage('✗ Sync failed: ' + error);
    } finally {
      setLoading(false);
      setTimeout(() => setMessage(''), 3000);
    }
  }, [loading, syncStatus, fetchPeers, userId, onMemoriesSynced]);

  // Refresh peer list (no sync)
  const handleRefresh = useCallback(async () => {
    setMessage('Refreshing...');
    await fetchPeers();
    setMessage('✓ Refreshed');
    setTimeout(() => setMessage(''), 2000);
  }, [fetchPeers]);

  // Leave the entire sync network
  const handleUnsync = useCallback(async () => {
    if (!window.confirm(
      'Leave the sync network?\n\n' +
      'This device will disconnect from all other devices. ' +
      'Your memories stay on this device — nothing is deleted. ' +
      'Other devices will also remove this device from their lists.\n\n' +
      'You can re-join the network at any time by entering a pairing code.'
    )) return;

    setLoading(true);
    setMessage('Leaving sync network...');
    try {
      const newUserId = await invoke('unsync_and_reset_identity');
      setPeers([]);
      setPairingCode('');
      setLocalIp('');
      setShowAddDevice(false);
      setMessage('✓ Left the sync network. Your memories are safe on this device.');
      if (onIdentityAdopted) onIdentityAdopted(newUserId);
    } catch (error) {
      setMessage('✗ Failed to unsync: ' + error);
    } finally {
      setLoading(false);
    }
  }, [onIdentityAdopted]);

  const handleExpelDevice = useCallback(async (deviceId, deviceName) => {
    if (!window.confirm(
      `Remove "${deviceName}" from the network?\n\n` +
      'This device will be removed from all other devices in the network. ' +
      'If the device is online it will also be notified.\n\n' +
      'It can rejoin at any time by entering a new pairing code.'
    )) return;

    try {
      await invoke('expel_zynksync_device', { deviceId });
      setPeers(prev => prev.filter(p => p.device_id !== deviceId));
      setMessage(`✓ Removed ${deviceName} from the network.`);
    } catch (error) {
      setMessage(`✗ Failed to remove ${deviceName}: ` + error);
    }
  }, []);

  // Get pairing code and IP
  const handleGetPairingCode = async () => {
    try {
      const [code, ip] = await Promise.all([
        invoke('get_zynksync_pairing_code'),
        invoke('get_local_ip')
      ]);
      setPairingCode(code);
      setLocalIp(ip);
      setMessage('✓ Share the IP:code below with the other device');
    } catch (error) {
      setMessage(`✗ Failed to get pairing info: ${error}`);
    }
  };

  const handleCopyPairingInfo = () => {
    if (pairingCode && localIp) {
      navigator.clipboard.writeText(`${localIp}:${pairingCode}`);
      setMessage('✓ IP:Code copied to clipboard');
    }
  };

  // Add a device by entering a pairing code
  const handleAddDevice = async () => {
    const input = pairingInput.trim();
    if (!input) {
      setMessage('✗ Please enter the pairing code in IP:code format');
      return;
    }
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

      if (peer.user_id && peer.user_id !== userId) {
        const confirmed = window.confirm(
          `⚠️ IDENTITY SYNC\n\n` +
          `Joining this network will:\n` +
          `• Migrate your memories to match the host's identity\n` +
          `• Start syncing with all devices on the network\n\n` +
          `Host: ${peer.device_name}\n\n` +
          `Continue?`
        );
        if (!confirmed) {
          setMessage('✗ Pairing cancelled');
          setPairingInput('');
          setShowAddDevice(false);
          setLoading(false);
          return;
        }
        setMessage('⏳ Migrating memories to host identity...');
        try {
          const migratedCount = await invoke('migrate_user_memories', {
            oldUserId: userId,
            newUserId: peer.user_id
          });
          console.log('[ZynkSync] Migrated', migratedCount, 'memories');
          await invoke('set_user_identity', { userId: peer.user_id });
          setMessage(`✓ Joined network — syncing with ${peer.device_name}`);
          if (onIdentityAdopted) onIdentityAdopted(peer.user_id);
        } catch (identityError) {
          setMessage(`✗ Identity sync failed: ${identityError}`);
        }
      } else {
        setMessage(`✓ Joined network — syncing with ${peer.device_name}`);
      }

      setPairingInput('');
      setPairingIPPart('');
      setPairingNumPart('');
      setShowAddDevice(false);
      fetchPeers();
    } catch (error) {
      setMessage(`✗ Failed to join: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Check status on mount — auto-start if peers exist
  useEffect(() => {
    const checkServiceStatus = async () => {
      try {
        const isRunning = await invoke('get_zynksync_status');
        if (isRunning) {
          setSyncStatus('running');
          fetchPeers();
        } else {
          const existingPeers = await invoke('get_zynksync_peers');
          if (existingPeers && existingPeers.length > 0) {
            await invoke('start_zynksync', { syncIntervalSecs: 60 });
            setSyncStatus('running');
            fetchPeers();
          } else {
            setSyncStatus('stopped');
          }
        }
      } catch (error) {
        console.error('[ZynkSync] Failed to check status:', error);
        setSyncStatus('stopped');
      }
    };
    checkServiceStatus();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for pairing warnings
  useEffect(() => {
    let unlisten;
    const setup = async () => {
      unlisten = await listen('zynksync://warning', (event) => {
        const warning = event.payload;
        if (warning?.message) {
          setMessage(`${warning.severity === 'high' ? '⚠️' : 'ℹ️'} ${warning.message}`);
        }
      });
    };
    setup();
    return () => { if (typeof unlisten === 'function') unlisten(); };
  }, []);

  // Listen for remote unsync
  useEffect(() => {
    let unlisten;
    const setup = async () => {
      unlisten = await listen('zynksync-device-removed', () => {
        fetchPeers();
        setMessage('✓ A device left the network — peer list updated.');
      });
    };
    setup();
    return () => { if (typeof unlisten === 'function') unlisten(); };
  }, [fetchPeers]);

  // Listen for remote pause/resume
  useEffect(() => {
    let unlisten;
    const setup = async () => {
      unlisten = await listen('zynksync-status-changed', (event) => {
        const { status } = event.payload;
        if (status === 'paused') {
          setSyncStatus('stopped');
          setMessage('ZynkSync paused by another device. Press Sync Now to resume.');
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

  // Auto-refresh peer list every 30s when running
  useEffect(() => {
    if (syncStatus === 'running' && autoRefresh) {
      const interval = setInterval(fetchPeers, 30000);
      return () => clearInterval(interval);
    }
  }, [syncStatus, autoRefresh, fetchPeers]);

  const isRunning = syncStatus === 'running';
  const isAndroid = !!window.AndroidPaths;

  return (
    <div style={{
      background: '#282a36',
      border: '1px solid #44475a',
      borderRadius: '8px',
      padding: '15px',
      marginBottom: '20px'
    }}>

      {/* 4-Button Control Row */}
      <div style={{ display: 'flex', flexWrap: isAndroid ? 'wrap' : 'nowrap', gap: '8px', marginBottom: '15px' }}>

        {/* Pause / Sync Now */}
        <button
          onClick={isRunning ? handlePause : handleSyncNow}
          disabled={loading}
          style={{
            flex: isAndroid ? '1 1 calc(50% - 4px)' : 1,
            padding: '8px 10px',
            background: isRunning ? '#ffb86c' : '#50fa7b',
            color: '#282a36',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'wait' : 'pointer',
            fontSize: '0.82rem',
            fontWeight: 'bold',
            opacity: loading ? 0.5 : 1
          }}
          title={isRunning ? 'Pause auto-sync' : 'Sync now and resume auto-sync'}
        >
          {isRunning ? '⏸ Pause' : '▶ Sync Now'}
        </button>

        {/* Refresh */}
        <button
          onClick={handleRefresh}
          disabled={loading}
          style={{
            flex: isAndroid ? '1 1 calc(50% - 4px)' : 1,
            padding: '8px 10px',
            background: '#6272a4',
            color: '#f8f8f2',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'wait' : 'pointer',
            fontSize: '0.82rem',
            fontWeight: 'bold',
            opacity: loading ? 0.5 : 1
          }}
          title="Refresh peer list"
        >
          🔄 Refresh
        </button>

        {/* Unsync */}
        <button
          onClick={handleUnsync}
          disabled={loading || peers.length === 0}
          style={{
            flex: isAndroid ? '1 1 calc(50% - 4px)' : 1,
            padding: '8px 10px',
            background: peers.length > 0 ? '#ff5555' : '#3a2a2a',
            color: peers.length > 0 ? '#f8f8f2' : '#6272a4',
            border: 'none',
            borderRadius: '4px',
            cursor: (loading || peers.length === 0) ? 'default' : 'pointer',
            fontSize: '0.82rem',
            fontWeight: 'bold',
            opacity: loading ? 0.5 : 1
          }}
          title="Leave the sync network"
        >
          ⏏ Unsync
        </button>

        {/* Identity */}
        <button
          onClick={onOpenUserIdentity}
          style={{
            flex: isAndroid ? '1 1 calc(50% - 4px)' : 1,
            padding: '8px 10px',
            background: '#bd93f9',
            color: '#282a36',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '0.82rem',
            fontWeight: 'bold'
          }}
          title="Manage your identity"
        >
          👤 Identity
        </button>

      </div>

      {/* Status Badge */}
      <div style={{
        display: 'inline-block',
        padding: '5px 10px',
        background: isRunning ? '#1e3a1e' : '#2a2a1e',
        border: `1px solid ${isRunning ? '#50fa7b' : '#ffb86c'}`,
        borderRadius: '4px',
        fontSize: '0.78rem',
        fontWeight: 'bold',
        color: isRunning ? '#50fa7b' : '#ffb86c',
        marginBottom: '15px'
      }}>
        {isRunning ? '● Syncing automatically' : '○ Paused — press Sync Now to resume'}
      </div>

      {/* Pairing: Generate Code */}
      {isRunning && (
        <div style={{
          marginBottom: '15px',
          padding: '12px',
          background: '#1e1f29',
          borderRadius: '6px',
          border: '1px solid #44475a'
        }}>
          <div style={{ color: '#ffb86c', fontWeight: 'bold', fontSize: '0.9rem', marginBottom: '8px' }}>
            🔑 Generate Code
          </div>
          <p style={{ fontSize: '0.82rem', color: '#9aa5c4', marginBottom: '6px', lineHeight: '1.5' }}>
            Any device can generate a code. The device that <em>enters</em> the code joins the existing network.
            Code expires in 10 minutes.
          </p>
          <p style={{ fontSize: '0.8rem', color: '#ff5555', marginBottom: '10px', lineHeight: '1.5' }}>
            ⚠️ <strong>Tip:</strong> The device with existing memories you care about should generate the code, not enter it.
          </p>
          {pairingCode && localIp ? (
            <div>
              <div style={{ marginBottom: '4px' }}>
                <div style={{ fontSize: '0.75rem', color: '#6272a4', marginBottom: '2px' }}>IP address</div>
                <div style={{
                  padding: '10px 14px',
                  background: '#282a36',
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '1.15rem',
                  color: '#50fa7b',
                  textAlign: 'center',
                  letterSpacing: '2px',
                  border: '2px solid #50fa7b',
                  whiteSpace: 'nowrap',
                  overflowX: 'auto'
                }}>
                  {localIp}
                </div>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <div style={{ fontSize: '0.75rem', color: '#6272a4', marginBottom: '2px' }}>6-digit code</div>
                <div style={{
                  padding: '10px 14px',
                  background: '#282a36',
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '1.15rem',
                  color: '#50fa7b',
                  textAlign: 'center',
                  letterSpacing: '4px',
                  border: '2px solid #50fa7b'
                }}>
                  {pairingCode}
                </div>
              </div>
              <button
                onClick={handleCopyPairingInfo}
                style={{
                  width: '100%',
                  padding: '9px',
                  background: '#50fa7b',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: 'bold'
                }}
              >
                📋 Copy IP Address + Code
              </button>
            </div>
          ) : (
            <button
              onClick={handleGetPairingCode}
              style={{
                width: '100%',
                padding: '9px',
                background: '#50fa7b',
                color: '#282a36',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.88rem'
              }}
            >
              🔑 Generate Code
            </button>
          )}
        </div>
      )}

      {/* Enter Code from Another Device */}
      {isRunning && (
        <div style={{
          marginBottom: '15px',
          padding: '12px',
          background: '#1e1f29',
          borderRadius: '6px',
          border: '1px solid #44475a'
        }}>
          <div style={{ color: '#ffb86c', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
            ➕ Enter Code from Another Device
          </div>
          {!showAddDevice ? (
            <button
              onClick={() => setShowAddDevice(true)}
              style={{
                width: '100%',
                padding: '8px',
                background: '#ffb86c',
                color: '#282a36',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.85rem',
                fontWeight: 'bold'
              }}
            >
              ➕ Enter IP:Code
            </button>
          ) : (
            <div>
              <div style={{ fontSize: '0.82rem', color: '#9aa5c4', marginBottom: '8px' }}>
                Enter the IP address and 6-digit code shown on the other device — they're displayed together as <span style={{ fontFamily: 'monospace', color: '#50fa7b' }}>192.168.x.x:123456</span>. Enter each part in its own box below.
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '8px' }}>
                <div style={{ fontSize: '0.75rem', color: '#6272a4' }}>IP address (the numbers before the colon)</div>
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
                    padding: '11px', background: '#282a36',
                    border: '1px solid #44475a', borderRadius: '4px',
                    color: '#f8f8f2', fontSize: '1rem', fontFamily: 'monospace'
                  }}
                />
                <div style={{ fontSize: '0.75rem', color: '#6272a4' }}>6-digit code (the numbers after the colon)</div>
                <input
                  type="text"
                  inputMode="numeric"
                  autoComplete="off"
                  placeholder="123456"
                  value={pairingNumPart}
                  onChange={(e) => { setPairingNumPart(e.target.value); setPairingInput(pairingIPPart + ':' + e.target.value); }}
                  onKeyPress={(e) => { if (e.key === 'Enter' && !loading) handleAddDevice(); }}
                  disabled={loading}
                  style={{
                    width: '100%', boxSizing: 'border-box',
                    padding: '11px', background: '#282a36',
                    border: '1px solid #44475a', borderRadius: '4px',
                    color: '#f8f8f2', fontSize: '1rem', fontFamily: 'monospace'
                  }}
                />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={handleAddDevice}
                    disabled={loading}
                    style={{
                      flex: 1, padding: '8px',
                      background: '#50fa7b', color: '#282a36',
                      border: 'none', borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem', fontWeight: 'bold',
                      opacity: loading ? 0.5 : 1
                    }}
                  >
                    ➕ Join Network
                  </button>
                  <button
                    onClick={() => { setShowAddDevice(false); setPairingInput(''); setPairingIPPart(''); setPairingNumPart(''); }}
                    disabled={loading}
                    style={{
                      flex: 1, padding: '8px',
                      background: '#44475a', color: '#f8f8f2',
                      border: 'none', borderRadius: '4px',
                      cursor: loading ? 'wait' : 'pointer',
                      fontSize: '0.85rem', fontWeight: 'bold',
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

      {/* Synced Devices List */}
      <div style={{ marginBottom: '15px' }}>
        <div style={{ color: '#ffb86c', fontWeight: 'bold', marginBottom: '8px', fontSize: '0.9rem' }}>
          📡 Synced Devices ({peers.length})
        </div>

        {!isRunning ? (
          <div style={{
            background: '#1e1f29', padding: '14px', borderRadius: '6px',
            fontSize: '0.85rem', color: '#9aa5c4', lineHeight: '1.6'
          }}>
            Press <strong style={{ color: '#50fa7b' }}>Sync Now</strong> to resume syncing with your devices.
          </div>
        ) : peers.length === 0 ? (
          <div style={{
            background: '#1e1f29', padding: '14px', borderRadius: '6px',
            fontSize: '0.85rem', lineHeight: '1.6'
          }}>
            <div style={{ color: '#8be9fd', marginBottom: '8px' }}>No devices connected yet.</div>
            <div style={{ color: '#9aa5c4', marginBottom: '4px' }}>
              <strong>To join devices into a sync network:</strong>
            </div>
            <ol style={{ color: '#9aa5c4', marginLeft: '18px', marginTop: '4px', marginBottom: '10px' }}>
              <li>On the device with existing memories: Generate a code above</li>
              <li>On the other device: enter that IP:code here</li>
              <li>Repeat for each device you want in the network</li>
            </ol>
            <div style={{ color: '#6272a4', fontSize: '0.78rem', borderTop: '1px solid #44475a', paddingTop: '8px' }}>
              All devices on the same network share the same memory pool and sync automatically.
              Press <strong>Unsync</strong> above to leave the network at any time.
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
                  marginBottom: '8px',
                  fontSize: '0.85rem',
                  border: `2px solid ${peer.is_online ? '#50fa7b' : '#44475a'}`
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{
                    width: '9px', height: '9px', borderRadius: '50%', flexShrink: 0,
                    background: peer.is_online ? '#50fa7b' : '#6272a4'
                  }} />
                  <div style={{ color: '#f8f8f2', fontWeight: 'bold', flex: 1 }}>
                    {peer.device_name}
                  </div>
                  <div style={{ fontSize: '0.73rem', color: peer.is_online ? '#50fa7b' : '#6272a4' }}>
                    {peer.is_online ? 'Online' : 'Offline'}
                  </div>
                  <button
                    onClick={() => handleExpelDevice(peer.device_id, peer.device_name)}
                    title={`Remove ${peer.device_name} from the network`}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#6272a4',
                      cursor: 'pointer',
                      fontSize: '0.85rem',
                      padding: '2px 5px',
                      borderRadius: '3px',
                      lineHeight: 1,
                      flexShrink: 0
                    }}
                    onMouseOver={e => e.currentTarget.style.color = '#ff5555'}
                    onMouseOut={e => e.currentTarget.style.color = '#6272a4'}
                  >✕</button>
                </div>
                <div style={{ color: '#6272a4', fontSize: '0.72rem', marginTop: '5px', paddingLeft: '17px' }}>
                  {peer.host}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Message */}
      {message && (
        <div style={{
          marginTop: '10px',
          padding: '10px',
          background: message.includes('✓') ? '#1e3a1e' : message.includes('⏳') ? '#1e1f29' : '#3a1e1e',
          border: `1px solid ${message.includes('✓') ? '#50fa7b' : message.includes('⏳') ? '#6272a4' : '#ff5555'}`,
          borderRadius: '4px',
          color: message.includes('✓') ? '#50fa7b' : message.includes('⏳') ? '#9aa5c4' : '#ff5555',
          fontSize: '0.85rem',
          whiteSpace: 'pre-line'
        }}>
          {message}
        </div>
      )}
    </div>
  );
}
