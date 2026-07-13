import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import '../styles/UserIdentityModal.css';

export default function UserIdentityModal({ isOpen, onClose, apiBaseUrl, sessionId }) {
  const [identity, setIdentity] = useState(null);
  const [deviceIp, setDeviceIp] = useState('Loading...');
  const [copyFeedback, setCopyFeedback] = useState('');
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (isOpen) {
      fetchIdentity();
      fetchDeviceIp();
    }
  }, [isOpen]);

  const fetchIdentity = async () => {
    try {
      setError(null);
      const data = await invoke('get_user_identity');
      setIdentity(data);
    } catch (error) {
      console.error('Failed to fetch identity:', error);
      setError(error.toString() || 'Failed to connect to backend. Make sure the backend is running.');
    }
  };

  const fetchDeviceIp = async () => {
    try {
      const ip = await invoke('get_local_ip');
      setDeviceIp(ip);
    } catch (error) {
      console.error('Failed to fetch device IP:', error);
      setDeviceIp('Unable to determine');
    }
  };

  const handleCopyUserId = () => {
    if (identity?.user_id) {
      navigator.clipboard.writeText(identity.user_id);
      setCopyFeedback('user_id');
      setTimeout(() => setCopyFeedback(''), 2000);
    }
  };

  const handleCopyDeviceId = () => {
    if (identity?.device_id) {
      navigator.clipboard.writeText(identity.device_id);
      setCopyFeedback('device_id');
      setTimeout(() => setCopyFeedback(''), 2000);
    }
  };

  const handleCopyIp = () => {
    if (deviceIp && deviceIp !== 'Loading...' && deviceIp !== 'Unable to determine') {
      navigator.clipboard.writeText(deviceIp);
      setCopyFeedback('ip');
      setTimeout(() => setCopyFeedback(''), 2000);
    }
  };

  const handleCopySessionId = () => {
    if (sessionId) {
      navigator.clipboard.writeText(sessionId);
      setCopyFeedback('session_id');
      setTimeout(() => setCopyFeedback(''), 2000);
    }
  };

  const handleResetIdentity = async () => {
    const confirmed = window.confirm(
      '⚠️ DANGER: This will create a NEW User ID for this device!\n\n' +
      'Current memories will remain in the database but will be associated with your OLD User ID.\n' +
      'You will start fresh with a new identity.\n\n' +
      'Are you absolutely sure?'
    );

    if (!confirmed) return;

    try {
      setMessage('⏳ Creating new identity...');
      const newIdentity = await invoke('reset_user_identity');
      setIdentity(newIdentity);
      setMessage('✅ New identity created successfully! User ID: ' + newIdentity.user_id);

      // Reload after showing success message
      setTimeout(() => {
        window.location.reload();
      }, 2000);
    } catch (error) {
      console.error('Failed to reset identity:', error);
      setError(`❌ Failed to reset identity: ${error.toString()}`);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="user-identity-modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>👤 User Identity</h2>
          <button onClick={onClose} className="close-button">×</button>
        </div>

        <div className="modal-body">
          {/* Error Display */}
          {error && (
            <section className="identity-section error-section" style={{
              background: '#ff5555',
              color: '#f8f8f2',
              padding: '15px',
              borderRadius: '6px',
              marginBottom: '20px'
            }}>
              <h3 style={{ margin: '0 0 10px 0' }}>⚠️ Error</h3>
              <p style={{ margin: 0 }}>{error}</p>
            </section>
          )}

          {/* Success Message Display */}
          {message && (
            <section className="identity-section" style={{
              background: '#1e3a1e',
              border: '1px solid #50fa7b',
              color: '#50fa7b',
              padding: '15px',
              borderRadius: '6px',
              marginBottom: '20px'
            }}>
              <p style={{ margin: 0 }}>{message}</p>
            </section>
          )}

          {/* Current Identity Section */}
          <section className="identity-section">
            <h3>📱 Device Information</h3>
            {identity ? (
              <>
                <div className="identity-field" style={{ marginBottom: '20px' }}>
                  <label>User ID:</label>
                  <div className="id-display">
                    <code style={{ wordBreak: 'break-all', whiteSpace: 'normal' }}>{identity.user_id}</code>
                    <button onClick={handleCopyUserId} className="copy-button" title="Copy User ID">
                      {copyFeedback === 'user_id' ? '✅' : '📋'}
                    </button>
                  </div>
                  <p className="help-text">
                    Shared across all your synced devices. Use the ZynkSync panel in Settings to link devices.
                  </p>
                </div>

                <div className="identity-field" style={{ marginBottom: '20px' }}>
                  <label>Device ID:</label>
                  <div className="id-display">
                    <code style={{ wordBreak: 'break-all', whiteSpace: 'normal' }}>{identity.device_id}</code>
                    <button onClick={handleCopyDeviceId} className="copy-button" title="Copy Device ID">
                      {copyFeedback === 'device_id' ? '✅' : '📋'}
                    </button>
                  </div>
                  <p className="help-text">
                    Unique to this specific device.
                  </p>
                </div>

                <div className="identity-field" style={{ marginBottom: '20px' }}>
                  <label>Session ID:</label>
                  <div className="id-display">
                    <code style={{ wordBreak: 'break-all', whiteSpace: 'normal' }}>{sessionId || '—'}</code>
                    {sessionId && (
                      <button onClick={handleCopySessionId} className="copy-button" title="Copy Session ID">
                        {copyFeedback === 'session_id' ? '✅' : '📋'}
                      </button>
                    )}
                  </div>
                  <p className="help-text">
                    Identifies the current conversation session.
                  </p>
                </div>

                <div className="identity-field">
                  <label>IP Address:</label>
                  <div className="id-display">
                    <code>{deviceIp}</code>
                    {deviceIp && deviceIp !== 'Loading...' && deviceIp !== 'Unable to determine' && (
                      <button onClick={handleCopyIp} className="copy-button" title="Copy IP Address">
                        {copyFeedback === 'ip' ? '✅' : '📋'}
                      </button>
                    )}
                  </div>
                  <p className="help-text">
                    Your device's local network IP address. Share this when pairing devices.
                  </p>
                </div>
              </>
            ) : !error ? (
              <p>Loading identity...</p>
            ) : null}
          </section>

          {/* Info Section */}
          <section className="identity-section info-section">
            <h3>ℹ️ About Identity</h3>
            <div style={{ fontSize: '0.9rem', lineHeight: '1.6', color: '#f8f8f2' }}>
              <div style={{ marginBottom: '12px' }}>
                <strong style={{ color: '#50fa7b' }}>User ID:</strong> Identifies YOU across all your devices.
                Devices with the same User ID automatically sync memories through ZynkSync.
              </div>
              <div style={{ marginBottom: '12px' }}>
                <strong style={{ color: '#8be9fd' }}>Device ID:</strong> Identifies THIS specific device.
                Used for ZynkLink file sharing with other users.
              </div>
              <div>
                <strong style={{ color: '#bd93f9' }}>Session ID:</strong> Identifies the current conversation session. Changes when you start a new chat.
              </div>
              <div style={{ marginTop: '12px' }}>
                <strong style={{ color: '#ffb86c' }}>Syncing Devices:</strong> Use the ZynkSync panel in Settings
                to generate pairing codes and link your devices.
              </div>
            </div>
          </section>

          {/* Danger Zone */}
          <section className="identity-section danger-section">
            <h3>⚠️ Danger Zone</h3>
            <button
              onClick={handleResetIdentity}
              className="danger-button"
            >
              🔄 Create New User ID
            </button>
            <p className="help-text">
              This creates a completely new User ID for this device. Use only if you want to start fresh or disconnect from synced devices.
              Your current memories will remain but won't sync to the new User ID.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
