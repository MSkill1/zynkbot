import React, { useState } from 'react';
import '../styles/SessionInfoModal.css';
import ControlsPage from './ControlsPage';

export default function SessionInfoModal({ show, onClose, userId, sessionId }) {
  const [copiedUser, setCopiedUser] = useState(false);
  const [copiedSession, setCopiedSession] = useState(false);

  if (!show) return null;

  const copyToClipboard = (text, type) => {
    navigator.clipboard.writeText(text).then(() => {
      if (type === 'user') {
        setCopiedUser(true);
        setTimeout(() => setCopiedUser(false), 2000);
      } else {
        setCopiedSession(true);
        setTimeout(() => setCopiedSession(false), 2000);
      }
    });
  };

  return (
    <ControlsPage icon="🪪" title="Session Information" accent="#8be9fd" closeLabel="Close" zIndex={2000} size="small" onClose={onClose}>
      {/* The shared page shell (ui-redo, 2026-11-18). */}
        <div className="info-section">
          <div className="info-label">User ID</div>
          <div className="info-value-row">
            <code className="info-value">{userId}</code>
            <button
              className="copy-btn"
              onClick={() => copyToClipboard(userId, 'user')}
            >
              {copiedUser ? '✓ Copied' : '📋 Copy'}
            </button>
          </div>
        </div>

        <div className="info-section">
          <div className="info-label">Session ID</div>
          <div className="info-value-row">
            <code className="info-value">{sessionId}</code>
            <button
              className="copy-btn"
              onClick={() => copyToClipboard(sessionId, 'session')}
            >
              {copiedSession ? '✓ Copied' : '📋 Copy'}
            </button>
          </div>
        </div>
    </ControlsPage>
  );
}
