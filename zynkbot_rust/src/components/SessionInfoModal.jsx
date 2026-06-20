import React, { useState } from 'react';
import '../styles/SessionInfoModal.css';

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
    <div className="modal-overlay" onClick={onClose}>
      <div className="session-info-modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2 style={{color: '#8be9fd', marginBottom: '1.5rem'}}>Session Information</h2>

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
      </div>
    </div>
  );
}
