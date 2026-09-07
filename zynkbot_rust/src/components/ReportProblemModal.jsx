import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

const ISSUES_URL = 'https://github.com/MSkill1/zynkbot/issues/new';

// "Report a problem": builds a plain-text report on the device (version, build,
// device, model backend, the user's description, optionally the current
// conversation, and the last few hundred log lines with credentials masked).
// Nothing is sent anywhere; the user copies it into a GitHub issue themselves.
export default function ReportProblemModal({ isOpen, onClose, context, threadText, backend }) {
  const [description, setDescription] = useState('');
  const [includeThread, setIncludeThread] = useState(false);
  const [report, setReport] = useState('');
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setDescription(context ? `About this reply:\n"${context.slice(0, 300)}${context.length > 300 ? '…' : ''}"\n\n` : '');
    setIncludeThread(false);
    setReport('');
    setCopied(false);
    setError('');
  }, [isOpen, context]);

  if (!isOpen) return null;

  const deviceInfo = () => {
    try { if (window.AndroidPaths?.getDeviceInfo) return window.AndroidPaths.getDeviceInfo(); } catch (_) {}
    return navigator.userAgent;
  };

  const build = async () => {
    setBusy(true); setError('');
    try {
      const text = await invoke('build_problem_report', {
        description,
        device: deviceInfo(),
        backend: backend || '',
        thread: includeThread ? (threadText || '') : null,
        logLines: 300,
      });
      setReport(text);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) {
      setError('Could not copy. Select the text and copy it by hand.');
    }
  };

  const openIssues = async () => {
    try { await invoke('open_external_url', { url: ISSUES_URL }); }
    catch (_) { window.open(ISSUES_URL, '_blank'); }
  };

  const btn = {
    padding: '8px 14px', borderRadius: '6px', border: '1px solid #6272a4',
    background: 'rgba(98,114,164,0.25)', color: '#8be9fd', cursor: 'pointer', fontWeight: 600, fontSize: '0.85rem',
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#22232a', color: '#f8f8f2', borderRadius: '10px', padding: '18px',
          width: 'min(720px, 94vw)', maxHeight: '90vh', overflowY: 'auto', position: 'relative',
          border: '1px solid #44475a',
        }}
      >
        <button className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2 style={{ margin: '0 0 6px 0', color: '#8be9fd' }}>Report a problem</h2>
        <p style={{ margin: '0 0 12px 0', color: '#9aa5c4', fontSize: '0.9rem', lineHeight: 1.5 }}>
          This builds a text report on your device: app version, build, device, the model in use,
          your description, and the last few hundred lines of Zynkbot's own log with any keys masked.
          Nothing is sent anywhere. You copy it and paste it into a GitHub issue.
        </p>

        <label style={{ display: 'block', fontSize: '0.85rem', color: '#9aa5c4', marginBottom: '4px' }}>
          What happened? What did you expect?
        </label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={5}
          style={{ width: '100%', boxSizing: 'border-box', background: '#1e1f29', color: '#f8f8f2', border: '1px solid #44475a', borderRadius: '6px', padding: '8px', fontSize: '0.9rem' }}
          placeholder="e.g. Said 'Hey Zynk, set a timer for two minutes'. It confirmed but no timer appeared."
        />

        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '10px 0', fontSize: '0.85rem', color: '#f8f8f2' }}>
          <input type="checkbox" checked={includeThread} onChange={(e) => setIncludeThread(e.target.checked)} />
          Include the current conversation text in the report
        </label>

        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
          <button style={btn} onClick={build} disabled={busy}>{busy ? 'Building…' : '🧾 Build report'}</button>
          {report && <button style={btn} onClick={copy}>{copied ? '✓ Copied' : '📋 Copy report'}</button>}
          {report && <button style={btn} onClick={openIssues}>Open GitHub issues</button>}
        </div>
        {error && <p style={{ color: '#ff5555', fontSize: '0.85rem' }}>{error}</p>}

        {report && (
          <>
            <p style={{ margin: '6px 0', color: '#9aa5c4', fontSize: '0.8rem' }}>
              Read it before you paste it. Keys are masked automatically, but check for anything else you would rather not share.
            </p>
            <textarea
              readOnly
              value={report}
              rows={14}
              style={{ width: '100%', boxSizing: 'border-box', background: '#1e1f29', color: '#ececec', border: '1px solid #44475a', borderRadius: '6px', padding: '8px', fontFamily: 'monospace', fontSize: '0.75rem' }}
            />
          </>
        )}
      </div>
    </div>
  );
}
