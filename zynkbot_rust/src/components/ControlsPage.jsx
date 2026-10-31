import React from 'react';
import '../styles/ControlsPage.css';
import PageIntro from './PageIntro';

/**
 * One System Controls section, shown on a page of its own on top of the sidebar.
 * Only one page is open at a time; ✕ (or a tap outside the card) returns to the
 * System Controls list. Introduced in 0.9.6-beta5 to replace the stack of
 * fold-out sections, which had grown too tall to scan on a phone.
 */
export default function ControlsPage({ icon, title, subtitle, accent = '#44475a', onClose, intro, introKey, children }) {
  return (
    <div className="controls-page-overlay" onClick={onClose}>
      <div
        className="controls-page"
        style={{ borderColor: accent }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={title}
      >
        <div className="controls-page-head" style={{ borderBottomColor: accent }}>
          <span className="controls-page-icon">{icon}</span>
          <div className="controls-page-title">
            <b style={{ color: accent === '#44475a' ? '#f8f8f2' : accent }}>{title}</b>
            {subtitle && <small>{subtitle}</small>}
          </div>
          <button className="controls-page-close" onClick={onClose} title="Back to System Controls" aria-label="Close">✕</button>
        </div>
        <div className="controls-page-body">
          {intro && <PageIntro introKey={introKey || title} accent={accent === '#44475a' ? '#6272a4' : accent}>{intro}</PageIntro>}
          {children}
        </div>
      </div>
    </div>
  );
}
