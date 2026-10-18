import React from 'react';
import '../styles/ControlsPage.css';

/** A row in the System Controls list: icon, name, one-line status, chevron. */
export default function ControlsMenuButton({ icon, title, subtitle, accent, dim, onClick }) {
  return (
    <button
      className={`controls-menu-btn${dim ? ' dim' : ''}`}
      style={accent ? { borderColor: accent } : undefined}
      onClick={onClick}
    >
      <span className="controls-menu-icon">{icon}</span>
      <span className="controls-menu-text">
        <b>{title}</b>
        {subtitle && <small>{subtitle}</small>}
      </span>
      <span className="controls-menu-chevron">›</span>
    </button>
  );
}
