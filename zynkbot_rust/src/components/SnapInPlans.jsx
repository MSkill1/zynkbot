import React, { useState } from 'react';
import '../styles/ControlsPage.css';

/**
 * The Snap-ins page (ui-redo, 2026-10-25). Snap-ins come after 1.0; the first three are
 * named here, each with a description that opens on tap: what it would look like and
 * why it helps in that profession. Nothing here is built; the wording says so.
 */
const PLANS = [
  {
    key: 'therapist', icon: '🧠', name: 'Therapist', color: '#ff79c6',
    what: 'A workspace per client: dated session notes, kept and indexed on this device. Between sessions, ask what came up last time, which themes keep returning, what the homework was.',
    why: 'Clinical notes are the most sensitive text a person writes. With a model on the device nothing leaves it, and HIPAA containment mode keeps chat history off.',
  },
  {
    key: 'legal', icon: '⚖️', name: 'Legal', color: '#bd93f9',
    what: 'A workspace per matter: its documents in the knowledge base, a timeline, deadlines, and your own precedents to draft from. Ask across a matter in plain words: what the lease says about renewal, when notice was served.',
    why: 'Privileged material stays on your own machine instead of a vendor\'s servers, and answers point at your documents, not the open internet.',
  },
  {
    key: 'journalist', icon: '📰', name: 'Journalist', color: '#8be9fd',
    what: 'Sources, interview notes and documents organised by story, with a timeline to check claims against. Ask what a source said in March, or which documents mention a name.',
    why: 'A reporter\'s notes protect the people in them. Kept on your device, no third-party account holds them. In a hostile environment or country, an AI assistant that runs on your own device and keeps no account anywhere may be the only one you can use.',
  },
];

export default function SnapInPlans({ onTrySample }) {
  const [open, setOpen] = useState(null);
  return (
    <div>
      <p style={{ fontSize: '0.9rem', color: '#e6e6e6', lineHeight: 1.6, margin: '0 0 14px' }}>
        A snap-in is a workspace for one kind of work, built on Zynkbot's memory and knowledge base.
        They come after version 1.0. The first three are planned; tap one to see what it will do.
      </p>
      {PLANS.map(p => {
        const isOpen = open === p.key;
        return (
          <div key={p.key} className="controls-menu-btn controls-menu-row" style={{ borderColor: p.color, flexDirection: 'column', alignItems: 'stretch', cursor: 'pointer' }} onClick={() => setOpen(isOpen ? null : p.key)}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className="controls-menu-icon">{p.icon}</span>
              <span className="controls-menu-text">
                <b>{p.name}</b>
                <small>Planned</small>
              </span>
              <span className="controls-menu-chevron">{isOpen ? '▾' : '›'}</span>
            </div>
            {isOpen && (
              <div style={{ marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #44475a', fontSize: '0.88rem', lineHeight: 1.55, color: '#e6e6e6' }}>
                <p style={{ margin: '0 0 8px' }}><b style={{ color: p.color }}>What it looks like.</b> {p.what}</p>
                <p style={{ margin: 0 }}><b style={{ color: p.color }}>Why it helps.</b> {p.why}</p>
              </div>
            )}
          </div>
        );
      })}
      {onTrySample && (
        <button onClick={onTrySample}
          style={{ width: '100%', marginTop: '6px', padding: '10px', background: 'none', border: '1px solid #6272a4', borderRadius: '8px', color: '#9aa5c4', cursor: 'pointer', fontSize: '0.85rem' }}>
          Sample: therapist journal (early demo of the idea)
        </button>
      )}
    </div>
  );
}
