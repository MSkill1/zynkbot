import React, { useState } from 'react';
import '../styles/ControlsPage.css';

const seenKey = (k) => `zynkbot_intro_seen_${k}`;
export function introSeen(k) {
  try { return localStorage.getItem(seenKey(k)) === '1'; } catch (_) { return false; }
}

/**
 * The first time a section is opened, a short explanation sits at the top of it:
 * two or three sentences and one "try this", with a Got it button. After that it
 * folds to a small "What is this?" link, one tap away and out of the way. Never a
 * forced tour (Matt, 2026-10-25). Per device, in localStorage; `noLink` means the
 * card shows once and then nothing (the chat screen).
 */
export default function PageIntro({ introKey, accent = '#6272a4', noLink = false, children }) {
  const [seen, setSeen] = useState(() => introSeen(introKey));
  const [open, setOpen] = useState(false);

  const dismiss = () => {
    try { localStorage.setItem(seenKey(introKey), '1'); } catch (_) {}
    setSeen(true);
    setOpen(false);
  };

  if (seen && !open) {
    if (noLink) return null;
    return (
      <div className="page-intro-link">
        <button onClick={() => setOpen(true)}>What is this?</button>
      </div>
    );
  }
  return (
    <div className="page-intro" style={{ borderColor: accent }}>
      <div className="page-intro-text">{children}</div>
      <button className="page-intro-ok" onClick={dismiss}>Got it</button>
    </div>
  );
}
