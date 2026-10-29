import React, { useState } from 'react';

// A check-in question (labs/check_in, 2026-10-29): one line from Zynkbot about something
// the user said they would do, with the three answers that close it. Shown above the
// conversation, not stored in it. Experimental in beta 5; the switch is on the System
// Controls page under Experimental.
export default function CheckInCard({ checkin, onAnswer, onDismiss }) {
  const [busy, setBusy] = useState(false);
  if (!checkin) return null;
  const answer = async (a) => {
    if (busy) return;
    setBusy(true);
    try { await onAnswer(a); } finally { setBusy(false); }
  };
  const btn = (label, a, title) => (
    <button
      type="button"
      onClick={() => answer(a)}
      disabled={busy}
      title={title}
      style={{ background: 'transparent', border: '1px solid #44475a', color: '#8be9fd', borderRadius: '6px', padding: '4px 10px', fontSize: '0.8rem', cursor: 'pointer' }}
    >{label}</button>
  );
  return (
    <div
      role="status"
      style={{ margin: '0 0 8px', padding: '10px 12px', background: '#22232a', border: '1px solid #44475a', borderLeft: '3px solid #f1fa8c', borderRadius: '8px' }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
        <div style={{ flex: 1, color: '#ececec', fontSize: '0.95rem', lineHeight: 1.5 }}>
          <strong style={{ color: '#8be9fd' }}>Zynkbot:</strong> {checkin.question}
        </div>
        <button
          type="button"
          onClick={onDismiss}
          title="Leave it as it is"
          aria-label="Dismiss"
          style={{ background: 'transparent', border: 'none', color: '#9aa5c4', cursor: 'pointer', fontSize: '1rem', lineHeight: 1, padding: '0 2px' }}
        >✕</button>
      </div>
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '8px' }}>
        {btn('Done', 'done', 'It happened: the memory is kept as history and leaves recall')}
        {btn('Not yet', 'later', 'Still planned: kept for another two weeks')}
        {btn('Never mind', 'never', 'Dropped: the memory is deleted')}
      </div>
    </div>
  );
}
