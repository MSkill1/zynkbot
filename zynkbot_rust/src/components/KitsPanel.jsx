import React, { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

/**
 * The Kits page in System Controls (2026-11-27; design in labs/kits/KITS_DESIGN.md).
 * One row per kit with a switch. Switching a kit on asks its opening questions once, in
 * place; the answers can be changed later from the row. The kit's standing instructions
 * are shown in full so nothing is hidden. "Forget everything from this kit" deletes its
 * memories on every device. Kits that do not exist yet are listed greyed, labels only.
 *
 * Props: userId; onChanged() after any switch or forget, so the buttons above the message
 * box follow; notify(text) for errors.
 */
export default function KitsPanel({ userId, onChanged, notify }) {
  const [view, setView] = useState(null);
  const [asking, setAsking] = useState(null);        // kit id whose questions are open
  const [answers, setAnswers] = useState({});
  const [showText, setShowText] = useState({});
  const [confirmForget, setConfirmForget] = useState(null);

  const load = async () => {
    try { setView(await invoke('list_kits')); } catch (e) { notify?.(`Could not load the kits: ${e}`); }
  };
  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (kit, enabled, a) => {
    try {
      await invoke('set_kit_enabled', { kitId: kit.id, enabled, answers: a ?? null });
      await load();
      onChanged?.();
    } catch (e) { notify?.(`Could not save: ${e}`); }
  };

  const toggle = (kit) => {
    if (kit.enabled) { save(kit, false); return; }
    if (kit.questions.length === 0) { save(kit, true, {}); return; }
    setAnswers(kit.answers || {});
    setAsking(kit.id);
  };

  const forget = async (kit) => {
    try {
      const n = await invoke('forget_kit', { kitId: kit.id, userId });
      setConfirmForget(null);
      notify?.(n === 0 ? 'Nothing to forget yet.' : `Forgot ${n} ${n === 1 ? 'memory' : 'memories'} from the ${kit.name} kit.`);
      await load();
      onChanged?.();
    } catch (e) { notify?.(`Could not forget: ${e}`); }
  };

  if (!view) return <div style={{ color: '#9aa5c4', fontSize: '0.85rem' }}>Loading…</div>;

  const row = { background: '#1e1f29', border: '1px solid #44475a', borderRadius: '10px', padding: '10px 12px', marginBottom: '8px' };
  const small = { color: '#9aa5c4', fontSize: '0.8rem', lineHeight: 1.45 };
  const linkBtn = { background: 'none', border: 'none', color: '#8be9fd', cursor: 'pointer', font: 'inherit', fontSize: '0.8rem', padding: 0 };
  const btn = { background: '#44475a', color: '#f8f8f2', border: 'none', borderRadius: '6px', padding: '7px 12px', cursor: 'pointer', fontSize: '0.85rem' };

  return (
    <div>
      {view.kits.map((kit) => (
        <div key={kit.id} style={row}>
          <label style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', cursor: 'pointer' }}>
            <input type="checkbox" checked={kit.enabled} onChange={() => toggle(kit)} style={{ width: 'auto', margin: '4px 0 0' }} />
            <span style={{ flex: 1 }}>
              <strong>{kit.icon} {kit.name}</strong>
              <div style={small}>{kit.tagline}</div>
            </span>
          </label>

          {asking === kit.id && (
            <div style={{ marginTop: '10px', paddingLeft: '26px' }}>
              <div style={{ ...small, marginBottom: '6px' }}>A couple of questions first. Zynkbot keeps the answers in mind while the kit is on. Leave one blank if you like.</div>
              {kit.questions.map((q) => (
                <div key={q.id} style={{ marginBottom: '8px' }}>
                  <div style={{ fontSize: '0.85rem', marginBottom: '3px' }}>{q.text}</div>
                  <input type="text" value={answers[q.id] || ''} onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
                    style={{ width: '100%', boxSizing: 'border-box', background: '#282a36', color: '#f8f8f2', border: '1px solid #44475a', borderRadius: '6px', padding: '6px 8px', fontSize: '0.85rem' }} />
                </div>
              ))}
              <div style={{ display: 'flex', gap: '8px' }}>
                <button type="button" style={{ ...btn, background: '#50fa7b', color: '#282a36' }} onClick={() => { setAsking(null); save(kit, true, answers); }}>
                  {kit.enabled ? 'Save' : 'Turn on'}
                </button>
                <button type="button" style={btn} onClick={() => setAsking(null)}>Cancel</button>
              </div>
            </div>
          )}

          {/* The links under a kit. "Forget everything" shows while the kit is off too (2026-11-28):
              the page says to turn a kit off and then forget, and the link used to go with the switch. */}
          {asking !== kit.id && (
            <div style={{ marginTop: '8px', paddingLeft: '26px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
              {kit.enabled && kit.questions.length > 0 && <button type="button" style={linkBtn} onClick={() => { setAnswers(kit.answers || {}); setAsking(kit.id); }}>Your answers</button>}
              {kit.enabled && <button type="button" style={linkBtn} onClick={() => setShowText({ ...showText, [kit.id]: !showText[kit.id] })}>
                {showText[kit.id] ? 'Hide what it tells Zynkbot' : 'What it tells Zynkbot'}
              </button>}
              {confirmForget === kit.id
                ? <span style={{ fontSize: '0.8rem' }}>Delete every {kit.name} memory on every device? <button type="button" style={{ ...linkBtn, color: '#ff5555' }} onClick={() => forget(kit)}>Yes, forget it all</button> · <button type="button" style={linkBtn} onClick={() => setConfirmForget(null)}>No</button></span>
                : <button type="button" style={{ ...linkBtn, color: '#ffb86c' }} onClick={() => setConfirmForget(kit.id)}>Forget everything from this kit</button>}
            </div>
          )}

          {showText[kit.id] && (
            <div style={{ ...small, marginTop: '8px', paddingLeft: '26px', whiteSpace: 'pre-wrap' }}>{kit.instructions}</div>
          )}
        </div>
      ))}

      <div style={{ ...small, margin: '14px 0 6px' }}>Later</div>
      {view.future.map(([name, icon]) => (
        <div key={name} style={{ ...row, opacity: 0.45, padding: '8px 12px' }}>
          <span>{icon} {name}</span>
        </div>
      ))}
    </div>
  );
}
