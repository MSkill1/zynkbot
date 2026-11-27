import React, { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import ControlsPage from './ControlsPage';

/**
 * The list window of a kit with actions (the Grocery kit, 2026-11-27): items still needed,
 * grouped by section in store order, a tick box each; ticked items drop to "In the cart";
 * a line at the top adds an item; "Usual items" below shows what has been bought before,
 * with when, and a + to put it back on the list. Items are memories, so a change made here
 * reaches the other devices like any other memory.
 *
 * Props: kit (from list_kits), userId, onClose, notify.
 */
export default function GroceryListModal({ kit, userId, onClose, notify }) {
  const [items, setItems] = useState(null);
  const [newItem, setNewItem] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try { setItems(await invoke('kit_items', { kitId: kit.id, userId })); }
    catch (e) { notify?.(`Could not load the list: ${e}`); }
  }, [kit.id, userId, notify]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    // A change made by a reply here, or by another device through sync, redraws the list.
    const uns = [];
    listen('kit-items-changed', () => load()).then((u) => uns.push(u));
    listen('zynksync-memories-updated', () => load()).then((u) => uns.push(u));
    return () => { uns.forEach((u) => u()); };
  }, [load]);

  const setState = async (item, state) => {
    try { await invoke('kit_set_item_state', { memoryId: item.id, state }); await load(); }
    catch (e) { notify?.(`Could not change ${item.name}: ${e}`); }
  };
  const remove = async (item) => {
    try { await invoke('kit_remove_item', { memoryId: item.id }); await load(); }
    catch (e) { notify?.(`Could not remove ${item.name}: ${e}`); }
  };
  const add = async () => {
    const name = newItem.trim();
    if (!name || busy) return;
    setBusy(true);
    try { await invoke('kit_add_item', { kitId: kit.id, userId, name, section: null, quantity: null }); setNewItem(''); await load(); }
    catch (e) { notify?.(`Could not add ${name}: ${e}`); }
    finally { setBusy(false); }
  };

  const sections = kit.sections?.length ? kit.sections : ['other'];
  const needed = (items || []).filter((i) => i.state === 'needed');
  const today = new Date().toISOString().slice(0, 10);
  const inCart = (items || []).filter((i) => i.state === 'bought' && i.last_bought === today);
  const usual = (items || []).filter((i) => i.state === 'bought' && i.last_bought !== today)
    .sort((a, b) => (b.bought_count - a.bought_count) || (b.last_bought || '').localeCompare(a.last_bought || ''));
  const bySection = sections.map((s) => [s, needed.filter((i) => i.section === s)]).filter(([, v]) => v.length);
  const stray = needed.filter((i) => !sections.includes(i.section));
  if (stray.length) bySection.push(['other', stray]);

  const sec = { color: '#bd93f9', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', margin: '12px 0 4px' };
  const line = { display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 4px', borderBottom: '1px solid #2b2d3a', fontSize: '0.95rem' };
  const small = { color: '#9aa5c4', fontSize: '0.8rem' };
  const x = { background: 'none', border: 'none', color: '#6272a4', cursor: 'pointer', fontSize: '1rem', padding: '0 4px', marginLeft: 'auto' };
  const when = (d) => {
    if (!d) return '';
    const days = Math.round((Date.parse(today) - Date.parse(d)) / 86400000);
    return days <= 0 ? 'today' : days === 1 ? 'yesterday' : days < 14 ? `${days} days ago` : days < 60 ? `${Math.round(days / 7)} weeks ago` : `${Math.round(days / 30)} months ago`;
  };

  return (
    <ControlsPage icon={kit.icon || '🛒'} title={`${kit.name} list`} subtitle={needed.length ? `${needed.length} to get` : 'Nothing to get'} accent="#50fa7b" zIndex={1200} closeLabel="Close" onClose={onClose}>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
        <input type="text" value={newItem} onChange={(e) => setNewItem(e.target.value)} placeholder="Add something…"
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } }}
          style={{ flex: 1, background: '#282a36', color: '#f8f8f2', border: '1px solid #44475a', borderRadius: '8px', padding: '10px 12px', fontSize: '0.95rem' }} />
        <button type="button" onClick={add} disabled={busy || !newItem.trim()}
          style={{ background: '#50fa7b', color: '#282a36', border: 'none', borderRadius: '8px', padding: '10px 14px', fontWeight: 'bold', cursor: 'pointer', opacity: newItem.trim() ? 1 : 0.5 }}>Add</button>
      </div>
      <div style={small}>Or say it: "Hey Zynk, we're out of milk." Section and quantity come from how you say it.</div>

      {items === null && <div style={{ ...small, marginTop: '12px' }}>Loading…</div>}
      {items !== null && needed.length === 0 && <div style={{ ...small, marginTop: '16px' }}>The list is empty. Add something above, or tell Zynkbot what you need.</div>}

      {bySection.map(([s, list]) => (
        <div key={s}>
          <div style={sec}>{s}</div>
          {list.map((i) => (
            <label key={i.id} style={{ ...line, cursor: 'pointer' }}>
              <input type="checkbox" checked={false} onChange={() => setState(i, 'bought')} style={{ width: '22px', height: '22px', margin: 0, flexShrink: 0 }} />
              <span>{i.name}{i.quantity ? <span style={small}> · {i.quantity}</span> : null}{i.price ? <span style={small}> · {i.price}</span> : null}</span>
              <button type="button" style={x} title="Remove from the list" onClick={() => remove(i)}>✕</button>
            </label>
          ))}
        </div>
      ))}

      {inCart.length > 0 && (
        <div>
          <div style={{ ...sec, color: '#50fa7b' }}>In the cart</div>
          {inCart.map((i) => (
            <label key={i.id} style={{ ...line, cursor: 'pointer', opacity: 0.75 }}>
              <input type="checkbox" checked onChange={() => setState(i, 'needed')} style={{ width: '22px', height: '22px', margin: 0, flexShrink: 0 }} />
              <span style={{ textDecoration: 'line-through' }}>{i.name}</span>
            </label>
          ))}
        </div>
      )}

      {usual.length > 0 && (
        <div>
          <div style={{ ...sec, color: '#9aa5c4' }}>Usual items</div>
          {usual.slice(0, 30).map((i) => (
            <div key={i.id} style={{ ...line, fontSize: '0.88rem' }}>
              <button type="button" title="Put it back on the list" onClick={() => setState(i, 'needed')}
                style={{ background: '#44475a', color: '#f8f8f2', border: 'none', borderRadius: '6px', width: '26px', height: '26px', cursor: 'pointer', flexShrink: 0 }}>+</button>
              <span>{i.name} <span style={small}>· bought {when(i.last_bought)}{i.bought_count > 1 ? `, ${i.bought_count} times` : ''}</span></span>
              <button type="button" style={x} title="Forget this item" onClick={() => remove(i)}>✕</button>
            </div>
          ))}
        </div>
      )}

      {inCart.length > 0 && (
        <button type="button" onClick={onClose}
          style={{ width: '100%', marginTop: '16px', padding: '14px', borderRadius: '10px', border: 'none', background: '#50fa7b', color: '#282a36', fontSize: '1rem', fontWeight: 'bold', cursor: 'pointer' }}>
          Done
        </button>
      )}
    </ControlsPage>
  );
}
