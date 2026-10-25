import React, { useState, useEffect } from "react";
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { openUrl } from '@tauri-apps/plugin-opener';
import { confirmDialog } from '../utils/confirmDialog';

const inputStyle = { padding: '8px', borderRadius: '4px', border: '1px solid #44475a', background: '#282a36', color: '#f8f8f2', fontSize: '0.85rem', width: '100%', boxSizing: 'border-box' };
const labelStyle = { fontSize: '0.8rem', color: '#8be9fd' };

/**
 * Where the cloud backup goes: the Cloudflare R2 storage details. Lived at the bottom
 * of the API Keys window until 2026-10-25; moved into the Memory Manager, next to
 * Backup and Restore, where the backup itself is (Matt). The values are API keys to
 * the app (set_api_key / get_api_keys) so the existing push-to-devices path carries them.
 */
export default function CloudBackupSettings({ onShowKey, onClearAll }) {
  const [endpoint, setEndpoint] = useState("");
  const [accessKey, setAccessKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [bucket, setBucket] = useState("zynkbot-backups");
  const [status, setStatus] = useState({ type: "idle", message: "" });

  const load = async () => {
    try {
      const keys = await invoke('get_api_keys');
      if (keys.R2_ENDPOINT) setEndpoint(keys.R2_ENDPOINT);
      if (keys.R2_ACCESS_KEY_ID) setAccessKey(keys.R2_ACCESS_KEY_ID);
      if (keys.R2_SECRET_ACCESS_KEY) setSecretKey(keys.R2_SECRET_ACCESS_KEY);
      if (keys.R2_BUCKET) setBucket(keys.R2_BUCKET);
    } catch (e) { console.debug('[Cloud backup] load:', e); }
  };
  useEffect(() => {
    load();
    let unlisten;
    listen('api-keys-updated', () => { load(); }).then(fn => { unlisten = fn; }).catch(() => {});
    return () => { if (typeof unlisten === 'function') unlisten(); };
  }, []);

  const save = async () => {
    if (!endpoint.trim() || !accessKey.trim() || !secretKey.trim()) {
      setStatus({ type: 'error', message: 'Endpoint, Access Key ID, and Secret Key are required.' });
      return;
    }
    setStatus({ type: 'saving', message: 'Saving…' });
    try {
      const vals = {
        R2_ENDPOINT: endpoint.trim(),
        R2_ACCESS_KEY_ID: accessKey.trim(),
        R2_SECRET_ACCESS_KEY: secretKey.trim(),
        R2_BUCKET: bucket.trim() || 'zynkbot-backups',
      };
      for (const [k, v] of Object.entries(vals)) {
        await invoke('set_api_key', { key: k, value: v });
      }
      setStatus({ type: 'ok', message: 'Saved!' });
      setTimeout(() => setStatus({ type: 'idle', message: '' }), 3000);
      // Offer to push to paired devices
      try {
        const peers = await invoke('get_zynksync_peers');
        const onlinePeers = (peers || []).filter(p => p.is_online);
        if (onlinePeers.length > 0) {
          const confirmed = await confirmDialog(
            `Send these backup storage details to your ${onlinePeers.length} other device${onlinePeers.length > 1 ? 's' : ''}?\n\n` +
            `They travel encrypted, device to device, and never touch the internet.`
          );
          if (confirmed) {
            for (const [k, v] of Object.entries(vals)) {
              await invoke('propagate_api_key', { key: k, value: v });
            }
          }
        }
      } catch (e) {
        console.debug('[Cloud backup] propagation skipped:', e);
      }
    } catch (err) {
      setStatus({ type: 'error', message: String(err) });
    }
  };

  const configured = Boolean(endpoint && accessKey && secretKey);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', background: '#282a36', border: '1px solid #44475a', borderRadius: '8px', padding: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <strong style={{ color: '#f8f8f2', fontSize: '0.95rem' }}>☁ Backup storage</strong>
        <span style={{ fontSize: '0.75rem', padding: '2px 7px', borderRadius: '10px', background: configured ? '#1a3a1a' : '#3a2a1a', color: configured ? '#50fa7b' : '#ffb86c' }}>
          {configured ? 'Set up' : 'Not set up'}
        </span>
        <span style={{ fontSize: '0.72rem', background: '#1a3a1a', color: '#50fa7b', border: '1px solid #50fa7b44', padding: '2px 7px', borderRadius: '10px' }}>Free Beta</span>
      </div>
      <div style={{ fontSize: '0.82rem', color: '#9aa5c4', lineHeight: 1.5 }}>
        <b style={{ color: '#f8f8f2' }}>Backup</b> copies every memory on this device to the cloud. <b style={{ color: '#f8f8f2' }}>Restore</b> brings a backup back onto this device, or onto a new one; memories already here are kept. You can also clear everything on the device and restore from the backup later.
      </div>
      <div style={{ fontSize: '0.82rem', color: '#9aa5c4', lineHeight: 1.5 }}>
        Backups are encrypted before they leave this device, with a key that stays on your own devices.
        They are stored in a Cloudflare R2 bucket of your own.{' '}
        <button onClick={() => openUrl('https://dash.cloudflare.com/?to=/:account/r2/api-tokens')}
          style={{ background: 'none', border: 'none', color: '#8be9fd', cursor: 'pointer', padding: 0, font: 'inherit', textDecoration: 'underline' }}>
          Get your free Cloudflare account
        </button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <label style={labelStyle}>Endpoint URL</label>
        <input type="text" value={endpoint} onChange={e => setEndpoint(e.target.value)} placeholder="https://<account-id>.r2.cloudflarestorage.com" style={inputStyle} />
      </div>
      <div style={{ display: 'flex', gap: '10px' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={labelStyle}>Access Key ID</label>
          <input type="text" value={accessKey} onChange={e => setAccessKey(e.target.value)} placeholder="Access Key ID" style={inputStyle} />
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={labelStyle}>Secret Access Key</label>
          <input type="password" value={secretKey} onChange={e => setSecretKey(e.target.value)} placeholder="Secret Access Key" style={inputStyle} />
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '10px' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={labelStyle}>Bucket Name</label>
          <input type="text" value={bucket} onChange={e => setBucket(e.target.value)} placeholder="zynkbot-backups" style={inputStyle} />
        </div>
        <button onClick={save}
          style={{ padding: '8px 20px', borderRadius: '4px', border: 'none', background: status.type === 'ok' ? '#50fa7b' : '#6272a4', color: status.type === 'ok' ? '#282a36' : '#fff', cursor: 'pointer', fontWeight: 'bold', whiteSpace: 'nowrap', minHeight: '36px' }}>
          {status.type === 'saving' ? 'Saving…' : status.type === 'ok' ? '✓ Saved' : 'Save'}
        </button>
      </div>
      <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start', borderTop: '1px solid #44475a', paddingTop: '10px' }}>
        <div style={{ flex: 1, fontSize: '0.8rem', color: '#9aa5c4', lineHeight: 1.5 }}>
          <b style={{ color: '#ffb86c' }}>The key.</b> Every backup is locked with one key that your paired devices share among themselves, so day to day you never touch it. Save it somewhere safe anyway: if you lose every device, or set up a new one without pairing it to an old one, that key is the only way to open the backup.
        </div>
        {onShowKey && (
          <button onClick={onShowKey} title="Show the encryption key so you can store it somewhere safe"
            style={{ background: '#ffb86c', border: 'none', color: '#282a36', fontSize: '0.8rem', fontWeight: 'bold', padding: '8px 12px', borderRadius: '4px', cursor: 'pointer', whiteSpace: 'nowrap', minHeight: '40px', flexShrink: 0 }}>
            🔑 Key
          </button>
        )}
      </div>
      {onClearAll && (
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', borderTop: '1px solid #44475a', paddingTop: '10px' }}>
          <div style={{ flex: 1, fontSize: '0.8rem', color: '#9aa5c4', lineHeight: 1.5 }}>
            Delete every memory on this device. It asks twice first.
          </div>
          <button onClick={onClearAll}
            style={{ background: 'none', border: '1px solid #ff5555', color: '#ff5555', fontSize: '0.8rem', padding: '8px 12px', borderRadius: '4px', cursor: 'pointer', whiteSpace: 'nowrap', minHeight: '40px', flexShrink: 0 }}>
            Clear All
          </button>
        </div>
      )}
      {status.message && status.type !== 'ok' && (
        <div style={{ fontSize: '0.8rem', color: status.type === 'error' ? '#ff5555' : '#6272a4' }}>{status.message}</div>
      )}
    </div>
  );
}
