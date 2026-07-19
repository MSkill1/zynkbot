import React, { useState, useEffect, useRef } from "react";
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { openUrl } from '@tauri-apps/plugin-opener';
import CostGuideModal from "./CostGuideModal";
import "../styles/APIKeyModal.css";

const POPULAR_MODELS = [
  "llama3.2:3b",
  "llama3.1:8b",
  "mistral:7b",
  "phi4-mini:3.8b",
  "gemma3:4b",
  "codellama:7b",
];

const PROVIDERS = [
  {
    name: "Anthropic (Claude)",
    key: "ANTHROPIC_API_KEY",
    placeholder: "sk-ant-...",
    link: "https://console.anthropic.com/settings/keys",
    description: "Claude 3.5 Sonnet/Haiku/Opus"
  },
  {
    name: "OpenAI (GPT)",
    key: "OPENAI_API_KEY",
    placeholder: "sk-...",
    link: "https://platform.openai.com/api-keys",
    description: "GPT-4o, GPT-4o-mini"
  },
  {
    name: "xAI (Grok)",
    key: "XAI_API_KEY",
    placeholder: "xai-...",
    link: "https://console.x.ai/",
    description: "Grok (X Premium+ required)"
  }
];

export default function APIKeyModal({ isOpen, onClose, onKeysChanged }) {
  const [apiKeys, setApiKeys] = useState({});
  const [editingKey, setEditingKey] = useState(null);
  const [tempValue, setTempValue] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [showKeys, setShowKeys] = useState({});
  const [showCostGuide, setShowCostGuide] = useState(false);

  // Custom endpoint state
  const [customUrl, setCustomUrl] = useState("");
  const [customApiKey, setCustomApiKey] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [availableCustomModels, setAvailableCustomModels] = useState([]);
  const [customStatus, setCustomStatus] = useState({ type: "idle", message: "" });
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const [isSavingCustom, setIsSavingCustom] = useState(false);
  const [syncPeers, setSyncPeers] = useState([]);
  const [isConnecting, setIsConnecting] = useState(false);

  const isAndroid = /Android/i.test(navigator.userAgent);

  // Pull model state
  const [pullModelName, setPullModelName] = useState("");
  const [isPulling, setIsPulling] = useState(false);
  const [pullLog, setPullLog] = useState([]);
  const pullLogRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      loadAPIKeys();
    }
  }, [isOpen]);

  const loadAPIKeys = async () => {
    try {
      const keys = await invoke('get_api_keys');
      setApiKeys(keys);
      if (keys.CUSTOM_API_URL) setCustomUrl(keys.CUSTOM_API_URL);
      if (keys.CUSTOM_API_KEY) setCustomApiKey(keys.CUSTOM_API_KEY);
      if (keys.CUSTOM_MODEL) setCustomModel(keys.CUSTOM_MODEL);

      // Pre-fill Ollama default URL on desktop if nothing configured yet
      if (!keys.CUSTOM_API_URL && !isAndroid) {
        setCustomUrl("http://localhost:11434/v1");
      }

      // Auto-populate model list if URL is already configured
      if (keys.CUSTOM_API_URL) {
        autoFetchModels(keys.CUSTOM_API_URL, keys.CUSTOM_API_KEY || "", keys.CUSTOM_MODEL || "");
      }
    } catch (error) {
      console.error("Error loading API keys:", error);
    }
    try {
      // Collect peer-aware Ollama buttons from both ZynkSync peers and ZynkLink-only devices
      const [syncPeers, linkedDevices] = await Promise.allSettled([
        invoke('get_zynksync_peers'),
        invoke('get_linked_devices'),
      ]);

      const seen = new Set();
      const merged = [];

      const addDevice = (d, host, port, name, id) => {
        if (!id || !host || seen.has(id)) return;
        seen.add(id);
        merged.push({ device_id: id, device_name: name, host, port });
      };

      if (syncPeers.status === 'fulfilled') {
        for (const p of (syncPeers.value || [])) {
          if (p.paired) addDevice(p, p.host, p.port, p.device_name, p.device_id);
        }
      }
      if (linkedDevices.status === 'fulfilled') {
        for (const d of (linkedDevices.value || [])) {
          addDevice(d, d.host, d.port, d.device_name, d.device_id);
        }
      }

      setSyncPeers(merged);
    } catch (_) {}
  };

  const autoFetchModels = async (url, apiKey, savedModel) => {
    try {
      const models = await invoke('fetch_custom_models', {
        baseUrl: url.trim(),
        apiKey: apiKey.trim()
      });
      setAvailableCustomModels(models);
      if (models.length > 0) {
        setCustomStatus({ type: "success", message: `✓ Connected — ${models.length} model(s) available` });
        if (!savedModel) setCustomModel(models[0]);
      }
    } catch (_) {
      // Silent — user can manually fetch if offline
    }
  };

  const handleSave = async (providerKey) => {
    if (!tempValue.trim()) {
      alert("API key cannot be empty");
      return;
    }

    setIsSaving(true);
    try {
      await invoke('set_api_key', {
        key: providerKey,
        value: tempValue.trim()
      });

      setApiKeys(prev => ({
        ...prev,
        [providerKey]: tempValue.trim()
      }));

      setEditingKey(null);
      setTempValue("");
      console.log(`✅ Saved ${providerKey}`);

      if (onKeysChanged) {
        onKeysChanged();
      }
    } catch (error) {
      console.error(`Error saving ${providerKey}:`, error);
      alert(`Failed to save API key: ${error}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async (providerKey) => {
    const confirmed = window.confirm(
      `Remove ${providerKey}?\n\n` +
      `You can add it back later if needed.`
    );

    if (!confirmed) return;

    try {
      await invoke('remove_api_key', { key: providerKey });

      setApiKeys(prev => {
        const updated = { ...prev };
        delete updated[providerKey];
        return updated;
      });

      console.log(`🗑️ Removed ${providerKey}`);

      if (onKeysChanged) {
        onKeysChanged();
      }
    } catch (error) {
      console.error(`Error removing ${providerKey}:`, error);
      alert(`Failed to remove API key: ${error}`);
    }
  };

  const handleFetchCustomModels = async () => {
    if (!customUrl.trim()) {
      setCustomStatus({ type: "error", message: "Enter a base URL first (e.g. http://localhost:11434/v1)" });
      return;
    }
    setIsFetchingModels(true);
    setCustomStatus({ type: "idle", message: "" });
    try {
      const models = await invoke('fetch_custom_models', {
        baseUrl: customUrl.trim(),
        apiKey: customApiKey.trim()
      });
      setAvailableCustomModels(models);
      if (models.length === 0) {
        setCustomStatus({ type: "error", message: "Connected but no models found. Pull a model first (e.g. ollama pull llama3.1:8b)" });
      } else {
        setCustomStatus({ type: "success", message: `✓ Connected — ${models.length} model(s) available` });
        if (!customModel) setCustomModel(models[0]);
      }
    } catch (err) {
      setCustomStatus({ type: "error", message: String(err) });
      setAvailableCustomModels([]);
    } finally {
      setIsFetchingModels(false);
    }
  };

  const handleSaveCustom = async () => {
    if (!customUrl.trim()) {
      setCustomStatus({ type: "error", message: "Base URL is required" });
      return;
    }
    if (!customModel.trim()) {
      setCustomStatus({ type: "error", message: "Select a model first" });
      return;
    }
    setIsSavingCustom(true);
    try {
      await invoke('set_api_key', { key: 'CUSTOM_API_URL', value: customUrl.trim() });
      if (customApiKey.trim()) {
        await invoke('set_api_key', { key: 'CUSTOM_API_KEY', value: customApiKey.trim() });
      } else {
        await invoke('remove_api_key', { key: 'CUSTOM_API_KEY' });
      }
      await invoke('set_api_key', { key: 'CUSTOM_MODEL', value: customModel.trim() });
      setApiKeys(prev => ({ ...prev, CUSTOM_API_URL: customUrl.trim(), CUSTOM_MODEL: customModel.trim() }));
      setCustomStatus({ type: "success", message: "✓ Saved — select \"Custom / Ollama\" from the model picker" });
      if (onKeysChanged) onKeysChanged();
    } catch (err) {
      setCustomStatus({ type: "error", message: `Failed to save: ${err}` });
    } finally {
      setIsSavingCustom(false);
    }
  };

  const handleRemoveCustom = async () => {
    if (!window.confirm("Remove custom endpoint configuration?")) return;
    try {
      await invoke('remove_api_key', { key: 'CUSTOM_API_URL' });
      await invoke('remove_api_key', { key: 'CUSTOM_API_KEY' });
      await invoke('remove_api_key', { key: 'CUSTOM_MODEL' });
      setCustomUrl("");
      setCustomApiKey("");
      setCustomModel("");
      setAvailableCustomModels([]);
      setCustomStatus({ type: "idle", message: "" });
      setApiKeys(prev => {
        const u = { ...prev };
        delete u.CUSTOM_API_URL; delete u.CUSTOM_API_KEY; delete u.CUSTOM_MODEL;
        return u;
      });
      if (onKeysChanged) onKeysChanged();
    } catch (err) {
      alert(`Failed to remove: ${err}`);
    }
  };

  const handleConnectToDesktop = async (peer) => {
    setIsConnecting(true);
    setCustomStatus({ type: "idle", message: "" });
    const proxyUrl = `https://${peer.host}:${peer.port}/api/ollama/v1`;
    try {
      const model = await invoke('get_peer_ollama_config', { host: peer.host, port: peer.port });
      await invoke('set_api_key', { key: 'CUSTOM_API_URL', value: proxyUrl });
      await invoke('set_api_key', { key: 'CUSTOM_MODEL', value: model });
      await invoke('remove_api_key', { key: 'CUSTOM_API_KEY' });
      setCustomUrl(proxyUrl);
      setCustomModel(model);
      setApiKeys(prev => ({ ...prev, CUSTOM_API_URL: proxyUrl, CUSTOM_MODEL: model }));
      delete apiKeys.CUSTOM_API_KEY;
      setCustomStatus({ type: "success", message: `✓ Connected to ${peer.device_name} — using ${model}` });
      if (onKeysChanged) onKeysChanged();
    } catch (err) {
      setCustomStatus({ type: "error", message: String(err) });
    } finally {
      setIsConnecting(false);
    }
  };

  const handlePullModel = async () => {
    const name = pullModelName.trim();
    if (!name) return;
    setIsPulling(true);
    setPullLog([]);

    const unlisten = await listen('ollama-pull-progress', (event) => {
      setPullLog(prev => {
        const next = [...prev, event.payload];
        setTimeout(() => {
          if (pullLogRef.current) {
            pullLogRef.current.scrollTop = pullLogRef.current.scrollHeight;
          }
        }, 0);
        return next;
      });
    });

    try {
      await invoke('pull_ollama_model', { modelName: name });
      // Refresh available models list if URL is set
      if (customUrl.trim()) {
        handleFetchCustomModels();
      }
    } catch (err) {
      setPullLog(prev => [...prev, `❌ ${err}`]);
    } finally {
      unlisten();
      setIsPulling(false);
    }
  };

  const maskKey = (key) => {
    if (!key) return "";
    if (key.length <= 4) return "•".repeat(key.length);
    return key.substring(0, 4) + "•".repeat(20);
  };

  const isConfigured = (providerKey) => {
    return apiKeys[providerKey] && apiKeys[providerKey].length > 0;
  };

  const isCustomConfigured = () => apiKeys.CUSTOM_API_URL && apiKeys.CUSTOM_MODEL;

  if (!isOpen) return null;

  return (
    <>
    <CostGuideModal isOpen={showCostGuide} onClose={() => setShowCostGuide(false)} />
    <div className="modal-overlay" onClick={onClose}>
      <div className="api-key-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2>API Key Management</h2>
        <p className="modal-subtitle">
          Configure your AI provider API keys to enable cloud models.{' '}
          <button
            onClick={() => setShowCostGuide(true)}
            style={{ background: 'none', border: 'none', color: '#50fa7b', cursor: 'pointer', textDecoration: 'underline', fontSize: 'inherit', padding: 0 }}
          >💰 What will this cost?</button>
        </p>

        <div className="api-keys-list">
          {PROVIDERS.map(provider => (
            <div key={provider.key} className="api-key-item">
              <div className="provider-header">
                <div className="provider-info">
                  <span className="provider-name">{provider.name}</span>
                  <span className="provider-description">{provider.description}</span>
                </div>
                <div className="provider-status">
                  {isConfigured(provider.key) ? (
                    <span className="status-configured">✅ Configured</span>
                  ) : (
                    <span className="status-missing">⚠️ Not set</span>
                  )}
                </div>
              </div>

              {editingKey === provider.key ? (
                <div className="key-edit-mode">
                  <input
                    type="text"
                    value={tempValue}
                    onChange={(e) => setTempValue(e.target.value)}
                    placeholder={provider.placeholder}
                    className="key-input"
                    autoFocus
                  />
                  <div className="key-actions">
                    <button
                      onClick={() => handleSave(provider.key)}
                      disabled={isSaving || !tempValue.trim()}
                      className="btn-save"
                    >
                      💾 Save
                    </button>
                    <button
                      onClick={() => {
                        setEditingKey(null);
                        setTempValue("");
                      }}
                      className="btn-cancel"
                    >
                      ✕ Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="key-display-mode">
                  {isConfigured(provider.key) ? (
                    <>
                      <div className="key-value">
                        <span className={`key-value-text${showKeys[provider.key] ? ' revealed' : ''}`}>
                          {showKeys[provider.key]
                            ? apiKeys[provider.key]
                            : maskKey(apiKeys[provider.key])}
                        </span>
                        <button
                          onClick={() => setShowKeys(prev => ({
                            ...prev,
                            [provider.key]: !prev[provider.key]
                          }))}
                          className="btn-toggle-visibility"
                          title={showKeys[provider.key] ? "Hide key" : "Show key"}
                        >
                          👁️
                        </button>
                      </div>
                      <div className="key-actions">
                        <button
                          onClick={() => {
                            setEditingKey(provider.key);
                            setTempValue(apiKeys[provider.key] || "");
                          }}
                          className="btn-edit"
                        >
                          ✏️ Edit
                        </button>
                        <button
                          onClick={() => handleDelete(provider.key)}
                          className="btn-delete"
                        >
                          🗑️ Remove
                        </button>
                        <button
                          onClick={() => openUrl(provider.link)}
                          className="btn-get-key"
                        >
                          🔗 Get Key
                        </button>
                      </div>
                    </>
                  ) : (
                    <div className="key-actions">
                      <button
                        onClick={() => setEditingKey(provider.key)}
                        className="btn-add"
                      >
                        ➕ Add Key
                      </button>
                      <button
                        onClick={() => openUrl(provider.link)}
                        className="btn-get-key"
                      >
                        🔗 Get Key
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Custom / Ollama section */}
          <div className="api-key-item">
            <div className="provider-header">
              <div className="provider-info">
                <span className="provider-name">
                  {isAndroid ? "Ollama (Local AI)" : "Custom / Ollama"}
                </span>
                <span className="provider-description">
                  {isAndroid
                    ? "Use your desktop's Ollama models over your home network"
                    : "Any OpenAI-compatible server (Ollama, llama-server, LM Studio)"}
                </span>
              </div>
              <div className="provider-status">
                {isCustomConfigured() ? (
                  <span className="status-configured">✅ Configured</span>
                ) : (
                  <span className="status-missing">⚠️ Not set</span>
                )}
              </div>
            </div>

            {isAndroid ? (
              /* ── Android: one-tap connect, no manual fields ── */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '8px' }}>
                {syncPeers.length === 0 ? (
                  <div style={{ fontSize: '0.85rem', color: '#9aa5c4', padding: '10px', background: '#282a36', borderRadius: '6px' }}>
                    No paired desktops found. Open ZynkSync on your desktop and pair with this device to use Ollama.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {syncPeers.map(peer => (
                      <button
                        key={peer.device_id}
                        onClick={() => handleConnectToDesktop(peer)}
                        disabled={isConnecting}
                        style={{
                          padding: '12px 16px', background: '#2d2f3f', border: '1px solid #6272a4',
                          borderRadius: '8px', color: '#f8f8f2', cursor: isConnecting ? 'wait' : 'pointer',
                          fontSize: '0.9rem', textAlign: 'left', display: 'flex', alignItems: 'center', gap: '10px'
                        }}
                      >
                        <span style={{ fontSize: '1.3rem' }}>🖥</span>
                        <span>
                          {isConnecting ? "Connecting..." : "Connect to Ollama on "}
                          <strong>{peer.device_name}</strong>
                        </span>
                      </button>
                    ))}
                  </div>
                )}

                {customStatus.message && (
                  <div style={{
                    fontSize: '0.85rem', padding: '8px 10px', borderRadius: '6px',
                    background: customStatus.type === 'success' ? '#1a3a1a' : '#3a1a1a',
                    color: customStatus.type === 'success' ? '#50fa7b' : '#ff5555',
                    border: `1px solid ${customStatus.type === 'success' ? '#50fa7b44' : '#ff555544'}`
                  }}>
                    {customStatus.message}
                  </div>
                )}

                {isCustomConfigured() && (
                  <button onClick={handleRemoveCustom} className="btn-delete" style={{ alignSelf: 'flex-start' }}>
                    🗑️ Disconnect
                  </button>
                )}
              </div>
            ) : (
              /* ── Desktop: full configuration form ── */
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px', maxWidth: '420px' }}>
                <input
                  type="text"
                  className="key-input"
                  placeholder="http://localhost:11434/v1"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                />
                <input
                  type="text"
                  className="key-input"
                  placeholder="API key (leave blank for Ollama)"
                  value={customApiKey}
                  onChange={(e) => setCustomApiKey(e.target.value)}
                />
                <div style={{ fontSize: '0.78rem', color: '#9aa5c4', marginTop: '-4px' }}>
                  Ollama doesn't require an API key — leave it blank.
                  Other servers (LM Studio, vLLM) may require one.
                </div>
                <button
                  onClick={handleFetchCustomModels}
                  disabled={isFetchingModels || !customUrl.trim()}
                  className="btn-save"
                  style={{ alignSelf: 'flex-start' }}
                >
                  {isFetchingModels ? "Connecting..." : "🔍 Fetch Models"}
                </button>

                {availableCustomModels.length > 0 && (
                  <select
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    style={{
                      padding: '8px', background: '#ffffff', color: '#000000',
                      border: '1px solid #44475a', borderRadius: '4px', fontSize: '0.9rem',
                      width: '100%'
                    }}
                  >
                    {availableCustomModels.map(m => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                )}

                {customStatus.message && (
                  <div style={{
                    fontSize: '0.85rem', padding: '6px 8px', borderRadius: '4px',
                    background: customStatus.type === 'success' ? '#1a3a1a' : '#3a1a1a',
                    color: customStatus.type === 'success' ? '#50fa7b' : '#ff5555',
                    border: `1px solid ${customStatus.type === 'success' ? '#50fa7b44' : '#ff555544'}`
                  }}>
                    {customStatus.message}
                  </div>
                )}

                <div className="key-actions">
                  <button
                    onClick={handleSaveCustom}
                    disabled={isSavingCustom || !customUrl.trim() || !customModel.trim()}
                    className="btn-save"
                  >
                    {isSavingCustom ? "Saving..." : "💾 Save"}
                  </button>
                  {isCustomConfigured() && (
                    <button onClick={handleRemoveCustom} className="btn-delete">
                      🗑️ Remove
                    </button>
                  )}
                </div>

                {/* Pull Model — desktop only */}
                <div style={{ marginTop: '16px', paddingTop: '14px', borderTop: '1px solid #44475a' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f8f8f2', marginBottom: '6px' }}>
                    Pull a Model
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#9aa5c4', marginBottom: '10px' }}>
                    Ollama models are downloaded on demand. Type a model name and click Pull — it's the same as running{' '}
                    <code style={{ background: '#282a36', padding: '1px 4px', borderRadius: '3px', fontFamily: 'monospace' }}>
                      ollama pull &lt;name&gt;
                    </code>{' '}
                    in a terminal.
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', marginBottom: '8px' }}>
                    {POPULAR_MODELS.map(m => (
                      <button
                        key={m}
                        onClick={() => setPullModelName(m)}
                        style={{
                          padding: '3px 9px', fontSize: '0.78rem', background: pullModelName === m ? '#44475a' : '#282a36',
                          border: '1px solid #6272a4', borderRadius: '12px', color: '#f8f8f2',
                          cursor: 'pointer'
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <input
                      type="text"
                      className="key-input"
                      placeholder="llama3.1:8b"
                      value={pullModelName}
                      onChange={(e) => setPullModelName(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter' && !isPulling) handlePullModel(); }}
                      style={{ flex: 1 }}
                    />
                    <button
                      onClick={handlePullModel}
                      disabled={isPulling || !pullModelName.trim()}
                      className="btn-save"
                    >
                      {isPulling ? "Pulling..." : "⬇ Pull"}
                    </button>
                  </div>
                  {pullLog.length > 0 && (
                    <div
                      ref={pullLogRef}
                      style={{
                        marginTop: '8px', padding: '8px', background: '#1e1f29',
                        border: '1px solid #44475a', borderRadius: '4px',
                        fontFamily: 'monospace', fontSize: '0.78rem', color: '#f8f8f2',
                        maxHeight: '160px', overflowY: 'auto', whiteSpace: 'pre-wrap',
                        wordBreak: 'break-all'
                      }}
                    >
                      {pullLog.map((line, i) => (
                        <div key={i}>{line}</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="api-key-note">
          <strong>Privacy Note:</strong> API keys are stored locally in your .env file.
          They are never sent to any server except the respective AI provider when you use that model.
          Custom endpoint traffic goes directly to your server — no cloud involved.
        </div>

        <div className="modal-footer">
          <button className="btn-close" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
    </>
  );
}
