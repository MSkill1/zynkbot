import React, { useState, useEffect } from "react";
import { invoke } from '@tauri-apps/api/core';
import { openUrl } from '@tauri-apps/plugin-opener';
import CostGuideModal from "./CostGuideModal";
import "../styles/APIKeyModal.css";

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
    } catch (error) {
      console.error("Error loading API keys:", error);
    }
    try {
      const peers = await invoke('get_zynksync_peers');
      setSyncPeers((peers || []).filter(p => p.paired));
    } catch (_) {}
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
        if (!customModel || !models.includes(customModel)) {
          setCustomModel(models[0]);
        }
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
                <span className="provider-name">Custom / Ollama</span>
                <span className="provider-description">Any OpenAI-compatible server (Ollama, llama-server, LM Studio)</span>
              </div>
              <div className="provider-status">
                {isCustomConfigured() ? (
                  <span className="status-configured">✅ Configured</span>
                ) : (
                  <span className="status-missing">⚠️ Not set</span>
                )}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px', maxWidth: '420px' }}>
              {/* Peer-aware quick-connect: show paired devices as one-tap options */}
              {syncPeers.length > 0 && (
                <div>
                  <div style={{ fontSize: '0.8rem', color: '#9aa5c4', marginBottom: '6px' }}>
                    Use Ollama on a paired device:
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {syncPeers.map(peer => (
                      <button
                        key={peer.device_id}
                        onClick={() => {
                          const proxyUrl = `https://${peer.host}:${peer.port}/api/ollama/v1`;
                          setCustomUrl(proxyUrl);
                          setCustomStatus({ type: "idle", message: "" });
                          setAvailableCustomModels([]);
                        }}
                        style={{
                          padding: '8px 12px', background: '#2d2f3f', border: '1px solid #6272a4',
                          borderRadius: '4px', color: '#f8f8f2', cursor: 'pointer',
                          fontSize: '0.85rem', textAlign: 'left'
                        }}
                      >
                        🖥 Use Ollama on <strong>{peer.device_name}</strong>
                        <span style={{ color: '#9aa5c4', marginLeft: '8px' }}>({peer.host})</span>
                      </button>
                    ))}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#6272a4', margin: '4px 0 6px' }}>
                    — or enter a URL manually —
                  </div>
                </div>
              )}
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
            </div>
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
