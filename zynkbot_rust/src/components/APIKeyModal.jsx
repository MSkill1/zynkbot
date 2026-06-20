import React, { useState, useEffect } from "react";
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-shell';
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

  useEffect(() => {
    if (isOpen) {
      loadAPIKeys();
    }
  }, [isOpen]);

  const loadAPIKeys = async () => {
    try {
      const keys = await invoke('get_api_keys');
      setApiKeys(keys);
    } catch (error) {
      console.error("Error loading API keys:", error);
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

      // Notify parent to refresh models
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

      // Notify parent to refresh models
      if (onKeysChanged) {
        onKeysChanged();
      }
    } catch (error) {
      console.error(`Error removing ${providerKey}:`, error);
      alert(`Failed to remove API key: ${error}`);
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

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="api-key-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2>API Key Management</h2>
        <p className="modal-subtitle">
          Configure your AI provider API keys to enable cloud models
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
                          onClick={() => open(provider.link)}
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
                        onClick={() => open(provider.link)}
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
        </div>

        <div className="api-key-note">
          <strong>Privacy Note:</strong> API keys are stored locally in your .env file.
          They are never sent to any server except the respective AI provider when you use that model.
        </div>

        <div className="modal-footer">
          <button className="btn-close" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
