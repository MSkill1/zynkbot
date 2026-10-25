import React, { useState, useEffect } from "react";
import { invoke } from '@tauri-apps/api/core';
import "../styles/KnowledgeBasePanel.css";

export default function KnowledgeBasePanel({ userId }) {
  const [kbFolderPath, setKbFolderPath] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  // Load KB folder path on mount
  useEffect(() => {
    loadKBSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  const loadKBSettings = async () => {
    setIsLoading(true);
    try {
      // Get KB folder path from backend
      const path = await invoke('get_kb_folder_path', { userId });
      setKbFolderPath(path);

    } catch (error) {
      console.error('[KB Panel] Error loading settings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Open KB folder in file explorer
  const handleOpenFolder = async () => {
    try {
      await invoke('open_kb_folder_in_explorer', { userId });
    } catch (error) {
      console.error('[KB Panel] Error opening folder:', error);
      alert(`Failed to open folder: ${error}`);
    }
  };

  if (isLoading) {
    return (
      <div className="knowledge-base-panel">
        <div className="panel-header">
          <h3>Knowledge Base</h3>
        </div>
        <div className="panel-content">
          <div className="loading-state">Loading...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="knowledge-base-panel">
      <div className="panel-content">
        {/* KB Folder Path (Read-only). Not on the phone: the folder is inside the app's
            storage and nothing else can open it, so the path told the user nothing
            (Matt, 2026-10-25). The desktop keeps it, since folders are chosen there. */}
        {!window.AndroidFolderPicker && (
        <div className="setting-row">
          <label className="setting-label">KB Folder:</label>
          <div className="directory-selector">
            <input
              type="text"
              value={kbFolderPath}
              readOnly
              className="directory-input"
              title={kbFolderPath}
            />
            {!window.AndroidFolderPicker && (
            <button
              onClick={handleOpenFolder}
              className="browse-button"
              title="Open folder in file explorer"
            >
              Open Folder
            </button>
            )}
          </div>
        </div>
        )}

        {/* Help Text */}
        <div className="help-text">
          <p>
            <strong>How it works:</strong>{' '}
            {window.AndroidFolderPicker
              ? 'Share a file to the Zynkbot drive, open it under ZynkLink → Browse shared files on this device, and tap "→ KB". Then index it below.'
              : 'Put documents in the folder above, then index them below.'}
            To search them, turn on the 📚 KB button under the message box in chat and ask your question; Zynkbot answers from your documents.
          </p>
          <p className="help-examples">
            <strong>Examples:</strong> Tourist guides, API documentation, research papers, course materials, technical specifications
          </p>
        </div>
      </div>
    </div>
  );
}
