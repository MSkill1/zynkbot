import React, { useState, useEffect } from "react";
import { invoke } from '@tauri-apps/api/core';
import "../styles/KnowledgeBasePanel.css";

const isMobile = () => window.innerWidth <= 768;

export default function KnowledgeBasePanel({ userId, onManageDocuments }) {
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
        {/* KB Folder Path (Read-only) */}
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

        {/* Action Buttons */}
        <div className="setting-row button-row">
          <button
            onClick={onManageDocuments}
            style={{
              width: '100%',
              padding: '10px',
              background: '#ffb86c',
              color: '#282a36',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.9rem'
            }}
          >
            Knowledge Base Manager
          </button>
        </div>

        {/* Help Text */}
        <div className="help-text">
          <p>
            <strong>How it works:</strong> Place plain text files in your KB folder, then click "Knowledge Base Manager" to index them.
            To search your documents, click the 📚 KB button in the message input area, then send your question.
            Zynkbot will search for relevant content using semantic similarity.
            (PDF and DOCX support coming soon)
          </p>
          <p className="help-examples">
            <strong>Examples:</strong> Tourist guides, API documentation, research papers, course materials, technical specifications
          </p>
        </div>
      </div>
    </div>
  );
}
