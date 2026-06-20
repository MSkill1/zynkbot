import React, { useState } from "react";
import { invoke } from '@tauri-apps/api/core';
import "../styles/ContainmentModeSelector.css";

const MODES = [
  { value: "witness", label: "Witness", description: "No filtering" },
  { value: "sovereign", label: "Sovereign", description: "Warnings only" },
  { value: "guardian", label: "Guardian", description: "Block dangerous content" },
  { value: "child", label: "Child", description: "⚠️ Strongest safety - requires OpenAI API + internet" },
  { value: "hipaa", label: "HIPAA", description: "PHI protection (see documentation)" }
];

export default function ContainmentModeSelector({ currentMode, onModeChange }) {
  const [isChanging, setIsChanging] = useState(false);

  const handleModeChange = async (newMode) => {
    // Show warning when switching to Child mode
    if (newMode === 'child') {
      const confirmed = window.confirm(
        `Child Safety Mode\n\n` +
        `This mode turns on the strongest content filtering available — designed for households where children will be using Zynkbot.\n\n` +
        `What it does:\n` +
        `Blocks inappropriate content more aggressively than other modes, including attempts to work around the filter.\n\n` +
        `What it requires:\n` +
        `• An OpenAI API key (added in Settings → API Keys)\n` +
        `• An internet connection when chatting\n\n` +
        `If you don't have an OpenAI API key, Child Safety Mode will still apply basic filtering, but protection will be reduced. For the strongest protection, the API key is recommended.\n\n` +
        `Note: A dedicated local model trained specifically for child safety may be available in the future if there is enough interest.\n\n` +
        `Enable Child Safety Mode?`
      );

      if (!confirmed) {
        return; // User cancelled
      }
    }

    setIsChanging(true);
    try {
      // Use Tauri invoke instead of fetch
      await invoke('set_containment_mode', { mode: newMode });
      console.log(`✅ Containment mode changed to: ${newMode}`);
      onModeChange(newMode);
    } catch (error) {
      console.error("Error changing containment mode:", error);
      alert(`Failed to change containment mode: ${error}`);
    } finally {
      setIsChanging(false);
    }
  };

  const currentModeObj = MODES.find(m => m.value === currentMode) || MODES[2];

  return (
    <div className="containment-selector">
      <label htmlFor="mode-select">Containment Mode:</label>
      <select
        id="mode-select"
        value={currentMode}
        onChange={(e) => handleModeChange(e.target.value)}
        disabled={isChanging}
        className="mode-dropdown"
      >
        {MODES.map(mode => (
          <option key={mode.value} value={mode.value}>
            {mode.label}
          </option>
        ))}
      </select>
      <span className="mode-description">{currentModeObj.description}</span>
    </div>
  );
}
