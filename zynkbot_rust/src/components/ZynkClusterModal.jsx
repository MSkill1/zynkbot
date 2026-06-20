import React from "react";
import { invoke } from '@tauri-apps/api/core';
import "../styles/AboutModal.css";

export default function ZynkClusterModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const openLabsFolder = async () => {
    try {
      await invoke('open_external_folder', { path: 'labs/moe_poc' });
    } catch (error) {
      console.error('Failed to open labs folder:', error);
      alert('Failed to open documentation folder. Please check if the folder exists.');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="about-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2 style={{color: '#8be9fd', marginBottom: '20px'}}>🧠 ZynkCluster: Distributed MoE Inference</h2>

        <div className="about-section">
          <h3>What is this?</h3>
          <p style={{lineHeight: '1.6', color: '#f8f8f2'}}>
            ZynkCluster is a proposed deployment model for running large <strong>Mixture of Experts (MoE)</strong> models
            across multiple consumer devices on a local network. MoE models like Mixtral already activate only 2 experts
            per token per layer — ZynkCluster maps those experts onto separate devices so they execute in parallel,
            rather than routing everything through one machine.
          </p>
          <p style={{lineHeight: '1.6', color: '#9aa5c4', fontSize: '0.92rem', marginTop: '10px'}}>
            The MoE architecture and parallel expert execution are established research concepts. What's being
            proposed here is the deployment model: expert-level distribution over consumer WiFi/LAN, built on
            top of Zynkbot's existing ZynkSync device-pairing infrastructure.
          </p>
        </div>

        <div className="about-section" style={{
          background: 'rgba(255, 121, 198, 0.1)',
          border: '1px solid #ff79c6',
          borderRadius: '8px',
          padding: '15px',
          marginBottom: '20px'
        }}>
          <h3 style={{color: '#ff79c6', marginTop: '0'}}>⚠️ Current Status: Design Phase</h3>

          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#ffb86c'}}>Rust Implementation Required:</strong>
            <p style={{margin: '5px 0 0 0', fontSize: '0.95rem', lineHeight: '1.5'}}>
              Zynkbot is designed to be fully Rust-based for performance and safety. However,
              distributed MoE requires Python prototyping first, as all ML libraries (Transformers,
              PyTorch) are Python-based.
            </p>
          </div>

          <div style={{marginBottom: '15px'}}>
            <strong style={{color: '#ffb86c'}}>Hardware Required for Testing:</strong>
            <p style={{margin: '5px 0 0 0', fontSize: '0.95rem', lineHeight: '1.5'}}>
              Proof of concept requires 2+ GPU-enabled devices (NVIDIA/AMD) on the same local network,
              with ~13GB VRAM per node for Mixtral 8x7B.
            </p>
          </div>

          <div>
            <strong style={{color: '#50fa7b'}}>What's Included:</strong>
            <ul style={{margin: '5px 0 0 0', paddingLeft: '20px', fontSize: '0.95rem', lineHeight: '1.6'}}>
              <li>Detailed architecture documentation</li>
              <li>Code examples showing proposed implementation</li>
              <li>Comparison with existing approaches (Petals)</li>
              <li>Future roadmap for Rust integration</li>
            </ul>
          </div>
        </div>

        <div className="about-section">
          <h3>How is this different from Petals?</h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '15px',
            marginTop: '15px'
          }}>
            <div style={{
              background: 'rgba(189, 147, 249, 0.1)',
              border: '1px solid #bd93f9',
              borderRadius: '8px',
              padding: '15px'
            }}>
              <h4 style={{color: '#bd93f9', marginTop: '0', fontSize: '1rem'}}>Petals (Existing)</h4>
              <ul style={{fontSize: '0.9rem', lineHeight: '1.6', paddingLeft: '20px', margin: '10px 0 0 0'}}>
                <li>Sequential layer processing</li>
                <li>All devices in pipeline</li>
                <li>High cumulative latency</li>
                <li>Every token through all devices</li>
              </ul>
            </div>

            <div style={{
              background: 'rgba(80, 250, 123, 0.1)',
              border: '1px solid #50fa7b',
              borderRadius: '8px',
              padding: '15px'
            }}>
              <h4 style={{color: '#50fa7b', marginTop: '0', fontSize: '1rem'}}>ZynkCluster (Proposed)</h4>
              <ul style={{fontSize: '0.9rem', lineHeight: '1.6', paddingLeft: '20px', margin: '10px 0 0 0'}}>
                <li>Parallel expert execution</li>
                <li>Number of devices used reflects model's active expert count</li>
                <li>Lower latency</li>
                <li>Sparse activation (efficient)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="about-section">
          <h3>Integration with Existing Features</h3>
          <p style={{lineHeight: '1.6', fontSize: '0.95rem'}}>
            ZynkCluster leverages existing ZynkSync device pairing infrastructure. The same device
            discovery and networking code used for memory synchronization can be extended for
            compute clustering - making this a natural evolution of the existing architecture.
          </p>
        </div>

        <div className="about-links" style={{marginTop: '25px', display: 'flex', flexDirection: 'column', gap: '10px'}}>
          <button
            onClick={openLabsFolder}
            className="about-link"
            style={{
              width: '100%',
              padding: '12px',
              background: '#50fa7b',
              color: '#282a36',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              textAlign: 'center',
              flex: '0 0 auto'
            }}
          >
            📂 Documentation & Code Examples
          </button>
        </div>

        <div className="about-footer" style={{marginTop: '25px', paddingTop: '20px', borderTop: '1px solid #44475a'}}>
          <p style={{fontSize: '0.9rem', color: '#9aa5c4', lineHeight: '1.6', margin: '0'}}>
            Design phase. The contribution is the deployment model — expert-level distribution
            over consumer devices using existing ZynkSync infrastructure — not the underlying
            MoE concepts, which are established research. Community feedback and hardware
            collaboration welcome.
          </p>
        </div>
      </div>
    </div>
  );
}
