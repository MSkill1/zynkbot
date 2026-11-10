import React, { useState } from "react";
import CostGuideModal from "./CostGuideModal";
import "../styles/GettingStartedModal.css";

export default function GettingStartedModal({ isOpen, onClose, onOpenAPIKeys }) {
  const [showCostGuide, setShowCostGuide] = useState(false);

  if (!isOpen) return null;

  return (
    <>
    <CostGuideModal isOpen={showCostGuide} onClose={() => setShowCostGuide(false)} />
    <div className="modal-overlay" onClick={onClose}>
      <div className="getting-started-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2>Getting Started with Zynkbot</h2>

        <div className="guide-intro" style={{ marginBottom: '16px' }}>
          <p style={{ marginBottom: '10px' }}>
            Zynkbot is different from ChatGPT and other AI assistants — instead of hiding memories in a black box,
            it gives you complete transparency and control over what it knows about you.
          </p>
          <p style={{ marginBottom: '12px' }}>
            <strong>Two ways to use it:</strong>
          </p>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '12px' }}>
            <div style={{ flex: '1', minWidth: '200px', background: '#21222c', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #50fa7b' }}>
              <strong style={{ color: '#50fa7b' }}>🖥️ Local model (free)</strong>
              <p style={{ margin: '6px 0 0', fontSize: '0.88rem', color: '#ececec' }}>Download the Zynkbot model and run it entirely on your device. No API key, no cost, and nothing leaves the device. Slower on most hardware.</p>
            </div>
            <div style={{ flex: '1', minWidth: '200px', background: '#21222c', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #8be9fd' }}>
              <strong style={{ color: '#8be9fd' }}>⚡ API key (pay-per-use)</strong>
              <p style={{ margin: '6px 0 0', fontSize: '0.88rem', color: '#ececec' }}>Connect Claude, GPT, Grok, or Mistral via API key for fast, capable responses. Your data is still protected: your memories and conversations are stored only on this device, and the providers keep API requests for a limited time, for security reasons only. Most users spend $2–15/month.</p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {onOpenAPIKeys && (
              <button
                onClick={() => { onClose(); onOpenAPIKeys(); }}
                style={{ padding: '8px 18px', background: '#8be9fd', color: '#282a36', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.9rem' }}
              >🔑 Set up a model</button>
            )}
            <button
              onClick={() => setShowCostGuide(true)}
              style={{ padding: '8px 18px', background: 'none', color: '#50fa7b', border: '1px solid #50fa7b', borderRadius: '8px', cursor: 'pointer', fontSize: '0.9rem' }}
            >💰 What will this cost? →</button>
          </div>
        </div>

        <div className="guide-step">
          <div className="step-header">
            <span className="step-number">1</span>
            <h3>Load the Einstein Demo</h3>
          </div>
          <p>First, we'll load a pre-built memory set so you can see how Zynkbot works with real data:</p>
          <ol>
            <li>Click the <strong>⚙️</strong> button in the bottom left corner to open <strong>System Controls</strong></li>
            <li>Open <strong>🚀 Getting Started</strong> at the top of the list</li>
            <li>Click <strong>Load Einstein Demo</strong> and confirm</li>
            <li>Wait a few seconds while the memories and their links load</li>
            <li>The page will reload automatically when complete</li>
          </ol>
          <div className="info-box">
            <strong>What this does:</strong> Loads Einstein's life as Zynkbot would have remembered it had he typed it in:
            what he said, the facts kept from each message, and the links between them.
            Zynkbot will treat you as Einstein and respond as your personal AI assistant.
          </div>
        </div>

        {/* Step 2: View & Edit Memories */}
        <div className="guide-step">
          <div className="step-header">
            <span className="step-number">2</span>
            <h3>View & Edit Memories</h3>
          </div>
          <p>Unlike ChatGPT's hidden memory, you can see and control everything Zynkbot knows:</p>
          <ol>
            <li>Click <strong>📚 Memory Manager</strong> in the Recent Memories section (on a phone, <strong>📚 Memories</strong> beside the message box)</li>
            <li>Browse through Einstein's memories and relationships</li>
            <li>Click <strong>🗺️ Map</strong> to see the people, places and things they share</li>
            <li>Try editing or deleting a memory, then ask a related question</li>
            <li>Notice how Zynkbot's response changes based on your edits</li>
          </ol>
          <div className="info-box warning">
            <strong>This is the key difference:</strong> With ChatGPT, if it remembers something wrong,
            you can't see or fix it. With Zynkbot, you have complete visibility and control.
          </div>
        </div>

        {/* Step 3: Test Memory Recall */}
        <div className="guide-step">
          <div className="step-header">
            <span className="step-number">3</span>
            <h3>Test Memory Recall</h3>
          </div>
          <p>Now ask questions to see how Zynkbot recalls and uses memories:</p>
          <div className="example-queries">
            <div className="query-box">
              <strong>Try asking:</strong>
              <ul>
                <li><code>"What is my theory about the photoelectric effect?"</code></li>
                <li><code>"Who did I work with at the patent office?"</code></li>
                <li><code>"What did I think about quantum mechanics?"</code></li>
                <li><code>"Tell me about my relationship with my wife Mileva"</code></li>
              </ul>
            </div>
          </div>
          <p>Open <strong>View recalled memories</strong> under Zynkbot's response to see which memories
          were retrieved for that question.</p>
          <div className="info-box">
            <strong>Ready to use Zynkbot for yourself?</strong> When you're done exploring, open
            <strong> 📚 Memory Manager</strong>, open <strong>Storage</strong> and click <strong>Clear All</strong> to clear the Einstein demo.
            Then go to <strong>⚙️ System Controls → Getting Started → Start 5-Minute Onboarding</strong> to set up your
            own profile.
          </div>
        </div>

        {/* Step 4: Explore More Features */}
        <div className="guide-step">
          <div className="step-header">
            <span className="step-number">4</span>
            <h3>Explore More Features</h3>
          </div>
          <p>Try these other powerful features:</p>

          <div className="feature-list">
            <div className="feature-item">
              <strong>🛡️ Containment Modes</strong>
              <p>Switch between Guardian, Child, Sovereign, Witness, and HIPAA modes in System Controls.
              Each has different safety filtering levels. <em>You</em> control what gets filtered, not a corporation.</p>
            </div>

            <div className="feature-item">
              <strong>🎭 Model Switching</strong>
              <p>Change AI models on the fly. Try Anthropic Claude, OpenAI GPT, Grok, Mistral, or add your own local .gguf models.
              Your memories persist across all models.</p>
            </div>

            <div className="feature-item">
              <strong>🤝 Ensemble Mode</strong>
              <p>Click the <strong>Ensemble</strong> button to query multiple models simultaneously.
              Get a synthesized answer combining Claude, GPT, Grok, Mistral, and local models. Reduces hallucinations
              and improves clarity by cross-validating responses from different AI systems.</p>
            </div>

            <div className="feature-item">
              <strong>🌐 Web Search</strong>
              <p>Ask questions requiring current information (e.g., "What is in the news today?").
              Zynkbot will detect the need and offer to search the web. Review and edit the
              search query before it runs.</p>
            </div>

            <div className="feature-item">
              <strong>📚 Knowledge Base (RAG)</strong>
              <p>Open <strong>⚙️ System Controls → Knowledge Base</strong> to add documents (text, PDF, Word, code files).
              Turn on <strong>📚 KB</strong> under the message box, ask about your documents, and Zynkbot searches them by meaning.</p>
            </div>

            <div className="feature-item">
              <strong>🔄 ZynkSync - Memory Sharing</strong>
              <p>Open <strong>⚙️ System Controls → ZynkSync</strong> to pair devices and sync memories
              across your network. Your work laptop and home desktop can share the same AI context.</p>
            </div>

            <div className="feature-item">
              <strong>🔗 ZynkLink - File Sharing</strong>
              <p>Open <strong>⚙️ System Controls → ZynkLink</strong> to share directories across paired devices
              without cloud storage. All transfers happen peer-to-peer.</p>
            </div>

            <div className="feature-item">
              <strong>💬 ZChat - Device Messaging</strong>
              <p>Once linked via ZynkLink, open the ZynkLink panel and click <strong>💬 Chat</strong> beside a linked device. Your own paired devices have the same button in ZynkSync.
              Messages go directly between devices — no external servers involved.</p>
            </div>

            <div className="feature-item">
              <strong>🧩 Snap-ins (Planned)</strong>
              <p>Open <strong>⚙️ System Controls → Snap-ins</strong> to see the professional tool modules planned
              for after version 1.0: therapist, legal and journalist. An early sample of the therapist
              journal is there to try.</p>
            </div>

            <div className="feature-item">
              <strong>🔬 ZynkCluster (Upcoming)</strong>
              <p>Open <strong>⚙️ System Controls → ZynkCluster</strong> to read about research on distributed
              Mixture of Experts inference. It is in the design phase; a button there opens the
              architecture notes and code examples on GitHub.</p>
            </div>

            <div className="feature-item">
              <strong>👤 User Identity</strong>
              <p>Click <strong>Info</strong> at the top of System Controls to see your user ID and this device's ID.</p>
            </div>
          </div>
        </div>

        <div className="guide-footer">
          <p>
            <strong>Remember:</strong> Zynkbot is about giving <em>you</em> control.
            Your data stays local, your memories are transparent, and your choices matter.
          </p>
          <p className="guide-footer-tip">
            <strong>💡 Tip:</strong> To ask about Zynkbot's own features, include "Zynkbot" in your question — e.g., <em>"Zynkbot, how does Ensemble Mode work?"</em>
          </p>
        </div>
      </div>
    </div>
    </>
  );
}
