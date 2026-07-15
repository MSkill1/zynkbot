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
              <p style={{ margin: '6px 0 0', fontSize: '0.88rem', color: '#ececec' }}>Download a GGUF model and run entirely on your device. No API keys, no cost, complete privacy. Slower on most hardware.</p>
            </div>
            <div style={{ flex: '1', minWidth: '200px', background: '#21222c', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #8be9fd' }}>
              <strong style={{ color: '#8be9fd' }}>⚡ API key (pay-per-use)</strong>
              <p style={{ margin: '6px 0 0', fontSize: '0.88rem', color: '#ececec' }}>Connect Claude, GPT, or Grok via API key for fast, capable responses. Most users spend $2–15/month.</p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {onOpenAPIKeys && (
              <button
                onClick={() => { onClose(); onOpenAPIKeys(); }}
                style={{ padding: '8px 18px', background: '#8be9fd', color: '#282a36', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.9rem' }}
              >🔑 Set Up API Keys</button>
            )}
            <button
              onClick={() => setShowCostGuide(true)}
              style={{ padding: '8px 18px', background: 'none', color: '#50fa7b', border: '1px solid #50fa7b', borderRadius: '8px', cursor: 'pointer', fontSize: '0.9rem' }}
            >💰 What will this cost? →</button>
          </div>
        </div>

        {/* Step 1: Load Einstein Demo */}
        <div className="guide-step">
          <div className="step-header">
            <span className="step-number">1</span>
            <h3>Load the Einstein Demo</h3>
          </div>
          <p>First, we'll load a pre-built memory set so you can see how Zynkbot works with real data:</p>
          <ol>
            <li>Click the <strong>⚙️ Settings</strong> button in the bottom left corner</li>
            <li>Scroll down to the demo section</li>
            <li>Click <strong>👨‍🔬 Load Einstein Demo</strong></li>
            <li>Wait a few seconds while 59 memories and relationships load</li>
            <li>The page will reload automatically when complete</li>
          </ol>
          <div className="info-box">
            <strong>What this does:</strong> Loads Einstein's memories from a first-person perspective.
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
            <li>Click <strong>📚 Memory Manager</strong> button in the Recent Memories section</li>
            <li>Browse through Einstein's memories and relationships</li>
            <li>Click the <strong>🕸️ Graph View</strong> tab to see memory connections</li>
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
          <p>Watch for the <strong>recalled memories dropdown</strong> in Zynkbot's response to see:</p>
          <ul>
            <li>Which memories were retrieved for each query</li>
            <li>How Zynkbot used them in the response</li>
            <li>The similarity scores showing relevance</li>
          </ul>
          <div className="info-box">
            <strong>Ready to use Zynkbot for yourself?</strong> When you're done exploring, open
            <strong> 📚 Memory Manager</strong> and click <strong>Clear All Memories</strong> to clear the Einstein demo.
            Then go to <strong>⚙️ Settings → Getting Started → Onboarding</strong> to set up your
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
              <p>Switch between Guardian, Child, Sovereign, Witness, and HIPAA modes in Settings.
              Each has different safety filtering levels. <em>You</em> control what gets filtered, not a corporation.</p>
            </div>

            <div className="feature-item">
              <strong>🎭 Model Switching</strong>
              <p>Change AI models on the fly. Try Anthropic Claude, OpenAI GPT, Grok, or add your own local .gguf models.
              Your memories persist across all models.</p>
            </div>

            <div className="feature-item">
              <strong>🤝 Ensemble Mode</strong>
              <p>Click the <strong>Ensemble</strong> button to query multiple models simultaneously.
              Get a synthesized answer combining Claude, GPT, and local models. Reduces hallucinations
              and improves clarity by cross-validating responses from different AI systems.</p>
            </div>

            <div className="feature-item">
              <strong>🌐 Web Search</strong>
              <p>Ask questions requiring current information (e.g., "What's the weather in Tokyo?").
              Zynkbot will detect the need and offer to search the web via SearXNG. Review and edit the
              search query before execution.</p>
            </div>

            <div className="feature-item">
              <strong>📚 Knowledge Base (RAG)</strong>
              <p>Click <strong>⚙️ Settings → Knowledge Base</strong> to upload documents (txt, md, json, code files).
              Ask questions about your documents and Zynkbot will search them semantically.</p>
            </div>

            <div className="feature-item">
              <strong>🔄 ZynkSync - Memory Sharing</strong>
              <p>Click <strong>⚙️ Settings → ZynkSync</strong> to pair devices and sync memories
              across your network. Your work laptop and home desktop can share the same AI context.</p>
            </div>

            <div className="feature-item">
              <strong>🔗 ZynkLink - File Sharing</strong>
              <p>Click <strong>⚙️ Settings → ZynkLink</strong> to share directories across paired devices
              without cloud storage. All transfers happen peer-to-peer.</p>
            </div>

            <div className="feature-item">
              <strong>💬 ZChat - Device Messaging</strong>
              <p>Once linked via ZynkLink, open the ZynkLink panel and click a linked contact to open chat.
              Messages go directly between devices — no external servers involved.</p>
            </div>

            <div className="feature-item">
              <strong>🧩 Snap-Ins (Experimental)</strong>
              <p>Click <strong>⚙️ Settings → Snap-Ins</strong> to access professional tool modules.
              Try the Therapist demo: create patient notes with structured data entry and isolated
              storage. Snap-ins can be customized for your workflow—medical, legal, research, etc.</p>
            </div>

            <div className="feature-item">
              <strong>🔬 ZynkCluster (Upcoming)</strong>
              <p>Click <strong>⚙️ Settings → ZynkCluster</strong> to view research on distributed
              Mixture of Experts inference. See architecture docs and code examples for novel
              parallel execution approach.</p>
            </div>

            <div className="feature-item">
              <strong>👤 User Identity</strong>
              <p>Click your user ID in the top right to manage your identity and see paired devices.</p>
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
