import React from "react";
import "../styles/WhyZynkbotModal.css";

export default function WhyZynkbotModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="why-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h1 className="modal-title">Why Zynkbot?</h1>

        {/* Data Privacy Comparison */}
        <section className="privacy-section">
          <h2>Data Privacy: Zynkbot vs. ChatGPT</h2>
          <table className="comparison-table">
            <thead>
              <tr>
                <th className="feature-col">Feature</th>
                <th className="chatgpt-col">ChatGPT</th>
                <th className="zynkbot-col">Zynkbot</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="feature-cell">Data Storage</td>
                <td className="chatgpt-cell">Corporate servers</td>
                <td className="zynkbot-cell">Your private database</td>
              </tr>
              <tr>
                <td className="feature-cell">Training Data Usage</td>
                <td className="chatgpt-cell">Your data may be used for training</td>
                <td className="zynkbot-cell">Never used for training</td>
              </tr>
              <tr>
                <td className="feature-cell">Memory</td>
                <td className="chatgpt-cell">Hidden — you can't see what it knows about you</td>
                <td className="zynkbot-cell">Fully visible & editable</td>
              </tr>
              <tr>
                <td className="feature-cell">Recall Transparency</td>
                <td className="chatgpt-cell">No visibility into what influenced a response</td>
                <td className="zynkbot-cell">Each response shows exactly which memories were retrieved and used</td>
              </tr>
              <tr>
                <td className="feature-cell">Memory Conflicts</td>
                <td className="chatgpt-cell">AI resolves contradictions silently, you're not told</td>
                <td className="zynkbot-cell">You're prompted to resolve — keep old, keep new, or explain both</td>
              </tr>
              <tr>
                <td className="feature-cell">Safety Controls</td>
                <td className="chatgpt-cell">Fixed by OpenAI</td>
                <td className="zynkbot-cell">User-controlled (custom safety modes)</td>
              </tr>
              <tr>
                <td className="feature-cell">Offline Capable</td>
                <td className="chatgpt-cell">No (requires internet)</td>
                <td className="zynkbot-cell">Yes (with local models)</td>
              </tr>
              <tr>
                <td className="feature-cell">Data Deletion</td>
                <td className="chatgpt-cell">Request required, not immediate</td>
                <td className="zynkbot-cell">Instant, complete control</td>
              </tr>
              <tr>
                <td className="feature-cell">Account Required</td>
                <td className="chatgpt-cell">Yes — email, verification codes, password resets</td>
                <td className="zynkbot-cell">No account, no login — just open it</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* Containment Modes */}
        <section className="modes-section">
          <h2>User-Controlled Safety: Containment Modes</h2>
          <p className="section-intro">
            Unlike ChatGPT's one-size-fits-all approach, Zynkbot lets <strong>you</strong> choose
            the level of safety and filtering that makes sense for your use case.
          </p>

          <div className="modes-grid">
            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">🛡️</span>
                <span className="mode-name">Guardian</span>
              </div>
              <p>Default mode: Blocks harmful content while allowing thoughtful conversation. Similar to corporate content filtering - no potentially risky or controversial responses.</p>
            </div>

            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">👶</span>
                <span className="mode-name">Child</span>
              </div>
              <p>Enhanced safety for minors: Explicit content filtering + semantic analysis. Let Zynkbot filter your child's access to the internet for harmful or mature content.</p>
            </div>

            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">👑</span>
                <span className="mode-name">Sovereign</span>
              </div>
              <p>Model responses unfiltered.  Shows a warning if info is potentially harmful.</p>
            </div>

            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">👁️</span>
                <span className="mode-name">Witness</span>
              </div>
              <p>Research/testing mode: No filtering. Used for debugging, simulations, and academic work.</p>
            </div>

            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">🏥</span>
                <span className="mode-name">HIPAA</span>
              </div>
              <p>Healthcare compliance mode: Blocks PHI (Protected Health Information) at routing layer, prevents diagnostic/dosing advice, enforces ephemeral memory. Check documentation for limitations.</p>
            </div>

            <div className="mode-card">
              <div className="mode-header">
                <span className="mode-icon">🔮</span>
                <span className="mode-name">Potential Modes</span>
              </div>
              <p>Future compliance-focused modes under consideration: Mental health professionals, Legal (lawyers/law offices), Elder (cognitive support), GDPR (EU privacy), COPPA (children's privacy) - build a mode for your use case.</p>
            </div>
          </div>
        </section>

        {/* Platform Capabilities */}
        <section className="networks-section">
          <h2>Platform Capabilities</h2>
          <p className="section-intro">
            <strong>Beyond the conversation:</strong> Zynkbot includes a full set of tools for AI-assisted
            research, local networking, and professional workflows — all running on your hardware,
            with no cloud dependency required.
          </p>

          <div className="feature-cards">
            <div className="feature-card">
              <div className="feature-icon">🤝</div>
              <h3>Ensemble Mode</h3>
              <p>
                <strong>✅ Available:</strong> Query multiple AI models simultaneously and synthesize
                their answers into a consensus response. Compare Claude, GPT, local models, and more
                side by side — great for fact-checking and catching AI hallucinations.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🌐</div>
              <h3>Web Search</h3>
              <p>
                <strong>✅ Available:</strong> Built-in DuckDuckGo web search. When the AI needs
                current information, it displays the query to edit first and synthesizes results into its
                answer with source context included.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">📚</div>
              <h3>Knowledge Base (RAG)</h3>
              <p>
                <strong>✅ Available:</strong> Upload your own documents (.txt, .md, code files, and
                more) for semantic retrieval — automatic chunking, embedding, and search included. Unlike most AI assistants,
                you can see exactly what's indexed, remove individual documents, and control what your AI knows.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🧩</div>
              <h3>Snap-Ins</h3>
              <p>
                <strong>🧩 Ongoing Development:</strong> Industry specific tools with isolated data and specialized interfaces. Proof-of-concept: Therapist notes with patient session. Zynkbot is a development platform. Build custom snap-ins for your workflow.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔄</div>
              <h3>ZynkSync — Memory Sync</h3>
              <p>
                <strong>✅ Available:</strong> Pair your devices and keep memories in sync across
                your laptop, desktop, and phone over your local network. No cloud required —
                everything stays on your hardware.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔗</div>
              <h3>ZynkLink — File Sharing</h3>
              <p>
                <strong>✅ Available:</strong> Share directories between paired devices without cloud
                storage. Download shared files directly into your bot's knowledge base with one click.
                Large file sharing supported - share AI models without the internet.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">💬</div>
              <h3>ZChat — Device Messaging</h3>
              <p>
                <strong>✅ Available:</strong> Direct text messaging with other Zynkbot users you've linked to via ZynkLink. Messages travel directly between devices over your local network — no intermediary servers, no cloud storage. Messages are saved locally in your database.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔬</div>
              <h3>ZynkCluster — Distributed Inference</h3>
              <p>
                <strong>🔬 Upcoming:</strong> Distributed Mixture-of-Experts inference across local
                networks, allowing multiple machines to collaboratively run models too large for any
                single device. Active research — see in-app documentation.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🏢</div>
              <h3>Offline &amp; Air-Gapped Deployment</h3>
              <p>
                <strong>Enterprise Solutions:</strong> Enterprise AI for environments where cloud
                services aren't an option — developing regions, post-disaster scenarios, law firms
                that want AI locally without exposing sensitive client data to third parties. The
                full stack runs offline: inference, memory, search, and networking over a local
                hotspot or LAN. See case studies on GitHub.
              </p>
            </div>
          </div>

          <div className="highlight-box">
            <strong>No corporate AI can copy this:</strong> Their business model requires centralization.
            Zynkbot is built for people who want capable AI without giving a corporation permanent
            access to their data.
          </div>
        </section>

        {/* CTA Section */}
        <section className="cta-section">
          <h2>Experience the Difference</h2>
          <p>
            Open Beta 0.9 — Zynkbot's core features are complete: containment modes, transparent memory,
            networking, and model flexibility. Experiment with different models and containment
            modes to see how responses change.
          </p>
          <p>
            How Zynkbot develops as a personal AI companion is genuinely uncharted territory — it won't
            be clear until people use it over time. Volunteer testers and community feedback are needed
            before mobile development begins. Contact matt@containai.ai if you're interested.
          </p>

          <div className="cta-checklist">
            <div className="cta-item">✓ Switch between 5 containment modes (Guardian, Child, Sovereign, Witness, HIPAA)</div>
            <div className="cta-item">✓ View and edit your memories in real-time with graph visualization</div>
            <div className="cta-item">✓ Test API models (Claude, GPT, Grok) or add local .gguf models</div>
            <div className="cta-item">✓ Pair devices with ZynkSync for memory synchronization</div>
            <div className="cta-item">✓ Share files with ZynkLink across your network</div>
            <div className="cta-item">✓ Send private messages with ZChat directly between devices</div>
            <div className="cta-item">✓ Search the web with built-in DuckDuckGo integration</div>
            <div className="cta-item">✓ Upload documents for Knowledge Base RAG search</div>
            <div className="cta-item">✓ Query multiple models simultaneously with Ensemble mode</div>
          </div>

          <button className="cta-button" onClick={onClose}>
            Start Chatting
          </button>

          <p className="cta-note">
            Want complete privacy? Add local GGUF models to the models/ directory. Fork on GitHub to customize.
          </p>
        </section>

        {/* Footer */}
        <footer className="modal-footer">
          <i>Memory without surveillance, intelligence without manipulation.</i>
        </footer>
      </div>
    </div>
  );
}
