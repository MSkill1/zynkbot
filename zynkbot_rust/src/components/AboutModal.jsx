import React from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import "../styles/AboutModal.css";

// A plain <a target="_blank"> does nothing inside the app window: the address has to be
// handed to the system browser (2026-11-10). The phone's own bridge is tried first.
async function openInBrowser(e, url) {
  e.preventDefault();
  try { if (window.AndroidPaths?.openUrl && window.AndroidPaths.openUrl(url)) return; } catch (_) {}
  try { await openUrl(url); } catch (_) { window.open(url, "_blank"); }
}

export default function AboutModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="about-modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>

        <h2>About Zynkbot</h2>

        <div className="about-section">
          <h3>Privacy-First AI Assistant</h3>

          <div className="feature-item">
            <strong>Your Data Stays Local - Zero-Trust Architecture:</strong>
            <p>All conversations and memories are stored securely in a local SQLite database on your device — never on external corporate servers. Unlike ChatGPT where you must trust OpenAI with your data, Zynkbot requires zero trust in third parties. You control the database, you control the models, you control the memory. Unlike websites that require user profiles, Zynkbot uses API calls which are not retained on corporate servers.  Easy setup.</p>
          </div>

          <div className="feature-item">
            <strong>Full Memory Control:</strong>
            <p>View, edit, and delete any stored memory in real time. You own your data completely and control what your Zynkbot knows.</p>
          </div>

          <div className="feature-item">
            <strong>Transparent Recall:</strong>
            <p>See exactly which memories the AI retrieved for each response.</p>
          </div>

          <div className="feature-item">
            <strong>Intelligent Memory Processing:</strong>
            <p>Automatically detects contradictions and prompts you to resolve conflicts between new and existing memories—choose which is correct or explain how both can be true. Every memory undergoes factual extraction, relationship classification, semantic search, and duplicate prevention to maintain a consistent, organized memory vault without silent overwrites or preventable hallucinations.</p>
          </div>

          <div className="feature-item">
            <strong>Customizable Safety:</strong>
            <p>Choose your containment mode—from unrestricted conversation to child-safe filtering tailored to your needs.</p>
          </div>

          <div className="feature-item">
            <strong>Zynking Networks (Complete Ecosystem):</strong>
            <p>
              <strong>ZynkSync:</strong> Secure peer-to-peer memory sharing between paired devices.<br/>
              <strong>ZynkLink:</strong> Distributed file sharing across your private network.<br/>
              <strong>ZChat:</strong> Direct device-to-device messaging over your local network.<br/>
              No corporate servers required. See <em>Why Zynkbot?</em> for details.
            </p>
          </div>

          <div className="feature-item">
            <strong>Web Search & Knowledge Base:</strong>
            <p>Built-in web search for current information. Add documents (text, PDF, Word, code files) to your Knowledge Base, turn on KB under the message box, and ask about them in conversation.</p>
          </div>

          <div className="feature-item">
            <strong>Ensemble Mode:</strong>
            <p>Query multiple AI models simultaneously (Claude, GPT, Grok, Mistral, local models) and get a synthesized consensus answer. Reduces hallucinations and improves response clarity by cross-validating across different AI systems.</p>
          </div>

          <div className="feature-item">
            <strong>Snap-ins (Planned):</strong>
            <p>Professional tool modules that come after version 1.0. Therapist, legal and journalist tools are planned; an early sample of the therapist journal is included.</p>
          </div>
        </div>

        <div className="about-section">
          <h3>Technical Architecture</h3>

          <p className="tech-description">
            Zynkbot uses vector embeddings for semantic memory search and
            supports both local inference and API backends.
          </p>

          <p className="tech-description">
            <strong>Conversational Memory and RAG:</strong> Most AI assistants use RAG
            (Retrieval-Augmented Generation) to look things up in a fixed set of documents.
            Zynkbot is built around conversational memory for a personal AI: it learns from your
            interactions, preferences, and life context, and you can read, edit and delete
            everything it keeps. It has document retrieval too: add files to the Knowledge Base,
            turn on KB, and Zynkbot answers from them alongside your memories.
          </p>

          <div className="performance-note">
            <strong>Note on Local Model Performance:</strong>
            <p>Local models work today on desktop computers and on Android phones. On a desktop the model runs on the processor (CPU) by default, which works on any computer. When Zynkbot is built with CUDA enabled on a computer with an NVIDIA graphics card, the model runs on the card instead, which is much faster. On Android the model runs on the phone itself. A local model is slower than a cloud model, and how much slower depends on your hardware.</p>
          </div>
        </div>

        <div className="about-links">
          <a
            href="https://github.com/MSkill1/zynkbot"
            onClick={(e) => openInBrowser(e, "https://github.com/MSkill1/zynkbot")}
            className="about-link"
          >
            View on GitHub
          </a>
          <a
            href="https://github.com/MSkill1/zynkbot/tree/main/docs"
            onClick={(e) => openInBrowser(e, "https://github.com/MSkill1/zynkbot/tree/main/docs")}
            className="about-link"
          >
            Documentation
          </a>
        </div>

        <div className="about-footer">
          <p>The AI that works for you — not the other way around.</p>
        </div>
      </div>
    </div>
  );
}
