import React from "react";
import "../styles/AboutModal.css";

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
            <p>See exactly which memories the AI retrieved and how it used them in each response.</p>
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
            <p>Built-in web search via SearXNG for current information. Upload text documents (.txt, .md, .json, etc.) for semantic RAG search. Query your personal knowledge base naturally in conversation.</p>
          </div>

          <div className="feature-item">
            <strong>Ensemble Mode:</strong>
            <p>Query multiple AI models simultaneously (Claude, GPT, local models) and get a synthesized consensus answer. Reduces hallucinations and improves response clarity by cross-validating across different AI systems.</p>
          </div>

          <div className="feature-item">
            <strong>Snap-Ins (Experimental):</strong>
            <p>Professional tool modules with isolated data storage and specialized interfaces. Demo available: Therapist notes system with patient session tracking, structured data entry, and HIPAA-mode integration. Designed for extensibility—build custom snap-ins for your professional workflow.</p>
          </div>
        </div>

        <div className="about-section">
          <h3>Technical Architecture</h3>

          <p className="tech-description">
            Zynkbot uses vector embeddings for semantic memory search,
            supports both local inference and API backends, and provides schema-based
            routing for personal, utility, and ethical queries.
          </p>

          <p className="tech-description">
            <strong>Conversational Memory vs. RAG:</strong> Most AI assistants use RAG
            (Retrieval-Augmented Generation) for document retrieval in enterprise knowledge bases.
            Zynkbot uses conversational memory designed for personal AI—learning from your
            interactions, preferences, and life context rather than searching static documents.
            Different architecture for a fundamentally different purpose.
          </p>

          <div className="performance-note">
            <strong>Note on Local Model Performance:</strong>
            <p>The local inference option is intentionally unoptimized in this prototype to demonstrate functionality on basic hardware. With GPU acceleration and model optimization, local inference can run efficiently on mobile devices - as proven by apps like LM Studio and Ollama running on smartphones today.</p>
          </div>
        </div>

        <div className="about-links">
          <a
            href="https://github.com/MSkill1/zynkbot"
            target="_blank"
            rel="noopener noreferrer"
            className="about-link"
          >
            View on GitHub
          </a>
          <a
            href="https://github.com/MSkill1/zynkbot/tree/main/docs"
            target="_blank"
            rel="noopener noreferrer"
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
