import React from "react";

export default function CostGuideModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const tableRows = [
    ["Light (a few exchanges a day)", "100–150", "$1.50–4"],
    ["Moderate (real daily use)", "400–600", "$5–15"],
    ["Heavy (hours daily, long conversations)", "1,500+", "$25–60+"],
  ];

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '16px' }}
      onClick={onClose}
    >
      <div
        style={{ background: '#282a36', borderRadius: '12px', width: '100%', maxWidth: '680px', maxHeight: '85vh', overflowY: 'auto', padding: '28px', position: 'relative' }}
        onClick={e => e.stopPropagation()}
      >
        <button onClick={onClose} style={{ position: 'absolute', top: '16px', right: '16px', background: 'none', border: 'none', color: '#6272a4', fontSize: '1.4rem', cursor: 'pointer', lineHeight: 1 }}>×</button>

        <h2 style={{ color: '#50fa7b', marginTop: 0, marginBottom: '6px' }}>💰 What Will API Keys Cost Me?</h2>
        <p style={{ color: '#ff5555', fontSize: '0.85rem', marginBottom: '20px', lineHeight: 1.5 }}>
          <strong>⚠️ Speculative estimates.</strong> Prices change frequently. Always check each provider's site and watch your usage dashboard during your first month.
        </p>

        <h3 style={{ color: '#8be9fd', marginBottom: '10px' }}>The short version</h3>
        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '16px' }}>
          Zynkbot uses a <strong>bring-your-own-key</strong> model — instead of paying $20/month per provider, you pay per use. For most people this is significantly cheaper, and you get all three providers in one app with your memory staying local.
        </p>

        <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '20px', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ background: '#44475a' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: '#f8f8f2' }}>Usage pattern</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: '#f8f8f2' }}>Messages/month</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: '#50fa7b' }}>Est. cost</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map(([usage, msgs, cost], i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#21222c' : '#282a36' }}>
                <td style={{ padding: '8px 12px', color: '#ececec' }}>{usage}</td>
                <td style={{ padding: '8px 12px', color: '#ececec' }}>{msgs}</td>
                <td style={{ padding: '8px 12px', color: '#50fa7b', fontWeight: 'bold' }}>{cost}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={{ background: '#21222c', borderRadius: '8px', padding: '14px', marginBottom: '20px', borderLeft: '3px solid #ffb86c' }}>
          <strong style={{ color: '#ffb86c' }}>Honest caveat:</strong>
          <span style={{ color: '#ececec', fontSize: '0.9rem' }}> Heavy users on top-tier models may exceed subscription pricing. Subscriptions are flat-rate; APIs are metered. Light and moderate users usually win with APIs.</span>
        </div>

        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '6px' }}>
          <strong style={{ color: '#8be9fd' }}>Comparison:</strong> Two subscription apps (Claude + ChatGPT) = <strong>$40/month</strong>, two separate memories, no Grok.<br />
          Zynkbot moderate use ≈ <strong style={{ color: '#50fa7b' }}>$5–15/month</strong>, all three providers, one memory graph on your device.
        </p>

        <h3 style={{ color: '#8be9fd', marginTop: '24px', marginBottom: '10px' }}>What one exchange actually costs</h3>
        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '10px' }}>
          Each Zynkbot message is heavier than a bare chat message because the memory system retrieves relevant context:
        </p>
        <ul style={{ color: '#ececec', lineHeight: 1.8, paddingLeft: '20px', marginBottom: '10px' }}>
          <li>System prompt + retrieved memories + conversation context + your message → ~1,500–3,000 input tokens</li>
          <li>Model reply → ~300–800 output tokens</li>
          <li>Small memory processing calls (what to store, contradiction detection)</li>
        </ul>
        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '16px' }}>
          Total: <strong>~2,500–4,000 tokens per exchange.</strong> Mid-tier models (Sonnet/GPT-4o-class) cost roughly <strong style={{ color: '#50fa7b' }}>$0.01–0.03 per exchange</strong>. Budget tiers (Haiku/mini) are often under a cent.
        </p>

        <h3 style={{ color: '#8be9fd', marginBottom: '10px' }}>Ensemble mode multiplier</h3>
        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '16px' }}>
          Ensemble queries multiple models and synthesizes answers — roughly <strong>4–5× the cost</strong> of a normal exchange (~$0.05–0.15 per question on mid-tier models). Use it for questions where accuracy matters, not as a default.
        </p>

        <h3 style={{ color: '#8be9fd', marginBottom: '10px' }}>Local models cost $0</h3>
        <p style={{ color: '#ececec', lineHeight: 1.6, marginBottom: '20px' }}>
          Local GGUF models running on your hardware have no per-message cost. Tradeoff is speed and capability. Mixing modes (local for casual use, API for hard questions) is a legitimate cost strategy.
        </p>

        <h3 style={{ color: '#8be9fd', marginBottom: '10px' }}>Practical tips</h3>
        <ul style={{ color: '#ececec', lineHeight: 1.8, paddingLeft: '20px', marginBottom: '20px' }}>
          <li><strong>Set spending limits</strong> in each provider's console on day one — caps worst-case to a fixed dollar amount.</li>
          <li><strong>Default to mid- or budget-tier models</strong>; switch to top-tier only when a question deserves it.</li>
          <li><strong>Start new conversations</strong> when the topic changes — message #40 in a long thread costs several times message #2.</li>
          <li><strong>Use Ensemble deliberately</strong>, not habitually.</li>
          <li><strong>Use local models for casual chatter</strong> if your hardware supports them.</li>
        </ul>

        <p style={{ color: '#6272a4', fontSize: '0.8rem', lineHeight: 1.5 }}>
          Estimates written mid-2026. Token prices have historically fallen over time — if your real costs differ meaningfully, open an issue. Real user data beats estimation.
        </p>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px' }}>
          <button onClick={onClose} style={{ padding: '10px 24px', background: '#50fa7b', color: '#282a36', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer', fontSize: '0.95rem' }}>Got it</button>
        </div>
      </div>
    </div>
  );
}
