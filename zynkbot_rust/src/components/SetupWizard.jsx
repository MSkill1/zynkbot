import React, { useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

const SYSTEM_MODELS = [
  { id: 'minilm',     name: 'all-MiniLM-L6-v2', desc: 'Semantic embeddings for memory search',         size: '~90MB'  },
  { id: 'bert-ner',   name: 'BERT NER',           desc: 'Extracts names, dates, and entities from text', size: '~440MB' },
  { id: 'toxic-bert', name: 'ToxicBERT',           desc: 'Local safety classifier for content filtering', size: '~440MB' },
];

const ZB_MODEL_ID = 'zb-qwen3-4b';
const USER_MODELS = [
  // The Zynkbot model (2026-10-27): the model the app is built around, phone-sized; offered
  // on every platform, the only one offered on a phone.
  { id: ZB_MODEL_ID,      name: 'Zynkbot model (Qwen3 4B)',          desc: 'The model Zynkbot is built around. Phone-sized, fast, no key needed', size: '2.4GB' },
  { id: 'qwen3-8b',       name: 'Qwen3 8B',                        desc: 'Best all-around — recommended for most users', size: '5.0GB' },
  { id: 'deepseek-r1-8b', name: 'DeepSeek R1 Distill Llama 8B',    desc: 'Reasoning and analytical tasks',               size: '4.7GB' },
  { id: 'llama-lexi-8b',  name: 'Llama 3.1 8B Lexi Uncensored',    desc: 'Creative, unfiltered responses',               size: '4.9GB' },
];

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '';
  const gb = bytes / 1024 / 1024 / 1024;
  if (gb >= 0.1) return gb.toFixed(2) + ' GB';
  const mb = bytes / 1024 / 1024;
  return mb.toFixed(1) + ' MB';
}

const S = {
  overlay: {
    position: 'fixed', inset: 0, background: '#0d0d0d',
    display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999,
  },
  card: {
    background: '#1a1a1a', border: '1px solid #333', borderRadius: '12px',
    padding: '36px 40px', maxWidth: '520px', width: '90%',
    color: '#e0e0e0', fontFamily: 'monospace',
  },
  title:    { fontSize: '20px', fontWeight: 'bold', marginBottom: '12px', color: '#fff' },
  body:     { fontSize: '13px', color: '#999', lineHeight: '1.7', marginBottom: '20px' },
  label:    { fontSize: '11px', color: '#888', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '10px' },
  row:      { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' },
  name:     { fontSize: '13px', fontWeight: 'bold', color: '#ccc' },
  desc:     { fontSize: '11px', color: '#999', marginTop: '2px' },
  size:     { fontSize: '11px', color: '#888', whiteSpace: 'nowrap', marginLeft: '12px' },
  bar:      { width: '100%', height: '4px', background: '#2a2a2a', borderRadius: '2px', marginTop: '6px', overflow: 'hidden' },
  divider:  { border: 'none', borderTop: '1px solid #252525', margin: '20px 0' },
  btn:      { background: '#7c5cbf', color: '#fff', border: 'none', borderRadius: '6px', padding: '10px 24px', cursor: 'pointer', fontSize: '14px', marginTop: '24px' },
  btnGhost: { background: 'transparent', color: '#777', border: '1px solid #3a3a3a', borderRadius: '6px', padding: '10px 24px', cursor: 'pointer', fontSize: '14px', marginTop: '24px', marginLeft: '12px' },
  checkRow: (selected) => ({ display: 'flex', alignItems: 'flex-start', padding: '12px', border: `1px solid ${selected ? '#7c5cbf' : '#252525'}`, borderRadius: '8px', marginBottom: '8px', cursor: 'pointer', userSelect: 'none' }),
  errorBanner: {
    background: '#2a0a0a', border: '1px solid #c33', borderRadius: '6px',
    padding: '12px 14px', marginBottom: '18px', color: '#f77', fontSize: '13px', lineHeight: '1.6',
  },
  errorDismiss: {
    background: 'transparent', color: '#f77', border: '1px solid #c33',
    borderRadius: '4px', padding: '4px 10px', cursor: 'pointer', fontSize: '12px', marginTop: '8px',
  },
};

// Fresh-install bug (OnePlus, 2026-09-11): width alone misdetected Android. At module
// load the WebView has not applied width=device-width yet, so innerWidth reads the raw
// physical width (1264px OnePlus, 1080px Pixel) — both > 768 — and the phone was shown
// the desktop GGUF "optional local model" flow. The Android bridge global is the reliable
// platform signal; the width test stays only as the narrow-desktop-window fallback.
const isMobile = !!window.AndroidPaths || window.innerWidth <= 768;

export default function SetupWizard({ onComplete }) {
  const [screen, setScreen]           = useState('welcome');
  const [progress, setProgress]       = useState({ minilm: 0, 'bert-ner': 0, 'toxic-bert': 0 });
  const [done, setDone]               = useState({ minilm: false, 'bert-ner': false, 'toxic-bert': false });
  const [selected, setSelected]       = useState([]);
  const [currentLLM, setCurrentLLM]   = useState(null);
  const [llmProgress, setLlmProgress] = useState(0);
  const [llmBytes, setLlmBytes]       = useState({ downloaded: 0, total: 0 });
  const [llmVerifying, setLlmVerifying] = useState(false);
  const [llmQueue, setLlmQueue]       = useState([]);
  const [error, setError]             = useState(null);

  const startRequired = useCallback(async () => {
    setScreen('downloading_required');
    setError(null);
    const unlisteners = [];

    for (const m of SYSTEM_MODELS) {
      unlisteners.push(await listen(`setup:progress:${m.id}`, (e) =>
        setProgress(p => ({ ...p, [m.id]: e.payload.percent }))
      ));
      unlisteners.push(await listen(`setup:complete:${m.id}`, () =>
        setDone(d => ({ ...d, [m.id]: true }))
      ));
    }

    try {
      await invoke('download_system_models');
      setScreen(isMobile ? 'zb_offer' : 'llm_selection');
    } catch (e) {
      setError(String(e));
    } finally {
      unlisteners.forEach(fn => fn());
    }
  }, []);

  const startLLMs = useCallback(async (queue) => {
    if (queue.length === 0) { setScreen('complete'); return; }

    setLlmQueue(queue);
    setScreen('downloading_llm');
    setError(null);

    const ul1 = await listen('setup:llm_progress', (e) => {
      setLlmProgress(e.payload.percent);
      setLlmBytes({ downloaded: e.payload.downloaded, total: e.payload.total });
      setLlmVerifying(!!e.payload.verifying);
    });

    try {
      for (const modelId of queue) {
        setCurrentLLM(modelId);
        setLlmProgress(0);
        setLlmBytes({ downloaded: 0, total: 0 });
        setLlmVerifying(false);
        await invoke('download_user_model', { modelId });
      }
      setScreen('complete');
    } catch (e) {
      setError(String(e));
    } finally {
      ul1();
    }
  }, []);

  const toggle = (id) =>
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);

  const fill = (pct) => ({
    height: '100%', width: `${pct}%`,
    background: pct >= 100 ? '#4caf50' : '#7c5cbf',
    transition: 'width 0.3s ease',
  });

  const ErrorBanner = ({ msg }) => (
    <div style={S.errorBanner}>
      <strong>Download failed</strong><br />{msg}
      <div><button style={S.errorDismiss} onClick={() => setError(null)}>Dismiss</button></div>
    </div>
  );

  if (screen === 'welcome') return (
    <div data-modal="" style={S.overlay}>
      <div style={S.card}>
        <div style={S.title}>Welcome to Zynkbot</div>
        <p style={S.body}>
          Before first use, Zynkbot needs to download several files.
          Everything runs locally — no data leaves your device.
        </p>

        <div style={S.label}>Required — downloads automatically (~970MB total)</div>
        {SYSTEM_MODELS.map(m => (
          <div key={m.id} style={S.row}>
            <div><div style={S.name}>{m.name}</div><div style={S.desc}>{m.desc}</div></div>
            <div style={S.size}>{m.size}</div>
          </div>
        ))}

        <hr style={S.divider} />

        {isMobile ? (
          <p style={{ ...S.body, marginBottom: 0 }}>
            Next, Zynkbot offers its own model (2.4 GB): it runs on this phone with no key and nothing sent anywhere.
            A cloud model is the other choice; add its key after setup via{' '}
            <strong style={{ color: '#ccc' }}>⚙ Settings → Keys and servers</strong>.
          </p>
        ) : (
          <>
            <div style={S.label}>Optional — local LLM for offline chat</div>
            <p style={{ ...S.body, marginBottom: 0 }}>
              Choose one or more on the next screen (4.7–5.0GB each), or skip and connect
              an API key (OpenAI, Anthropic, xAI, or Mistral) instead. Local models work without internet;
              API models require a connection. You can add local models at any time from Settings.
            </p>
          </>
        )}

        <div><button style={S.btn} onClick={startRequired}>Continue</button></div>
      </div>
    </div>
  );

  if (screen === 'downloading_required') return (
    <div data-modal="" style={S.overlay}>
      <div style={S.card}>
        <div style={S.title}>Downloading required models</div>
        <p style={S.body}>One-time download. Please keep the app open.</p>

        {error && <ErrorBanner msg={error} />}

        {SYSTEM_MODELS.map(m => (
          <div key={m.id} style={{ marginBottom: '18px' }}>
            <div style={S.row}>
              <div><div style={S.name}>{m.name}</div><div style={S.desc}>{m.desc}</div></div>
              <div style={{ ...S.size, color: done[m.id] ? '#4caf50' : '#888' }}>
                {done[m.id] ? '✓ Done' : `${progress[m.id]}%`}
              </div>
            </div>
            <div style={S.bar}><div style={fill(done[m.id] ? 100 : progress[m.id])} /></div>
          </div>
        ))}
      </div>
    </div>
  );

  if (screen === 'llm_selection') return (
    <div data-modal="" style={S.overlay}>
      <div style={S.card}>
        <div style={S.title}>Choose a local model (optional)</div>
        <p style={S.body}>
          Local models run entirely on your device with no internet needed.
          Select any you want to download now — 2.4 to 5 GB each.
          You can skip this and use an API key instead — add yours after setup
          via the <strong style={{ color: '#ccc' }}>⚙ Settings</strong> button in the bottom-left corner.
          Local models can also be added from Settings at any time.
        </p>

        {USER_MODELS.map(m => (
          <div key={m.id} style={S.checkRow(selected.includes(m.id))} onClick={() => toggle(m.id)}>
            <input
              type="checkbox" checked={selected.includes(m.id)}
              onChange={() => toggle(m.id)} onClick={e => e.stopPropagation()}
              style={{ marginRight: '12px', marginTop: '2px', accentColor: '#7c5cbf' }}
            />
            <div>
              <div style={S.name}>{m.name} <span style={{ color: '#555', fontWeight: 'normal' }}>({m.size})</span></div>
              <div style={S.desc}>{m.desc}</div>
            </div>
          </div>
        ))}

        <div style={{ margin: '16px 0', padding: '12px 14px', background: '#1a1a2e', borderRadius: '8px', border: '1px solid #2a2a4a' }}>
          <div style={{ color: '#aaa', fontSize: '13px', marginBottom: '4px' }}>
            <strong style={{ color: '#ccc' }}>Prefer Ollama?</strong> Ollama is free, open-source, and gives you access to hundreds of models.
            Install it from <strong style={{ color: '#7c5cbf' }}>ollama.com</strong> before or after setup — once running, configure it in{' '}
            <strong style={{ color: '#ccc' }}>⚙ Settings → API Keys</strong>.
          </div>
        </div>

        <div>
          <button style={S.btn} onClick={() => startLLMs(selected)}>
            {selected.length > 0 ? `Download ${selected.length} model${selected.length > 1 ? 's' : ''}` : 'Skip — use API key instead'}
          </button>
          {selected.length > 0 && (
            <button style={S.btnGhost} onClick={() => startLLMs([])}>Skip</button>
          )}
        </div>
      </div>
    </div>
  );

  if (screen === 'zb_offer') {
    const zb = USER_MODELS.find(m => m.id === ZB_MODEL_ID);
    return (
      <div data-modal="" style={S.overlay}>
        <div style={S.card}>
          <div style={S.title}>The Zynkbot model</div>
          <p style={S.body}>
            {zb.name}, {zb.size}. It answers on this phone with no key and nothing sent anywhere;
            a first question in a thread takes 10 to 20 seconds, later ones under 10.
            Download on Wi-Fi and keep the app open. If the download is interrupted it continues
            where it stopped next time, and the file is checked before it is used.
          </p>
          <p style={{ ...S.body, marginBottom: 0 }}>
            Choose Later to use a cloud model instead; the download stays on the Current Model page.
          </p>
          <div>
            <button style={S.btn} onClick={() => startLLMs([ZB_MODEL_ID])}>Download ({zb.size})</button>
            <button style={S.btnGhost} onClick={() => setScreen('complete')}>Later</button>
          </div>
        </div>
      </div>
    );
  }
  if (screen === 'downloading_llm') {
    const currentIdx = llmQueue.indexOf(currentLLM);
    return (
      <div data-modal="" style={S.overlay}>
        <div style={S.card}>
          <div style={S.title}>Downloading local models</div>
          <p style={S.body}>This may take a while. Please keep the app open.</p>

          {error && <ErrorBanner msg={error} />}

          {llmQueue.map((id, i) => {
            const m = USER_MODELS.find(x => x.id === id);
            const isDone = i < currentIdx;
            const isCurrent = id === currentLLM;
            const pct = isDone ? 100 : isCurrent ? llmProgress : 0;
            return (
              <div key={id} style={{ marginBottom: '18px' }}>
                <div style={S.row}>
                  <div>
                    <div style={S.name}>{m.name}</div>
                    {isCurrent && llmVerifying && <div style={S.desc}>Checking the file…</div>}
                    {isCurrent && !llmVerifying && llmBytes.total > 0 && (
                      <div style={S.desc}>{formatBytes(llmBytes.downloaded)} / {formatBytes(llmBytes.total)}</div>
                    )}
                  </div>
                  <div style={{ ...S.size, color: isDone ? '#4caf50' : '#888' }}>
                    {isDone ? '✓ Done' : isCurrent ? `${pct}%` : 'Waiting'}
                  </div>
                </div>
                <div style={S.bar}><div style={fill(pct)} /></div>
              </div>
            );
          })}

          {!error && (
            <button style={{ ...S.btnGhost, marginTop: '16px' }} onClick={() => setScreen('complete')}>
              Skip remaining downloads
            </button>
          )}
        </div>
      </div>
    );
  }

  if (screen === 'complete') return (
    <div data-modal="" style={S.overlay}>
      <div style={S.card}>
        <div style={S.title}>Setup complete</div>
        <p style={S.body}>
          Zynkbot is ready to use. All required models are downloaded and configured.
          You can download additional local models at any time from Settings.
        </p>
        <div><button style={S.btn} onClick={onComplete}>Launch Zynkbot</button></div>
      </div>
    </div>
  );

  return null;
}
