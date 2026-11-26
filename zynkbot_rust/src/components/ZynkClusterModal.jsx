import React from "react";
import { invoke } from '@tauri-apps/api/core';
import "../styles/AboutModal.css";
import ControlsPage from "./ControlsPage";

/**
 * The ZynkCluster window. Rewritten 2026-11-26 (Matt) to say what has been measured
 * rather than what was proposed: the text until then described a graphics-card cluster
 * in its design phase, while labs/moe_poc had moved to the RAM-first plan on
 * 2026-10-17 and measured its first step on two home machines on 29 Oct – 2 Nov 2026.
 * Numbers here come from labs/moe_poc/ZYNKCLUSTER_RAM_FIRST_PLAN.md, "Step 1 results".
 */
export default function ZynkClusterModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const openLabsFolder = async () => {
    try {
      await invoke('open_external_url', { url: 'https://github.com/MSkill1/zynkbot/tree/main/labs/moe_poc' });
    } catch (error) {
      console.error('Failed to open URL:', error);
    }
  };

  const box = (border, bg) => ({ background: bg, border: `1px solid ${border}`, borderRadius: '8px', padding: '15px', marginBottom: '20px' });
  const p = { lineHeight: '1.6', fontSize: '0.95rem', margin: '0 0 10px 0' };
  const ul = { margin: '5px 0 0 0', paddingLeft: '20px', fontSize: '0.95rem', lineHeight: '1.6' };

  return (
    <ControlsPage icon="🧠" title="ZynkCluster" subtitle="A model too big for one computer, run across your own" accent="#8be9fd" closeLabel="Close" zIndex={2000} onClose={onClose}>
        <div className="about-section">
          <h3>What it is</h3>
          <p style={p}>
            ZynkCluster runs an open model that is bigger than any one of your computers can hold,
            by spreading it over several ordinary PCs on your own network. Every large open model
            today is a <strong>mixture of experts</strong>: for each word it reads only a small
            slice of itself, so the work per word stays close to what one PC already does. What
            you need is somewhere to <em>keep</em> the rest of the model. Used office computers
            full of cheap memory are that somewhere. Graphics cards buy speed and can come later.
          </p>
          <p style={p}>
            Speed is deliberately second. "Ask the big one and wait" is the product: a question
            that deserves a frontier-class answer goes to the cluster, and the answer comes back
            in a minute instead of a second, from a model that stays in your house.
          </p>
        </div>

        <div className="about-section" style={box('#50fa7b', 'rgba(80, 250, 123, 0.1)')}>
          <h3 style={{ color: '#50fa7b', marginTop: 0 }}>Where it stands: measured, November 2026</h3>
          <ul style={ul}>
            <li>A 120-billion-parameter open model (63 GB) answered across a desktop with 62 GB of
              memory and a laptop with 16 GB, joined by one network cable, at about 5 words a second.
              Neither machine could hold it alone.</li>
            <li>A 20-billion-parameter model split the same way ran at 17 words a second.</li>
            <li>A long prompt asked a second time reused its cache and skipped the slow part entirely.</li>
            <li>On a cable the hop between machines costs under a hundredth of a second per word.
              On Wi-Fi it costs ten times that, so the cluster is a wired thing.</li>
            <li>The real bottleneck is reading the prompt, not producing the answer: the slowest
              machine's share of that work sets the pace while the others wait.</li>
          </ul>
          <p style={{ ...p, margin: '10px 0 0 0' }}>
            All of this ran on stock llama.cpp, splitting the model by layers. The finer split that
            ZynkCluster was first designed around, each word's experts read in parallel on different
            machines, is designed but not built, and may come after prompt caching, which the
            measurements suggest will be felt more.
          </p>
        </div>

        <div className="about-section">
          <h3>What comes next</h3>
          <ul style={ul}>
            <li><strong>Measure the three costs apart</strong> on a wired cluster: memory reading,
              network hops and arithmetic, so the speed of hardware you have not bought yet can be
              predicted to within a third.</li>
            <li><strong>Prompt caching and the expert split</strong>, in whichever order the
              measurements favour.</li>
            <li><strong>A "think hard" path in Zynkbot:</strong> the question and its memory context go
              to the cluster, the model on this device keeps answering as usual in the meantime, and
              the result arrives as a notification. The cluster is an optional back end, never a
              dependency. With it off, Zynkbot behaves exactly as it does today.</li>
          </ul>
        </div>

        <div className="about-section">
          <h3>What it needs</h3>
          <ul style={ul}>
            <li>Ordinary PCs with a lot of memory. Count about 50 GB usable on a 64 GB machine.</li>
            <li>A wired network between them.</li>
            <li>Today, llama.cpp's RPC mode, which is experimental and has no authentication: a trusted
              wired network only, never exposed. The plan is to put it behind ZynkSync's device
              pairing and encrypted links once it leaves the bench.</li>
            <li>No graphics card is required. One makes the main machine faster, nothing more.</li>
          </ul>
        </div>

        <div className="about-section" style={box('#ffb86c', 'rgba(255, 184, 108, 0.1)')}>
          <h3 style={{ color: '#ffb86c', marginTop: 0 }}>What this does not claim</h3>
          <p style={{ ...p, margin: 0 }}>
            Only the first step has been measured, on two machines. Every other speed figure in the
            plan is arithmetic from bandwidth and packet sizes, and such figures err on the side of
            optimism. Hardware prices in the plan are October 2026 observations in a rising memory market.
          </p>
        </div>

        <div className="about-links" style={{ marginTop: '25px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
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
            📂 The plan and the measurements, on GitHub
          </button>
        </div>
    </ControlsPage>
  );
}
