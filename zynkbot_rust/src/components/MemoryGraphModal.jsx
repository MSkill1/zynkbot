import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { invoke } from '@tauri-apps/api/core';
import ForceGraph2D from 'react-force-graph-2d';
import "../styles/MemoryGraphModal.css";

// The memory map (2026-11-03, Matt): a hub for each person, place or thing that two or
// more memories mention, memories hanging off their hubs and tinted by namespace, and
// the real relationships (updates, contradicts, related) drawn on top. A life clusters
// around its people and places; the lines between facts alone were a scatter of dots.

// Relationship line colours (module-level so the callbacks keep their identity)
const relationColors = {
  'related': '#bd93f9',        // Lavender
  'contradicts': '#ff5555',    // Red
  'updates': '#50fa7b',        // Green
  'resolves': '#ff79c6',       // Pink
};

// Hub rings by kind
const hubColors = { person: '#ffb86c', place: '#8be9fd', org: '#ff79c6', thing: '#f1fa8c' };
const hubWords = { person: 'Person', place: 'Place', org: 'Organisation', thing: 'Thing' };

// Memory dots by namespace: named ones keep a colour, the rest take the next free one
const namespaceColors = {
  personal: '#a78bfa', hobbies: '#34d399', politics: '#f87171', kitchen: '#fbbf24',
  health: '#60a5fa', technology: '#22d3ee', family: '#f472b6', work: '#fb923c', _zynkbot: '#6272a4',
};
const sparePalette = ['#c084fc', '#4ade80', '#2dd4bf', '#e879f9', '#a3e635', '#38bdf8', '#fca5a5'];

const HUB_MIN_MEMORIES = 3;      // a thing mentioned once or twice is not a hub
const HUB_MAX = 80;              // the biggest hubs; a phone cannot read more
const LEGEND_NAMESPACES = 6;     // the biggest namespaces named in the legend
const MEMBER_DISTANCE = 26;      // a memory sits this close to its hub

// A pull along one axis toward a point that depends on the node (d3's forceX/forceY,
// written out so the map does not import d3 directly)
function pullTo(axis, target, strength) {
  let nodes = [];
  const force = (alpha) => {
    const v = axis === 'x' ? 'vx' : 'vy';
    for (const n of nodes) {
      const t = target(n);
      if (t == null) continue;
      n[v] += (t - n[axis]) * strength(n) * alpha;
    }
  };
  force.initialize = (ns) => { nodes = ns; };
  return force;
}

// Hubs keep their distance from each other (rings plus room for a name); 80 hubs are
// a few thousand pairs a tick, nothing
function spreadHubs(gap) {
  let hubs = [];
  const force = (alpha) => {
    for (let i = 0; i < hubs.length; i++) {
      for (let j = i + 1; j < hubs.length; j++) {
        const a = hubs[i], b = hubs[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const want = a.val + b.val + gap;
        if (d >= want) continue;
        const push = ((want - d) / d) * 0.5 * alpha;
        a.vx -= dx * push; a.vy -= dy * push;
        b.vx += dx * push; b.vy += dy * push;
      }
    }
  };
  force.initialize = (ns) => { hubs = ns.filter(n => n.hub); };
  return force;
}

// Where the map is, ignoring a stray node: the middle of the nodes and the distance that
// takes in nineteen of every twenty of them
function mapExtent(nodes) {
  const pts = nodes.filter(n => Number.isFinite(n.x) && Number.isFinite(n.y));
  if (pts.length === 0) return null;
  const xs = pts.map(n => n.x).sort((a, b) => a - b);
  const ys = pts.map(n => n.y).sort((a, b) => a - b);
  const cx = xs[Math.floor(xs.length / 2)];
  const cy = ys[Math.floor(ys.length / 2)];
  const dists = pts.map(n => Math.hypot(n.x - cx, n.y - cy)).sort((a, b) => a - b);
  const r = dists[Math.min(dists.length - 1, Math.floor(dists.length * 0.95))];
  return { cx, cy, r: Math.max(r, 40) };
}

export default function MemoryGraphModal({ isOpen, onClose, memoryId, userId, onNavigate }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [], namespaces: [], hubCount: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hasInitiallyZoomed, setHasInitiallyZoomed] = useState(false);
  const graphRef = useRef();
  const containerRef = useRef();
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!isOpen) return;
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [isOpen, isLoading]);

  // Show the whole map: centre on the middle of it and zoom so the bulk fits
  const fitView = useCallback((ms = 800) => {
    const fg = graphRef.current;
    const ext = mapExtent(graphData.nodes);
    if (!fg || !ext || !size.width || !size.height) return;
    fg.centerAt(ext.cx, ext.cy, ms);
    fg.zoom(Math.max(0.05, Math.min(size.width, size.height) / (2.3 * ext.r)), ms);
  }, [graphData.nodes, size]);

  const fetchGraphData = useCallback(async () => {
    setIsLoading(true);
    try {
      const graphResult = await invoke('get_full_memory_graph', { userId: userId, namespace: null });
      const allMemories = graphResult.memories || [];
      const allLinks = graphResult.links || [];
      const allEntities = graphResult.entities || [];

      // Namespaces by size, each with a colour and a home angle on the map
      const nsCount = new Map();
      for (const m of allMemories) nsCount.set(m.namespace || 'personal', (nsCount.get(m.namespace || 'personal') || 0) + 1);
      const nsOrder = [...nsCount.entries()].sort((a, b) => b[1] - a[1]).map(([ns]) => ns);
      let spare = 0;
      const namespaces = nsOrder.map((ns, i) => ({
        name: ns,
        count: nsCount.get(ns),
        color: namespaceColors[ns] || sparePalette[spare++ % sparePalette.length],
        angle: (2 * Math.PI * i) / Math.max(1, nsOrder.length),
      }));
      const nsInfo = new Map(namespaces.map(n => [n.name, n]));
      const radius = 120 + 14 * Math.sqrt(allMemories.length);

      // Hubs: a person, place or thing with two or more memories
      const present = new Set(allMemories.map(m => m.id));
      const hubs = allEntities
        .map(e => ({ ...e, memory_ids: (e.memory_ids || []).filter(id => present.has(id)) }))
        .filter(e => e.memory_ids.length >= HUB_MIN_MEMORIES)
        .sort((a, b) => b.memory_ids.length - a.memory_ids.length)
        .slice(0, HUB_MAX);
      const hubsOf = new Map(); // memory id -> [hub node id]
      const hubNodes = hubs.map((e, rank) => {
        const id = `hub:${e.key}`;
        for (const mid of e.memory_ids) {
          if (!hubsOf.has(mid)) hubsOf.set(mid, []);
          hubsOf.get(mid).push(id);
        }
        return {
          id, hub: true, name: e.name, kind: e.kind in hubColors ? e.kind : 'thing', rank,
          count: e.memory_ids.length, memberIds: e.memory_ids,
          val: Math.min(16, 5 + 2 * Math.sqrt(e.memory_ids.length)),
        };
      });

      // Facts from one message share original_text (one memory per fact since
      // 2026-10-24); they keep a short tie to each other so a message stays together.
      const normalise = (t) => t.split(/\s+/).join(' ').toLowerCase();
      const groupOf = new Map();
      let nextGroup = 0;
      const nodes = allMemories.map(memory => {
        const said = (memory.original_text || '').trim();
        let group = null;
        if (said) {
          const key = normalise(said);
          if (!groupOf.has(key)) groupOf.set(key, { n: nextGroup++, ids: [] });
          group = groupOf.get(key);
          group.ids.push(memory.id);
        }
        const ns = memory.namespace || 'personal';
        return {
          id: memory.id,
          hub: false,
          name: memory.title || `Memory ${memory.id}`,
          content: memory.content,
          namespace: ns,
          color: nsInfo.get(ns)?.color || '#8be9fd',
          said,
          group: group ? group.n : null,
          hubIds: hubsOf.get(memory.id) || [],
          replaced: memory.temporal_status === 'superseded',
          isCenter: memory.id === memoryId,
          val: memory.id === memoryId ? 6 : 3,
        };
      });

      const links = allLinks
        .filter(link => present.has(link.source_memory_id) && present.has(link.target_memory_id))
        .map(link => ({
          id: `${link.source_memory_id}-${link.target_memory_id}-${link.relation_type}`,
          source: link.source_memory_id,
          target: link.target_memory_id,
          label: link.relation_type,
          confidence: link.confidence,
          color: relationColors[link.relation_type] || '#6272a4',
        }));
      for (const { ids } of groupOf.values()) {
        for (let i = 1; i < ids.length; i++) {
          links.push({ id: `same-${ids[i - 1]}-${ids[i]}`, source: ids[i - 1], target: ids[i], label: 'same message', sibling: true, color: 'rgba(98, 114, 164, 0.55)' });
        }
      }
      for (const h of hubNodes) {
        for (const mid of h.memberIds) {
          links.push({ id: `member-${mid}-${h.id}`, source: mid, target: h.id, member: true, color: 'rgba(248, 248, 242, 0.16)' });
        }
      }

      setGraphData({ nodes: [...hubNodes, ...nodes], links, namespaces, hubCount: hubNodes.length, radius });
    } catch (error) {
      console.error('Failed to fetch the memory map:', error);
      alert(`Error loading the memory map: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [memoryId, userId]);

  useEffect(() => {
    if (isOpen) {
      fetchGraphData();
      setHasInitiallyZoomed(false);
      setSelectedNode(null);
    }
  }, [isOpen, fetchGraphData]);

  // Layout: hubs push harder than memories, members sit close to their hub, and every
  // memory is drawn gently toward its namespace's home on a ring, so health, kitchen and
  // politics form regions even where nothing links them.
  useEffect(() => {
    const fg = graphRef.current;
    if (!fg || graphData.nodes.length === 0) return;
    const home = new Map(graphData.namespaces.map(n => [n.name, { x: Math.cos(n.angle) * graphData.radius, y: Math.sin(n.angle) * graphData.radius }]));
    fg.d3Force('charge').strength(n => n.hub ? -Math.min(400, 120 + 12 * n.count) : -28);
    fg.d3Force('link')
      .distance(l => l.member ? MEMBER_DISTANCE + 2 * Math.sqrt(l.target.count || 1) : l.sibling ? 12 : 48)
      .strength(l => l.member ? 0.5 : l.sibling ? 1 : 0.4);
    const target = (axis) => (n) => n.hub ? 0 : (home.get(n.namespace) || { x: 0, y: 0 })[axis];
    const strength = (n) => n.hub ? 0.01 : (n.hubIds.length ? 0.02 : 0.06);
    fg.d3Force('homeX', pullTo('x', target('x'), strength));
    fg.d3Force('homeY', pullTo('y', target('y'), strength));
    fg.d3Force('spread', spreadHubs(30));
    fg.d3ReheatSimulation();
  }, [graphData]);

  const handleEngineStop = useCallback(() => {
    const fg = graphRef.current;
    if (hasInitiallyZoomed || !fg || graphData.nodes.length === 0) return;
    const centerNode = graphData.nodes.find(n => n.isCenter);
    if (centerNode && Number.isFinite(centerNode.x)) {
      fg.centerAt(centerNode.x, centerNode.y, 800);
      fg.zoom(4, 800);
    } else {
      fitView(800);
    }
    setHasInitiallyZoomed(true);
  }, [hasInitiallyZoomed, graphData.nodes, fitView]);

  const selectAndGo = useCallback((node, zoomLevel) => {
    setSelectedNode(node);
    const fg = graphRef.current;
    if (fg && node && node.x !== undefined) {
      fg.centerAt(node.x, node.y, 800);
      if (zoomLevel) fg.zoom(zoomLevel, 800);
    }
  }, []);

  const handleNodeClick = useCallback((node) => selectAndGo(node, null), [selectAndGo]);
  const handleNodeHover = useCallback((node) => { document.body.style.cursor = node ? 'pointer' : 'default'; }, []);

  const nodeById = useMemo(() => new Map(graphData.nodes.map(n => [n.id, n])), [graphData.nodes]);

  // Names drawn so far in this frame, in map units, so a name is left out rather than
  // printed over another; hubs are painted biggest first, so the big ones win
  const labelBoxes = useRef([]);
  const handleFramePre = useCallback(() => { labelBoxes.current = []; }, []);
  const claimLabel = (box) => {
    if (labelBoxes.current.some(b => box.x1 < b.x2 && box.x2 > b.x1 && box.y1 < b.y2 && box.y2 > b.y1)) return false;
    labelBoxes.current.push(box);
    return true;
  };

  const paintNode = useCallback((node, ctx, globalScale) => {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return;
    const selected = node === selectedNode;
    if (node.hub) {
      // A hub is a ring in its kind's colour, with its name
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
      ctx.fillStyle = 'rgba(30, 31, 41, 0.9)';
      ctx.fill();
      ctx.strokeStyle = selected ? '#f8f8f2' : hubColors[node.kind];
      ctx.lineWidth = (selected ? 3 : 2) / globalScale;
      ctx.stroke();
      // The biggest hubs are named at any zoom; the rest as you come closer
      const named = selected || node.rank < 12 || (globalScale > 0.7 && node.rank < 40) || globalScale > 1.3;
      if (!named) return;
      const fontSize = Math.min(13, 10 + node.count / 4) / globalScale;
      ctx.font = `bold ${fontSize}px Sans-Serif`;
      const w = ctx.measureText(node.name).width;
      const ly = node.y + node.val + fontSize * 0.8;
      const pad = 2 / globalScale;
      if (!selected && !claimLabel({ x1: node.x - w / 2 - pad, x2: node.x + w / 2 + pad, y1: ly - fontSize / 2 - pad, y2: ly + fontSize / 2 + pad })) return;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = selected ? '#f8f8f2' : hubColors[node.kind];
      ctx.fillText(node.name, node.x, ly);
      return;
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
    ctx.fillStyle = node.isCenter ? '#50fa7b' : node.replaced ? '#44475a' : node.color;
    ctx.fill();
    ctx.strokeStyle = selected ? '#f8f8f2' : node.replaced ? '#6272a4' : 'rgba(30, 31, 41, 0.9)';
    ctx.lineWidth = (selected ? 2 : 1) / globalScale;
    ctx.stroke();

    // Hundreds of titles at once are unreadable: show them zoomed in, or for the
    // selected and centre memories
    if (globalScale < 1.6 && !selected && !node.isCenter) return;
    const label = node.name;
    const fontSize = (node.isCenter ? 13 : 11) / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const textWidth = ctx.measureText(label).width;
    const padding = 3 / globalScale;
    const labelY = node.y + node.val + 4 / globalScale;
    ctx.fillStyle = 'rgba(40, 42, 54, 0.92)';
    ctx.fillRect(node.x - textWidth / 2 - padding, labelY - padding, textWidth + padding * 2, fontSize + padding * 2);
    ctx.fillStyle = node.isCenter ? '#50fa7b' : '#f8f8f2';
    ctx.fillText(label, node.x, labelY);
  }, [selectedNode]);

  // A finger-sized touch area on every node, whatever the zoom
  const paintPointerArea = useCallback((node, color, ctx, globalScale) => {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) return;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.val + (node.hub ? 16 : 10) / globalScale, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
  }, []);

  const paintLink = useCallback((link, ctx, globalScale) => {
    const start = link.source;
    const end = link.target;
    if (!start || !end || typeof start !== 'object' || typeof end !== 'object') return;
    if (![start.x, start.y, end.x, end.y].every(Number.isFinite)) return;

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = link.color;
    ctx.lineWidth = (link.member ? 0.8 : link.sibling ? 1 : 2) / globalScale;
    ctx.stroke();
    if (link.member || link.sibling) return; // a tie, not a relationship: no arrow, no word

    const arrowLength = 8 / globalScale;
    const arrowWidth = 5 / globalScale;
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const arrowX = end.x - end.val * Math.cos(angle);
    const arrowY = end.y - end.val * Math.sin(angle);
    ctx.save();
    ctx.translate(arrowX, arrowY);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-arrowLength, arrowWidth);
    ctx.lineTo(-arrowLength, -arrowWidth);
    ctx.closePath();
    ctx.fillStyle = link.color;
    ctx.fill();
    ctx.restore();

    if (globalScale < 1.2) return; // the word on the line, once close enough to read it
    const midX = (start.x + end.x) / 2;
    const midY = (start.y + end.y) / 2;
    const fontSize = 9 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const text = link.label;
    const textWidth = ctx.measureText(text).width;
    const padding = 3 / globalScale;
    ctx.fillStyle = 'rgba(40, 42, 54, 0.9)';
    ctx.fillRect(midX - textWidth / 2 - padding, midY - fontSize / 2 - padding, textWidth + padding * 2, fontSize + padding * 2);
    ctx.fillStyle = link.color;
    ctx.fillText(text, midX, midY);
  }, []);

  if (!isOpen) return null;

  const memoryNodes = graphData.nodes.filter(n => !n.hub);
  const hubsPresent = new Set(graphData.nodes.filter(n => n.hub).map(n => n.kind));

  return (
    <div className="graph-modal-overlay" onClick={onClose}>
      <div className="memory-graph-modal" onClick={(e) => e.stopPropagation()}>
        <div className="graph-modal-header">
          <h2>Memory Map</h2>
          <button onClick={onClose} className="close-button">✕</button>
        </div>

        <div className="graph-controls">
          <button
            onClick={() => {
              const targetNode = selectedNode || graphData.nodes.find(n => n.isCenter);
              if (targetNode) selectAndGo(targetNode, 4);
            }}
            className="control-button"
          >
            📐 Zoom to Selected
          </button>
          <button onClick={() => fitView(800)} className="control-button">
            🎯 Whole map
          </button>
          <button onClick={fetchGraphData} className="control-button">
            🔄 Refresh
          </button>
        </div>

        <div className="graph-container" ref={containerRef}>
          {!isLoading && memoryNodes.length > 0 && (
          <div className="graph-legend">
            {graphData.namespaces.filter(n => n.name !== '_zynkbot').slice(0, LEGEND_NAMESPACES).map(n => (
              <div className="legend-item" key={n.name} title={`${n.count} memories`}>
                <div className="legend-dot" style={{ background: n.color }}></div>
                <span>{n.name}</span>
              </div>
            ))}
            {['person', 'place', 'org', 'thing'].filter(k => hubsPresent.has(k)).map(k => (
              <div className="legend-item" key={k}>
                <div className="legend-ring" style={{ borderColor: hubColors[k] }}></div>
                <span>{hubWords[k]}</span>
              </div>
            ))}
            <div className="legend-item"><div className="legend-line" style={{ background: relationColors.updates }}></div><span>Updates</span></div>
            <div className="legend-item"><div className="legend-line" style={{ background: relationColors.contradicts }}></div><span>Contradicts</span></div>
            <div className="legend-item"><div className="legend-line" style={{ background: relationColors.related }}></div><span>Related</span></div>
            <div className="legend-item"><div className="legend-dot" style={{ background: '#44475a', border: '1px solid #6272a4' }}></div><span>Replaced</span></div>
          </div>
          )}
          {isLoading ? (
            <div className="graph-loading">Drawing the map...</div>
          ) : memoryNodes.length === 0 ? (
            <div className="graph-empty">No memories to show yet</div>
          ) : (
            <ForceGraph2D
              ref={graphRef}
              width={size.width || undefined}
              height={size.height || undefined}
              graphData={graphData}
              nodeLabel={node => node.hub ? `${node.name} · ${node.count} memories` : `${node.name}\n${(node.content || '').substring(0, 100)}...`}
              nodeCanvasObject={paintNode}
              nodePointerAreaPaint={paintPointerArea}
              linkCanvasObject={paintLink}
              linkDirectionalParticles={link => (link.sibling || link.member) ? 0 : 2}
              linkDirectionalParticleWidth={2}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              onEngineStop={handleEngineStop}
              onRenderFramePre={handleFramePre}
              cooldownTicks={200}
              warmupTicks={40}
              backgroundColor="#1e1f29"
              d3AlphaDecay={0.025}
              d3VelocityDecay={0.35}
              enableZoomInteraction={true}
            />
          )}
        </div>

        {selectedNode && (
          <div className="node-details-panel">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '8px' }}>
              <h3 style={{ margin: 0, flex: 1 }}>{selectedNode.name}</h3>
              <button
                onClick={() => setSelectedNode(null)}
                style={{ background: 'none', border: 'none', color: '#9aa5c4', fontSize: '1.2rem', cursor: 'pointer', padding: '0 0 0 8px', lineHeight: 1 }}
              >✕</button>
            </div>

            {selectedNode.hub ? (
              <>
                <div className="node-detail-row">
                  <strong>{hubWords[selectedNode.kind]}:</strong> {selectedNode.count} memories mention this
                </div>
                <div className="related-memories-section">
                  <div className="related-memories-list">
                    {selectedNode.memberIds.map(id => nodeById.get(id)).filter(Boolean).map(n => (
                      <div key={n.id} className="related-memory-item" onClick={() => selectAndGo(n, 4)}>
                        <span className="legend-dot" style={{ background: n.color, flex: 'none' }}></span>
                        <span className="related-memory-name">{n.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="node-detail-row">
                  <strong>Namespace:</strong> {selectedNode.namespace}{selectedNode.replaced ? ' · replaced by a newer memory' : ''}
                </div>
                <div className="node-content">
                  <p>{selectedNode.content}</p>
                </div>
                {selectedNode.hubIds.length > 0 && (
                  <div className="node-detail-row hub-chips">
                    <strong>Mentions:</strong>
                    {selectedNode.hubIds.map(id => nodeById.get(id)).filter(Boolean).map(h => (
                      <button key={h.id} className="hub-chip" style={{ borderColor: hubColors[h.kind], color: hubColors[h.kind] }} onClick={() => selectAndGo(h, 2.5)}>
                        {h.name}
                      </button>
                    ))}
                  </div>
                )}
                {selectedNode.group !== null && (() => {
                  const siblings = memoryNodes.filter(n => n.group === selectedNode.group && n.id !== selectedNode.id);
                  if (siblings.length === 0) return null;
                  return (
                    <div className="related-memories-section">
                      <strong>From the same message:</strong>
                      <p style={{ color: '#9aa5c4', fontStyle: 'italic', fontSize: '0.85rem', margin: '4px 0 6px' }}>“{selectedNode.said.length > 160 ? selectedNode.said.substring(0, 160) + '…' : selectedNode.said}”</p>
                      <div className="related-memories-list">
                        {siblings.map(n => (
                          <div key={n.id} className="related-memory-item" onClick={() => selectAndGo(n, 4)}>
                            <span className="related-memory-name">{n.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
                {(() => {
                  const seen = new Set();
                  const rows = [];
                  for (const link of graphData.links) {
                    if (link.sibling || link.member) continue;
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    if (sourceId !== selectedNode.id && targetId !== selectedNode.id) continue;
                    const otherId = sourceId === selectedNode.id ? targetId : sourceId;
                    if (seen.has(otherId)) continue;
                    seen.add(otherId);
                    const other = nodeById.get(otherId);
                    if (other) rows.push({ link, other });
                  }
                  if (rows.length === 0) return null;
                  return (
                    <div className="related-memories-section">
                      <strong>Relationships:</strong>
                      <div className="related-memories-list">
                        {rows.map(({ link, other }) => (
                          <div key={other.id} className="related-memory-item" onClick={() => selectAndGo(other, 4)}>
                            <span className="relation-type" style={{ color: link.color }}>{link.label}</span>
                            <span className="related-memory-name">{other.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })()}
                <button
                  onClick={() => {
                    if (onNavigate) onNavigate(selectedNode.id);
                    onClose();
                  }}
                  className="view-memory-button"
                >
                  View Full Memory
                </button>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
