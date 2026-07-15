import React, { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from '@tauri-apps/api/core';
import ForceGraph2D from 'react-force-graph-2d';
import "../styles/MemoryGraphModal.css";

// Relationship type colors - MOVED OUTSIDE component to prevent infinite loop
const relationColors = {
  'supports': '#50fa7b',       // Green
  'supported_by': '#50fa7b',
  'contradicts': '#ff5555',    // Red
  'contradicted_by': '#ff5555',
  'elaborates': '#8be9fd',     // Cyan
  'elaborated_by': '#8be9fd',
  'reminds_of': '#f1fa8c',     // Yellow
  'caused_by': '#bd93f9',      // Purple
  'causes': '#bd93f9',
  'quotes': '#ffb86c',         // Orange
  'quoted_by': '#ffb86c',
  'resolves': '#ff79c6',       // Pink
  'resolved_by': '#ff79c6',
};

export default function MemoryGraphModal({ isOpen, onClose, memoryId, userId, onNavigate }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hasInitiallyZoomed, setHasInitiallyZoomed] = useState(false);
  const graphRef = useRef();

  // Fetch memory graph data - FULL GRAPH
  const fetchGraphData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Get the FULL memory graph (all memories + all relationships)
      const graphResult = await invoke('get_full_memory_graph', {
        userId: userId,
        namespace: null
      });
      console.log('Full graph data:', graphResult);

      const allMemories = graphResult.memories || [];
      const allLinks = graphResult.links || [];

      // Build nodes from ALL memories (smaller nodes)
      const nodes = allMemories.map(memory => ({
        id: memory.id,
        name: memory.title || `Memory ${memory.id}`,
        content: memory.content,
        namespace: memory.namespace,
        isCenter: memory.id === memoryId,  // Highlight the selected memory
        val: memory.id === memoryId ? 8 : 4  // Much smaller nodes to reduce overlap
      }));

      // Build links from ALL relationships
      const links = allLinks.map(link => ({
        id: `${link.source_memory_id}-${link.target_memory_id}-${link.relation_type}`,
        source: link.source_memory_id,
        target: link.target_memory_id,
        label: link.relation_type,
        confidence: link.confidence,
        color: relationColors[link.relation_type] || '#6272a4'
      }));

      setGraphData({ nodes, links });
    } catch (error) {
      console.error('Failed to fetch full graph data:', error);
      alert(`Error loading full graph: ${error}`);
    } finally {
      setIsLoading(false);
    }
  }, [memoryId, userId]);

  useEffect(() => {
    if (isOpen) {
      fetchGraphData();
      setHasInitiallyZoomed(false);  // Reset zoom flag when modal opens
    }
  }, [isOpen, fetchGraphData]);

  // Center graph on selected memory after force simulation stabilizes
  const handleEngineStop = useCallback(() => {
    if (!hasInitiallyZoomed && graphRef.current && graphData.nodes.length > 0) {
      const centerNode = graphData.nodes.find(n => n.isCenter);
      if (centerNode && centerNode.x !== undefined && centerNode.y !== undefined) {
        // Center on the selected memory node
        graphRef.current.centerAt(centerNode.x, centerNode.y, 1000);
        graphRef.current.zoom(6, 1000);
        setHasInitiallyZoomed(true);
      }
    }
  }, [hasInitiallyZoomed, graphData.nodes]);

  // Handle node click
  const handleNodeClick = useCallback((node) => {
    console.log('[MemoryGraph] Node clicked:', node);
    console.log('[MemoryGraph] Node ID:', node?.id);
    setSelectedNode(node);

    // Center camera on clicked node
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
    }
  }, []);

  // Handle node hover
  const handleNodeHover = useCallback((node) => {
    document.body.style.cursor = node ? 'pointer' : 'default';
  }, []);

  // Custom node rendering
  const paintNode = useCallback((node, ctx, globalScale) => {
    const label = node.name;
    const fontSize = node.isCenter ? 14 / globalScale : 12 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;

    // Draw node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
    ctx.fillStyle = node.isCenter ? '#50fa7b' : '#8be9fd';
    ctx.fill();

    // Add border
    ctx.strokeStyle = node === selectedNode ? '#f1fa8c' : '#44475a';
    ctx.lineWidth = node === selectedNode ? 2 / globalScale : 1 / globalScale;
    ctx.stroke();

    // Draw label below node
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Add background for better readability
    const textWidth = ctx.measureText(label).width;
    const padding = 4 / globalScale;
    const labelY = node.y + node.val + 5 / globalScale;

    ctx.fillStyle = 'rgba(40, 42, 54, 0.95)';
    ctx.fillRect(
      node.x - textWidth / 2 - padding,
      labelY - padding,
      textWidth + padding * 2,
      fontSize + padding * 2
    );

    // Draw label text
    ctx.fillStyle = node.isCenter ? '#50fa7b' : '#8be9fd';
    ctx.fillText(label, node.x, labelY);
  }, [selectedNode]);

  // Custom link rendering
  const paintLink = useCallback((link, ctx, globalScale) => {
    const start = link.source;
    const end = link.target;

    // Don't draw if nodes don't have positions yet
    if (!start || !end || typeof start !== 'object' || typeof end !== 'object') return;

    // Draw line
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = link.color;
    ctx.lineWidth = 2 / globalScale;
    ctx.stroke();

    // Draw arrow
    const arrowLength = 10 / globalScale;
    const arrowWidth = 6 / globalScale;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const angle = Math.atan2(dy, dx);

    // Arrow at target node
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

    // Draw label at midpoint
    const midX = (start.x + end.x) / 2;
    const midY = (start.y + end.y) / 2;
    const fontSize = 10 / globalScale;
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Background for label
    const text = link.label;
    const textWidth = ctx.measureText(text).width;
    const padding = 4 / globalScale;
    ctx.fillStyle = 'rgba(40, 42, 54, 0.9)';
    ctx.fillRect(
      midX - textWidth / 2 - padding,
      midY - fontSize / 2 - padding,
      textWidth + padding * 2,
      fontSize + padding * 2
    );

    // Label text
    ctx.fillStyle = link.color;
    ctx.fillText(text, midX, midY);
  }, []);

  if (!isOpen) return null;

  return (
    <div className="graph-modal-overlay" onClick={onClose}>
      <div className="memory-graph-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="graph-modal-header">
          <h2>Memory Relationship Graph</h2>
          <button onClick={onClose} className="close-button">✕</button>
        </div>

        {/* Controls */}
        <div className="graph-controls">
          <button
            onClick={() => {
              const targetNode = selectedNode || graphData.nodes.find(n => n.isCenter);
              if (targetNode && graphRef.current) {
                graphRef.current.centerAt(targetNode.x, targetNode.y, 1000);
                graphRef.current.zoom(6, 1000);
              }
            }}
            className="control-button"
          >
            📐 Zoom to Selected
          </button>
          <button
            onClick={() => {
              if (graphRef.current) graphRef.current.centerAt(0, 0, 1000);
            }}
            className="control-button"
          >
            🎯 Center
          </button>
          <button onClick={fetchGraphData} className="control-button">
            🔄 Refresh
          </button>
          <div className="graph-legend">
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#50fa7b' }}></div>
              <span>Supports</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#ff5555' }}></div>
              <span>Contradicts</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#8be9fd' }}></div>
              <span>Elaborates</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#f1fa8c' }}></div>
              <span>Reminds Of</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#bd93f9' }}></div>
              <span>Causes</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ background: '#ff79c6' }}></div>
              <span>Resolves</span>
            </div>
          </div>
        </div>

        {/* Graph */}
        <div className="graph-container">
          {isLoading ? (
            <div className="graph-loading">Loading graph...</div>
          ) : graphData.nodes.length === 0 ? (
            <div className="graph-empty">No relationships to display</div>
          ) : (
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              nodeLabel={node => `${node.name}\n${node.content.substring(0, 100)}...`}
              nodeCanvasObject={paintNode}
              linkCanvasObject={paintLink}
              linkDirectionalParticles={2}
              linkDirectionalParticleWidth={2}
              onNodeClick={handleNodeClick}
              onNodeHover={handleNodeHover}
              onEngineStop={handleEngineStop}
              cooldownTicks={100}
              backgroundColor="#1e1f29"
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
              d3Force={{
                center: { strength: 0.3 },     // Stronger center force (0.05 → 0.3) keeps weakly-connected components together
                charge: { strength: -120 },    // Reduce repulsion (-300 default → -120) allows tighter clustering
                link: {
                  strength: link => 0.7,       // Strong link force (default 1/Math.min(degree)) keeps all connections visible
                  distance: 50                 // Minimum link distance for readability
                },
                radial: {
                  strength: 0.1,               // Reduced (was 0.15) since center force is now stronger
                  radius: 350,                 // Keep nodes within ~350px of center
                  x: 0,                        // Center X coordinate
                  y: 0                         // Center Y coordinate
                }
              }}
              enableZoomInteraction={true}
            />
          )}
        </div>

        {/* Selected Node Details */}
        {selectedNode && (
          <div className="node-details-panel">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '8px' }}>
              <h3 style={{ margin: 0, flex: 1 }}>{selectedNode.name}</h3>
              <button
                onClick={() => setSelectedNode(null)}
                style={{
                  background: 'none', border: 'none', color: '#9aa5c4',
                  fontSize: '1.2rem', cursor: 'pointer', padding: '0 0 0 8px', lineHeight: 1,
                }}
              >✕</button>
            </div>
            <div className="node-detail-row">
              <strong>ID:</strong> {selectedNode.id}
            </div>
            <div className="node-detail-row">
              <strong>Namespace:</strong> {selectedNode.namespace}
            </div>
            <div className="node-content">
              <strong>Content:</strong>
              <p>{selectedNode.content}</p>
            </div>

            {/* Related Memories List */}
            <div className="related-memories-section">
              <strong>Related Memories:</strong>
              <div className="related-memories-list">
                {(() => {
                  // Filter links connected to selected node
                  const connectedLinks = graphData.links.filter(link => {
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    return sourceId === selectedNode.id || targetId === selectedNode.id;
                  });

                  // Deduplicate: Only show each connected memory once
                  // If bidirectional relationships exist (A->B and B->A), only show one
                  const seenMemories = new Set();
                  const uniqueLinks = [];

                  for (const link of connectedLinks) {
                    const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                    const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                    const connectedNodeId = sourceId === selectedNode.id ? targetId : sourceId;

                    // Skip if we've already shown this memory
                    if (seenMemories.has(connectedNodeId)) {
                      continue;
                    }

                    seenMemories.add(connectedNodeId);
                    uniqueLinks.push({ link, connectedNodeId });
                  }

                  return uniqueLinks.map(({ link, connectedNodeId }) => {
                    const connectedNode = graphData.nodes.find(n => n.id === connectedNodeId);
                    if (!connectedNode) return null;

                    return (
                      <div
                        key={connectedNodeId}
                        className="related-memory-item"
                        onClick={() => {
                          // Zoom to the connected memory
                          if (graphRef.current && connectedNode) {
                            setSelectedNode(connectedNode);
                            graphRef.current.centerAt(connectedNode.x, connectedNode.y, 1000);
                            graphRef.current.zoom(6, 1000);
                          }
                        }}
                      >
                        <span className="relation-type" style={{ color: link.color }}>
                          {link.label}
                        </span>
                        <span className="related-memory-name">{connectedNode.name}</span>
                      </div>
                    );
                  });
                })()}
              </div>
            </div>

            <button
              onClick={() => {
                // Navigate to this memory in the main modal
                console.log('[MemoryGraph] View Full Memory clicked, selectedNode:', selectedNode);
                console.log('[MemoryGraph] selectedNode.id:', selectedNode?.id);

                if (!selectedNode || !selectedNode.id) {
                  console.error('[MemoryGraph] Invalid selectedNode or missing ID');
                  alert('Error: Cannot navigate to memory - invalid node selected');
                  return;
                }

                if (onNavigate) {
                  onNavigate(selectedNode.id);
                }
                onClose();
              }}
              className="view-memory-button"
            >
              View Full Memory
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
