import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open as openFileDialog } from '@tauri-apps/plugin-dialog';
import { open as openUrl } from '@tauri-apps/plugin-shell';
import VoiceButton from './VoiceButton';

export default function EnsembleModal({
  isOpen,
  onClose,
  availableModels,
  userId,
  sessionId,
  containmentMode,
  onEnsembleComplete
}) {
  const [selectedModels, setSelectedModels] = useState([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [searchKBEnabled, setSearchKBEnabled] = useState(false);
  const [attachedFile, setAttachedFile] = useState(null);

  useEffect(() => {
    if (!isLoading) {
      setElapsedSeconds(0);
      return;
    }
    const interval = setInterval(() => setElapsedSeconds(s => s + 1), 1000);
    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isOpen) return null;

  // Debug logging
  console.log('[EnsembleModal] Render - result:', result ? 'EXISTS' : 'NULL', 'isLoading:', isLoading);

  // Child mode check
  const isChildMode = containmentMode === 'child';

  const IMAGE_EXTENSIONS = ['jpg','jpeg','png','gif','webp','bmp'];
  const MIME_TYPES = { jpg:'image/jpeg', jpeg:'image/jpeg', png:'image/png', gif:'image/gif', webp:'image/webp', bmp:'image/bmp' };

  const handleAttachFile = async () => {
    const path = await openFileDialog({
      multiple: false,
      filters: [
        { name: 'Files', extensions: [...IMAGE_EXTENSIONS, 'txt','md','rs','js','jsx','ts','tsx','py','json','toml','yaml','yml','sh','css','html','c','cpp','h','go','java','rb','php','swift','kt'] }
      ]
    });
    if (!path) return;
    try {
      const name = path.split('/').pop();
      const ext = name.split('.').pop().toLowerCase();
      if (IMAGE_EXTENSIONS.includes(ext)) {
        const base64 = await invoke('read_file_base64', { path });
        const mimeType = MIME_TYPES[ext] || 'image/jpeg';
        setAttachedFile({ name, base64, mimeType, size: base64.length, isImage: true });
      } else {
        const content = await invoke('read_text_file', { path });
        setAttachedFile({ name, content, size: content.length, isImage: false });
      }
    } catch (e) {
      alert(`Could not read file: ${e}`);
    }
  };

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const handleRunEnsemble = async () => {
    if (selectedModels.length < 2) {
      setError('Please select at least 2 models for ensemble');
      return;
    }

    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      let messageToSend = question;
      let userQuery = null;
      let imageData = null;
      if (attachedFile) {
        if (attachedFile.isImage) {
          imageData = { base64: attachedFile.base64, mimeType: attachedFile.mimeType };
          userQuery = question;
        } else {
          messageToSend = `[Attached file: ${attachedFile.name}]\n\`\`\`\n${attachedFile.content}\n\`\`\`\n\nUser question: ${question}`;
          userQuery = question;
        }
      }

      const data = await invoke('run_ensemble', {
        message: messageToSend,
        models: selectedModels,
        userId,
        sessionId,
        containmentMode,
        kbEnabled: searchKBEnabled,
        userQuery,
        imageData
      });

      console.log('[EnsembleModal] Received data from backend:', data);
      console.log('[EnsembleModal] Synthesized response:', data?.synthesized_response);
      setResult(data);
      console.log('[EnsembleModal] Result state set successfully');
    } catch (err) {
      console.error('Ensemble error:', err);
      setError(err.toString());
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    if (result && onEnsembleComplete) {
      onEnsembleComplete({ ...result, question });
    }
    setSelectedModels([]);
    setQuestion('');
    setResult(null);
    setError(null);
    setSearchKBEnabled(false);
    setAttachedFile(null);
    onClose();
  };

  const handleCancel = () => {
    // Close without adding to conversation
    setSelectedModels([]);
    setQuestion('');
    setResult(null);
    setError(null);
    setSearchKBEnabled(false);
    setAttachedFile(null);
    onClose();
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '20px'
      }}
    >
      <div
        style={{
          background: '#1e1f29',
          borderRadius: '12px',
          padding: '30px',
          maxWidth: '900px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '1px solid #44475a',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
          position: 'relative'
        }}
      >
        {/* X button in top right */}
        <button
          onClick={handleCancel}
          style={{
            position: 'absolute',
            top: '15px',
            right: '15px',
            background: 'rgba(255, 255, 255, 0.1)',
            border: 'none',
            color: '#f8f8f2',
            fontSize: '1.5rem',
            cursor: 'pointer',
            borderRadius: '6px',
            width: '35px',
            height: '35px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'background 0.2s'
          }}
          onMouseOver={(e) => e.target.style.background = 'rgba(255, 85, 85, 0.3)'}
          onMouseOut={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.1)'}
        >
          ✕
        </button>
        <h2 style={{ color: '#8be9fd', marginBottom: '10px' }}>
          🤝 Multi-Model Ensemble
        </h2>

        {isChildMode ? (
          <>
            <div style={{
              padding: '20px',
              background: '#ffb86c33',
              borderRadius: '8px',
              border: '2px solid #ffb86c',
              marginBottom: '20px',
              textAlign: 'center'
            }}>
              <p style={{ color: '#ffb86c', fontSize: '1.1rem', marginBottom: '10px', fontWeight: 'bold' }}>
                🔒 Ensemble Mode Disabled
              </p>
              <p style={{ color: '#f8f8f2', fontSize: '0.95rem', lineHeight: '1.6' }}>
                Ensemble mode is not available in Child safety mode. This feature requires multiple AI models and is designed for research and comparison tasks.
              </p>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <button
                onClick={onClose}
                style={{
                  padding: '10px 20px',
                  background: '#44475a',
                  color: '#f8f8f2',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Close
              </button>
            </div>
          </>
        ) : (
          <>
            <div style={{ color: '#9aa5c4', fontSize: '0.9rem', marginBottom: '20px' }}>
              <p style={{ marginBottom: '8px' }}>
                <strong style={{ color: '#8be9fd' }}>How Ensemble Works:</strong>
              </p>
              <ol style={{ paddingLeft: '20px', margin: 0, lineHeight: '1.6' }}>
                <li><strong>Phase 0:</strong> Coordinator assesses if the question needs current information (e.g., "latest version of React") and performs web search if needed</li>
                <li><strong>Phase 1:</strong> Each AI model answers independently with relevant memory context (plus web search results if retrieved)</li>
                <li><strong>Phase 2:</strong> Coordinator judges (not averages) the responses, identifies consensus & uncertainty, and synthesizes the best answer</li>
              </ol>
              <p style={{ marginTop: '8px', color: '#50fa7b', fontSize: '0.85rem' }}>
                ✨ Purpose: Research and fact-checking tool - responses are not stored as memories
              </p>
              <p style={{ marginTop: '4px', color: '#9aa5c4', fontSize: '0.8rem', fontStyle: 'italic' }}>
                Use ensemble for questions like "Compare X vs Y", "What's the latest version of Z?", or when you want multiple AI perspectives on a research question. For personal conversation with memory, use regular chat.
              </p>
            </div>

        {isLoading ? (
          <>
            <style>{`
              @keyframes ensembleSpin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
              }
              @keyframes ensemblePulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.35; }
              }
            `}</style>
            <div style={{ textAlign: 'center', padding: '30px 20px' }}>
              <div style={{
                fontSize: '2.5rem',
                display: 'inline-block',
                animation: 'ensembleSpin 1.4s linear infinite',
                marginBottom: '16px'
              }}>
                ⚙️
              </div>
              <h3 style={{ color: '#8be9fd', marginBottom: '6px' }}>Running Ensemble...</h3>
              <p style={{ color: '#6272a4', fontSize: '0.9rem', marginBottom: '20px' }}>
                Elapsed: {String(Math.floor(elapsedSeconds / 60)).padStart(2, '0')}:{String(elapsedSeconds % 60).padStart(2, '0')}
              </p>
              <div style={{
                background: '#282a36',
                borderRadius: '8px',
                padding: '16px',
                marginBottom: '16px',
                border: '1px solid #44475a',
                textAlign: 'left'
              }}>
                <p style={{ color: '#6272a4', fontSize: '0.8rem', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Querying models
                </p>
                {selectedModels.map((model, idx) => (
                  <div key={model} style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    padding: '8px 0',
                    borderBottom: idx < selectedModels.length - 1 ? '1px solid #44475a' : 'none',
                    animation: `ensemblePulse ${1.2 + idx * 0.3}s ease-in-out infinite`
                  }}>
                    <span style={{ fontSize: '1rem' }}>⟳</span>
                    <span style={{ color: '#f8f8f2', fontSize: '0.9rem', wordBreak: 'break-all' }}>
                      {model.includes('/') ? model.split('/').pop() : model}
                    </span>
                  </div>
                ))}
              </div>
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                gap: '6px',
                alignItems: 'center',
                fontSize: '0.8rem',
                color: '#6272a4'
              }}>
                <span style={{ color: elapsedSeconds < 15 ? '#50fa7b' : '#6272a4' }}>Phase 0</span>
                <span>→</span>
                <span style={{ color: elapsedSeconds >= 15 ? '#ffb86c' : '#44475a' }}>Phase 1</span>
                <span>→</span>
                <span style={{ color: '#44475a' }}>Phase 2</span>
              </div>
              <p style={{ color: '#44475a', fontSize: '0.75rem', marginTop: '12px', fontStyle: 'italic' }}>
                Local models may take several minutes
              </p>
            </div>
          </>
        ) : !result ? (
          <>
            {/* Model Selection */}
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#f8f8f2', fontSize: '1rem', marginBottom: '10px' }}>
                Select Models (minimum 2):
              </h3>
              {process.env.NODE_ENV === 'production' && availableModels.some(m => m.type === 'local') && (
                <div style={{
                  padding: '8px 12px',
                  marginBottom: '10px',
                  background: '#21222c',
                  border: '1px solid #44475a',
                  borderRadius: '6px',
                  color: '#6272a4',
                  fontSize: '0.82rem',
                  lineHeight: '1.4'
                }}>
                  🔒 <strong style={{ color: '#9aa5c4' }}>Local models require the CUDA build</strong> (coming soon).
                  To use local models in ensemble now, build the developer version from source.
                </div>
              )}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {availableModels.map(model => {
                  const isLocalInProd = process.env.NODE_ENV === 'production' && model.type === 'local';
                  return (
                    <label
                      key={model.id}
                      title={isLocalInProd ? 'Local models require the CUDA build (coming soon). Build the developer version now to use CUDA-optimized local models.' : undefined}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: '10px',
                        background: isLocalInProd ? '#1a1b26' : selectedModels.includes(model.id) ? '#44475a' : '#282a36',
                        border: `2px solid ${isLocalInProd ? '#44475a' : selectedModels.includes(model.id) ? '#8be9fd' : '#44475a'}`,
                        borderRadius: '6px',
                        cursor: isLocalInProd ? 'not-allowed' : 'pointer',
                        opacity: isLocalInProd ? 0.5 : 1,
                        transition: 'all 0.2s'
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(model.id)}
                        onChange={() => handleModelToggle(model.id)}
                        disabled={isLocalInProd}
                        style={{ marginRight: '10px', cursor: isLocalInProd ? 'not-allowed' : 'pointer' }}
                      />
                      <span style={{ color: isLocalInProd ? '#6272a4' : '#f8f8f2', fontWeight: selectedModels.includes(model.id) ? 'bold' : 'normal' }}>
                        {model.name}
                      </span>
                      <span style={{ marginLeft: 'auto', color: '#9aa5c4', fontSize: '0.85rem' }}>
                        {isLocalInProd ? '🔒 CUDA required' : model.type === 'local' ? '🔒 Local' : '☁️ API'}
                      </span>
                    </label>
                  );
                })}
              </div>
              {selectedModels.length === 1 && (
                <p style={{ color: '#ffb86c', fontSize: '0.85rem', marginTop: '8px' }}>
                  ⚠️ Select at least one more model
                </p>
              )}
            </div>

            {/* Question Input */}
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#f8f8f2', fontSize: '1rem', marginBottom: '10px' }}>
                Your Question:
              </h3>

              {/* Attached file chip */}
              {attachedFile && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '8px',
                  padding: '6px 10px',
                  background: '#21222c',
                  border: `1px solid ${attachedFile.isImage ? '#bd93f9' : attachedFile.size > 50000 ? '#ffb86c' : '#50fa7b'}`,
                  borderRadius: '6px',
                  fontSize: '0.8rem',
                  color: '#f8f8f2',
                }}>
                  {attachedFile.isImage ? (
                    <img
                      src={`data:${attachedFile.mimeType};base64,${attachedFile.base64}`}
                      alt="preview"
                      style={{ width: '32px', height: '32px', objectFit: 'cover', borderRadius: '4px' }}
                    />
                  ) : <span>📎</span>}
                  <span style={{ fontWeight: 600 }}>{attachedFile.name}</span>
                  {attachedFile.isImage && (
                    <span style={{ color: '#bd93f9', fontSize: '0.75rem' }}>🖼️ Image — vision model required</span>
                  )}
                  {!attachedFile.isImage && attachedFile.size > 50000 && (
                    <span style={{ color: '#ffb86c', fontSize: '0.75rem' }}>⚠️ Large file — uses significant context</span>
                  )}
                  <button
                    onClick={() => setAttachedFile(null)}
                    style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#ff5555', cursor: 'pointer', fontSize: '1rem', lineHeight: 1, padding: '0 2px' }}
                    title="Remove attachment"
                  >×</button>
                </div>
              )}

              <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What question do you want the AI models to debate?"
                    style={{
                      width: '100%',
                      minHeight: '100px',
                      padding: '12px',
                      background: '#282a36',
                      border: '1px solid #44475a',
                      borderRadius: '6px',
                      color: '#f8f8f2',
                      fontSize: '0.95rem',
                      resize: 'vertical',
                      fontFamily: 'inherit',
                      boxSizing: 'border-box'
                    }}
                    disabled={isLoading}
                  />
                  {/* KB and attach buttons */}
                  <div style={{ display: 'flex', gap: '6px' }}>
                    <button
                      onClick={() => setSearchKBEnabled(!searchKBEnabled)}
                      disabled={isLoading}
                      title={searchKBEnabled ? 'Knowledge Base search enabled' : 'Click to search Knowledge Base'}
                      style={{
                        height: '28px',
                        padding: '0 10px',
                        background: searchKBEnabled
                          ? 'linear-gradient(135deg, #8be9fd 0%, #50fa7b 100%)'
                          : 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                        color: searchKBEnabled ? '#282a36' : '#f8f8f2',
                        border: searchKBEnabled ? '2px solid #50fa7b' : 'none',
                        borderRadius: '6px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        transition: 'all 0.2s',
                        opacity: isLoading ? 0.5 : 1,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        boxShadow: searchKBEnabled ? '0 0 8px rgba(139, 233, 253, 0.5)' : '0 2px 4px rgba(0,0,0,0.2)',
                      }}
                    >
                      {searchKBEnabled ? '📚 KB ON' : '📚 KB'}
                    </button>
                    <button
                      onClick={handleAttachFile}
                      disabled={isLoading}
                      title={attachedFile ? `Attached: ${attachedFile.name}` : 'Attach a file or image'}
                      style={{
                        height: '28px',
                        padding: '0 10px',
                        background: attachedFile
                          ? 'linear-gradient(135deg, #ffb86c 0%, #ff79c6 100%)'
                          : 'linear-gradient(135deg, #6272a4 0%, #44475a 100%)',
                        color: attachedFile ? '#282a36' : '#f8f8f2',
                        border: attachedFile ? '2px solid #ffb86c' : 'none',
                        borderRadius: '6px',
                        cursor: isLoading ? 'not-allowed' : 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.7rem',
                        transition: 'all 0.2s',
                        opacity: isLoading ? 0.5 : 1,
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                    >
                      {attachedFile ? '📎 1 file' : '📎'}
                    </button>
                  </div>
                </div>
                <VoiceButton
                  onTranscript={(text) => setQuestion(text)}
                  disabled={isLoading}
                  style={{
                    minWidth: '45px',
                    minHeight: '45px'
                  }}
                />
              </div>
            </div>

            {error && (
              <div style={{
                padding: '12px',
                background: '#ff5555',
                color: '#fff',
                borderRadius: '6px',
                marginBottom: '20px',
                fontSize: '0.9rem'
              }}>
                ❌ {error}
              </div>
            )}

            {/* Buttons */}
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={handleClose}
                style={{
                  padding: '10px 20px',
                  background: '#44475a',
                  color: '#f8f8f2',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                onClick={handleRunEnsemble}
                style={{
                  padding: '10px 20px',
                  background: isLoading ? '#6272a4' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: isLoading ? 'wait' : 'pointer',
                  fontWeight: 'bold',
                  opacity: (selectedModels.length < 2 || !question.trim()) ? 0.5 : 1
                }}
                disabled={isLoading || selectedModels.length < 2 || !question.trim()}
              >
                {isLoading ? '⏳ Running Ensemble...' : '🚀 Run Ensemble'}
              </button>
            </div>
          </>
        ) : (
          <>
            {/* Results Display */}
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#50fa7b', fontSize: '1.1rem', marginBottom: '15px' }}>
                ✅ Ensemble Complete
              </h3>

              {/* Question Asked */}
              <div style={{
                padding: '15px',
                background: '#282a36',
                borderRadius: '8px',
                marginBottom: '20px',
                border: '1px solid #44475a'
              }}>
                <strong style={{ color: '#8be9fd' }}>Question:</strong>
                <p style={{ color: '#f8f8f2', marginTop: '8px' }}>{question}</p>
              </div>

              {/* Individual Responses */}
              <h4 style={{ color: '#8be9fd', fontSize: '1rem', marginBottom: '10px' }}>
                Individual Model Responses:
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px' }}>
                {result.individual_responses?.map((resp, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '15px',
                      background: resp.success ? '#282a36' : '#ff555533',
                      borderRadius: '8px',
                      border: `1px solid ${resp.success ? '#44475a' : '#ff5555'}`
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}>
                      <strong style={{ color: resp.success ? '#50fa7b' : '#ff5555' }}>
                        {resp.model}
                      </strong>
                      <span style={{ fontSize: '0.85rem', color: '#9aa5c4' }}>
                        {resp.success ? '✓ Success' : '✗ Failed'}
                      </span>
                    </div>
                    <p style={{
                      color: '#f8f8f2',
                      whiteSpace: 'pre-wrap',
                      fontSize: '0.9rem',
                      lineHeight: '1.5'
                    }}>
                      {resp.response}
                    </p>
                  </div>
                ))}
              </div>

              {/* Synthesized Answer */}
              <h4 style={{ color: '#8be9fd', fontSize: '1rem', marginBottom: '10px' }}>
                🎯 Synthesized Answer (Coordinator: {result.coordinator_model}):
              </h4>
              <div style={{
                padding: '20px',
                background: '#44475a',
                borderRadius: '8px',
                border: '2px solid #50fa7b',
                marginBottom: '20px'
              }}>
                <p style={{
                  color: '#f8f8f2',
                  whiteSpace: 'pre-wrap',
                  fontSize: '0.95rem',
                  lineHeight: '1.6'
                }}>
                  {result.synthesized_response}
                </p>
              </div>

              {/* Web Search Results (if available) */}
              {result.search_results && result.search_results.results && result.search_results.results.length > 0 && (
                <details style={{ marginBottom: '20px' }}>
                  <summary style={{
                    color: '#8be9fd',
                    fontSize: '1rem',
                    cursor: 'pointer',
                    padding: '10px',
                    background: '#282a36',
                    borderRadius: '6px',
                    border: '1px solid #44475a',
                    marginBottom: '10px'
                  }}>
                    🔍 External Verification Sources ({result.search_results.num_results} results)
                  </summary>
                  <div style={{
                    padding: '15px',
                    background: '#282a36',
                    borderRadius: '8px',
                    border: '1px solid #44475a'
                  }}>
                    <p style={{ color: '#9aa5c4', fontSize: '0.85rem', marginBottom: '15px' }}>
                      Query: "{result.search_results.query}"
                    </p>
                    {result.search_results.results.map((searchResult, idx) => (
                      <div key={idx} style={{
                        padding: '12px',
                        background: '#1e1f29',
                        borderRadius: '6px',
                        marginBottom: '10px',
                        border: '1px solid #44475a'
                      }}>
                        <span
                          onClick={() => openUrl(searchResult.url)}
                          style={{
                            color: '#8be9fd',
                            textDecoration: 'underline',
                            cursor: 'pointer',
                            fontWeight: 'bold',
                            fontSize: '0.95rem'
                          }}
                        >
                          {searchResult.title}
                        </span>
                        <p style={{
                          color: '#9aa5c4',
                          fontSize: '0.8rem',
                          marginTop: '4px',
                          marginBottom: '8px'
                        }}>
                          {searchResult.url}
                        </p>
                        <p style={{
                          color: '#f8f8f2',
                          fontSize: '0.9rem',
                          lineHeight: '1.5'
                        }}>
                          {searchResult.snippet}
                        </p>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={handleCancel}
                style={{
                  padding: '10px 20px',
                  background: '#44475a',
                  color: '#f8f8f2',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                ✗ Cancel
              </button>
              <button
                onClick={handleClose}
                style={{
                  padding: '10px 20px',
                  background: 'linear-gradient(135deg, #50fa7b 0%, #2ecc71 100%)',
                  color: '#282a36',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontWeight: 'bold'
                }}
              >
                ✓ Add to Conversation
              </button>
            </div>
          </>
        )}
        </>
        )}
      </div>
    </div>
  );
}
