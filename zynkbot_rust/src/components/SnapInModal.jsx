import React, { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

export default function SnapInModal({ isOpen, onClose, userId }) {
  const [view, setView] = useState('intro'); // 'intro' or 'therapist'
  const [patientName, setPatientName] = useState('');
  const [sessionTitle, setSessionTitle] = useState('');
  const [notes, setNotes] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);

  if (!isOpen) return null;

  const handleIndexNotes = async () => {
    if (!patientName.trim() || !sessionTitle.trim() || !notes.trim()) {
      alert('Please fill in all fields');
      return;
    }

    setIsIndexing(true);
    try {
      const result = await invoke('index_snapin_notes', {
        patientName: patientName.trim(),
        sessionTitle: sessionTitle.trim(),
        notesContent: notes.trim(),
        userId: userId
      });

      alert(result);

      // Reset form
      setPatientName('');
      setSessionTitle('');
      setNotes('');
      setView('intro');
    } catch (error) {
      console.error('Failed to index notes:', error);
      alert('Failed to index notes: ' + error);
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 2000,
      padding: '20px'
    }}>
      <div style={{
        background: '#282a36',
        borderRadius: '12px',
        maxWidth: view === 'intro' ? '700px' : '800px',
        width: '100%',
        maxHeight: '90vh',
        overflow: 'auto',
        border: '2px solid #ff79c6',
        boxShadow: '0 8px 32px rgba(255, 121, 198, 0.3)'
      }}>
        {/* Header */}
        <div style={{
          padding: '20px',
          borderBottom: '1px solid #44475a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'linear-gradient(135deg, #ff79c6 0%, #bd93f9 100%)'
        }}>
          <h2 style={{margin: 0, color: '#fff', fontSize: '1.4rem'}}>
            {view === 'intro' ? '🧪 Snap-in Ecosystem' : '📝 Therapist Journal (Sample)'}
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'rgba(255, 255, 255, 0.2)',
              border: 'none',
              color: '#fff',
              fontSize: '1.5rem',
              cursor: 'pointer',
              borderRadius: '6px',
              width: '35px',
              height: '35px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div style={{padding: '25px'}}>
          {view === 'intro' ? (
            /* Introduction View */
            <>
              <div style={{marginBottom: '25px'}}>
                <h3 style={{color: '#ff79c6', marginTop: 0, marginBottom: '12px'}}>What Are Snap-ins?</h3>
                <p style={{color: '#f8f8f2', lineHeight: '1.7', fontSize: '0.95rem'}}>
                  <strong>Snap-ins</strong> are domain-specific workspaces designed for professional and personal use.
                  They integrate with Zynkbot's persistent memory system while maintaining strict privacy boundaries.
                </p>
              </div>

              <div style={{marginBottom: '25px'}}>
                <h3 style={{color: '#bd93f9', marginBottom: '12px'}}>Design Principles</h3>
                <ul style={{color: '#f8f8f2', lineHeight: '1.8', paddingLeft: '20px'}}>
                  <li><strong>Mode-aware:</strong> Respect containment modes (HIPAA, Sovereign, Guardian, etc.)</li>
                  <li><strong>Consent-driven:</strong> All data access requires explicit user permission</li>
                  <li><strong>Local-first:</strong> Data stays on your device by default</li>
                  <li><strong>Privacy-respecting:</strong> No surveillance, engagement loops, or dark patterns</li>
                  <li><strong>Structured workspaces:</strong> Organized contexts, not scattered notes</li>
                </ul>
              </div>

              <div style={{marginBottom: '25px', padding: '15px', background: '#44475a', borderRadius: '8px', borderLeft: '4px solid #50fa7b'}}>
                <h4 style={{color: '#50fa7b', marginTop: 0, marginBottom: '10px'}}>Professional Snap-ins (Future)</h4>
                <ul style={{color: '#f8f8f2', lineHeight: '1.7', margin: 0, paddingLeft: '20px', fontSize: '0.9rem'}}>
                  <li><strong>Therapy workspace:</strong> Session notes, patient tracking, pattern detection</li>
                  <li><strong>Investigative journalism:</strong> Source tracking, interview notes, research compilation, timeline verification</li>
                  <li><strong>Legal case management:</strong> Case files, research notes, timeline tracking</li>
                  <li><strong>Medical research:</strong> Literature review, hypothesis tracking, experiment notes</li>
                  <li><strong>Education planning:</strong> Curriculum development, student progress, lesson notes</li>
                  <li><strong>Humanitarian coordination:</strong> Project tracking, resource allocation, field reports</li>
                </ul>
              </div>

              <div style={{marginBottom: '25px', padding: '15px', background: '#44475a', borderRadius: '8px', borderLeft: '4px solid #8be9fd'}}>
                <h4 style={{color: '#8be9fd', marginTop: 0, marginBottom: '10px'}}>Personal Snap-ins (Future)</h4>
                <ul style={{color: '#f8f8f2', lineHeight: '1.7', margin: 0, paddingLeft: '20px', fontSize: '0.9rem'}}>
                  <li><strong>Journal mode:</strong> Guided reflection without emotional dependency</li>
                  <li><strong>Parenting companion:</strong> Development tracking, gentle reminders, resource suggestions</li>
                  <li><strong>Self-coaching:</strong> Goal tracking, pattern identification, accountability</li>
                  <li><strong>Mood drift reflector:</strong> Emotional pattern awareness (not diagnosis)</li>
                  <li><strong>Procrastination breaker:</strong> Task decomposition, friction analysis</li>
                </ul>
              </div>

              <div style={{marginBottom: '25px', padding: '15px', background: '#44475a', borderRadius: '8px'}}>
                <h4 style={{color: '#ffb86c', marginTop: 0, marginBottom: '10px'}}>⚠️ Current Status: Proof-of-Concept</h4>
                <p style={{color: '#f8f8f2', lineHeight: '1.7', margin: 0, fontSize: '0.9rem'}}>
                  The snap-in architecture exists, but only a <strong>sample therapist journal</strong> is implemented.
                  This demonstrates how snap-ins organize information using the existing knowledge base (RAG) system
                  with logical folder structures.
                </p>
              </div>

              <div style={{marginBottom: '25px'}}>
                <h3 style={{color: '#ff79c6', marginBottom: '12px'}}>How It Works (Therapist Example)</h3>
                <ol style={{color: '#f8f8f2', lineHeight: '1.8', paddingLeft: '20px'}}>
                  <li>Therapist creates patient/client entries (organized by name)</li>
                  <li>Session notes are stored as text documents with metadata</li>
                  <li>Notes are indexed into the vector database for semantic search</li>
                  <li>File paths provide logical organization: <code style={{background: '#44475a', padding: '2px 6px', borderRadius: '3px'}}>snap_ins/therapist/patient_name/session.txt</code></li>
                  <li>RAG search can filter by patient for privacy-aware recall</li>
                  <li><strong>Everything stays local</strong> - no cloud, no sharing without consent</li>
                </ol>
              </div>

              <div style={{padding: '15px', background: '#44475a', borderRadius: '8px', marginBottom: '20px', borderLeft: '4px solid #8be9fd'}}>
                <h4 style={{color: '#8be9fd', marginTop: 0, marginBottom: '10px'}}>💡 Once Notes Are Indexed</h4>
                <p style={{color: '#f8f8f2', lineHeight: '1.7', margin: 0, fontSize: '0.9rem'}}>
                  After indexing session notes, you can query them naturally through the Knowledge Base:
                </p>
                <ul style={{color: '#f8f8f2', margin: '10px 0 0 0', paddingLeft: '20px', fontSize: '0.9rem', lineHeight: '1.7'}}>
                  <li>"What did Sarah discuss in her last three sessions?"</li>
                  <li>"Show me all notes about coping strategies from February"</li>
                  <li>"When did John first mention his sleep problems?"</li>
                  <li>"Find sessions where we talked about family dynamics"</li>
                </ul>
                <p style={{color: '#b8c5db', margin: '10px 0 0 0', fontSize: '0.85rem', fontStyle: 'italic'}}>
                  RAG search finds relevant content even if you don't remember exact wording. As long as notes are dated and organized by patient, you can query by timeframe, topic, or patient name.
                </p>
              </div>

              <div style={{padding: '15px', background: '#44475a', borderRadius: '8px', marginBottom: '20px'}}>
                <h4 style={{color: '#f1fa8c', marginTop: 0, marginBottom: '10px'}}>🔒 Production vs. Demo</h4>
                <p style={{color: '#f8f8f2', lineHeight: '1.7', margin: 0, fontSize: '0.9rem'}}>
                  This demo stores everything under one user_id for simplicity. A production therapist snap-in would include:
                </p>
                <ul style={{color: '#f8f8f2', margin: '10px 0 0 0', paddingLeft: '20px', fontSize: '0.9rem'}}>
                  <li>Per-patient data isolation (separate encrypted containers)</li>
                  <li>HIPAA-compliant audit logs</li>
                  <li>Encrypted backup and export</li>
                  <li>Session templates and structured forms</li>
                  <li>Pattern detection (e.g., recurring themes across sessions)</li>
                </ul>
              </div>

              <button
                onClick={() => setView('therapist')}
                style={{
                  width: '100%',
                  padding: '15px',
                  background: 'linear-gradient(135deg, #ff79c6 0%, #bd93f9 100%)',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  fontSize: '1.05rem',
                  transition: 'transform 0.2s'
                }}
                onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
                onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
              >
                📝 Try Sample: Therapist Journal
              </button>
            </>
          ) : (
            /* Therapist Journal View */
            <>
              <button
                onClick={() => setView('intro')}
                style={{
                  marginBottom: '15px',
                  padding: '8px 16px',
                  background: '#44475a',
                  color: '#f8f8f2',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                ← Back to Overview
              </button>

              <div style={{marginBottom: '20px'}}>
                <h3 style={{color: '#ff79c6', marginTop: 0, marginBottom: '10px'}}>Session Notes Entry</h3>
                <p style={{color: '#9aa5c4', fontSize: '0.9rem', marginBottom: '15px'}}>
                  This demo shows how notes can be organized by patient and indexed for RAG search.
                </p>
              </div>

              <div style={{marginBottom: '15px'}}>
                <label style={{display: 'block', color: '#8be9fd', marginBottom: '8px', fontWeight: 'bold'}}>
                  Patient/Client Name:
                </label>
                <input
                  type="text"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="e.g., John Doe"
                  style={{
                    width: '100%',
                    padding: '10px',
                    background: '#44475a',
                    border: '1px solid #6272a4',
                    borderRadius: '6px',
                    color: '#f8f8f2',
                    fontSize: '0.95rem'
                  }}
                />
                <small style={{color: '#9aa5c4', fontSize: '0.85rem', marginTop: '5px', display: 'block'}}>
                  Creates a logical folder: snap_ins/therapist/{patientName.toLowerCase().replace(/ /g, '_')}/
                </small>
              </div>

              <div style={{marginBottom: '15px'}}>
                <label style={{display: 'block', color: '#8be9fd', marginBottom: '8px', fontWeight: 'bold'}}>
                  Session Title:
                </label>
                <input
                  type="text"
                  value={sessionTitle}
                  onChange={(e) => setSessionTitle(e.target.value)}
                  placeholder="e.g., Session 2024-02-14 or Initial Assessment"
                  style={{
                    width: '100%',
                    padding: '10px',
                    background: '#44475a',
                    border: '1px solid #6272a4',
                    borderRadius: '6px',
                    color: '#f8f8f2',
                    fontSize: '0.95rem'
                  }}
                />
              </div>

              <div style={{marginBottom: '15px'}}>
                <label style={{display: 'block', color: '#8be9fd', marginBottom: '8px', fontWeight: 'bold'}}>
                  Session Notes:
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Enter your session notes here...&#10;&#10;Example:&#10;Patient presented with continued anxiety around work deadlines.&#10;&#10;Discussed cognitive reframing techniques for negative self-talk.&#10;&#10;Homework: Practice 5-minute mindfulness exercise daily."
                  rows={12}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: '#44475a',
                    border: '1px solid #6272a4',
                    borderRadius: '6px',
                    color: '#f8f8f2',
                    fontSize: '0.95rem',
                    fontFamily: 'monospace',
                    resize: 'vertical'
                  }}
                />
              </div>

              <div style={{marginBottom: '20px', padding: '12px', background: '#44475a', borderRadius: '6px', borderLeft: '3px solid #50fa7b'}}>
                <p style={{color: '#50fa7b', margin: 0, fontSize: '0.9rem', lineHeight: '1.6'}}>
                  ✓ Notes will be indexed locally using RAG (vector embeddings)
                  <br />
                  ✓ Searchable via semantic similarity
                  <br />
                  ✓ Stored at: snap_ins/therapist/{patientName ? patientName.toLowerCase().replace(/ /g, '_') : 'patient'}/{sessionTitle || 'session'}.txt
                  <br />
                  ✓ Fully private, stays on your device
                </p>
              </div>

              <p style={{color: 'rgba(255,255,255,0.65)', fontSize: '0.82rem', margin: '0 0 12px 0', fontStyle: 'italic'}}>
                ⏱️ First-time indexing may take 1–2 minutes while the embedding model loads. Subsequent indexing is instant.
              </p>

              <div style={{display: 'flex', gap: '10px'}}>
                <button
                  onClick={onClose}
                  style={{
                    flex: 1,
                    padding: '12px',
                    background: '#44475a',
                    color: '#f8f8f2',
                    border: '1px solid #6272a4',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.95rem'
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleIndexNotes}
                  disabled={isIndexing}
                  style={{
                    flex: 2,
                    padding: '12px',
                    background: isIndexing ? '#6272a4' : 'linear-gradient(135deg, #50fa7b 0%, #3dd46b 100%)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: isIndexing ? 'wait' : 'pointer',
                    fontWeight: 'bold',
                    fontSize: '0.95rem',
                    transition: 'transform 0.2s'
                  }}
                  onMouseOver={(e) => !isIndexing && (e.target.style.transform = 'translateY(-2px)')}
                  onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
                >
                  {isIndexing ? '⏳ Indexing...' : '💾 Index Session Notes'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
