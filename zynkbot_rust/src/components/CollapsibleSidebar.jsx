import React, { useState } from 'react';

export default function CollapsibleSidebar({ children, icon, title, onInfoClick, onVoiceClick, onReportClick, hideToggle, onOpen }) {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    if (!isOpen && onOpen) onOpen();
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Icon button (always visible) - BOTTOM LEFT */}
      <button
        onClick={handleToggle}
        hidden={hideToggle}
        className={`sidebar-toggle-btn${isOpen ? ' sidebar-open' : ''}`}
        style={{
          position: 'fixed',
          left: isOpen ? '460px' : '10px',
          bottom: '20px',  // Changed from top to bottom
          width: '50px',
          height: '50px',
          borderRadius: '50%',
          background: isOpen ? '#44475a' : '#8be9fd',
          color: isOpen ? '#f8f8f2' : '#282a36',
          border: 'none',
          fontSize: '1.5rem',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.3s ease',
          zIndex: 1001,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
        title={isOpen ? 'Close sidebar' : `Open ${title}`}
      >
        {isOpen ? '✕' : icon}
      </button>

      {/* Sidebar panel - WIDER (460px) */}
      <div
        className="sidebar-panel"
        style={{
          position: 'fixed',
          left: '0',
          top: '0',
          width: '460px',
          height: '100vh',
          background: '#282a36',
          boxShadow: isOpen ? '4px 0 12px rgba(0, 0, 0, 0.5)' : 'none',
          transform: isOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition: 'transform 0.3s ease',
          zIndex: 1000,
          overflowY: 'auto',
          padding: '20px 15px 100px 15px', /* status bar reserved natively (build27) */
          borderRight: '1px solid #44475a'
        }}
      >
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '20px'
        }}>
          <h2 style={{
            margin: 0,
            color: '#8be9fd',
            fontSize: '1.3rem'
          }}>
            {title}
          </h2>
          {/* flexWrap: at ~360dp (the most common Android width) the 100vw drawer
              leaves ~330px of content, too little for the title and all three
              buttons on one line. Wrapping drops the button group to its own row
              instead of compressing the buttons until their labels break. */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px', flexWrap: 'wrap' }}>
            {onVoiceClick && (
              <button
                onClick={onVoiceClick}
                style={{
                  background: '#44475a',
                  color: '#8be9fd',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  minHeight: '32px',
                  boxSizing: 'border-box',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  whiteSpace: 'nowrap',
                }}
                onMouseOver={(e) => {
                  e.target.style.background = '#8be9fd';
                  e.target.style.color = '#22232a';
                }}
                onMouseOut={(e) => {
                  e.target.style.background = '#44475a';
                  e.target.style.color = '#8be9fd';
                }}
              >
                🎙️ Voice
              </button>
            )}
            {onReportClick && (
              <button
                onClick={onReportClick}
                title="Build a bug report to paste into GitHub"
                style={{
                  background: '#44475a',
                  color: '#8be9fd',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  minHeight: '32px',
                  boxSizing: 'border-box',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  whiteSpace: 'nowrap',
                }}
                onMouseOver={(e) => { e.target.style.background = '#8be9fd'; e.target.style.color = '#22232a'; }}
                onMouseOut={(e) => { e.target.style.background = '#44475a'; e.target.style.color = '#8be9fd'; }}
              >
                ⚑ Report
              </button>
            )}
            {onInfoClick && (
              <button
                onClick={onInfoClick}
                style={{
                  background: '#44475a',
                  color: '#8be9fd',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  minHeight: '32px',
                  boxSizing: 'border-box',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  whiteSpace: 'nowrap',
                }}
                onMouseOver={(e) => {
                  e.target.style.background = '#8be9fd';
                  e.target.style.color = '#22232a';
                }}
                onMouseOut={(e) => {
                  e.target.style.background = '#44475a';
                  e.target.style.color = '#8be9fd';
                }}
              >
                Info
              </button>
            )}
          </div>
        </div>
        {children}
      </div>

      {/* Backdrop (click to close) */}
      {isOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setIsOpen(false)}
          style={{
            position: 'fixed',
            top: '0',
            left: '460px',  // Updated for wider sidebar
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999,
            cursor: 'pointer'
          }}
        />
      )}
    </>
  );
}
