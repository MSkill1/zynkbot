import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom';
import { invoke } from '@tauri-apps/api/core';
import VoiceButton from './VoiceButton';

export default function ZChatModal({
  isOpen,
  onClose,
  apiBaseUrl,
  device,
  currentDeviceId
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [showEmoticons, setShowEmoticons] = useState(false);
  const messagesEndRef = useRef(null);

  // Standard emoji
  const emoticons = [
    '😊', '😢', '😉', '😄', '😛', '😮', '😐',
    '😕', '😎', '🤓', '😈', '😭', '😆', '❤️'
  ];

  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchMessages = useCallback(async () => {
    if (!device || !currentDeviceId) return;

    try {
      // Note: Tauri converts Rust snake_case params to JS camelCase automatically
      const data = await invoke('zchat_get_messages', {
        deviceId: device.device_id,
        since: null
      });

      if (data.messages) {
        setMessages(data.messages);
      }
    } catch (error) {
      console.error('[ZChat] Failed to fetch messages:', error);
    }
  }, [device, currentDeviceId]);

  // Mark all messages from this device as read when chat opens
  const markMessagesAsRead = useCallback(async () => {
    if (!device) return;

    try {
      // Note: Tauri converts Rust snake_case params to JS camelCase automatically
      await invoke('zchat_mark_all_read_from_device', {
        fromDeviceId: device.device_id
      });
    } catch (error) {
      console.debug('[ZChat] Failed to mark messages as read:', error);
    }
  }, [device]);

  // Fetch messages on open
  useEffect(() => {
    if (isOpen && device) {
      fetchMessages();
      markMessagesAsRead(); // Mark as read when opening chat
      // Poll every 3 seconds; mark read each cycle so messages arriving mid-chat are cleared
      const interval = setInterval(() => { fetchMessages(); markMessagesAsRead(); }, 3000);
      return () => clearInterval(interval);
    }
  }, [isOpen, device, fetchMessages, markMessagesAsRead]);

  const handleSend = async () => {
    if (!input.trim() || isSending) return;

    setIsSending(true);
    const messageText = input.trim();
    setInput(''); // Clear immediately for responsiveness

    try {
      // Note: Tauri converts Rust snake_case params to JS camelCase automatically
      const data = await invoke('zchat_send_message', {
        toDeviceId: device.device_id,
        messageText: messageText
      });

      if (data.success) {
        // Immediately fetch messages to show sent message
        await fetchMessages();
      } else {
        console.error('[ZChat] Failed to send:', data.error || 'Unknown error');
        // Restore message on failure
        setInput(messageText);
      }
    } catch (error) {
      console.error('[ZChat] Send error:', error);
      // Restore message on failure
      setInput(messageText);
    } finally {
      setIsSending(false);
    }
  };

  const handleClearHistory = async () => {
    if (!window.confirm('Clear all chat history with this user?')) return;
    try {
      await invoke('zchat_clear_history', { withDeviceId: device.device_id });
      setMessages([]);
    } catch (error) {
      console.error('[ZChat] Failed to clear history:', error);
    }
  };

  const handleEmoticonClick = (emoticon) => {
    setInput(prev => prev + emoticon);
    setShowEmoticons(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!isOpen || !device) return null;

  // Render modal to document.body to avoid clipping by parent containers
  return ReactDOM.createPortal(
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
        zIndex: 9999,
        padding: '20px'
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#1e1f29',
          borderRadius: '12px',
          padding: '20px',
          maxWidth: '600px',
          width: '100%',
          height: '600px',
          display: 'flex',
          flexDirection: 'column',
          border: '1px solid #44475a',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '15px',
          paddingBottom: '15px',
          borderBottom: '2px solid #44475a'
        }}>
          <h3 style={{ margin: 0, color: '#8be9fd' }}>
            💬 Chat with {device.device_name}
          </h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleClearHistory}
              style={{
                background: '#44475a',
                color: '#9aa5c4',
                border: 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                cursor: 'pointer',
                fontSize: '0.85rem'
              }}
              title="Clear chat history"
            >
              🗑 Clear
            </button>
            <button
              onClick={onClose}
              style={{
                background: '#ff5555',
                color: '#f8f8f2',
                border: 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9rem'
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Messages Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          marginBottom: '15px',
          padding: '10px',
          background: '#282a36',
          borderRadius: '8px',
          border: '1px solid #44475a'
        }}>
          {messages.length === 0 ? (
            <div style={{
              color: '#9aa5c4',
              textAlign: 'center',
              padding: '40px 20px',
              fontSize: '0.9rem'
            }}>
              No messages yet. Start the conversation!
            </div>
          ) : (
            messages.map((msg, idx) => {
              const isSent = msg.from_device_id === currentDeviceId;
              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    justifyContent: isSent ? 'flex-end' : 'flex-start',
                    marginBottom: '12px'
                  }}
                >
                  <div style={{
                    maxWidth: '70%',
                    padding: '10px 14px',
                    borderRadius: '12px',
                    background: isSent ? '#8be9fd' : '#44475a',
                    color: isSent ? '#282a36' : '#f8f8f2',
                    fontSize: '0.9rem',
                    wordWrap: 'break-word'
                  }}>
                    <div style={{ marginBottom: '4px' }}>
                      {msg.message_text}
                    </div>
                    <div style={{
                      fontSize: '0.7rem',
                      opacity: 0.7,
                      textAlign: 'right'
                    }}>
                      {new Date(msg.sent_at).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </div>
                  </div>
                </div>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Emoticons Popup */}
        {showEmoticons && (
          <div style={{
            background: '#282a36',
            border: '1px solid #44475a',
            borderRadius: '8px',
            padding: '10px',
            marginBottom: '10px',
            display: 'grid',
            gridTemplateColumns: 'repeat(7, 1fr)',
            gap: '8px'
          }}>
            {emoticons.map((emo, idx) => (
              <button
                key={idx}
                onClick={() => handleEmoticonClick(emo)}
                style={{
                  background: '#44475a',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '8px',
                  cursor: 'pointer',
                  fontSize: '1.2rem',
                  color: '#f8f8f2',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.target.style.background = '#6272a4'}
                onMouseOut={(e) => e.target.style.background = '#44475a'}
              >
                {emo}
              </button>
            ))}
          </div>
        )}

        {/* Input Area */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
          <button
            onClick={() => setShowEmoticons(!showEmoticons)}
            style={{
              padding: '10px',
              background: '#44475a',
              color: '#f8f8f2',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '1.2rem',
              minWidth: '45px',
              height: '45px'
            }}
            title="Toggle emoticons"
          >
            😊
          </button>

          <VoiceButton
            onTranscript={(text) => setInput(text)}
            disabled={isSending}
            style={{
              minWidth: '45px',
              minHeight: '45px',
              height: '45px'
            }}
          />

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            disabled={isSending}
            style={{
              flex: 1,
              padding: '12px',
              background: '#282a36',
              border: '1px solid #44475a',
              borderRadius: '6px',
              color: '#f8f8f2',
              fontSize: '0.95rem',
              outline: 'none'
            }}
          />

          <button
            onClick={handleSend}
            disabled={isSending || !input.trim()}
            style={{
              padding: '12px 20px',
              background: isSending || !input.trim() ? '#44475a' : '#50fa7b',
              color: isSending || !input.trim() ? '#6272a4' : '#282a36',
              border: 'none',
              borderRadius: '6px',
              cursor: isSending || !input.trim() ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
              fontSize: '0.95rem',
              minWidth: '80px',
              height: '45px'
            }}
          >
            {isSending ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}
