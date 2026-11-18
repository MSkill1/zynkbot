import React from 'react';
import '../styles/ControlsPage.css';
import PageIntro from './PageIntro';

/**
 * One System Controls section, shown on a page of its own on top of the sidebar.
 * Only one page is open at a time; ✕ (or a tap outside the card) returns to the
 * System Controls list. Introduced in 0.9.6-beta5 to replace the stack of
 * fold-out sections, which had grown too tall to scan on a phone.
 *
 * Also the shell for the windows that open from chat (ui-redo, 2026-11-18), so every
 * window in the app looks the same:
 * - `zIndex`: the layer the window sits on. The default (1000) is the sidebar's, for
 *   pages opened from System Controls; a window that must cover other windows (ZChat,
 *   the memory conflict card, the About me report) asks for a higher one.
 * - `closeLabel`: the ✕ button's tooltip and spoken name; "Back to System Controls"
 *   for a controls page, "Close" for a window opened from chat.
 * - `dismissOnOutside`: whether a tap outside the card closes it (default yes). Off for
 *   a window the user must answer (the conflict card) or wait for (the loading screen).
 * - `size`: "small" for a window with a line or two in it (a name prompt, a yes/no
 *   card), which stays a short centred card on the phone instead of filling the
 *   screen; "wide" for one that needs the room (the memory graph). Default: the
 *   560px card, full screen on a phone.
 */
export default function ControlsPage({ icon, title, subtitle, accent = '#44475a', onClose, intro, introKey, children,
  zIndex, closeLabel = 'Back to System Controls', dismissOnOutside = true, size }) {
  const sizeClass = size === 'small' || size === 'wide' ? ` ${size}` : '';
  return (
    <div className={`controls-page-overlay${sizeClass}`} style={zIndex != null ? { zIndex } : undefined} onClick={dismissOnOutside ? onClose : undefined}>
      <div
        className={`controls-page${sizeClass}`}
        style={{ borderColor: accent }}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label={title}
      >
        <div className="controls-page-head" style={{ borderBottomColor: accent }}>
          <span className="controls-page-icon">{icon}</span>
          <div className="controls-page-title">
            <b style={{ color: accent === '#44475a' ? '#f8f8f2' : accent }}>{title}</b>
            {subtitle && <small>{subtitle}</small>}
          </div>
          {onClose && <button className="controls-page-close" onClick={onClose} title={closeLabel} aria-label={closeLabel}>✕</button>}
        </div>
        <div className="controls-page-body">
          {intro && <PageIntro introKey={introKey || title} accent={accent === '#44475a' ? '#6272a4' : accent}>{intro}</PageIntro>}
          {children}
        </div>
      </div>
    </div>
  );
}
