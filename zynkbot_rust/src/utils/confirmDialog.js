// Native confirmation dialog that actually waits for the user on every platform.
//
// window.confirm() is synchronous. On Windows the WebView2 engine cannot block for a
// synchronous JavaScript dialog raised from an event handler (its ScriptDialogOpening
// handler is not awaited), so the call returned "true" before any dialog appeared.
// On 2026-09-12 that let the Einstein demo load and "Clear All" delete 59 memories
// with no confirmation shown (KI-047). Linux (WebKitGTK) and Android render the
// native dialog, which is why it was never seen there.
//
// The Tauri dialog plugin shows a real OS dialog from Rust and resolves with the
// user's actual answer on Windows, Linux, macOS and Android. dialog:default in
// tauri.conf.json already grants confirm, so no Rust rebuild is needed.
//
// Usage (the caller must be async):  if (!(await confirmDialog('Delete it?'))) return;
import { confirm, message } from '@tauri-apps/plugin-dialog';

export async function confirmDialog(message, options = {}) {
  const opts = { title: 'Zynkbot', kind: 'warning', ...options };
  try {
    return await confirm(message, opts);
  } catch (e) {
    // Outside the Tauri shell (plain browser at localhost:3000) the plugin has no
    // backend. The browser dialog works there, so fall back to it.
    console.warn('[confirmDialog] plugin unavailable, using window.confirm:', e);
    return window.confirm(message);
  }
}

// Same fix for alert(): on Windows a synchronous alert() does not block either, so code
// that alerts and then reloads the page (Einstein demo) reloaded with the dialog still
// up and showed it twice. Await this instead, then continue.
export async function messageDialog(text, options = {}) {
  const opts = { title: 'Zynkbot', kind: 'info', ...options };
  try {
    await message(text, opts);
  } catch (e) {
    console.warn('[messageDialog] plugin unavailable, using window.alert:', e);
    window.alert(text);
  }
}
