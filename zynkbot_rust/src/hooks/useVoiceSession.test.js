import { parseVoiceCommand, shouldSpeakReply, nativeTurnsToMessages, cleanForSpeech } from './useVoiceSession';

jest.mock('@tauri-apps/api/core', () => ({ invoke: jest.fn() }));

describe('parseVoiceCommand stop phrases', () => {
  test.each([
    'stop',
    'stop talking',
    'stop listening',
    'zynk stop',
    'hey zynk stop',
    'hey zynkbot stop',
  ])('recognizes %j as a TTS stop command', (phrase) => {
    expect(parseVoiceCommand(phrase)).toEqual({ type: 'stop_tts' });
  });

  test('does not consume a normal request containing the word stop', () => {
    expect(parseVoiceCommand('why did the train stop')).toBeNull();
  });
});

describe('parseVoiceCommand timers — what dictation actually produces', () => {
  test('"set a time for five minutes" is the same timer as "set a timer for five minutes" (#29)', () => {
    const withR = parseVoiceCommand('set a timer for five minutes');
    expect(withR).not.toBeNull();
    expect(parseVoiceCommand('set a time for five minutes')).toEqual(withR);
  });
});

describe('shouldSpeakReply — answer in the channel you used', () => {
  test('hands-free request is always spoken, even with the in-app toggle off', () => {
    expect(shouldSpeakReply({ handsFree: true, speakInApp: false })).toBe(true);
  });

  test('request made with the app open is text only by default', () => {
    expect(shouldSpeakReply({ handsFree: false, speakInApp: false })).toBe(false);
  });

  test('the in-app toggle opts into spoken replies while the app is open', () => {
    expect(shouldSpeakReply({ handsFree: false, speakInApp: true })).toBe(true);
  });

  test('missing flags mean text only', () => {
    expect(shouldSpeakReply({})).toBe(false);
  });
});

describe('nativeTurnsToMessages — hands-free exchanges join the thread on screen', () => {
  const turn = { sessionId: 's1', question: 'capital of arizona', answer: 'Phoenix.', at: 1000 };

  test('a turn for the current thread becomes a user + assistant pair', () => {
    const msgs = nativeTurnsToMessages([turn], 's1');
    expect(msgs.map((m) => m.role)).toEqual(['user', 'assistant']);
    expect(msgs[0].content).toBe('capital of arizona');
    expect(msgs[1].content).toBe('Phoenix.');
    expect(msgs[1].id).toBeGreaterThan(msgs[0].id);
  });

  test('what the model said before an automatic web search is its own message', () => {
    const msgs = nativeTurnsToMessages([{ ...turn, preAnswer: 'I think Phoenix, but let me check.' }], 's1');
    expect(msgs.map((m) => m.role)).toEqual(['user', 'assistant', 'assistant']);
    expect(msgs[1].content).toBe('I think Phoenix, but let me check.');
    expect(msgs[2].content).toBe('Phoenix.');
    expect(new Set(msgs.map((m) => m.id)).size).toBe(3);
  });

  test('turns for another thread are left to Conversation History', () => {
    expect(nativeTurnsToMessages([turn], 'other')).toEqual([]);
  });

  test('garbage input never throws', () => {
    expect(nativeTurnsToMessages(null, 's1')).toEqual([]);
    expect(nativeTurnsToMessages([null, {}, { sessionId: 's1' }], 's1')).toEqual([]);
    expect(nativeTurnsToMessages([turn], '')).toEqual([]);
  });
});

describe('cleanForSpeech — what the speech engine is handed', () => {
  test('markdown and arithmetic symbols become spoken words', () => {
    expect(cleanForSpeech('470 ÷ 20 = **23.5**')).toBe('470 divided by 20 equals 23.5');
  });

  test('headings, bullets, bold and links are stripped, text kept', () => {
    expect(cleanForSpeech('## Steps\n- **First** do [this](http://x)\n- then *that*'))
      .toBe('Steps\nFirst do this\nthen that');
  });

  test('plain prose passes through unchanged', () => {
    expect(cleanForSpeech('The capital of Colorado is Denver.')).toBe('The capital of Colorado is Denver.');
  });
});
