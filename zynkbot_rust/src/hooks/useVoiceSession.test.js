import { parseVoiceCommand, shouldSpeakReply, nativeTurnsToMessages } from './useVoiceSession';

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
    expect(msgs[1].id).toBe(msgs[0].id + 1);
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
