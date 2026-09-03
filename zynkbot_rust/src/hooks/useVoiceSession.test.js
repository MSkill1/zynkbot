import { parseVoiceCommand, shouldSpeakReply } from './useVoiceSession';

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
