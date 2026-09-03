import { parseVoiceCommand } from './useVoiceSession';

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
