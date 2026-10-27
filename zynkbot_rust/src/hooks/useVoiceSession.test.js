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

describe('parseVoiceCommand timers — one misheard word after "timer" (Matt, Pixel, 2026-10-12, KI-093)', () => {
  test('"set time or fifteen minutes" is a 15-minute timer, not a chat message', () => {
    expect(parseVoiceCommand('set time or fifteen minutes')).toEqual({ type: 'timer', seconds: 900 });
  });
  test('"four", "of", "fer" in place of "for"', () => {
    expect(parseVoiceCommand('set timer four fifteen minutes')).toEqual({ type: 'timer', seconds: 900 });
    expect(parseVoiceCommand('set a time of ten minutes')).toEqual({ type: 'timer', seconds: 600 });
    expect(parseVoiceCommand('timer fer thirty seconds')).toEqual({ type: 'timer', seconds: 30 });
    expect(parseVoiceCommand('set timer 4 minutes')).toEqual({ type: 'timer', seconds: 240 });
  });
  test('an ordinary sentence with "time" and a duration is still not a timer', () => {
    expect(parseVoiceCommand('what time is 5 minutes from now')).toBeNull();
    expect(parseVoiceCommand('the time was 10 minutes ago')).toBeNull();
  });
});

describe('parseVoiceCommand timers — spoken fractions (Matt, Pixel, 2026-10-07)', () => {
  test('"two and a half minutes" is 150 seconds, not a chat message', () => {
    expect(parseVoiceCommand('set a timer for two and a half minutes')).toEqual({ type: 'timer', seconds: 150 });
  });
  test('"a minute and a half", "half an hour", "a minute"', () => {
    expect(parseVoiceCommand('set a timer for a minute and a half')).toEqual({ type: 'timer', seconds: 90 });
    expect(parseVoiceCommand('timer for half an hour')).toEqual({ type: 'timer', seconds: 1800 });
    expect(parseVoiceCommand('set a timer for a minute')).toEqual({ type: 'timer', seconds: 60 });
    expect(parseVoiceCommand('set a timer for 2.5 minutes')).toEqual({ type: 'timer', seconds: 150 });
  });
  test('plain numbers still work and ordinary sentences are left alone', () => {
    expect(parseVoiceCommand('set a timer for ten minutes')).toEqual({ type: 'timer', seconds: 600 });
    expect(parseVoiceCommand('a minute ago I saw a hawk')).toBeNull();
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

  test('the sources of an automatic web search ride on the answer, as for a typed search', () => {
    const sources = { query: 'weather miami', results: [{ title: 'NWS Miami', url: 'https://example.org/miami', snippet: '' }], num_results: 1 };
    const msgs = nativeTurnsToMessages([{ ...turn, sources }], 's1');
    expect(msgs[1].web_search_results).toEqual(sources);
    expect(msgs[0].web_search_results).toBeUndefined();
  });

  test('a search with no results, or no search at all, attaches nothing', () => {
    expect(nativeTurnsToMessages([{ ...turn, sources: { query: 'x', results: [] } }], 's1')[1].web_search_results).toBeUndefined();
    expect(nativeTurnsToMessages([turn], 's1')[1].web_search_results).toBeUndefined();
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

describe('own timer and stopwatch commands (2026-10-25)', () => {
  test('"stop timer" and "cancel timer" cancel, never set', () => {
    expect(parseVoiceCommand('stop timer')).toEqual({ type: 'cancel_timer' });
    expect(parseVoiceCommand('cancel the timer')).toEqual({ type: 'cancel_timer' });
    expect(parseVoiceCommand('hey zynk cancel my timer')).toEqual({ type: 'cancel_timer' });
  });
  test('"check timer" and "how much time is left"', () => {
    expect(parseVoiceCommand('check timer')).toEqual({ type: 'check_timer' });
    expect(parseVoiceCommand('how much time is left')).toEqual({ type: 'check_timer' });
  });
  test('stopwatch stop, reset, check', () => {
    expect(parseVoiceCommand('stop stopwatch')).toEqual({ type: 'stopwatch_stop' });
    expect(parseVoiceCommand('reset the stopwatch')).toEqual({ type: 'stopwatch_reset' });
    expect(parseVoiceCommand('check stopwatch')).toEqual({ type: 'check_stopwatch' });
    expect(parseVoiceCommand('check stopwatch time')).toEqual({ type: 'check_stopwatch' });
    expect(parseVoiceCommand('start stopwatch')).toEqual({ type: 'stopwatch' });
  });
  test('a plain "stop" still stops speech, not the timer', () => {
    expect(parseVoiceCommand('stop')).toEqual({ type: 'stop_tts' });
  });
});

describe('dictation shapes from the 2026-10-25 phone test', () => {
  test('"Stop, stopwatch." and "stop, stop, watch" stop the stopwatch', () => {
    expect(parseVoiceCommand('Stop, stopwatch.')).toEqual({ type: 'stopwatch_stop' });
    expect(parseVoiceCommand('Stop, stop, watch.')).toEqual({ type: 'stopwatch_stop' });
  });
  test('"Set an alarm for 1026." is 10:26, "730 pm" is 19:30', () => {
    expect(parseVoiceCommand('Set an alarm for 1026.', new Date(2026, 9, 25, 9, 0))).toEqual({ type: 'alarm', hour: 10, minute: 26 });
    expect(parseVoiceCommand('set an alarm for 730 pm')).toEqual({ type: 'alarm', hour: 19, minute: 30 });
    expect(parseVoiceCommand('set an alarm for 7:30 am')).toEqual({ type: 'alarm', hour: 7, minute: 30 });
  });
  test('cancel alarm', () => {
    expect(parseVoiceCommand('cancel alarm')).toEqual({ type: 'cancel_alarm' });
    expect(parseVoiceCommand('stop the alarm')).toEqual({ type: 'cancel_alarm' });
  });
});

describe('several timers and alarms (2026-10-25)', () => {
  test('cancel a named timer', () => {
    expect(parseVoiceCommand('cancel the 10 minute timer')).toEqual({ type: 'cancel_timer', seconds: 600 });
    expect(parseVoiceCommand('stop the ten minute timer')).toEqual({ type: 'cancel_timer', seconds: 600 });
    expect(parseVoiceCommand('cancel timer')).toEqual({ type: 'cancel_timer' });
  });
  test('cancel a named alarm, check alarms', () => {
    expect(parseVoiceCommand('cancel the 730 alarm')).toEqual({ type: 'cancel_alarm', hour: 7, minute: 30 });
    expect(parseVoiceCommand('cancel the 7:30 pm alarm')).toEqual({ type: 'cancel_alarm', hour: 19, minute: 30 });
    expect(parseVoiceCommand('check alarms')).toEqual({ type: 'check_alarm' });
    expect(parseVoiceCommand('check my alarm')).toEqual({ type: 'check_alarm' });
  });
});

describe('"timer" heard for "alarm" with a clock time (Pixel, 2026-10-25)', () => {
  test('"Set a timer for 7.30 a.m. tomorrow." is the 7:30 alarm', () => {
    expect(parseVoiceCommand('Set a timer for 7.30 a.m. tomorrow.')).toEqual({ type: 'alarm', hour: 7, minute: 30 });
    expect(parseVoiceCommand('set a timer for 7 pm')).toEqual({ type: 'alarm', hour: 19, minute: 0 });
    expect(parseVoiceCommand('set an alarm for 7.30 p.m.')).toEqual({ type: 'alarm', hour: 19, minute: 30 });
  });
  test('a duration is still a timer', () => {
    expect(parseVoiceCommand('set a timer for 7 minutes')).toEqual({ type: 'timer', seconds: 420 });
    expect(parseVoiceCommand('set a timer for 2.5 minutes')).toEqual({ type: 'timer', seconds: 150 });
  });
});

describe('parseVoiceCommand alarms — what went wrong on the Pixel (Matt, 2026-10-26)', () => {
  const at = (h, m) => new Date(2026, 9, 26, h, m);
  test('"set an alarm four eight fifteen pm": a misheard "for" is still an alarm', () => {
    expect(parseVoiceCommand('set an alarm four eight fifteen pm', at(19, 26))).toEqual({ type: 'alarm', hour: 20, minute: 15 });
  });
  test('"alarm 4 15 pm" is a quarter past four, not a filler plus "15 pm"', () => {
    expect(parseVoiceCommand('set an alarm 4 15 pm', at(9, 0))).toEqual({ type: 'alarm', hour: 16, minute: 15 });
  });
  test('no am/pm: the next 8:15 to come, tonight at 7 pm, tomorrow morning at 9 pm', () => {
    expect(parseVoiceCommand('set an alarm for 8:15', at(19, 0))).toEqual({ type: 'alarm', hour: 20, minute: 15 });
    expect(parseVoiceCommand('set an alarm for 8:15', at(21, 0))).toEqual({ type: 'alarm', hour: 8, minute: 15 });
    expect(parseVoiceCommand('wake me up at 6', at(7, 0))).toEqual({ type: 'alarm', hour: 18, minute: 0 });
  });
  test('a spoken am/pm or a 24-hour time is taken as said', () => {
    expect(parseVoiceCommand('set an alarm for 8:15 am', at(19, 0))).toEqual({ type: 'alarm', hour: 8, minute: 15 });
    expect(parseVoiceCommand('set an alarm for 20:15', at(21, 0))).toEqual({ type: 'alarm', hour: 20, minute: 15 });
  });
  test('"set it for 8:15" after cancelling one is an alarm; "set it for 8" alone is not', () => {
    expect(parseVoiceCommand('set it for 8:15', at(19, 0))).toEqual({ type: 'alarm', hour: 20, minute: 15 });
    expect(parseVoiceCommand('set it for eight fifteen pm', at(9, 0))).toEqual({ type: 'alarm', hour: 20, minute: 15 });
    expect(parseVoiceCommand('set it for 8', at(19, 0))).toBeNull();
  });
});
