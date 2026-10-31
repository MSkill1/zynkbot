package ai.containai.zynkbot

import ai.containai.zynkbot.OpenAiRealtime.Event
import ai.containai.zynkbot.OpenAiRealtime.Resampler
import ai.containai.zynkbot.OpenAiRealtime.Transcript
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Base64
import kotlin.math.abs
import kotlin.math.sin

/** The pure parts of the live OpenAI dictation: event parsing, the running transcript,
 *  the resampler and the messages we send. The socket itself needs a key and a server. */
class OpenAiRealtimeTest {
    private fun parse(s: String) = OpenAiRealtime.parseEvent(s)

    @Test fun `parses delta, completed, committed, failed, error and unknown events`() {
        assertEquals(Event.Delta("Hello "), parse("""{"type":"conversation.item.input_audio_transcription.delta","item_id":"i1","delta":"Hello "}"""))
        assertEquals(Event.Completed("Hello there."), parse("""{"type":"conversation.item.input_audio_transcription.completed","item_id":"i1","transcript":"Hello there."}"""))
        assertEquals(Event.Committed, parse("""{"type":"input_audio_buffer.committed","item_id":"i1"}"""))
        assertEquals(Event.Failed("nope"), parse("""{"type":"conversation.item.input_audio_transcription.failed","error":{"message":"nope"}}"""))
        assertEquals(
            Event.Error("input_audio_buffer_commit_empty", "buffer too small"),
            parse("""{"type":"error","error":{"type":"invalid_request_error","code":"input_audio_buffer_commit_empty","message":"buffer too small"}}""")
        )
        assertEquals(Event.Error(null, "unknown error"), parse("""{"type":"error"}"""))
        assertEquals(Event.Other("session.created"), parse("""{"type":"session.created","session":{"id":"s1"}}"""))
        val bad = parse("not json")
        assertTrue(bad is Event.Error && bad.code == "bad_json")
        assertTrue((parse("[1,2]") as Event.Error).code == "bad_json")
    }

    @Test fun `accepts the other event-name family and escaped text`() {
        // Should the GA server name them differently, the suffix still decides.
        assertEquals(Event.Delta("a"), parse("""{"type":"conversation.item.audio_transcription.delta","delta":"a"}"""))
        assertEquals(Event.Completed("done"), parse("""{"type":"transcription.done","text":"done"}"""))
        assertEquals(Event.Delta("say \"hi\"\nnow – é"), parse("""{"type":"conversation.item.input_audio_transcription.delta","delta":"say \"hi\"\nnow – é"}"""))
    }

    @Test fun `transcript joins segments and tracks pending`() {
        val t = Transcript()
        assertTrue(t.apply(Event.Delta("Hel")))
        assertTrue(t.apply(Event.Delta("lo")))
        assertEquals("Hello", t.text())
        assertFalse(t.apply(Event.Committed))
        assertEquals(1, t.pending)
        assertTrue(t.apply(Event.Completed("Hello.")))
        assertEquals(0, t.pending)
        assertEquals("Hello.", t.text())
        t.apply(Event.Delta(" there"))
        assertEquals("Hello. there", t.text())
        t.apply(Event.Committed)
        t.apply(Event.Completed("There it is."))
        assertEquals("Hello. There it is.", t.text())
        assertFalse(t.apply(Event.Error(null, "x")))
        assertFalse(t.apply(Event.Delta("")))
        assertFalse(t.apply(Event.Other("session.updated")))
        // A failed segment drops its partial words and its pending slot.
        t.apply(Event.Delta("junk"))
        t.apply(Event.Committed)
        assertTrue(t.apply(Event.Failed("bad")))
        assertEquals("Hello. There it is.", t.text())
        assertEquals(0, t.pending)
        // A completed with no committed before it never goes negative.
        t.apply(Event.Completed("More."))
        assertEquals(0, t.pending)
    }

    @Test fun `resampler halves, passes through and interpolates up`() {
        val ramp = ShortArray(100) { (it * 10).toShort() }
        val out = Resampler(48_000, 24_000).process(ramp)
        assertEquals(50, out.size)
        assertArrayEquals(shortArrayOf(0, 20, 40, 60, 80), out.copyOf(5))

        assertArrayEquals(ramp, Resampler(16_000, 16_000).process(ramp))
        assertArrayEquals(ramp.copyOf(7), Resampler(16_000, 16_000).process(ramp, 7))
        assertEquals(0, Resampler(16_000, 24_000).process(ShortArray(0)).size)

        // Upsampling 16 → 24 kHz interpolates between neighbours and keeps its place
        // across chunks.
        val up = Resampler(16_000, 24_000)
        assertArrayEquals(shortArrayOf(0, 20, 40), up.process(shortArrayOf(0, 30, 60)))
        assertArrayEquals(shortArrayOf(60, 80, 100), up.process(shortArrayOf(90, 120)))
    }

    @Test fun `resampler chunks match the whole clip at the microphone rate`() {
        val signal = ShortArray(4800) { (sin(it * 0.05) * 10000.0).toInt().toShort() }
        val expected = Resampler(MicStream.SAMPLE_RATE, OpenAiRealtime.TARGET_RATE).process(signal)
        val chunked = Resampler(MicStream.SAMPLE_RATE, OpenAiRealtime.TARGET_RATE)
        val got = ArrayList<Short>()
        var off = 0
        while (off < signal.size) {
            val n = minOf(MicStream.CHUNK_SAMPLES, signal.size - off)
            got.addAll(chunked.process(signal.copyOfRange(off, off + n), n).toList())
            off += n
        }
        assertArrayEquals(expected, got.toShortArray())
        val ratio = got.size.toDouble() / signal.size
        assertTrue("ratio $ratio", abs(ratio - 1.5) < 0.01)
        // Full-scale input never leaves the 16-bit range.
        val loud = ShortArray(64) { if (it % 2 == 0) Short.MAX_VALUE else Short.MIN_VALUE }
        Resampler(16_000, 24_000).process(loud)
    }

    @Test fun `append event carries base64 pcm16 and the session update asks for pcm at 24 kHz`() {
        val pcm = OpenAiDictation.toBytes(shortArrayOf(1, -1), 2)
        val ev = MiniJson.parse(OpenAiRealtime.appendEvent(pcm)) as Map<*, *>
        assertEquals("input_audio_buffer.append", ev["type"])
        assertArrayEquals(byteArrayOf(1, 0, 0xFF.toByte(), 0xFF.toByte()), Base64.getDecoder().decode(ev["audio"] as String))
        val part = MiniJson.parse(OpenAiRealtime.appendEvent(pcm, 2)) as Map<*, *>
        assertArrayEquals(byteArrayOf(1, 0), Base64.getDecoder().decode(part["audio"] as String))

        val u = MiniJson.parse(OpenAiRealtime.sessionUpdate()) as Map<*, *>
        assertEquals("session.update", u["type"])
        assertEquals("transcription", MiniJson.string(u, "session", "type"))
        assertEquals("audio/pcm", MiniJson.string(u, "session", "audio", "input", "format", "type"))
        val format = ((((u["session"] as Map<*, *>)["audio"] as Map<*, *>)["input"] as Map<*, *>)["format"] as Map<*, *>)
        assertEquals(24000.0, format["rate"])
        assertEquals(OpenAiDictation.MODEL, MiniJson.string(u, "session", "audio", "input", "transcription", "model"))
        assertEquals("en", MiniJson.string(u, "session", "audio", "input", "transcription", "language"))
        val vad = (((u["session"] as Map<*, *>)["audio"] as Map<*, *>)["input"] as Map<*, *>)["turn_detection"] as Map<*, *>
        assertEquals("server_vad", vad["type"])
        assertEquals(700.0, vad["silence_duration_ms"])
        assertEquals(OpenAiRealtime.TARGET_RATE / 10 * 2, OpenAiRealtime.SEND_CHUNK_BYTES)
    }

    @Test fun `mini json reads nested objects, arrays, numbers, literals and escapes`() {
        val v = MiniJson.parse("""{"a": [1, -2.5, 3e2, true, false, null, "x"], "b": {"c": {"d": "deep"}}, "e": "tab\there \"q\" \\ \/ A"}""") as Map<*, *>
        assertEquals(listOf(1.0, -2.5, 300.0, true, false, null, "x"), v["a"])
        assertEquals("deep", MiniJson.string(v, "b", "c", "d"))
        assertEquals(null, MiniJson.string(v, "b", "zzz"))
        assertEquals(null, MiniJson.string(v, "a", "c"))
        assertEquals("tab\there \"q\" \\ / A", v["e"])
        assertEquals(emptyMap<String, Any?>(), MiniJson.parse("{}"))
        assertEquals(emptyList<Any?>(), MiniJson.parse(" [ ] "))
        for (bad in listOf("", "{", "{\"a\":}", "[1,]", "{\"a\":1} x", "\"open", "{\"a\" 1}", "nul")) {
            try { MiniJson.parse(bad); throw AssertionError("accepted $bad") } catch (_: IllegalArgumentException) {}
        }
    }

    @Test fun `silence guard drops the model's silence phrases on short clips only`() {
        assertTrue(OpenAiDictation.isSilencePhrase("Thank you."))
        assertFalse(OpenAiDictation.isSilencePhrase("Thank you for the ride home"))
    }
}
