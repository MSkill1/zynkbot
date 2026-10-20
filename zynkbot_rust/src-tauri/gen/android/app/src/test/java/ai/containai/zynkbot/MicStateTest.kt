package ai.containai.zynkbot

import ai.containai.zynkbot.MicState.Action.*
import ai.containai.zynkbot.MicState.Event
import ai.containai.zynkbot.MicState.State.*
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The microphone state machine, event by event. Each test is one real sequence. */
class MicStateTest {
    private fun listening(): MicState {
        val m = MicState()
        assertEquals(listOf(LOAD_MODELS), m.on(Event.PageOn))
        assertEquals(LOADING, m.state)
        assertEquals(listOf(OPEN_STREAM, SINK_DETECTOR), m.on(Event.ModelsLoaded))
        assertEquals(LISTENING, m.state)
        return m
    }

    @Test fun `toggle on loads then listens`() { listening() }

    @Test fun `a second start while on is refused and changes nothing`() {
        val m = listening()
        assertNull(m.on(Event.PageOn))
        assertEquals(LISTENING, m.state)
        m.on(Event.Wake)
        assertNull(m.on(Event.PageOn))
        assertEquals(TURN, m.state)
    }

    @Test fun `a wake runs a whole turn and comes back listening`() {
        val m = listening()
        assertEquals(listOf(SINK_NONE), m.on(Event.Wake))                       // chime
        assertEquals(listOf(SINK_DICTATION), m.on(Event.DictationStarted(ownsTurn = true)))
        assertEquals(listOf(SINK_NONE), m.on(Event.DictationEnded))             // thinking, speaking
        assertEquals(TURN, m.state)
        assertNull(m.on(Event.TurnBegun))                                        // the answerer's own begin is a no-op inside a turn
        assertEquals(listOf(SINK_DETECTOR), m.on(Event.TurnEnded))
        assertEquals(LISTENING, m.state)
        assertNull(m.on(Event.TurnEnded))                                        // a second end (two paths meeting) does nothing
    }

    @Test fun `the mic button dictates without a turn`() {
        val m = listening()
        assertEquals(listOf(SINK_DICTATION), m.on(Event.DictationStarted(ownsTurn = false)))
        assertEquals(listOf(SINK_DETECTOR), m.on(Event.DictationEnded))
        assertEquals(LISTENING, m.state)
    }

    @Test fun `a spoken confirmation outside a turn is its own turn`() {
        val m = listening()
        assertEquals(listOf(SINK_NONE), m.on(Event.TurnBegun))
        assertEquals(listOf(SINK_DETECTOR), m.on(Event.TurnEnded))
    }

    @Test fun `another app recording pauses and the end of it resumes`() {
        val m = listening()
        m.on(Event.Wake); m.on(Event.DictationStarted(ownsTurn = true))
        assertEquals(listOf(ABORT_DICTATION, CLOSE_STREAM), m.on(Event.ForeignStarted))
        assertEquals(PAUSED, m.state)
        assertNull(m.on(Event.TurnEnded))                                        // the aborted owner's late end is ignored
        assertEquals(listOf(OPEN_STREAM, SINK_DETECTOR), m.on(Event.ForeignEnded))
        assertEquals(LISTENING, m.state)
    }

    @Test fun `a foreign recording seen while loading leaves the stream closed until it ends`() {
        val m = MicState()
        m.on(Event.PageOn)
        assertEquals(emptyList<MicState.Action>(), m.on(Event.ForeignStarted))
        assertEquals(LOADING, m.state)
        assertEquals(emptyList<MicState.Action>(), m.on(Event.ModelsLoaded))
        assertEquals(PAUSED, m.state)
        assertEquals(listOf(OPEN_STREAM, SINK_DETECTOR), m.on(Event.ForeignEnded))
    }

    @Test fun `a turn that ends while another app records goes to paused`() {
        val m = listening()
        m.on(Event.Wake)
        assertEquals(listOf(CLOSE_STREAM), m.on(Event.ForeignStarted))
        assertEquals(PAUSED, m.state)
        m.on(Event.ForeignEnded)
        assertEquals(LISTENING, m.state)
    }

    @Test fun `toggle off from the middle of a dictation tears everything down`() {
        val m = listening()
        m.on(Event.Wake); m.on(Event.DictationStarted(ownsTurn = true))
        assertEquals(listOf(ABORT_DICTATION, CLOSE_STREAM, UNLOAD_MODELS), m.on(Event.PageOff))
        assertEquals(OFF, m.state)
        assertNull(m.on(Event.ModelsLoaded))                                     // a loader finishing late is ignored
        assertNull(m.on(Event.DictationEnded))
    }

    @Test fun `toggle off while loading`() {
        val m = MicState()
        m.on(Event.PageOn)
        assertEquals(listOf(UNLOAD_MODELS), m.on(Event.PageOff))
        assertEquals(OFF, m.state)
    }

    @Test fun `a lost stream reopens`() {
        val m = listening()
        assertEquals(listOf(CLOSE_STREAM, OPEN_STREAM, SINK_DETECTOR), m.on(Event.StreamLost))
        assertEquals(LISTENING, m.state)
        m.on(Event.Wake)
        assertEquals(listOf(CLOSE_STREAM, OPEN_STREAM, SINK_NONE), m.on(Event.StreamLost))
        assertEquals(TURN, m.state)
    }

    @Test fun `the watchdog closes a turn or a dictation that never ended`() {
        val m = listening()
        m.on(Event.Wake)
        assertEquals(listOf(SINK_DETECTOR), m.on(Event.Expired))
        assertEquals(LISTENING, m.state)
        m.on(Event.DictationStarted(ownsTurn = true))
        assertEquals(listOf(ABORT_DICTATION, SINK_DETECTOR), m.on(Event.Expired))
        assertEquals(LISTENING, m.state)
        assertNull(m.on(Event.Expired))
    }

    @Test fun `a dictation requested while paused opens the stream for it`() {
        val m = listening()
        m.on(Event.ForeignStarted)
        assertEquals(listOf(OPEN_STREAM, SINK_DICTATION), m.on(Event.DictationStarted(ownsTurn = false)))
        assertEquals(listOf(CLOSE_STREAM), m.on(Event.DictationEnded))          // still paused: the other app is still recording
        assertEquals(PAUSED, m.state)
    }
}
