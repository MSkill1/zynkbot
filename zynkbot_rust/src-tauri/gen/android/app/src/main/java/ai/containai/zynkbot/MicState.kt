package ai.containai.zynkbot

/**
 * Who has the microphone, as one state (2026-10-20 rebuild). Pure Kotlin, no Android
 * types, so it is unit-tested (MicStateTest). WakeWordService feeds it events and
 * carries out the actions it returns; nothing else decides when the stream opens or
 * what consumes it. An event that makes no sense in the current state is refused
 * (returns null, nothing changes) — that is what makes two paths meeting harmless.
 *
 * States:
 *   OFF        Hey Zynk is off: no models, no stream.
 *   LOADING    the page asked for it; the wake models are loading on a thread.
 *   LISTENING  stream open; chunks go to the wake-word detector.
 *   DICTATING  stream open; chunks go to a Dictation (after a wake, the assistant
 *              session, or the app's mic button).
 *   TURN       a turn is in progress (chime, thinking, speaking): stream open, chunks
 *              dropped, so the detector cannot hear the reply.
 *   PAUSED     another app is recording: stream closed, models kept.
 */
class MicState(private val log: (String) -> Unit = {}) {
    enum class State { OFF, LOADING, LISTENING, DICTATING, TURN, PAUSED }

    enum class Action { LOAD_MODELS, UNLOAD_MODELS, OPEN_STREAM, CLOSE_STREAM, SINK_DETECTOR, SINK_DICTATION, SINK_NONE, ABORT_DICTATION }

    sealed class Event(val name: String) {
        object PageOn : Event("page on")
        object PageOff : Event("page off")
        object ModelsLoaded : Event("models loaded")
        object ModelsFailed : Event("models failed")
        object Wake : Event("wake")
        /** ownsTurn: the caller ends the turn itself with TurnEnded (a wake, the
         *  assistant session). false: the listener comes back as soon as the
         *  dictation ends (the app's mic button). */
        class DictationStarted(val ownsTurn: Boolean) : Event("dictation started")
        object DictationEnded : Event("dictation ended")
        object TurnBegun : Event("turn begun")
        object TurnEnded : Event("turn ended")
        object ForeignStarted : Event("foreign recording started")
        object ForeignEnded : Event("foreign recording ended")
        object StreamLost : Event("stream lost")
        object Expired : Event("watchdog expired")
    }

    @Volatile var state = State.OFF
        private set
    @Volatile var foreignActive = false
        private set
    private var dictationOwnsTurn = false

    /** Apply one event. The actions to carry out, in order; null = refused. */
    @Synchronized
    fun on(event: Event): List<Action>? {
        val from = state
        val actions = ArrayList<Action>(4)
        val to: State = when (event) {
            Event.PageOn -> if (from == State.OFF) { actions += Action.LOAD_MODELS; State.LOADING } else return refuse(event)
            Event.PageOff -> {
                if (from == State.OFF) return refuse(event)
                if (from == State.DICTATING) actions += Action.ABORT_DICTATION
                if (streamOpenIn(from)) actions += Action.CLOSE_STREAM
                actions += Action.UNLOAD_MODELS
                State.OFF
            }
            Event.ModelsLoaded -> if (from == State.LOADING) idle(actions, streamOpen = false) else return refuse(event)
            Event.ModelsFailed -> if (from == State.LOADING) State.OFF else return refuse(event)
            Event.Wake -> if (from == State.LISTENING) { actions += Action.SINK_NONE; State.TURN } else return refuse(event)
            is Event.DictationStarted -> when (from) {
                State.LISTENING, State.TURN -> { dictationOwnsTurn = event.ownsTurn; actions += Action.SINK_DICTATION; State.DICTATING }
                // Another app records, but the user asked us to listen: open the stream for it.
                State.PAUSED -> { dictationOwnsTurn = event.ownsTurn; actions += Action.OPEN_STREAM; actions += Action.SINK_DICTATION; State.DICTATING }
                else -> return refuse(event)
            }
            Event.DictationEnded -> if (from == State.DICTATING) {
                if (dictationOwnsTurn) { actions += Action.SINK_NONE; State.TURN } else idle(actions, streamOpen = true)
            } else return refuse(event)
            Event.TurnBegun -> if (from == State.LISTENING) { actions += Action.SINK_NONE; State.TURN } else return refuse(event)
            Event.TurnEnded -> if (from == State.TURN) idle(actions, streamOpen = true) else return refuse(event)
            Event.ForeignStarted -> {
                foreignActive = true
                when (from) {
                    State.DICTATING -> { actions += Action.ABORT_DICTATION; actions += Action.CLOSE_STREAM; State.PAUSED }
                    State.LISTENING, State.TURN -> { actions += Action.CLOSE_STREAM; State.PAUSED }
                    else -> from   // OFF, LOADING, PAUSED: remembered, nothing to do now
                }
            }
            Event.ForeignEnded -> {
                foreignActive = false
                if (from == State.PAUSED) { actions += Action.OPEN_STREAM; actions += Action.SINK_DETECTOR; State.LISTENING } else from
            }
            Event.StreamLost -> {
                if (!streamOpenIn(from)) return refuse(event)
                if (from == State.DICTATING) actions += Action.ABORT_DICTATION
                actions += Action.CLOSE_STREAM
                when {
                    foreignActive -> State.PAUSED
                    from == State.TURN -> { actions += Action.OPEN_STREAM; actions += Action.SINK_NONE; State.TURN }
                    else -> { actions += Action.OPEN_STREAM; actions += Action.SINK_DETECTOR; State.LISTENING }
                }
            }
            Event.Expired -> when (from) {
                State.DICTATING -> { actions += Action.ABORT_DICTATION; idle(actions, streamOpen = true) }
                State.TURN -> idle(actions, streamOpen = true)
                else -> return refuse(event)
            }
        }
        state = to
        log("$from → $to (${event.name})" + if (actions.isEmpty()) "" else ": " + actions.joinToString(" ") { it.name.lowercase() })
        return actions
    }

    private fun streamOpenIn(s: State) = s == State.LISTENING || s == State.DICTATING || s == State.TURN

    /** Where to go when nothing is happening: listening, or paused while another app records. */
    private fun idle(actions: MutableList<Action>, streamOpen: Boolean): State =
        if (foreignActive) {
            if (streamOpen) actions += Action.CLOSE_STREAM
            State.PAUSED
        } else {
            if (!streamOpen) actions += Action.OPEN_STREAM
            actions += Action.SINK_DETECTOR
            State.LISTENING
        }

    private fun refuse(event: Event): List<Action>? {
        log("${event.name} ignored in $state")
        return null
    }
}
