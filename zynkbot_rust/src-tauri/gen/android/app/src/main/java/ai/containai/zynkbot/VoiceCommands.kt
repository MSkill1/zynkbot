package ai.containai.zynkbot

import android.content.Context
import android.content.Intent
import android.provider.AlarmClock
import android.util.Log

/**
 * Voice commands the assistant handles itself, before the model ever sees the
 * transcript: timers, alarms, the stopwatch. A port of parseVoiceCommand() in
 * useVoiceSession.js, so the hands-free (native) path behaves the way the in-app path
 * did before the assistant refactor. Without it (KI-027) "set a timer for ten minutes"
 * reached the model, which has no clock and either invented a confirmation or said it
 * couldn't. Vosk output is lowercase, unpunctuated, with numbers as words ("ten
 * minutes"), hence normalizeNumbers().
 */
object VoiceCommands {
    private const val TAG = "VoiceCommands"
    const val FAILED_LINE = "I couldn't reach the clock app to do that."

    sealed class Cmd {
        data class Timer(val seconds: Int) : Cmd()
        data class Alarm(val hour: Int, val minute: Int) : Cmd()
        object Stopwatch : Cmd()
        // Zynkbot's own timer and stopwatch (ZynkTimer, 2026-10-25). Words fixed by Matt:
        // "stop timer" / "cancel timer", "check timer", "check stopwatch [time]".
        data class CancelTimer(val seconds: Int = 0) : Cmd()   // 0: the only one, or ask which
        object CheckTimer : Cmd()
        object StopwatchStop : Cmd()
        object StopwatchReset : Cmd()
        object CheckStopwatch : Cmd()
        data class CancelAlarm(val hour: Int = -1, val minute: Int = 0) : Cmd()   // hour < 0: the only one, or ask which
        object CheckAlarm : Cmd()
    }

    private val NUMBER_WORDS = mapOf(
        "zero" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5,
        "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10,
        "eleven" to 11, "twelve" to 12, "thirteen" to 13, "fourteen" to 14, "fifteen" to 15,
        "sixteen" to 16, "seventeen" to 17, "eighteen" to 18, "nineteen" to 19,
        "twenty" to 20, "thirty" to 30, "forty" to 40, "fifty" to 50, "sixty" to 60,
        "seventy" to 70, "eighty" to 80, "ninety" to 90, "hundred" to 100,
    )
    private val TENS_ONES = Regex(
        "\\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]?(one|two|three|four|five|six|seven|eight|nine)\\b",
        RegexOption.IGNORE_CASE
    )
    private val SINGLE = Regex("\\b(" + NUMBER_WORDS.keys.joinToString("|") + ")\\b", RegexOption.IGNORE_CASE)

    fun normalizeNumbers(text: String): String {
        var t = TENS_ONES.replace(text) { m ->
            ((NUMBER_WORDS[m.groupValues[1].lowercase()] ?: 0) + (NUMBER_WORDS[m.groupValues[2].lowercase()] ?: 0)).toString()
        }
        t = SINGLE.replace(t) { m -> NUMBER_WORDS[m.value.lowercase()]?.toString() ?: m.value }
        return normalizeFractions(t)
    }

    /** "two and a half minutes", "a minute and a half", "half an hour", "a minute":
     *  the number words alone left "2 and a half minutes", which no timer pattern
     *  matched, so the request went to the model as chat and the model said "timer
     *  set" with no timer (Matt, Pixel, 2026-10-07; same shape as KI-027). Mirrors
     *  normalizeFractions() in useVoiceSession.js. */
    fun normalizeFractions(text: String): String {
        var t = Regex("(\\d+)\\s+and\\s+a\\s+half\\b", RegexOption.IGNORE_CASE).replace(text) { m -> m.groupValues[1] + ".5" }
        t = Regex("(\\d+)\\s+and\\s+a\\s+quarter\\b", RegexOption.IGNORE_CASE).replace(t) { m -> m.groupValues[1] + ".25" }
        t = Regex("\\bhalf\\s+an?\\s+(hour|minute|second)", RegexOption.IGNORE_CASE).replace(t) { m -> "0.5 " + m.groupValues[1] }
        t = Regex("\\ban?\\s+(hour|minute|second)\\s+and\\s+a\\s+half\\b", RegexOption.IGNORE_CASE).replace(t) { m -> "1.5 " + m.groupValues[1] }
        t = Regex("\\ban?\\s+(hour|minute|second)\\b", RegexOption.IGNORE_CASE).replace(t) { m -> "1 " + m.groupValues[1] }
        return t
    }

    private const val UNIT = "(hours?|hrs?|minutes?|mins?|seconds?|secs?)"
    private val TIMER_RES = listOf(
        // "time" too: dictation drops the r (tester, #29). One misheard word may sit
        // between "timer" and the number: Vosk heard "set timer for fifteen minutes"
        // as "set time or fifteen minutes" (Matt, Pixel, 2026-10-12, KI-093).
        Regex("(?:set\\s+(?:a\\s+)?)?time(?:r)?\\s+(?:(?:for|four|4|or|of|fer)\\s+)?(\\d+(?:\\.\\d+)?)\\s*$UNIT"),
        Regex("(\\d+(?:\\.\\d+)?)\\s*$UNIT\\s+timer"),
    )
    // (\\d{1,4}): dictation gives "1026" for ten twenty-six (Pixel, 2026-10-25); split below.
    private val ALARM_RE = Regex(
        "(?:set\\s+(?:an?\\s+)?alarm\\s+(?:for|at)|alarm\\s+(?:for|at)|wake\\s+me\\s+up\\s+at)\\s+(\\d{1,4})(?:[:\\s](\\d{1,2}))?\\s*(am|pm)?"
    )
    // "set a timer for 7:30 am": dictation hears "alarm" as "timer"; a clock time (a colon,
    // or am/pm) after "timer" is an alarm, never a duration.
    private val TIMER_AS_ALARM_RE = Regex("time(?:r)?\\s+(?:for|at)\\s+(\\d{1,2})(?::(\\d{2}))?\\s*(am|pm)?(?:\\s+tomorrow)?\\b")
    private val STOPWATCH_RE = Regex("(?:start|begin)\\s+(?:the\\s+|a\\s+)?stopwatch|^stopwatch$")
    private val CANCEL_TIMER_RE = Regex("\\b(?:stop|cancel)\\s+(?:the\\s+|my\\s+)?timer\\b")
    private val CHECK_TIMER_RE = Regex("\\bcheck\\s+(?:the\\s+|my\\s+)?timer\\b|how\\s+much\\s+time\\s+is\\s+left|how\\s+long\\s+(?:is\\s+)?left")
    private val STOPWATCH_STOP_RE = Regex("\\b(?:stop|pause)\\s+(?:the\\s+)?stopwatch\\b")
    private val STOPWATCH_RESET_RE = Regex("\\b(?:reset|clear)\\s+(?:the\\s+)?stopwatch\\b")
    private val CHECK_STOPWATCH_RE = Regex("\\bcheck\\s+(?:the\\s+)?stopwatch(?:\\s+time)?\\b")
    private val CANCEL_ALARM_RE = Regex("\\b(?:cancel|stop|delete)\\s+(?:the\\s+|my\\s+)?alarm\\b")
    private val CANCEL_ALARM_AT_RE = Regex("\\b(?:cancel|stop|delete)\\s+(?:the\\s+|my\\s+)?(\\d{1,4})(?:[:\\s](\\d{1,2}))?\\s*(am|pm)?\\s+alarm\\b")
    private val CANCEL_TIMER_LEN_RE = Regex("\\b(?:stop|cancel)\\s+(?:the\\s+|my\\s+)?(\\d+(?:\\.\\d+)?)\\s*$UNIT\\s+timer\\b")
    private val CHECK_ALARM_RE = Regex("\\bcheck\\s+(?:the\\s+|my\\s+)?alarms?\\b|what\\s+alarms")

    private fun toSeconds(value: String, unit: String): Int {
        val v = value.toDoubleOrNull() ?: return 0
        val secs = when { unit.startsWith("hour") || unit.startsWith("hr") -> v * 3600; unit.startsWith("min") -> v * 60; else -> v }
        return Math.round(secs).toInt()
    }
    /** "1026" -> 10:26, "730" -> 7:30, "7" + "30" + "pm" -> 19:30; null when not a clock time. */
    private fun clockFrom(d: String, m: String, ampm: String): Pair<Int, Int>? {
        var digits = d; var minuteText = m
        if (digits.length >= 3 && minuteText.isEmpty()) { minuteText = digits.takeLast(2); digits = digits.dropLast(2) }
        var hour = digits.toIntOrNull() ?: return null
        val minute = minuteText.ifEmpty { "0" }.toIntOrNull() ?: return null
        if (hour > 23 || minute > 59) return null
        if (ampm == "pm" && hour != 12) hour += 12
        if (ampm == "am" && hour == 12) hour = 0
        return Pair(hour, minute)
    }

    fun parse(text: String): Cmd? {
        // Dictation with punctuation heard "Stop, stopwatch." and "stop watch" as two
        // words (Pixel, 2026-10-25): punctuation becomes spaces, "stop watch" one word.
        // "7.30 a.m." (Pixel dictation, 2026-10-25): a.m./p.m. become am/pm, 7.30 becomes 7:30.
        val cleaned = text.lowercase()
            .replace(Regex("\\ba\\.m\\.?"), "am").replace(Regex("\\bp\\.m\\.?"), "pm")
            .replace(Regex("\\b(\\d{1,2})\\.(\\d{2})(?!\\d)"), "$1:$2")
            .replace(Regex("[,.!?;:](?!\\d)"), " ").replace(Regex("\\bstop\\s+watch\\b"), "stopwatch").replace(Regex("\\s+"), " ").trim()
        val t = normalizeNumbers(cleaned)
        if (CHECK_ALARM_RE.containsMatchIn(t)) return Cmd.CheckAlarm
        CANCEL_ALARM_AT_RE.find(t)?.let { m -> clockFrom(m.groupValues[1], m.groupValues[2], m.groupValues[3])?.let { return Cmd.CancelAlarm(it.first, it.second) } }
        if (CANCEL_ALARM_RE.containsMatchIn(t)) return Cmd.CancelAlarm()
        // Control words first: "stop timer" must never read as a timer request.
        CANCEL_TIMER_LEN_RE.find(t)?.let { m -> return Cmd.CancelTimer(toSeconds(m.groupValues[1], m.groupValues[2])) }
        if (CANCEL_TIMER_RE.containsMatchIn(t)) return Cmd.CancelTimer()
        if (CHECK_TIMER_RE.containsMatchIn(t)) return Cmd.CheckTimer
        if (STOPWATCH_STOP_RE.containsMatchIn(t)) return Cmd.StopwatchStop
        if (STOPWATCH_RESET_RE.containsMatchIn(t)) return Cmd.StopwatchReset
        if (CHECK_STOPWATCH_RE.containsMatchIn(t)) return Cmd.CheckStopwatch
        TIMER_AS_ALARM_RE.find(t)?.let { m ->
            if (m.groupValues[2].isNotEmpty() || m.groupValues[3].isNotEmpty()) {
                clockFrom(m.groupValues[1], m.groupValues[2], m.groupValues[3])?.let { return Cmd.Alarm(it.first, it.second) }
            }
        }
        for (re in TIMER_RES) {
            val m = re.find(t) ?: continue
            val secs = toSeconds(m.groupValues[1], m.groupValues[2])
            if (secs > 0) return Cmd.Timer(secs)
        }
        ALARM_RE.find(t)?.let { m ->
            val hm = clockFrom(m.groupValues[1], m.groupValues[2], m.groupValues[3]) ?: return null
            return Cmd.Alarm(hm.first, hm.second)
        }
        if (STOPWATCH_RE.containsMatchIn(t)) return Cmd.Stopwatch
        return null
    }

    /** Hands the command to the clock app. Returns true only if the clock app accepted
     *  the intent, so the spoken confirmation is never a lie. EXTRA_SKIP_UI: hands-free
     *  means no clock screen popping over the lock screen; the confirmation is spoken. */
    /** Set when the clock app took a timer/alarm because exact alarms are not allowed. */
    @Volatile private var viaClockApp = false

    fun execute(context: Context, cmd: Cmd): Boolean {
        viaClockApp = false
        when (cmd) {
            is Cmd.Timer -> if (ZynkTimer.setTimer(context, cmd.seconds)) return true
            is Cmd.Alarm -> if (ZynkTimer.setAlarm(context, cmd.hour, cmd.minute)) return true
            Cmd.Stopwatch -> { ZynkTimer.startStopwatch(context); return true }
            Cmd.StopwatchStop -> { ZynkTimer.stopStopwatch(context); return true }
            Cmd.StopwatchReset -> { ZynkTimer.resetStopwatch(context); return true }
            is Cmd.CancelTimer, Cmd.CheckTimer, Cmd.CheckStopwatch, is Cmd.CancelAlarm, Cmd.CheckAlarm -> return true  // answered in confirmation()
        }
        // Exact alarms not allowed on this phone: the clock app, as before 2026-10-25.
        viaClockApp = true
        val intent = when (cmd) {
            is Cmd.Timer -> Intent(AlarmClock.ACTION_SET_TIMER).apply {
                putExtra(AlarmClock.EXTRA_LENGTH, cmd.seconds)
                putExtra(AlarmClock.EXTRA_MESSAGE, "Zynkbot")
                putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            }
            is Cmd.Alarm -> Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_HOUR, cmd.hour)
                putExtra(AlarmClock.EXTRA_MINUTES, cmd.minute)
                putExtra(AlarmClock.EXTRA_MESSAGE, "Zynkbot")
                putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            }
            else -> return true
        }
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
        return try {
            context.startActivity(intent)
            true
        } catch (e: Exception) {
            Log.w(TAG, "${cmd::class.simpleName} failed: ${e.message}")
            // There is no public Android intent for a stopwatch (timers and alarms have
            // one; the stopwatch action above is honoured only by some clock apps — the
            // OnePlus clock ignored it, 2026-09-07). Open the clock app itself instead,
            // found through the alarm intent it must handle, so the user lands one tap away.
            false
        }
    }

    /** What to say once the command is done. Plain spoken words; state read from ZynkTimer. */
    fun confirmation(context: Context, cmd: Cmd): String = when (cmd) {
        is Cmd.Timer -> "Timer set for ${ZynkTimer.durationWords(cmd.seconds)}." + if (viaClockApp) " In the clock app, since Zynkbot is not allowed its own alarms yet." else ""
        is Cmd.Alarm -> {
            val h12 = if (cmd.hour % 12 == 0) 12 else cmd.hour % 12
            val ampm = if (cmd.hour < 12) "AM" else "PM"
            val minutes = if (cmd.minute > 0) " " + cmd.minute.toString().padStart(2, '0') else ""
            "Alarm set for $h12$minutes $ampm." + if (viaClockApp) " In the clock app, since Zynkbot is not allowed its own alarms yet." else ""
        }
        is Cmd.CancelTimer -> {
            val list = ZynkTimer.timers(context)
            when {
                list.isEmpty() -> "No timer is running."
                cmd.seconds > 0 -> ZynkTimer.cancelTimer(context, cmd.seconds)?.let { "The ${ZynkTimer.durationWords(it.totalSeconds)} timer is cancelled." } ?: "There is no ${ZynkTimer.durationWords(cmd.seconds)} timer."
                list.size == 1 -> { ZynkTimer.cancelTimer(context); "Timer cancelled." }
                else -> "You have ${list.size} timers: ${ZynkTimer.joinWords(list.map { ZynkTimer.durationWords(it.totalSeconds) })}. Say cancel the ${ZynkTimer.durationWords(list.first().totalSeconds)} timer."
            }
        }
        Cmd.CheckTimer -> {
            val list = ZynkTimer.timers(context)
            when (list.size) {
                0 -> "No timer is running."
                1 -> "${ZynkTimer.durationWords(ZynkTimer.secondsLeft(list[0]))} left."
                else -> list.joinToString(" ") { "${ZynkTimer.durationWords(it.totalSeconds)} timer: ${ZynkTimer.durationWords(ZynkTimer.secondsLeft(it))} left." }
            }
        }
        is Cmd.CancelAlarm -> {
            val list = ZynkTimer.alarms(context)
            when {
                list.isEmpty() -> "No alarm is set."
                cmd.hour >= 0 -> ZynkTimer.cancelAlarm(context, cmd.hour, cmd.minute)?.let { "The ${ZynkTimer.clockWords(it.atMs)} alarm is cancelled." } ?: "There is no alarm at that time."
                list.size == 1 -> { ZynkTimer.cancelAlarm(context); "Alarm cancelled." }
                else -> "You have ${list.size} alarms: ${ZynkTimer.joinWords(list.map { ZynkTimer.clockWords(it.atMs) })}. Say cancel the ${ZynkTimer.clockWords(list.first().atMs)} alarm."
            }
        }
        Cmd.CheckAlarm -> {
            val list = ZynkTimer.alarms(context)
            when (list.size) {
                0 -> "No alarm is set."
                1 -> "One alarm, at ${ZynkTimer.clockWords(list[0].atMs)}."
                else -> "${list.size} alarms: ${ZynkTimer.joinWords(list.map { ZynkTimer.clockWords(it.atMs) })}."
            }
        }
        Cmd.Stopwatch -> "Stopwatch started."
        Cmd.StopwatchStop -> "Stopwatch stopped at ${ZynkTimer.durationWords((ZynkTimer.stopwatchElapsedMs(context) / 1000).toInt())}."
        Cmd.StopwatchReset -> "Stopwatch reset."
        Cmd.CheckStopwatch -> {
            val secs = (ZynkTimer.stopwatchElapsedMs(context) / 1000).toInt()
            if (!ZynkTimer.stopwatchRunning(context) && secs == 0) "The stopwatch is not running."
            else ZynkTimer.durationWords(secs) + if (ZynkTimer.stopwatchRunning(context)) "." else ", stopped."
        }
    }
}
