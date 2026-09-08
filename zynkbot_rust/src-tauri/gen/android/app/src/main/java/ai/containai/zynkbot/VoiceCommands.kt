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
        return t
    }

    private const val UNIT = "(hours?|hrs?|minutes?|mins?|seconds?|secs?)"
    private val TIMER_RES = listOf(
        Regex("(?:set\\s+(?:a\\s+)?)?timer\\s+(?:for\\s+)?(\\d+(?:\\.\\d+)?)\\s*$UNIT"),
        Regex("(\\d+(?:\\.\\d+)?)\\s*$UNIT\\s+timer"),
    )
    private val ALARM_RE = Regex(
        "(?:set\\s+(?:an?\\s+)?alarm\\s+(?:for|at)|alarm\\s+(?:for|at)|wake\\s+me\\s+up\\s+at)\\s+(\\d{1,2})(?:[:\\s](\\d{1,2}))?\\s*(am|pm)?"
    )
    private val STOPWATCH_RE = Regex("(?:start|begin)\\s+(?:the\\s+|a\\s+)?stopwatch|^stopwatch$")

    fun parse(text: String): Cmd? {
        val t = normalizeNumbers(text.lowercase().trim())
        for (re in TIMER_RES) {
            val m = re.find(t) ?: continue
            val v = m.groupValues[1].toDoubleOrNull() ?: continue
            val u = m.groupValues[2]
            val secs = when {
                u.startsWith("hour") || u.startsWith("hr") -> v * 3600
                u.startsWith("min") -> v * 60
                else -> v
            }
            return Cmd.Timer(Math.round(secs).toInt())
        }
        ALARM_RE.find(t)?.let { m ->
            var hour = m.groupValues[1].toInt()
            val minute = m.groupValues[2].ifEmpty { "0" }.toInt()
            val ampm = m.groupValues[3]
            if (ampm == "pm" && hour != 12) hour += 12
            if (ampm == "am" && hour == 12) hour = 0
            return Cmd.Alarm(hour, minute)
        }
        if (STOPWATCH_RE.containsMatchIn(t)) return Cmd.Stopwatch
        return null
    }

    /** Hands the command to the clock app. Returns true only if the clock app accepted
     *  the intent, so the spoken confirmation is never a lie. EXTRA_SKIP_UI: hands-free
     *  means no clock screen popping over the lock screen; the confirmation is spoken. */
    fun execute(context: Context, cmd: Cmd): Boolean {
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
            Cmd.Stopwatch -> Intent("android.intent.action.START_STOPWATCH").apply {
                putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            }
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
            if (cmd == Cmd.Stopwatch) openClockApp(context) else false
        }
    }

    /** True if the device's clock app could be brought to the front. */
    private var openedClockInstead = false
    private fun openClockApp(context: Context): Boolean {
        val probe = Intent(AlarmClock.ACTION_SET_ALARM)
        val pkg = probe.resolveActivity(context.packageManager)?.packageName ?: return false
        val launch = context.packageManager.getLaunchIntentForPackage(pkg) ?: return false
        launch.flags = Intent.FLAG_ACTIVITY_NEW_TASK
        return try {
            context.startActivity(launch)
            openedClockInstead = true
            true
        } catch (e: Exception) {
            Log.w(TAG, "Opening the clock app failed: ${e.message}")
            false
        }
    }

    /** What to say once the clock app has taken the command. Plain spoken words. */
    fun confirmation(cmd: Cmd): String = when (cmd) {
        is Cmd.Timer -> {
            val h = cmd.seconds / 3600
            val m = (cmd.seconds % 3600) / 60
            val s = cmd.seconds % 60
            val parts = mutableListOf<String>()
            if (h > 0) parts += "$h hour" + if (h > 1) "s" else ""
            if (m > 0) parts += "$m minute" + if (m > 1) "s" else ""
            if (s > 0 && h == 0) parts += "$s second" + if (s > 1) "s" else ""
            "Timer set for ${parts.joinToString(" and ")}."
        }
        is Cmd.Alarm -> {
            val h12 = if (cmd.hour % 12 == 0) 12 else cmd.hour % 12
            val ampm = if (cmd.hour < 12) "AM" else "PM"
            val minutes = if (cmd.minute > 0) " " + cmd.minute.toString().padStart(2, '0') else ""
            "Alarm set for $h12$minutes $ampm."
        }
        Cmd.Stopwatch -> if (openedClockInstead) {
            openedClockInstead = false
            "This phone's clock app has no stopwatch shortcut, so I opened it for you. The stopwatch is one tap away."
        } else "Stopwatch started."
    }
}
