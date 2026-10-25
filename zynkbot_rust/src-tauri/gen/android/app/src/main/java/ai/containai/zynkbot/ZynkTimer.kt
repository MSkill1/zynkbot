package ai.containai.zynkbot

import android.app.AlarmManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import org.json.JSONObject
import java.util.Calendar

/**
 * Zynkbot's own timer, alarm and stopwatch (branch `timer`, 2026-10-25).
 *
 * Why not the clock app: a timer handed to it with the phone asleep was confirmed
 * aloud and never rang (KI-061, GitHub #29), nothing can ask it how long is left, and
 * neither test phone's clock cancels a running timer from outside. So the deadline is
 * an exact alarm of our own (AlarmManager.setAlarmClock: survives Doze and the process
 * being killed, shows the alarm icon in the status bar) that wakes TimerReceiver, which
 * starts TimerRingService to speak and ring. State lives in SharedPreferences so any
 * process (the app, the wake service, the assistant session) reads the same timer.
 *
 * One timer, one alarm, one stopwatch at a time (Matt: nobody has needed more).
 * Exact alarms need the "Alarms & reminders" permission on Android 12+; Android 14
 * denies it by default, so when it is missing the timer falls back to the clock app
 * (the pre-2026-10 path) and the Voice page offers the switch.
 */
object ZynkTimer {
    private const val TAG = "ZynkTimer"
    private const val PREFS = "zynk_timer"
    const val CHANNEL_COUNTDOWN = "zynk_timer_v2"   // v1 was IMPORTANCE_LOW and Android 16 folded it into the silent bundle (Matt, 2026-10-25)
    const val CHANNEL_RING = "zynk_timer_ring"
    const val NOTIF_TIMER = 74000      // + id
    const val NOTIF_ALARM = 75000      // + id
    const val NOTIF_RING = 7303
    const val RING_SECONDS = 30            // Matt, 2026-10-25: ring for 30 s, then stop

    const val ACTION_FIRE = "ai.containai.zynkbot.TIMER_FIRE"
    const val ACTION_CANCEL = "ai.containai.zynkbot.TIMER_CANCEL"
    const val ACTION_STOP_RING = "ai.containai.zynkbot.TIMER_STOP_RING"
    const val EXTRA_KIND = "kind"          // "timer" | "alarm"

    /** True while TimerRingService is ringing; WakeWordService lets a wake through the
     *  playback gate when the sound is ours, and any wake stops the ring. */
    @Volatile var ringing: Boolean = false
    /** The page's bridge sets this so the chip in the app follows every change. */
    @Volatile var onChanged: (() -> Unit)? = null

    private fun prefs(ctx: Context) = ctx.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private fun alarmManager(ctx: Context) = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
    private fun notifier(ctx: Context) = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    fun canScheduleExact(ctx: Context): Boolean =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) alarmManager(ctx).canScheduleExactAlarms() else true

    // ── Timers and alarms: several at a time (Matt, 2026-10-25) ────────────────

    data class TimerEntry(val id: Int, val endMs: Long, val totalSeconds: Int)
    data class AlarmEntry(val id: Int, val atMs: Long)

    private fun nextId(ctx: Context): Int {
        val p = prefs(ctx); val n = p.getInt("next_id", 1); p.edit().putInt("next_id", n + 1).apply(); return n
    }
    fun timers(ctx: Context): List<TimerEntry> = try {
        val arr = org.json.JSONArray(prefs(ctx).getString("timers", "[]"))
        (0 until arr.length()).map { arr.getJSONObject(it) }.map { TimerEntry(it.getInt("id"), it.getLong("end"), it.getInt("total")) }.sortedBy { it.endMs }
    } catch (_: Exception) { emptyList() }
    fun alarms(ctx: Context): List<AlarmEntry> = try {
        val arr = org.json.JSONArray(prefs(ctx).getString("alarms", "[]"))
        (0 until arr.length()).map { arr.getJSONObject(it) }.map { AlarmEntry(it.getInt("id"), it.getLong("at")) }.sortedBy { it.atMs }
    } catch (_: Exception) { emptyList() }
    private fun saveTimers(ctx: Context, list: List<TimerEntry>) {
        val arr = org.json.JSONArray(); list.forEach { arr.put(JSONObject().put("id", it.id).put("end", it.endMs).put("total", it.totalSeconds)) }
        prefs(ctx).edit().putString("timers", arr.toString()).apply()
    }
    private fun saveAlarms(ctx: Context, list: List<AlarmEntry>) {
        val arr = org.json.JSONArray(); list.forEach { arr.put(JSONObject().put("id", it.id).put("at", it.atMs)) }
        prefs(ctx).edit().putString("alarms", arr.toString()).apply()
    }

    fun secondsLeft(t: TimerEntry): Int = ((t.endMs - System.currentTimeMillis() + 999) / 1000).toInt().coerceAtLeast(0)

    /** False when exact alarms are not allowed: the caller falls back to the clock app. */
    fun setTimer(ctx: Context, seconds: Int): Boolean {
        if (!canScheduleExact(ctx)) return false
        val t = TimerEntry(nextId(ctx), System.currentTimeMillis() + seconds * 1000L, seconds)
        saveTimers(ctx, timers(ctx) + t)
        schedule(ctx, t.endMs, "timer", t.id)
        showCountdown(ctx, t)
        changed()
        return true
    }

    /** Cancel the timer whose length is `seconds` (0 = the only one, or the soonest when
     *  `any` is set). Returns the cancelled entry, or null. */
    fun cancelTimer(ctx: Context, seconds: Int = 0, any: Boolean = false): TimerEntry? {
        val list = timers(ctx)
        val victim = when {
            seconds > 0 -> list.firstOrNull { it.totalSeconds == seconds }
            list.size == 1 || any -> list.firstOrNull()
            else -> null
        } ?: return null
        removeTimer(ctx, victim)
        return victim
    }
    fun cancelAllTimers(ctx: Context): Int { val l = timers(ctx); l.forEach { removeTimer(ctx, it) }; return l.size }
    private fun removeTimer(ctx: Context, t: TimerEntry) {
        alarmManager(ctx).cancel(firePending(ctx, "timer", t.id))
        notifier(ctx).cancel(NOTIF_TIMER + t.id)
        saveTimers(ctx, timers(ctx).filter { it.id != t.id })
        if (ringing) stopRinging(ctx)
        changed()
    }

    /** Next occurrence of hour:minute. False when exact alarms are not allowed. */
    fun setAlarm(ctx: Context, hour: Int, minute: Int): Boolean {
        if (!canScheduleExact(ctx)) return false
        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, hour); set(Calendar.MINUTE, minute); set(Calendar.SECOND, 0); set(Calendar.MILLISECOND, 0)
            if (timeInMillis <= System.currentTimeMillis()) add(Calendar.DAY_OF_YEAR, 1)
        }
        // The same time twice is one alarm.
        alarms(ctx).firstOrNull { it.atMs == cal.timeInMillis }?.let { return true }
        val a = AlarmEntry(nextId(ctx), cal.timeInMillis)
        saveAlarms(ctx, alarms(ctx) + a)
        schedule(ctx, a.atMs, "alarm", a.id)
        showAlarmNotification(ctx, a)
        changed()
        return true
    }

    /** Cancel the alarm at hour:minute (hour < 0 = the only one). Returns it, or null. */
    fun cancelAlarm(ctx: Context, hour: Int = -1, minute: Int = 0): AlarmEntry? {
        val list = alarms(ctx)
        val victim = when {
            hour >= 0 -> list.firstOrNull { val c = Calendar.getInstance().apply { timeInMillis = it.atMs }; c.get(Calendar.HOUR_OF_DAY) == hour && c.get(Calendar.MINUTE) == minute }
                ?: list.firstOrNull { val c = Calendar.getInstance().apply { timeInMillis = it.atMs }; c.get(Calendar.HOUR_OF_DAY) % 12 == hour % 12 && c.get(Calendar.MINUTE) == minute }
            list.size == 1 -> list.first()
            else -> null
        } ?: return null
        removeAlarm(ctx, victim)
        return victim
    }
    private fun removeAlarm(ctx: Context, a: AlarmEntry) {
        alarmManager(ctx).cancel(firePending(ctx, "alarm", a.id))
        notifier(ctx).cancel(NOTIF_ALARM + a.id)
        saveAlarms(ctx, alarms(ctx).filter { it.id != a.id })
        changed()
    }

    // ── Stopwatch ───────────────────────────────────────────────────────────────

    fun stopwatchRunning(ctx: Context): Boolean = prefs(ctx).getLong("sw_start", 0L) > 0L
    fun stopwatchElapsedMs(ctx: Context): Long {
        val p = prefs(ctx)
        val base = p.getLong("sw_elapsed", 0L)
        val start = p.getLong("sw_start", 0L)
        return if (start > 0L) base + (System.currentTimeMillis() - start) else base
    }
    fun startStopwatch(ctx: Context) {
        if (!stopwatchRunning(ctx)) prefs(ctx).edit().putLong("sw_start", System.currentTimeMillis()).apply()
        changed()
    }
    fun stopStopwatch(ctx: Context) {
        if (stopwatchRunning(ctx)) {
            val elapsed = stopwatchElapsedMs(ctx)
            prefs(ctx).edit().putLong("sw_elapsed", elapsed).remove("sw_start").apply()
        }
        changed()
    }
    fun resetStopwatch(ctx: Context) {
        prefs(ctx).edit().remove("sw_elapsed").remove("sw_start").apply()
        changed()
    }

    // ── Firing and ringing ──────────────────────────────────────────────────────

    /** Called by TimerReceiver when the exact alarm goes off. */
    fun onFired(ctx: Context, kind: String, id: Int) {
        val line = when (kind) {
            "alarm" -> {
                val a = alarms(ctx).firstOrNull { it.id == id }
                if (a != null) { notifier(ctx).cancel(NOTIF_ALARM + a.id); saveAlarms(ctx, alarms(ctx).filter { it.id != id }) }
                "It's ${clockWords(a?.atMs ?: System.currentTimeMillis())}. Your alarm."
            }
            else -> {
                val t = timers(ctx).firstOrNull { it.id == id }
                if (t != null) { notifier(ctx).cancel(NOTIF_TIMER + t.id); saveTimers(ctx, timers(ctx).filter { it.id != id }) }
                "Your ${durationWords(t?.totalSeconds ?: 0)} timer is done."
            }
        }
        changed()
        val i = Intent(ctx, TimerRingService::class.java).putExtra("line", line).putExtra(EXTRA_KIND, kind)
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i) else ctx.startService(i)
        } catch (e: Exception) { Log.e(TAG, "Could not start the ring service: ${e.message}") }
    }

    fun stopRinging(ctx: Context) {
        try { ctx.startService(Intent(ctx, TimerRingService::class.java).setAction(ACTION_STOP_RING)) } catch (_: Exception) {}
    }

    /** After a reboot alarms are gone: put back the ones still in the future. */
    fun rescheduleAfterBoot(ctx: Context) {
        val now = System.currentTimeMillis()
        val keepT = timers(ctx).filter { it.endMs > now }
        saveTimers(ctx, keepT); keepT.forEach { schedule(ctx, it.endMs, "timer", it.id); showCountdown(ctx, it) }
        val keepA = alarms(ctx).filter { it.atMs > now }
        saveAlarms(ctx, keepA); keepA.forEach { schedule(ctx, it.atMs, "alarm", it.id); showAlarmNotification(ctx, it) }
    }

    private fun schedule(ctx: Context, atMs: Long, kind: String, id: Int) {
        val show = PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        alarmManager(ctx).setAlarmClock(AlarmManager.AlarmClockInfo(atMs, show), firePending(ctx, kind, id))
    }

    private fun firePending(ctx: Context, kind: String, id: Int): PendingIntent {
        val i = Intent(ctx, TimerReceiver::class.java).setAction(ACTION_FIRE).putExtra(EXTRA_KIND, kind).putExtra("id", id)
        val code = (if (kind == "alarm") 3000 else 1000) + id
        return PendingIntent.getBroadcast(ctx, code, i, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    private fun cancelPending(ctx: Context, kind: String, id: Int): PendingIntent {
        val i = Intent(ctx, TimerReceiver::class.java).setAction(ACTION_CANCEL).putExtra(EXTRA_KIND, kind).putExtra("id", id)
        val code = (if (kind == "alarm") 4000 else 2000) + id
        return PendingIntent.getBroadcast(ctx, code, i, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    /** Cancel by id, from the notification's Cancel button. */
    fun cancelById(ctx: Context, kind: String, id: Int) {
        if (kind == "alarm") alarms(ctx).firstOrNull { it.id == id }?.let { removeAlarm(ctx, it) }
        else timers(ctx).firstOrNull { it.id == id }?.let { removeTimer(ctx, it) }
    }

    fun ensureChannels(ctx: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = notifier(ctx)
        try { nm.deleteNotificationChannel("zynk_timer_countdown") } catch (_: Exception) {}
        if (nm.getNotificationChannel(CHANNEL_COUNTDOWN) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL_COUNTDOWN, "Zynkbot timer", NotificationManager.IMPORTANCE_DEFAULT).apply {
                description = "A running timer or a set alarm"; setShowBadge(false); setSound(null, null); enableVibration(false)
            })
        }
        if (nm.getNotificationChannel(CHANNEL_RING) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL_RING, "Zynkbot timer done", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "The timer or alarm going off"; setSound(null, null) // the service plays the sound itself
            })
        }
    }

    private fun showCountdown(ctx: Context, t: TimerEntry) {
        ensureChannels(ctx)
        val n: Notification = NotificationCompat.Builder(ctx, CHANNEL_COUNTDOWN)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Zynkbot timer: ${durationWords(t.totalSeconds)}")
            .setContentText("Time left")
            .setUsesChronometer(true).setChronometerCountDown(true).setWhen(t.endMs)
            .setOngoing(true).setOnlyAlertOnce(true).setSilent(true)
            .setContentIntent(PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
            .addAction(0, "Cancel", cancelPending(ctx, "timer", t.id))
            .build()
        try { notifier(ctx).notify(NOTIF_TIMER + t.id, n) } catch (e: Exception) { Log.w(TAG, "notify: ${e.message}") }
    }

    private fun showAlarmNotification(ctx: Context, a: AlarmEntry) {
        ensureChannels(ctx)
        val n = NotificationCompat.Builder(ctx, CHANNEL_COUNTDOWN)
            .setSmallIcon(android.R.drawable.ic_lock_idle_alarm)
            .setContentTitle("Zynkbot alarm: ${clockWords(a.atMs)}")
            .setOngoing(true).setOnlyAlertOnce(true).setSilent(true)
            .setContentIntent(PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
            .addAction(0, "Cancel", cancelPending(ctx, "alarm", a.id))
            .build()
        try { notifier(ctx).notify(NOTIF_ALARM + a.id, n) } catch (e: Exception) { Log.w(TAG, "notify: ${e.message}") }
    }

    // ── State for the page ──────────────────────────────────────────────────────

    fun stateJson(ctx: Context): String = JSONObject().apply {
        put("timers", org.json.JSONArray().also { arr -> timers(ctx).forEach { arr.put(JSONObject().put("id", it.id).put("end", it.endMs).put("total", it.totalSeconds)) } })
        put("alarms", org.json.JSONArray().also { arr -> alarms(ctx).forEach { arr.put(JSONObject().put("id", it.id).put("at", it.atMs)) } })
        put("sw_running", stopwatchRunning(ctx))
        put("sw_elapsed_ms", stopwatchElapsedMs(ctx))
        put("ringing", ringing)
        put("exact_allowed", canScheduleExact(ctx))
    }.toString()

    fun changed() { try { onChanged?.invoke() } catch (_: Exception) {} }

    // ── Words ───────────────────────────────────────────────────────────────────

    /** "ten minutes", "one hour and five minutes", "forty-five seconds". */
    fun durationWords(totalSeconds: Int): String {
        val h = totalSeconds / 3600
        val m = (totalSeconds % 3600) / 60
        val s = totalSeconds % 60
        val parts = mutableListOf<String>()
        if (h > 0) parts += "$h hour" + if (h > 1) "s" else ""
        if (m > 0) parts += "$m minute" + if (m > 1) "s" else ""
        if (s > 0 && h == 0) parts += "$s second" + if (s > 1) "s" else ""
        if (parts.isEmpty()) parts += "0 seconds"
        return parts.joinToString(" and ")
    }

    /** "10 minutes and 25 minutes" */
    fun joinWords(items: List<String>): String = when (items.size) { 0 -> ""; 1 -> items[0]; 2 -> items[0] + " and " + items[1]; else -> items.dropLast(1).joinToString(", ") + " and " + items.last() }

    /** "7 30 AM", "noon" style is overkill: "7:30 AM" read by TTS comes out fine. */
    fun clockWords(atMs: Long): String {
        val c = Calendar.getInstance().apply { timeInMillis = atMs }
        val h = c.get(Calendar.HOUR_OF_DAY); val mi = c.get(Calendar.MINUTE)
        val h12 = if (h % 12 == 0) 12 else h % 12
        val ampm = if (h < 12) "AM" else "PM"
        return if (mi == 0) "$h12 $ampm" else "$h12 ${mi.toString().padStart(2, '0')} $ampm"
    }
}
