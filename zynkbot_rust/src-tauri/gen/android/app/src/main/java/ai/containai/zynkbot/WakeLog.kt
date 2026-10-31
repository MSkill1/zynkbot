package ai.containai.zynkbot

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Hey Zynk's log lines, to logcat as before and to a file the app can export.
 *
 * Until 2026-10-24 the wake-word service wrote only to logcat, which a release build
 * cannot read back, so a tester's "Report a problem" showed the turns that got through
 * and nothing about the wakes that were ignored, or why (Mike, in the car). The file
 * sits beside the Rust side's log (files/zynkbot/logs/zynkbot.log) and the report
 * command appends its tail. Rotates at [MAX_BYTES], one older generation kept.
 */
object WakeLog {
    private const val MAX_BYTES = 1_000_000L
    private val stamp = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)
    @Volatile private var file: File? = null
    private var writer: FileWriter? = null
    private val lock = Any()

    /** Call once from the service's onCreate; before that, lines go to logcat only. */
    fun init(context: Context) {
        val dir = File(context.filesDir, "zynkbot/logs")
        dir.mkdirs()
        file = File(dir, "wake.log")
    }

    fun d(tag: String, msg: String) { Log.d(tag, msg); write('D', tag, msg) }
    fun i(tag: String, msg: String) { Log.i(tag, msg); write('I', tag, msg) }
    fun w(tag: String, msg: String) { Log.w(tag, msg); write('W', tag, msg) }
    fun e(tag: String, msg: String) { Log.e(tag, msg); write('E', tag, msg) }

    private fun write(level: Char, tag: String, msg: String) {
        val f = file ?: return
        synchronized(lock) {
            try {
                if (f.length() > MAX_BYTES) {
                    writer?.close(); writer = null
                    val older = File(f.parentFile, f.name + ".1")
                    older.delete()
                    f.renameTo(older)
                }
                val w = writer ?: FileWriter(f, true).also { writer = it }
                w.write("${stamp.format(Date())} $level/$tag: $msg\n")
                w.flush()
            } catch (ex: Exception) {
                // The file is a convenience for reports; never let it take the listener down.
                writer = null
                Log.w("WakeLog", "could not write: ${ex.message}")
            }
        }
    }
}
