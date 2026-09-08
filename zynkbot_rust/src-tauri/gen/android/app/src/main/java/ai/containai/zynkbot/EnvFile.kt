package ai.containai.zynkbot

import android.content.Context
import java.io.File

/**
 * Read-only access to the app's settings file, `files/zynkbot/.env` — the same
 * `KEY=value` lines the Rust side writes (lib.rs / zynksync.rs). Native code that
 * runs without the WebView (the assistant session) has no other way to reach the
 * API keys the user entered in Settings. Values are never logged.
 */
object EnvFile {
    fun file(context: Context): File = File(context.filesDir, "zynkbot/.env")

    /** The value for [name], or null when the file or the key is absent / blank. */
    fun read(context: Context, name: String): String? {
        val f = file(context)
        if (!f.isFile) return null
        return try {
            f.readLines()
                .asSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .firstOrNull { it.startsWith("$name=") }
                ?.substringAfter("=")
                ?.trim()
                ?.trim('"', '\'')   // tolerate a hand-edited quoted value
                ?.takeIf { it.isNotEmpty() }
        } catch (_: Exception) { null }
    }
}
