package ai.containai.zynkbot

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Toast
import androidx.core.content.IntentCompat
import java.io.File

/**
 * "Share to Zynkbot" (2026-09-19). Zynkbot appears in any app's Share menu; the
 * file(s) are copied into the share folder (see ZynkShareProvider) and this
 * activity finishes with a toast. Plain text shared without a file becomes a .txt.
 * Runs in its own task so it never pulls the main app to the front.
 */
class ShareReceiverActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val intent = intent
        val uris: List<Uri> = when (intent?.action) {
            Intent.ACTION_SEND -> listOfNotNull(IntentCompat.getParcelableExtra(intent, Intent.EXTRA_STREAM, Uri::class.java))
            Intent.ACTION_SEND_MULTIPLE -> IntentCompat.getParcelableArrayListExtra(intent, Intent.EXTRA_STREAM, Uri::class.java) ?: emptyList()
            else -> emptyList()
        }
        val text = if (uris.isEmpty()) intent?.getStringExtra(Intent.EXTRA_TEXT) else null

        Thread {
            val dir = ZynkShareProvider.shareDir(this)
            var added = 0
            for (uri in uris) {
                try {
                    var name = displayName(uri) ?: uri.lastPathSegment ?: "file"
                    // A photo shared from an album can arrive as a bare id; give it the
                    // extension its type implies so it is treated as an image later.
                    if (!name.contains('.')) {
                        val ext = contentResolver.getType(uri)?.let { android.webkit.MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
                        if (!ext.isNullOrBlank()) name = "$name.$ext"
                    }
                    val dest = ZynkShareProvider.uniqueFile(dir, name)
                    contentResolver.openInputStream(uri)?.use { input ->
                        dest.outputStream().use { output -> input.copyTo(output) }
                    } ?: continue
                    added++
                } catch (_: Exception) {}
            }
            if (text != null && text.isNotBlank()) {
                try {
                    val stamp = java.text.SimpleDateFormat("yyyy-MM-dd HH.mm", java.util.Locale.US).format(java.util.Date())
                    ZynkShareProvider.uniqueFile(dir, "Shared text $stamp.txt").writeText(text)
                    added++
                } catch (_: Exception) {}
            }
            if (added > 0) ZynkShareProvider.markChanged(this)
            runOnUiThread {
                val msg = when {
                    added == 0 -> "Nothing to add to Zynkbot"
                    added == 1 -> "Added to Zynkbot share"
                    else -> "Added $added files to Zynkbot share"
                }
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                finish()
            }
        }.start()
    }

    private fun displayName(uri: Uri): String? =
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
}
