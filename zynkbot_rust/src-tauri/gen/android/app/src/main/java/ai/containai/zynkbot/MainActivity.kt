package ai.containai.zynkbot

import android.Manifest
import android.app.NotificationManager
import android.app.role.RoleManager
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.AlarmClock
import android.provider.DocumentsContract
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.lang.ref.WeakReference
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream


class MainActivity : TauriActivity() {

    companion object {
        private const val REQ_WRITE_STORAGE = 100
        private const val REQ_LOCAL_NETWORK = 101
        private const val REQ_NOTIFICATIONS = 102
        private const val REQ_RECORD_AUDIO = 200
        // ACCESS_LOCAL_NETWORK is Android 16+ (SDK 36). Using string literal because the
        // constant may not be present when compiling against older SDK stubs, and because
        // GrapheneOS enforces it more strictly than stock Android — see hotfix for LAN
        // access on Pixel 10 / Android 16 devices.
        private const val PERM_ACCESS_LOCAL_NETWORK = "android.permission.ACCESS_LOCAL_NETWORK"

        // True only while the Activity is in the RESUMED state (app visible to user).
        // WakeWordService reads this to decide whether to call the JS detection callback
        // directly or route through the Kotlin-native path (chime + Vosk + notification).
        // evaluateJavascript() silently drops when the WebView is paused (Activity in
        // background), so this flag is the only reliable way to know if JS is reachable.
        @Volatile var isInForeground = false
    }

    private var webViewRef: WeakReference<WebView>? = null
    private var cameraOutputPath: String? = null

    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            launchCamera()
        } else {
            val wv = webViewRef?.get() ?: return@registerForActivityResult
            wv.post { wv.evaluateJavascript(
                "window.__camReject&&window.__camReject('Camera permission denied');window.__camResolve=null;window.__camReject=null;", null) }
        }
    }

    private val requestStoragePermission = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        pickFolderLauncher.launch(null)
    }

    private val pickFolderLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        val wv = webViewRef?.get() ?: return@registerForActivityResult
        val script = if (uri != null) {
            try {
                contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                    android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (e: Exception) { }
            val path = resolveUri(uri) ?: uri.toString()
            val escaped = path.replace("\\", "\\\\").replace("'", "\\'")
            "window.__fpResolve&&window.__fpResolve('$escaped');window.__fpResolve=null;window.__fpReject=null;"
        } else {
            "window.__fpReject&&window.__fpReject('cancelled');window.__fpResolve=null;window.__fpReject=null;"
        }
        wv.post { wv.evaluateJavascript(script, null) }
    }

    // File picker for adding files into ZynkbotShare
    private val pickFileLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        val wv = webViewRef?.get() ?: return@registerForActivityResult
        if (uri == null) {
            wv.post { wv.evaluateJavascript(
                "window.__zfpReject&&window.__zfpReject('cancelled');window.__zfpResolve=null;window.__zfpReject=null;", null) }
            return@registerForActivityResult
        }
        // Copy the file into ZynkbotShare on a background thread
        Thread {
            try {
                val destDir = zynkShareDir()
                destDir.mkdirs()
                // Resolve a display name for the file
                val fileName = contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    cursor.moveToFirst()
                    if (nameIndex >= 0) cursor.getString(nameIndex) else null
                } ?: uri.lastPathSegment ?: "file"
                val dest = File(destDir, fileName)
                contentResolver.openInputStream(uri)?.use { input ->
                    dest.outputStream().use { output -> input.copyTo(output) }
                }
                val escaped = dest.absolutePath.replace("\\", "\\\\").replace("'", "\\'")
                wv.post { wv.evaluateJavascript(
                    "window.__zfpResolve&&window.__zfpResolve('$escaped');window.__zfpResolve=null;window.__zfpReject=null;", null) }
            } catch (e: Exception) {
                val msg = (e.message ?: "copy failed").replace("'", "\\'")
                wv.post { wv.evaluateJavascript(
                    "window.__zfpReject&&window.__zfpReject('$msg');window.__zfpResolve=null;window.__zfpReject=null;", null) }
            }
        }.start()
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        val wv = webViewRef?.get() ?: return@registerForActivityResult
        val script = if (success && cameraOutputPath != null) {
            val escaped = cameraOutputPath!!.replace("\\", "\\\\").replace("'", "\\'")
            "window.__camResolve&&window.__camResolve('$escaped');window.__camResolve=null;window.__camReject=null;"
        } else {
            "window.__camReject&&window.__camReject('cancelled');window.__camResolve=null;window.__camReject=null;"
        }
        wv.post { wv.evaluateJavascript(script, null) }
    }

    inner class FolderPickerBridge {
        @JavascriptInterface
        fun pick() {
            runOnUiThread {
                val permsNeeded = mutableListOf<String>()
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                        permsNeeded.add(Manifest.permission.READ_MEDIA_IMAGES)
                    }
                } else if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
                    if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                        permsNeeded.add(Manifest.permission.READ_EXTERNAL_STORAGE)
                    }
                }
                if (permsNeeded.isNotEmpty()) {
                    requestStoragePermission.launch(permsNeeded.toTypedArray())
                } else {
                    pickFolderLauncher.launch(null)
                }
            }
        }
    }

    inner class ZynkbotPathsBridge {
        @JavascriptInterface
        fun getShareDir(): String {
            return try {
                val dir = zynkShareDir()
                dir.mkdirs()
                dir.absolutePath
            } catch (e: Exception) { "" }
        }

        @JavascriptInterface
        fun pickFile() {
            runOnUiThread {
                pickFileLauncher.launch(arrayOf("*/*"))
            }
        }

        @JavascriptInterface
        fun readFileBase64(uriStr: String): String {
            return try {
                val uri = Uri.parse(uriStr)
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return ""
                android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
            } catch (e: Exception) { "" }
        }

        @JavascriptInterface
        fun getFileName(uriStr: String): String {
            return try {
                val uri = Uri.parse(uriStr)
                contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    cursor.moveToFirst()
                    if (idx >= 0) cursor.getString(idx) else null
                } ?: uriStr.substringAfterLast('/').substringBefore('?')
            } catch (e: Exception) { uriStr.substringAfterLast('/').substringBefore('?') }
        }

        @JavascriptInterface
        fun readFileText(uriStr: String): String {
            return try {
                val uri = Uri.parse(uriStr)
                contentResolver.openInputStream(uri)?.use { it.bufferedReader(Charsets.UTF_8).readText() } ?: ""
            } catch (e: Exception) { "" }
        }

        @JavascriptInterface
        fun openShareFolder() {
            val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val dir = File(downloads, "ZynkbotShare").also { it.mkdirs() }
            runOnUiThread {
                var opened = false

                // Android 10+: raw: URI via Downloads provider
                if (!opened && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    try {
                        val uri = DocumentsContract.buildDocumentUri(
                            "com.android.providers.downloads.documents",
                            "raw:${dir.absolutePath}"
                        )
                        val intent = Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(uri, "vnd.android.document/directory")
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        startActivity(intent)
                        opened = true
                    } catch (_: Exception) {}
                }

                // Android 10+: external storage provider fallback
                // (Skipped on API <= 28 because documentsui on those versions crashes on this URI format)
                if (!opened && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    try {
                        val uri = DocumentsContract.buildDocumentUri(
                            "com.android.externalstorage.documents",
                            "primary:Download/ZynkbotShare"
                        )
                        val intent = Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(uri, "vnd.android.document/directory")
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                        }
                        startActivity(intent)
                        opened = true
                    } catch (_: Exception) {}
                }

                // Fallback: launch a file manager app
                if (!opened) {
                    try {
                        val intent = packageManager.getLaunchIntentForPackage("com.sec.android.app.myfiles")
                            ?: packageManager.getLaunchIntentForPackage("com.google.android.apps.nbu.files")
                            ?: packageManager.getLaunchIntentForPackage("com.android.documentsui")
                            ?: Intent(Intent.ACTION_VIEW).apply { type = "resource/folder" }
                        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        startActivity(intent)
                    } catch (_: Exception) {}
                }
            }
        }
    }

    inner class VoskBridge {
        @Volatile private var voskModel: org.vosk.Model? = null
        @Volatile private var speechService: org.vosk.android.SpeechService? = null
        private val accumulated = StringBuilder()

        private val listener = object : org.vosk.android.RecognitionListener {
            override fun onPartialResult(h: String?) {
                if (h.isNullOrBlank()) return
                val partial = try { org.json.JSONObject(h).optString("partial", "") } catch (_: Exception) { "" }
                if (partial.isNotBlank()) fire("window.__voskPartial&&window.__voskPartial('${esc(partial)}');")
            }
            override fun onResult(h: String?) {
                val t = parseText(h)
                if (t.isNotEmpty()) synchronized(accumulated) {
                    if (accumulated.isNotEmpty()) accumulated.append(" ")
                    accumulated.append(t)
                }
            }
            override fun onFinalResult(h: String?) {
                val last = parseText(h)
                val full = synchronized(accumulated) {
                    buildString {
                        append(accumulated)
                        if (accumulated.isNotEmpty() && last.isNotEmpty()) append(" ")
                        append(last)
                    }.trim().also { accumulated.clear() }
                }
                fire("window.__voskResult&&window.__voskResult('${esc(full)}');")
            }
            override fun onError(e: Exception?) {
                fire("window.__voskError&&window.__voskError('${esc(e?.message ?: "recognition error")}');")
            }
            override fun onTimeout() {
                fire("window.__voskError&&window.__voskError('timeout');")
            }
        }

        @JavascriptInterface
        fun isModelReady(): Boolean {
            val d = File(filesDir, "vosk-model")
            return d.exists() && d.isDirectory
        }

        @JavascriptInterface
        fun getModelDir(): String = File(filesDir, "vosk-model").absolutePath

        @JavascriptInterface
        fun startListening() {
            if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
                runOnUiThread {
                    requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_RECORD_AUDIO)
                }
                fire("window.__voskError&&window.__voskError('Microphone permission not granted — please allow in the prompt then try again');")
                return
            }
            Thread {
                try {
                    val modelDir = File(filesDir, "vosk-model")
                    if (!modelDir.exists()) {
                        fire("window.__voskError&&window.__voskError('Voice model not downloaded yet');")
                        return@Thread
                    }
                    if (voskModel == null) voskModel = org.vosk.Model(modelDir.absolutePath)
                    WakeWordService.sharedVoskModel = voskModel // share with screen-off dictation
                    val rec = org.vosk.Recognizer(voskModel, 16000.0f)
                    runOnUiThread {
                        try {
                            speechService = org.vosk.android.SpeechService(rec, 16000.0f)
                            speechService!!.startListening(listener)
                            fire("window.__voskStarted&&window.__voskStarted();")
                        } catch (e: Exception) {
                            fire("window.__voskError&&window.__voskError('${esc(e.message ?: "start failed")}');")
                        }
                    }
                } catch (e: Exception) {
                    fire("window.__voskError&&window.__voskError('${esc(e.message ?: "model load failed")}');")
                }
            }.start()
        }

        @JavascriptInterface
        fun stopListening() {
            runOnUiThread {
                speechService?.stop()
                speechService = null
            }
        }

        @JavascriptInterface
        fun downloadModel() {
            Thread {
                try {
                    val modelDir = File(filesDir, "vosk-model")
                    val tempZip = File(cacheDir, "vosk-model.zip")

                    fire("window.__voskDownloadProgress&&window.__voskDownloadProgress(0);")

                    val conn = URL("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                        .openConnection() as HttpURLConnection
                    conn.connect()
                    val total = conn.contentLength.toLong()
                    var read = 0L
                    var lastPct = -1

                    FileOutputStream(tempZip).use { fos ->
                        conn.inputStream.use { inp ->
                            val buf = ByteArray(8192)
                            var n = inp.read(buf)
                            while (n != -1) {
                                fos.write(buf, 0, n)
                                read += n
                                if (total > 0) {
                                    val pct = ((read * 70) / total).toInt()
                                    if (pct != lastPct) {
                                        lastPct = pct
                                        fire("window.__voskDownloadProgress&&window.__voskDownloadProgress($pct);")
                                    }
                                }
                                n = inp.read(buf)
                            }
                        }
                    }

                    fire("window.__voskDownloadProgress&&window.__voskDownloadProgress(70);")
                    modelDir.deleteRecursively()
                    modelDir.mkdirs()

                    ZipInputStream(BufferedInputStream(FileInputStream(tempZip))).use { zis ->
                        var entry = zis.nextEntry
                        while (entry != null) {
                            val rel = entry.name.split("/", limit = 2).let { if (it.size > 1) it[1] else "" }
                            if (rel.isNotEmpty()) {
                                val out = File(modelDir, rel)
                                if (entry.isDirectory) out.mkdirs()
                                else { out.parentFile?.mkdirs(); FileOutputStream(out).use { zis.copyTo(it) } }
                            }
                            zis.closeEntry()
                            entry = zis.nextEntry
                        }
                    }

                    tempZip.delete()
                    fire("window.__voskDownloadProgress&&window.__voskDownloadProgress(100);")
                    fire("window.__voskModelReady&&window.__voskModelReady();")
                } catch (e: Exception) {
                    fire("window.__voskDownloadError&&window.__voskDownloadError('${esc(e.message ?: "download failed")}');")
                }
            }.start()
        }

        private fun parseText(h: String?): String {
            if (h.isNullOrEmpty()) return ""
            return try { org.json.JSONObject(h).optString("text", "").trim() } catch (_: Exception) { "" }
        }

        private fun esc(s: String) = s.replace("\\", "\\\\").replace("'", "\\'")

        private fun fire(js: String) {
            val wv = webViewRef?.get() ?: return
            wv.post { wv.evaluateJavascript(js, null) }
        }
    }

    inner class WakeWordBridge {
        private val modelDir get() = File(filesDir, "wake-word-models")

        // The three ONNX models ship inside the APK under assets/wake-word-models/
        // (~3 MB). They are copied out to filesDir rather than read in place because
        // ONNX Runtime's createSession needs a real filesystem path, and assets live
        // inside the APK archive. Bundling means wake word works offline on first
        // launch instead of requiring a network fetch from a GitHub release.
        private val modelFiles = listOf("melspectrogram.onnx", "embedding_model.onnx", "hey_zynk.onnx")

        private fun modelsPresent(): Boolean =
            modelFiles.all { val f = File(modelDir, it); f.exists() && f.length() > 0L }

        /** Copy any missing model out of assets. Idempotent, and small enough to run inline. */
        private fun extractModelsFromAssets() {
            modelDir.mkdirs()
            for (name in modelFiles) {
                val dest = File(modelDir, name)
                if (dest.exists() && dest.length() > 0L) continue
                assets.open("wake-word-models/$name").use { input ->
                    FileOutputStream(dest).use { output -> input.copyTo(output) }
                }
            }
        }

        @JavascriptInterface
        fun isModelReady(): Boolean {
            if (modelsPresent()) return true
            // Extract on demand, so the "voice models needed" panel never has to appear.
            return try {
                extractModelsFromAssets()
                modelsPresent()
            } catch (_: Exception) {
                false
            }
        }

        /** The app's Stop button: cut native speech now. The web-side stopTts() only
         *  ever stopped the WebView's own audio, so a native reply was unstoppable
         *  short of force-closing the app (OnePlus, 2026-09-04). */
        @JavascriptInterface
        fun stopSpeaking() {
            NativeVoiceAnswerer.stopSpeaking()
        }

        @JavascriptInterface
        fun isNativeSpeaking(): Boolean = NativeVoiceAnswerer.speaking

        /** Hands-free exchanges finished since the page last asked, as a JSON array of
         *  {sessionId, question, answer, at}. The page appends those belonging to the
         *  thread on screen. Nudged via window.__nativeTurns when one completes, and
         *  drained again on resume because a nudge to a paused WebView is lost. */
        @JavascriptInterface
        fun drainNativeTurns(): String = NativeVoiceAnswerer.drainTurnsJson()

        /** Breadcrumbs from the page into logcat (tag ZynkWeb). WebView console output
         *  was not reaching logcat on the OnePlus, and some bugs (chat autoscroll) leave
         *  no other trace. */
        @JavascriptInterface
        fun log(message: String) { Log.i("ZynkWeb", message) }

        @JavascriptInterface
        fun start(threshold: Float) {
            if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
                fire("window.__wakeWordError&&window.__wakeWordError('Microphone permission required');")
                return
            }
            // Push native speaking state into the WebView so Stop enables/disables with it.
            NativeVoiceAnswerer.onSpeakingChanged = { on ->
                fire("window.__nativeSpeaking&&window.__nativeSpeaking(${if (on) "true" else "false"});")
            }
            NativeVoiceAnswerer.onTurnCompleted = { fire("window.__nativeTurns&&window.__nativeTurns();") }
            // A locked-screen "Hey Zynk" reply auto-opens the app via a full-screen
            // intent. Android 14+ denies that permission by default; without it the
            // second locked query only posts a notification instead of answering.
            ensureFullScreenIntentPermission()
            WakeWordService.detectionCallback = {
                fire("window.__wakeWordDetected&&window.__wakeWordDetected();")
            }
            val intent = Intent(this@MainActivity, WakeWordService::class.java).apply {
                putExtra("threshold", threshold)
                putExtra("modelDir", modelDir.absolutePath)
            }
            ContextCompat.startForegroundService(this@MainActivity, intent)
        }

        @JavascriptInterface
        fun stop() {
            WakeWordService.detectionCallback = null
            stopService(Intent(this@MainActivity, WakeWordService::class.java))
        }

        // Kept under its original name because App.jsx calls WakeWordBridge.downloadModels().
        // Nothing is downloaded in the normal path any more — the models are unpacked from
        // the APK, which is effectively instant. The remote fetch survives only as a
        // fallback for a build that somehow shipped without the assets.
        @JavascriptInterface
        fun downloadModels() {
            Thread {
                try {
                    fire("window.__wakeWordDownloadProgress&&window.__wakeWordDownloadProgress(0);")

                    try {
                        extractModelsFromAssets()
                    } catch (assetError: Exception) {
                        modelDir.mkdirs()
                        val base = "https://github.com/MSkill1/zynkbot/releases/download/wake-word-models"
                        modelFiles.forEachIndexed { idx, name ->
                            val dest = File(modelDir, name)
                            if (!dest.exists()) {
                                downloadFile("$base/$name", dest) { pct ->
                                    val overall = (idx * 33 + pct / 3)
                                    fire("window.__wakeWordDownloadProgress&&window.__wakeWordDownloadProgress($overall);")
                                }
                            }
                        }
                    }

                    fire("window.__wakeWordDownloadProgress&&window.__wakeWordDownloadProgress(100);")
                    fire("window.__wakeWordModelReady&&window.__wakeWordModelReady();")
                } catch (e: Exception) {
                    val msg = (e.message ?: "model setup failed").replace("'", "\\'")
                    fire("window.__wakeWordDownloadError&&window.__wakeWordDownloadError('$msg');")
                }
            }.start()
        }

        private fun downloadFile(url: String, dest: File, progress: (Int) -> Unit) {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.connect()
            val total = conn.contentLength.toLong()
            var read = 0L
            FileOutputStream(dest).use { fos ->
                conn.inputStream.use { inp ->
                    val buf = ByteArray(8192)
                    var n = inp.read(buf)
                    while (n != -1) {
                        fos.write(buf, 0, n)
                        read += n
                        if (total > 0) progress(((read * 100) / total).toInt())
                        n = inp.read(buf)
                    }
                }
            }
        }

        private fun fire(js: String) {
            val wv = webViewRef?.get() ?: return
            wv.post { wv.evaluateJavascript(js, null) }
        }
    }

    inner class VoiceCommandBridge {
        @JavascriptInterface
        fun setTimer(seconds: Int) {
            runOnUiThread {
                try {
                    val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
                        putExtra(AlarmClock.EXTRA_LENGTH, seconds)
                        putExtra(AlarmClock.EXTRA_SKIP_UI, false)
                        putExtra(AlarmClock.EXTRA_MESSAGE, "Zynkbot")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    startActivity(intent)
                } catch (e: Exception) {
                    android.util.Log.e("VoiceCmd", "setTimer failed: ${e.message}")
                }
            }
        }

        @JavascriptInterface
        fun setAlarm(hour: Int, minute: Int, label: String) {
            runOnUiThread {
                try {
                    val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                        putExtra(AlarmClock.EXTRA_HOUR, hour)
                        putExtra(AlarmClock.EXTRA_MINUTES, minute)
                        putExtra(AlarmClock.EXTRA_MESSAGE, label)
                        putExtra(AlarmClock.EXTRA_SKIP_UI, false)
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    startActivity(intent)
                } catch (e: Exception) {
                    android.util.Log.e("VoiceCmd", "setAlarm failed: ${e.message}")
                }
            }
        }

        @JavascriptInterface
        fun startStopwatch() {
            runOnUiThread {
                var launched = false
                // ACTION_START_STOPWATCH string literal (added API 33; use literal to avoid SDK floor issue)
                if (!launched) try {
                    val i = Intent("android.intent.action.START_STOPWATCH").apply {
                        putExtra(AlarmClock.EXTRA_SKIP_UI, true)
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    startActivity(i); launched = true
                } catch (_: Exception) {}
                // Fallback: open Google Clock or system clock
                if (!launched) try {
                    val i = packageManager.getLaunchIntentForPackage("com.google.android.deskclock")
                        ?: packageManager.getLaunchIntentForPackage("com.android.deskclock")
                        ?: Intent(Intent.ACTION_MAIN).apply { addCategory("android.intent.category.APP_CLOCK") }
                    i.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(i)
                } catch (_: Exception) {}
            }
        }
    }

    private fun launchCamera() {
        try {
            val photoFile = File.createTempFile("zynk_photo_", ".jpg", cacheDir)
            cameraOutputPath = photoFile.absolutePath
            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.fileprovider",
                photoFile
            )
            takePictureLauncher.launch(uri)
        } catch (e: Exception) {
            val wv = webViewRef?.get() ?: return
            val msg = (e.message ?: "camera failed").replace("'", "\\'")
            wv.post { wv.evaluateJavascript(
                "window.__camReject&&window.__camReject('$msg');window.__camResolve=null;window.__camReject=null;", null) }
        }
    }

    inner class AndroidCameraBridge {
        @JavascriptInterface
        fun takePicture() {
            runOnUiThread {
                if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                    requestCameraPermission.launch(Manifest.permission.CAMERA)
                } else {
                    launchCamera()
                }
            }
        }
    }

    private fun resolveUri(uri: Uri): String? {
        return try {
            val docId = DocumentsContract.getTreeDocumentId(uri)
            val colon = docId.indexOf(':')
            if (colon >= 0) {
                val volume = docId.substring(0, colon)
                val path = docId.substring(colon + 1)
                when (volume) {
                    "primary" -> "/storage/emulated/0/$path"
                    else -> "/storage/$volume/$path"
                }
            } else null
        } catch (e: Exception) { null }
    }

    // Class-level helpers used by lifecycle methods (inner class versions are private)
    private fun fireJs(js: String) {
        val wv = webViewRef?.get() ?: return
        wv.post { wv.evaluateJavascript(js, null) }
    }
    private fun escJs(s: String) = s.replace("\\", "\\\\").replace("'", "\\'")

    // Handle transcript delivered by WakeWordService after a screen-off detection.
    // Called from onNewIntent (app already running) and onResume (app just launched).
    private fun handleScreenOffTranscript(intent: Intent?) {
        val transcript = intent?.getStringExtra("wake_word_transcript") ?: return
        // Consume the extra so it doesn't fire twice
        intent.removeExtra("wake_word_transcript")

        // Wake the screen and dismiss the keyguard so the user sees the response
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
            )
        }

        // Give WebView ~800ms to resume after screen-on before firing the transcript
        Handler(Looper.getMainLooper()).postDelayed({
            fireJs("window.__handleScreenOffTranscript&&window.__handleScreenOffTranscript('${escJs(transcript)}');")
        }, 800)
    }

    // Holds a transcript received via onNewIntent() while the activity was stopped.
    // Delivered in onResume() once the WebView is live, not in onNewIntent() where
    // the WebView is paused and evaluateJavascript() silently drops.
    private var pendingWakeTranscript: String? = null

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        // Store transcript instead of firing immediately — the WebView is paused here.
        // onResume() will pick it up once the WebView is live.
        intent.getStringExtra("wake_word_transcript")?.let { transcript ->
            pendingWakeTranscript = transcript
            intent.removeExtra("wake_word_transcript")
            // Re-apply show-when-locked flags on every delivery. The flags set in onCreate()
            // may not survive the stop→resume cycle, so refresh them here before onResume().
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
            } else {
                @Suppress("DEPRECATION")
                window.addFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        isInForeground = true
        // Mid-reply and the user opened the app: the in-app Stop button takes over,
        // so drop the assistant session's Z overlay (it would sit over the UI).
        ZynkAssistantSession.current?.hideOverlay()
        // Deliver transcript stored by onNewIntent() (stopped-activity path), or
        // fall back to checking the creation intent (fresh-launch path).
        val pending = pendingWakeTranscript
        pendingWakeTranscript = null
        if (pending != null) {
            Handler(Looper.getMainLooper()).postDelayed({
                fireJs("window.__handleScreenOffTranscript&&window.__handleScreenOffTranscript('${escJs(pending)}');")
            }, 800)
        } else {
            handleScreenOffTranscript(intent)
        }
    }

    override fun onPause() {
        super.onPause()
        isInForeground = false
    }

    override fun onWebViewCreate(webView: WebView) {
        webViewRef = WeakReference(webView)
        webView.addJavascriptInterface(FolderPickerBridge(), "AndroidFolderPicker")
        webView.addJavascriptInterface(ZynkbotPathsBridge(), "AndroidPaths")
        webView.addJavascriptInterface(AndroidCameraBridge(), "AndroidCamera")
        webView.addJavascriptInterface(VoskBridge(), "VoskBridge")
        webView.addJavascriptInterface(WakeWordBridge(), "WakeWordBridge")
        webView.addJavascriptInterface(VoiceCommandBridge(), "VoiceCommandBridge")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        // Set before super so the window flag is in place before the activity is shown.
        // Allows the full-screen-intent to launch this activity over a PIN-locked screen.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), REQ_WRITE_STORAGE)
        } else {
            ensureShareDir()
        }
        // Requested one at a time: Android only shows one permission dialog per
        // Activity at once, and a navigate-away Settings screen (manage storage)
        // yanks focus from any dialog still pending. Firing all four together
        // silently dropped the microphone request — the one the wake word needs.
        // Microphone goes first since it's the most important permission in the app.
        permissionQueue.addLast { requestRecordAudioIfNeeded() }
        permissionQueue.addLast { requestNotificationPermissionIfNeeded() }
        permissionQueue.addLast { requestAssistantRoleIfNeeded() }
        permissionQueue.addLast { requestLocalNetworkPermissionIfNeeded() }
        permissionQueue.addLast { requestManageStorageIfNeeded() }
        runNextPermissionRequest()
        extractVoskModelIfNeeded()
        startSyncService()
    }

    // Ask, once per install, to become the phone's digital assistant. This is what
    // gives lock-screen answers and lets the wake-word service survive a reboot
    // without the app being reopened. Uses the launcher API like the other pickers in
    // this file so the sequenced queue advances on the dialog's result (accept,
    // decline, or dismiss alike). Skipped if the role is already held or unavailable.
    private val requestAssistantRole = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { runNextPermissionRequest() }

    private fun requestAssistantRoleIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val prefs = getSharedPreferences("zynkbot_setup", MODE_PRIVATE)
            val rm = getSystemService(RoleManager::class.java)
            val asked = prefs.getBoolean("asked_assistant_role", false)
            if (rm != null && !asked
                && rm.isRoleAvailable(RoleManager.ROLE_ASSISTANT)
                && !rm.isRoleHeld(RoleManager.ROLE_ASSISTANT)) {
                prefs.edit().putBoolean("asked_assistant_role", true).apply()
                try {
                    requestAssistantRole.launch(rm.createRequestRoleIntent(RoleManager.ROLE_ASSISTANT))
                    return
                } catch (e: Exception) {
                    Log.w("MainActivity", "Assistant role request failed: ${e.message}")
                }
            }
        }
        runNextPermissionRequest()
    }

    private val permissionQueue = ArrayDeque<() -> Unit>()

    private fun runNextPermissionRequest() {
        permissionQueue.removeFirstOrNull()?.invoke()
    }

    private fun requestManageStorageIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (_: Exception) {
                    try {
                        startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
                    } catch (_: Exception) {}
                }
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_WRITE_STORAGE &&
            grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            ensureShareDir()
        }
        if (requestCode == REQ_RECORD_AUDIO || requestCode == REQ_NOTIFICATIONS || requestCode == REQ_LOCAL_NETWORK) {
            runNextPermissionRequest()
        }
    }

    private fun zynkShareDir(): File {
        val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val dir = File(downloads, "ZynkbotShare")
        if (!dir.mkdirs() && !dir.exists()) {
            error("Could not create ZynkbotShare in Downloads — check storage permissions")
        }
        return dir
    }

    private fun ensureShareDir() {
        zynkShareDir()
    }

    private fun requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQ_NOTIFICATIONS)
                return
            }
        }
        runNextPermissionRequest()
    }

    // Android 16 (SDK 36) makes LAN access a runtime permission. Without an explicit
    // request, GrapheneOS keeps the auto-grant in REVOKE_WHEN_REQUESTED state and
    // silently blocks connections to 192.168.x.x / 10.x.x.x, breaking ZynkSync pairing.
    // Requesting it here converts the compat auto-grant into a user-affirmed grant.
    private fun requestLocalNetworkPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= 36) {
            try {
                if (ContextCompat.checkSelfPermission(this, PERM_ACCESS_LOCAL_NETWORK)
                    != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(arrayOf(PERM_ACCESS_LOCAL_NETWORK), REQ_LOCAL_NETWORK)
                    return
                }
            } catch (_: Exception) {
                // Older devices without the permission constant will throw — ignore.
            }
        }
        runNextPermissionRequest()
    }

    private fun extractVoskModelIfNeeded() {
        val modelDir = File(filesDir, "vosk-model")
        if (modelDir.exists() && modelDir.list()?.isNotEmpty() == true) return
        Thread {
            try {
                copyAssetDir("vosk-model", modelDir)
            } catch (e: Exception) {
                android.util.Log.e("VoskModel", "Failed to extract bundled model: ${e.message}")
            }
        }.start()
    }

    private fun copyAssetDir(assetPath: String, destDir: File) {
        val children = assets.list(assetPath) ?: return
        if (children.isEmpty()) {
            destDir.parentFile?.mkdirs()
            assets.open(assetPath).use { input -> FileOutputStream(destDir).use { input.copyTo(it) } }
        } else {
            destDir.mkdirs()
            for (child in children) {
                copyAssetDir("$assetPath/$child", File(destDir, child))
            }
        }
    }

    // Prompt once per app launch for USE_FULL_SCREEN_INTENT if it isn't already
    // allowed. There is no runtime-permission dialog for it on Android 14+; the app
    // must send the user to a dedicated Settings page. canUseFullScreenIntent()
    // means we never nag once it's granted. Wrapped in try/catch because some OEM
    // ROMs (e.g. ColorOS) have thrown on the appops path historically.
    private var promptedFullScreenIntent = false
    private fun ensureFullScreenIntentPermission() {
        if (promptedFullScreenIntent) return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) return
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.canUseFullScreenIntent()) { promptedFullScreenIntent = true; return }
        promptedFullScreenIntent = true
        try {
            startActivity(
                Intent(Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT).apply {
                    data = Uri.parse("package:$packageName")
                }
            )
        } catch (e: Exception) {
            Log.w("MainActivity", "Could not open full-screen-intent settings: ${e.message}")
        }
    }

    private fun requestRecordAudioIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_RECORD_AUDIO)
        } else {
            runNextPermissionRequest()
        }
    }

    private fun startSyncService() {
        val intent = Intent(this, SyncForegroundService::class.java)
        ContextCompat.startForegroundService(this, intent)
    }
}
