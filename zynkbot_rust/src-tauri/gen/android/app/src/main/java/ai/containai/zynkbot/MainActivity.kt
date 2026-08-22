package ai.containai.zynkbot

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.provider.Settings
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
        private const val REQ_RECORD_AUDIO = 200
        // ACCESS_LOCAL_NETWORK is Android 16+ (SDK 36). Using string literal because the
        // constant may not be present when compiling against older SDK stubs, and because
        // GrapheneOS enforces it more strictly than stock Android — see hotfix for LAN
        // access on Pixel 10 / Android 16 devices.
        private const val PERM_ACCESS_LOCAL_NETWORK = "android.permission.ACCESS_LOCAL_NETWORK"
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
            override fun onPartialResult(h: String?) {}
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
                fire("window.__voskError&&window.__voskError('Microphone permission not granted');")
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

    override fun onWebViewCreate(webView: WebView) {
        webViewRef = WeakReference(webView)
        webView.addJavascriptInterface(FolderPickerBridge(), "AndroidFolderPicker")
        webView.addJavascriptInterface(ZynkbotPathsBridge(), "AndroidPaths")
        webView.addJavascriptInterface(AndroidCameraBridge(), "AndroidCamera")
        webView.addJavascriptInterface(VoskBridge(), "VoskBridge")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), REQ_WRITE_STORAGE)
        } else {
            ensureShareDir()
        }
        requestNotificationPermissionIfNeeded()
        requestLocalNetworkPermissionIfNeeded()
        requestManageStorageIfNeeded()
        requestRecordAudioIfNeeded()
        extractVoskModelIfNeeded()
        startSyncService()
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
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0)
            }
        }
    }

    // Android 16 (SDK 36) makes LAN access a runtime permission. Without an explicit
    // request, GrapheneOS keeps the auto-grant in REVOKE_WHEN_REQUESTED state and
    // silently blocks connections to 192.168.x.x / 10.x.x.x, breaking ZynkSync pairing.
    // Requesting it here converts the compat auto-grant into a user-affirmed grant.
    private fun requestLocalNetworkPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < 36) return
        try {
            if (ContextCompat.checkSelfPermission(this, PERM_ACCESS_LOCAL_NETWORK)
                != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(PERM_ACCESS_LOCAL_NETWORK), REQ_LOCAL_NETWORK)
            }
        } catch (_: Exception) {
            // Older devices without the permission constant will throw — ignore.
        }
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

    private fun requestRecordAudioIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_RECORD_AUDIO)
        }
    }

    private fun startSyncService() {
        val intent = Intent(this, SyncForegroundService::class.java)
        ContextCompat.startForegroundService(this, intent)
    }
}
