package com.zynkbot.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.DocumentsContract
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.lang.ref.WeakReference


class MainActivity : TauriActivity() {

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
            val dir = zynkShareDir()
            dir.mkdirs()
            return dir.absolutePath
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
            zynkShareDir().mkdirs()
            runOnUiThread {
                try {
                    val intent = Intent("android.intent.action.VIEW_DOWNLOADS")
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    startActivity(intent)
                } catch (_: Exception) {
                    try {
                        val intent = packageManager.getLaunchIntentForPackage("com.google.android.apps.nbu.files")
                            ?: packageManager.getLaunchIntentForPackage("com.android.documentsui")
                            ?: Intent(Intent.ACTION_VIEW).apply { type = "resource/folder" }
                        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        startActivity(intent)
                    } catch (_: Exception) {}
                }
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

    override fun onWebViewCreate(webView: WebView) {
        webViewRef = WeakReference(webView)
        webView.addJavascriptInterface(FolderPickerBridge(), "AndroidFolderPicker")
        webView.addJavascriptInterface(ZynkbotPathsBridge(), "AndroidPaths")
        webView.addJavascriptInterface(AndroidCameraBridge(), "AndroidCamera")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        ensureShareDir()
        requestNotificationPermissionIfNeeded()
        startSyncService()
    }

    private fun zynkShareDir(): File {
        val downloads = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val preferred = File(downloads, "Zynkbot")
        if (preferred.mkdirs() || preferred.exists()) return preferred
        val fallback = File(getExternalFilesDir(null) ?: filesDir, "ZynkbotShare")
        fallback.mkdirs()
        return fallback
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

    private fun startSyncService() {
        val intent = Intent(this, SyncForegroundService::class.java)
        ContextCompat.startForegroundService(this, intent)
    }
}
