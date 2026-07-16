package com.zynkbot.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.DocumentsContract
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import java.io.File
import java.lang.ref.WeakReference


class MainActivity : TauriActivity() {

    private var webViewRef: WeakReference<WebView>? = null

    private val requestStoragePermission = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        // After permission result (granted or denied), launch the folder picker regardless
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
            } catch (e: Exception) { /* ignore non-persistable URIs */ }
            val path = resolveUri(uri) ?: uri.toString()
            val escaped = path.replace("\\", "\\\\").replace("'", "\\'")
            "window.__fpResolve&&window.__fpResolve('$escaped');window.__fpResolve=null;window.__fpReject=null;"
        } else {
            "window.__fpReject&&window.__fpReject('cancelled');window.__fpResolve=null;window.__fpReject=null;"
        }
        wv.post { wv.evaluateJavascript(script, null) }
    }

    inner class FolderPickerBridge {
        @JavascriptInterface
        fun pick() {
            runOnUiThread {
                val permsNeeded = mutableListOf<String>()
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    // Android 13+: only request image permission (video/audio not used)
                    if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                        permsNeeded.add(Manifest.permission.READ_MEDIA_IMAGES)
                    }
                } else if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.S_V2) {
                    // Android 12 and below
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
            val base = getExternalFilesDir(null) ?: filesDir
            val dir = File(base, "ZynkbotShare")
            dir.mkdirs()
            return dir.absolutePath
        }

        @JavascriptInterface
        fun openShareFolder() {
            val base = getExternalFilesDir(null) ?: filesDir
            val dir = File(base, "ZynkbotShare")
            dir.mkdirs()
            runOnUiThread {
                // Try DocumentsUI browsing to the exact directory
                var opened = false
                try {
                    val docId = "primary:Android/data/${packageName}/files/ZynkbotShare"
                    val uri = DocumentsContract.buildDocumentUri(
                        "com.android.externalstorage.documents", docId)
                    val intent = Intent(Intent.ACTION_VIEW).apply {
                        data = uri
                        type = DocumentsContract.Document.MIME_TYPE_DIR
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
                    }
                    startActivity(intent)
                    opened = true
                } catch (_: Exception) {}

                if (!opened) {
                    // Fallback: open the system Files app
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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        ensureShareDir()
        requestNotificationPermissionIfNeeded()
        startSyncService()
    }

    private fun ensureShareDir() {
        val base = getExternalFilesDir(null) ?: filesDir
        File(base, "ZynkbotShare").mkdirs()
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
