package com.zynkbot.app

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
                    // Android 13+: granular media permissions
                    listOf(
                        Manifest.permission.READ_MEDIA_IMAGES,
                        Manifest.permission.READ_MEDIA_VIDEO,
                        Manifest.permission.READ_MEDIA_AUDIO
                    ).forEach { perm ->
                        if (ContextCompat.checkSelfPermission(this@MainActivity, perm) != PackageManager.PERMISSION_GRANTED) {
                            permsNeeded.add(perm)
                        }
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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        requestAllFilesAccessIfNeeded()
        requestNotificationPermissionIfNeeded()
        startSyncService()
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

    private fun requestAllFilesAccessIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                    intent.data = Uri.parse("package:$packageName")
                    startActivity(intent)
                } catch (e: Exception) {
                    val fallback = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                    startActivity(fallback)
                }
            }
        }
    }
}
