package ai.containai.zynkbot

import android.content.Context
import android.database.Cursor
import android.database.MatrixCursor
import android.os.CancellationSignal
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.DocumentsContract.Document
import android.provider.DocumentsContract.Root
import android.provider.DocumentsProvider
import android.webkit.MimeTypeMap
import java.io.File
import java.io.FileNotFoundException

/**
 * Zynkbot as a storage location (2026-09-19, KI-015).
 *
 * The share folder used to be Download/ZynkbotShare. Since Android 11 an app can read
 * only the files it created itself in a public folder, so anything another app put
 * there was invisible to Zynkbot and the only way in was the in-app picker. This
 * provider publishes the share folder the way Drive or a USB stick appears in the
 * Files app: a "Zynkbot" entry in the drawer, browsable by any app's picker. Every
 * file that arrives through it is written by this process, so Zynkbot owns it and
 * the Rust scan, the share routes and the Knowledge Base see it unchanged.
 *
 * The folder lives in app storage (filesDir/ZynkbotShare): the provider is the one
 * route in, and Rust reads it by plain path as before. It is not a media folder, so
 * photos in it do not appear in the gallery; they are one tap away in Files.
 *
 * Document ids are paths relative to the folder ("Reports/x.pdf"); the root is "root".
 */
class ZynkShareProvider : DocumentsProvider() {

    companion object {
        const val AUTHORITY = "ai.containai.zynkbot.share"
        const val ROOT_ID = "zynkbot"
        const val ROOT_DOC_ID = "root"
        private const val PREFS = "zynkbot_share"
        private const val PREF_CHANGED = "changed"

        private val ROOT_PROJECTION = arrayOf(
            Root.COLUMN_ROOT_ID, Root.COLUMN_FLAGS, Root.COLUMN_ICON, Root.COLUMN_TITLE,
            Root.COLUMN_SUMMARY, Root.COLUMN_DOCUMENT_ID, Root.COLUMN_AVAILABLE_BYTES,
        )
        private val DOC_PROJECTION = arrayOf(
            Document.COLUMN_DOCUMENT_ID, Document.COLUMN_DISPLAY_NAME, Document.COLUMN_MIME_TYPE,
            Document.COLUMN_FLAGS, Document.COLUMN_SIZE, Document.COLUMN_LAST_MODIFIED,
        )

        /** The share folder; created on first use. Same path MainActivity hands to the page. */
        fun shareDir(context: Context): File = File(context.filesDir, "ZynkbotShare").also { it.mkdirs() }

        /** "notes.txt" -> "notes (2).txt" while the name is taken. Never overwrites. */
        fun uniqueFile(dir: File, name: String): File {
            val safe = name.replace('/', '_').ifBlank { "file" }
            var f = File(dir, safe)
            if (!f.exists()) return f
            val dot = safe.lastIndexOf('.')
            val stem = if (dot > 0) safe.substring(0, dot) else safe
            val ext = if (dot > 0) safe.substring(dot) else ""
            var n = 2
            while (f.exists()) { f = File(dir, "$stem ($n)$ext"); n++ }
            return f
        }

        /**
         * Something outside the Rust index changed the folder (a file arrived through
         * the Files app, the Share button, or a rename/delete). MainActivity checks this
         * on resume and asks the page to re-index; the Files app is told to refresh.
         */
        fun markChanged(context: Context) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(PREF_CHANGED, true).apply()
            notifyFilesApp(context)
        }

        fun takeChanged(context: Context): Boolean {
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val changed = prefs.getBoolean(PREF_CHANGED, false)
            if (changed) prefs.edit().putBoolean(PREF_CHANGED, false).apply()
            return changed
        }

        /** Refresh any Files-app view of the root (after a peer download lands, for instance). */
        fun notifyFilesApp(context: Context) {
            try {
                context.contentResolver.notifyChange(
                    DocumentsContract.buildChildDocumentsUri(AUTHORITY, ROOT_DOC_ID), null)
            } catch (_: Exception) {}
        }
    }

    private lateinit var base: File

    override fun onCreate(): Boolean {
        base = shareDir(context!!)
        return true
    }

    // ---- id <-> file -----------------------------------------------------------

    private fun fileFor(docId: String): File {
        if (docId == ROOT_DOC_ID) return base
        val f = File(base, docId)
        if (!f.canonicalPath.startsWith(base.canonicalPath + File.separator)) throw FileNotFoundException(docId)
        if (!f.exists()) throw FileNotFoundException(docId)
        return f
    }

    private fun docIdFor(f: File): String {
        val rel = f.canonicalPath.removePrefix(base.canonicalPath).trimStart(File.separatorChar)
        return if (rel.isEmpty()) ROOT_DOC_ID else rel
    }

    private fun mimeFor(f: File): String {
        if (f.isDirectory) return Document.MIME_TYPE_DIR
        val ext = f.name.substringAfterLast('.', "").lowercase()
        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext) ?: "application/octet-stream"
    }

    private fun hidden(f: File): Boolean = f.name.startsWith(".") || f.name.endsWith(".part")

    private fun includeFile(cursor: MatrixCursor, f: File) {
        val isRoot = f.canonicalPath == base.canonicalPath
        var flags = 0
        if (f.isDirectory) {
            flags = flags or Document.FLAG_DIR_SUPPORTS_CREATE
            if (!isRoot) flags = flags or Document.FLAG_SUPPORTS_DELETE or Document.FLAG_SUPPORTS_RENAME
        } else {
            flags = flags or Document.FLAG_SUPPORTS_WRITE or Document.FLAG_SUPPORTS_DELETE or Document.FLAG_SUPPORTS_RENAME
        }
        cursor.newRow()
            .add(Document.COLUMN_DOCUMENT_ID, docIdFor(f))
            .add(Document.COLUMN_DISPLAY_NAME, if (isRoot) "Zynkbot" else f.name)
            .add(Document.COLUMN_MIME_TYPE, mimeFor(f))
            .add(Document.COLUMN_FLAGS, flags)
            .add(Document.COLUMN_SIZE, if (f.isFile) f.length() else null)
            .add(Document.COLUMN_LAST_MODIFIED, f.lastModified())
    }

    // ---- DocumentsProvider ------------------------------------------------------

    override fun queryRoots(projection: Array<out String>?): Cursor {
        val cursor = MatrixCursor(projection ?: ROOT_PROJECTION)
        cursor.newRow()
            .add(Root.COLUMN_ROOT_ID, ROOT_ID)
            .add(Root.COLUMN_FLAGS, Root.FLAG_SUPPORTS_CREATE or Root.FLAG_SUPPORTS_IS_CHILD or Root.FLAG_LOCAL_ONLY)
            .add(Root.COLUMN_ICON, R.mipmap.ic_launcher)
            .add(Root.COLUMN_TITLE, "Zynkbot")
            .add(Root.COLUMN_SUMMARY, "Shared files")
            .add(Root.COLUMN_DOCUMENT_ID, ROOT_DOC_ID)
            .add(Root.COLUMN_AVAILABLE_BYTES, base.freeSpace)
        return cursor
    }

    override fun queryDocument(documentId: String, projection: Array<out String>?): Cursor {
        android.util.Log.i("ZynkShareProvider", "queryDocument $documentId")
        val cursor = MatrixCursor(projection ?: DOC_PROJECTION)
        includeFile(cursor, fileFor(documentId))
        return cursor
    }

    override fun queryChildDocuments(parentDocumentId: String, projection: Array<out String>?, sortOrder: String?): Cursor {
        val parent = fileFor(parentDocumentId)
        val cursor = MatrixCursor(projection ?: DOC_PROJECTION)
        val children = (parent.listFiles() ?: emptyArray())
            .filter { !hidden(it) }
            .sortedWith(compareBy({ !it.isDirectory }, { it.name.lowercase() }))
        android.util.Log.i("ZynkShareProvider", "list $parentDocumentId: " +
            children.joinToString { "${it.name} (${if (it.isDirectory) "dir" else "${it.length()} B"})" })
        children.forEach { includeFile(cursor, it) }
        cursor.setNotificationUri(context!!.contentResolver,
            DocumentsContract.buildChildDocumentsUri(AUTHORITY, parentDocumentId))
        return cursor
    }

    override fun isChildDocument(parentDocumentId: String, documentId: String): Boolean {
        val parent = fileFor(parentDocumentId).canonicalPath
        return try { fileFor(documentId).canonicalPath.startsWith(parent + File.separator) } catch (_: FileNotFoundException) { false }
    }

    override fun getDocumentType(documentId: String): String = mimeFor(fileFor(documentId))

    override fun openDocument(documentId: String, mode: String, signal: CancellationSignal?): ParcelFileDescriptor {
        android.util.Log.i("ZynkShareProvider", "openDocument $documentId mode=$mode")
        val f = fileFor(documentId)
        val pfdMode = ParcelFileDescriptor.parseMode(mode)
        val writing = mode.contains('w') || mode.contains('t')
        return if (writing) {
            // Copying a file in through the Files app writes here; index it once closed.
            ParcelFileDescriptor.open(f, pfdMode, Handler(Looper.getMainLooper())) { _ ->
                android.util.Log.i("ZynkShareProvider", "closed after write: ${f.name} (${f.length()} B)")
                markChanged(context!!)
            }
        } else {
            ParcelFileDescriptor.open(f, pfdMode)
        }
    }

    override fun createDocument(parentDocumentId: String, mimeType: String, displayName: String): String {
        val parent = fileFor(parentDocumentId)
        if (!parent.isDirectory) throw FileNotFoundException("not a folder: $parentDocumentId")
        val f = uniqueFile(parent, displayName)
        val ok = if (mimeType == Document.MIME_TYPE_DIR) f.mkdir() else f.createNewFile()
        if (!ok) throw FileNotFoundException("could not create $displayName")
        markChanged(context!!)
        return docIdFor(f)
    }

    override fun deleteDocument(documentId: String) {
        val f = fileFor(documentId)
        if (f.canonicalPath == base.canonicalPath) throw FileNotFoundException("cannot delete the root")
        if (!f.deleteRecursively()) throw FileNotFoundException("could not delete $documentId")
        markChanged(context!!)
    }

    override fun renameDocument(documentId: String, displayName: String): String {
        val f = fileFor(documentId)
        if (f.canonicalPath == base.canonicalPath) throw FileNotFoundException("cannot rename the root")
        val target = uniqueFile(f.parentFile ?: base, displayName)
        if (!f.renameTo(target)) throw FileNotFoundException("could not rename $documentId")
        markChanged(context!!)
        return docIdFor(target)
    }
}
