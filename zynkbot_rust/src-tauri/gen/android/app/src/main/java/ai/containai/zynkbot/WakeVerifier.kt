package ai.containai.zynkbot

import android.os.Build
import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.max

/**
 * Personal wake-word verifier (2026-09-08): a logistic regression over the same
 * 16×96 embedding window the classifier fires on, trained on the desktop from
 * this user's own "Hey Zynk" clips against every false-trigger clip collected.
 * Runs entirely on the phone; nothing is uploaded. The weights are the file
 * zynkbot/wake-word-models/hey_zynk_verifier.json in the app data folder
 * (standardiser mean/scale, coefficients, intercept, threshold), written by the
 * on-phone trainer or received from the owner's other devices through sync or the
 * cloud backup (2026-11-03; nothing ships inside the APK any more, so no one's
 * voice is in the download). No profile that enforces: the wake word answers nothing.
 */
class WakeVerifier private constructor(
    private val mean: FloatArray, private val scale: FloatArray,
    private val coef: FloatArray, private val intercept: Float, val threshold: Float, val trained: String,
    private val owners: Set<String>,
    /** The phone model that taught this profile (2026-11-26). Each phone keeps its own:
     *  a profile taught on one microphone rejected the owner on another (OnePlus,
     *  Pixel-taught, 6 of 8 real wakes rejected on 2026-11-26). */
    val device: String,
) {
    /** True when this file was taught on another phone model: it scores and logs but
     *  blocks nothing, and the Voice page asks the owner to teach this phone. */
    val taughtElsewhere: Boolean get() = device.isNotEmpty() && device != Build.MODEL

    /** A verifier is personal. "owners" is "*" (whoever holds this file: the phone
     *  wrote it from its owner's clips, or it came through the owner's own account or
     *  backup) or a list of Zynkbot user ids, as older files wrote. The user id is
     *  the same on every device that user pairs and survives a reinstall once the
     *  phone is re-paired; a device id does not (a fresh install on 2026-09-09 got a
     *  new device id and the verifier silently stopped enforcing). For anyone else
     *  it scores and logs so labelled clips accumulate, but blocks nothing. */
    fun enforcesOn(context: Context): Boolean {
        if (taughtElsewhere) return false
        // "*" means "whoever this device belongs to": a verifier trained for one
        // person and delivered to that person's phone (custom build or imported file).
        if (owners.contains("*")) return true
        val id = try { java.io.File(context.filesDir, "zynkbot/.zynk_user_id").readText().trim() } catch (_: Exception) { "" }
        return id.isNotEmpty() && owners.contains(id)
    }
    companion object {
        private const val TAG = "WakeVerifier"
        fun profileFile(context: Context) = java.io.File(context.filesDir, "zynkbot/wake-word-models/hey_zynk_verifier.json")

        /** The phone's profile file, or null when there is none (never taught, and nothing
         *  received). Until 2026-11-03 a profile built into the APK stood in; it was one
         *  person's voice and enforced only for that person's user id, which a fresh
         *  install changes, so it silently stopped working (2026-09-09 and 2026-11-02). */
        fun load(context: Context, quiet: Boolean = false): WakeVerifier? {
            val local = profileFile(context)
            if (!local.isFile) { if (!quiet) WakeLog.i(TAG, "No voice profile on this phone yet"); return null }
            return try {
            val text = local.readText(); val source = "data folder"
            val j = JSONObject(text)
            fun arr(k: String): FloatArray { val a = j.getJSONArray(k); return FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val owners = mutableSetOf<String>()
            j.optJSONArray("owners")?.let { a -> for (i in 0 until a.length()) owners.add(a.getString(i)) }
            // Which phone taught it (2026-11-26). A file from before that date has no
            // "device": if this phone's training state says it trained it, stamp it with
            // this model and keep going; otherwise it came from another phone.
            var device = j.optString("device", "")
            if (device.isEmpty()) {
                val trainedHere = try {
                    JSONObject(java.io.File(context.filesDir, "zynkbot/wake-word-models/verifier_training.json").readText()).has("trained")
                } catch (_: Exception) { false }
                if (trainedHere) {
                    device = Build.MODEL
                    try { local.writeText(j.put("device", device).toString()); WakeLog.i(TAG, "Profile stamped as taught on $device (this phone trained it)") } catch (_: Exception) {}
                } else {
                    // Taught before each phone kept its own, with no training state here: it
                    // may be this phone's (restored from the backup onto a fresh install) or
                    // another phone's; the two cannot be told apart, and a foreign profile
                    // that enforces is the worse mistake, so it does not enforce.
                    device = "unknown"
                }
            }
            WakeVerifier(arr("mean"), arr("scale"), arr("coef"), j.getDouble("intercept").toFloat(),
                j.optDouble("threshold", 0.3).toFloat(), j.optString("trained", "?"), owners, device)
                .also { if (!quiet) WakeLog.i(TAG, "Loaded verifier ($source, ${it.coef.size} dims, threshold ${it.threshold}, ${it.trained}); taught on ${it.device}, this phone is ${Build.MODEL}; enforcing here: ${it.enforcesOn(context)}" +
                    (if (it.taughtElsewhere) " (a profile from another phone blocks nothing here; teach this phone)" else "")) }
        } catch (e: Exception) { WakeLog.w(TAG, "No verifier: ${e.message}"); null }
        }

        /** The shift-invariant part of the features: per-dimension mean and max over the
         *  window (2 × dim numbers). This is what the on-phone trainer stores per clip
         *  and what a v3 profile scores. */
        fun pooledFeatures(flatEmb: FloatArray, window: Int, dim: Int): FloatArray {
            val out = FloatArray(dim + dim)
            for (d in 0 until dim) {
                var sum = 0f; var mx = Float.NEGATIVE_INFINITY
                for (w in 0 until window) { val v = flatEmb[w * dim + d]; sum += v; mx = max(mx, v) }
                out[d] = sum / window
                out[dim + d] = mx
            }
            return out
        }
    }

    /** Probability that this embedding window is the owner saying the wake word.
     *  `flatEmb` is the classifier's input: 16 windows × 96 values, oldest first.
     *  Two file formats: v2 uses all 16×96 values in time order plus per-dimension
     *  mean and max (1728 numbers) — that part made it sensitive to where in the
     *  window the phrase sat (2026-09-15 analysis: one 80 ms frame of shift lost
     *  3 of 33 real clips, three frames lost 23). v3 ("meanmax") uses only the
     *  mean and max (192 numbers), which do not move with the phrase. */
    fun score(flatEmb: FloatArray, window: Int, dim: Int): Float {
        val meanMaxOnly = coef.size == dim + dim
        val feats = FloatArray(if (meanMaxOnly) dim + dim else window * dim + dim + dim)
        val base = if (meanMaxOnly) 0 else window * dim
        if (!meanMaxOnly) System.arraycopy(flatEmb, 0, feats, 0, window * dim)
        for (d in 0 until dim) {
            var sum = 0f; var mx = Float.NEGATIVE_INFINITY
            for (w in 0 until window) { val v = flatEmb[w * dim + d]; sum += v; mx = max(mx, v) }
            feats[base + d] = sum / window
            feats[base + dim + d] = mx
        }
        if (feats.size != coef.size) return -1f
        var z = intercept
        for (i in feats.indices) {
            val s = if (scale[i] == 0f) 1f else scale[i]
            z += coef[i] * ((feats[i] - mean[i]) / s)
        }
        return (1.0 / (1.0 + exp(-z.toDouble()))).toFloat()
    }
}
