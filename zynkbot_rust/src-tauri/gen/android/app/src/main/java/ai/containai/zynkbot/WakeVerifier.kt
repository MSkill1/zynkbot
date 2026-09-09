package ai.containai.zynkbot

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.max

/**
 * Personal wake-word verifier (2026-09-08): a logistic regression over the same
 * 16×96 embedding window the classifier fires on, trained on the desktop from
 * this user's own "Hey Zynk" clips against every false-trigger clip collected.
 * Runs entirely on the phone; nothing is uploaded. Weights ship as
 * assets/wake-word-models/hey_zynk_verifier.json (standardiser mean/scale,
 * coefficients, intercept, threshold). While `enforce` is false it only logs.
 */
class WakeVerifier private constructor(
    private val mean: FloatArray, private val scale: FloatArray,
    private val coef: FloatArray, private val intercept: Float, val threshold: Float, val trained: String,
    private val owners: Set<String>,
) {
    /** A verifier is personal: it enforces only for the user it was trained on
     *  (the asset's "owners" lists Zynkbot user ids, .zynk_user_id). The user id is
     *  the same on every device that user pairs and survives a reinstall once the
     *  phone is re-paired; a device id does not (a fresh install on 2026-09-09 got a
     *  new device id and the verifier silently stopped enforcing). For anyone else
     *  it scores and logs so labelled clips accumulate, but blocks nothing. */
    fun enforcesOn(context: Context): Boolean {
        val id = try { java.io.File(context.filesDir, "zynkbot/.zynk_user_id").readText().trim() } catch (_: Exception) { "" }
        return id.isNotEmpty() && owners.contains(id)
    }
    companion object {
        private const val TAG = "WakeVerifier"
        fun load(context: Context): WakeVerifier? = try {
            val text = context.assets.open("wake-word-models/hey_zynk_verifier.json").bufferedReader().readText()
            val j = JSONObject(text)
            fun arr(k: String): FloatArray { val a = j.getJSONArray(k); return FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val owners = mutableSetOf<String>()
            j.optJSONArray("owners")?.let { a -> for (i in 0 until a.length()) owners.add(a.getString(i)) }
            WakeVerifier(arr("mean"), arr("scale"), arr("coef"), j.getDouble("intercept").toFloat(),
                j.optDouble("threshold", 0.3).toFloat(), j.optString("trained", "?"), owners)
                .also { Log.i(TAG, "Loaded verifier (${it.coef.size} dims, threshold ${it.threshold}, ${it.trained}); enforcing here: ${it.enforcesOn(context)}") }
        } catch (e: Exception) { Log.w(TAG, "No verifier: ${e.message}"); null }
    }

    /** Probability that this embedding window is the owner saying the wake word.
     *  `flatEmb` is the classifier's input: 16 windows × 96 values, oldest first. */
    fun score(flatEmb: FloatArray, window: Int, dim: Int): Float {
        val feats = FloatArray(window * dim + dim + dim)
        System.arraycopy(flatEmb, 0, feats, 0, window * dim)
        for (d in 0 until dim) {
            var sum = 0f; var mx = Float.NEGATIVE_INFINITY
            for (w in 0 until window) { val v = flatEmb[w * dim + d]; sum += v; mx = max(mx, v) }
            feats[window * dim + d] = sum / window
            feats[window * dim + dim + d] = mx
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
