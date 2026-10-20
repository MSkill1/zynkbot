package ai.containai.zynkbot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.Collections
import java.util.Date
import java.util.Locale
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Trains the personal wake-word verifier on the phone (2026-10-19), the way Siri keeps
 * refining its "Hey Siri" voice profile from accepted utterances: every trigger that
 * ended in an answered question is a positive example, every one that ended in
 * nothing is a negative, and the profile is retrained as they accumulate. Nothing
 * leaves the phone.
 *
 * Features: the 192 "mean + max over the 16-embedding window" numbers the service
 * already has at the fire point (WakeVerifier.pooledFeatures), saved next to each
 * clip as <stem>.feat. Clips from before this existed get theirs computed once from
 * the saved WAV with the resident mel/embedding models (same pipeline, same numbers).
 *
 * Negatives: the phone's own `.false` clips plus a generic set shipped in the APK
 * (assets/wake-word-models/generic_negatives.json: 110 false-trigger clips — noise,
 * television, electronic sounds — collected on the developer's phones; no personal
 * voice). Measured on the desktop 2026-10-19, held out 5-fold: at the 0.5 threshold
 * the profile kept 40 of 45 of the owner's real wakes, stopped 100 of 110 noise
 * clips, and let 75% of a *second person's* "Hey Zynk" through. It is a noise and
 * TV filter, not a voice lock, and must never be described as security.
 *
 * Model: standardised logistic regression, class-balanced, L2 (C = 0.05 as the
 * desktop trainer used), full-batch gradient descent — 192 weights on ~200 rows,
 * well under a second on the phone. Threshold: the 5th percentile of the owner's own
 * training scores, clamped to [0.05, 0.5], so at least 95% of the clips it was
 * trained on pass.
 *
 * The written profile has an empty "owners" list, so WakeVerifier only *logs*
 * ("would reject") until enforcement is switched on deliberately after the field
 * numbers are in (ROADMAP: run log-only, enforce only if under 5% of answered wakes
 * would have been rejected).
 */
object WakeVerifierTrainer {
    private const val TAG = "WakeVerifierTrainer"
    const val MIN_REAL_CLIPS = 20          // fewer positives than this: no profile yet
    const val RETRAIN_EVERY_NEW_REAL = 5   // retrain once this many new real clips have arrived
    private const val DIM = 96
    private const val WINDOW = 16
    private const val FEATS = DIM + DIM
    private const val MEL_WINDOW = 76
    private const val MEL_BINS = 32
    private const val CHUNK = 1280
    private const val MEL_FRAMES_PER_CHUNK = 5

    private fun clipDir(context: Context) = File(context.filesDir, "zynkbot/wake_triggers")
    private fun profileFile(context: Context) = File(context.filesDir, "zynkbot/wake-word-models/hey_zynk_verifier.json")
    private fun stateFile(context: Context) = File(context.filesDir, "zynkbot/wake-word-models/verifier_training.json")

    @Volatile private var busy = false

    /** Save the fire-point features next to a clip (called by the service right after it
     *  writes the WAV). One line of 192 numbers. */
    fun saveFeatures(context: Context, stem: String, pooled: FloatArray) {
        try {
            File(clipDir(context), "$stem.feat").writeText(pooled.joinToString(" ") { "%.5f".format(Locale.US, it) })
        } catch (e: Exception) { Log.w(TAG, "Could not save features for $stem: ${e.message}") }
    }

    /** Train if there is enough new labelled material. Safe to call often; it returns at
     *  once when nothing changed, and runs the work on the calling thread (call it from a
     *  background thread). Returns true when a new profile was written. */
    fun trainIfDue(context: Context, modelDir: String?): Boolean {
        if (busy) return false
        busy = true
        try {
            val dir = clipDir(context)
            if (!dir.isDirectory) return false
            if (modelDir != null) backfillFeatures(dir, File(modelDir))
            val pos = ArrayList<FloatArray>(); val neg = ArrayList<FloatArray>()
            for (f in dir.listFiles { x -> x.name.endsWith(".feat") } ?: emptyArray()) {
                val stem = f.name.removeSuffix(".feat")
                val real = File(dir, "$stem.real").exists(); val falseL = File(dir, "$stem.false").exists()
                if (real == falseL) continue    // unlabelled, or both (KI-076): not training material
                val row = readFeat(f) ?: continue
                if (real) pos.add(row) else neg.add(row)
            }
            if (pos.size < MIN_REAL_CLIPS) {
                Log.i(TAG, "Not training: ${pos.size} real clips with features, need $MIN_REAL_CLIPS")
                return false
            }
            val state = try { JSONObject(stateFile(context).readText()) } catch (_: Exception) { JSONObject() }
            val trainedOn = state.optInt("real_clips", 0)
            if (profileFile(context).isFile && pos.size < trainedOn + RETRAIN_EVERY_NEW_REAL) return false

            val generic = loadGenericNegatives(context)
            val negAll = ArrayList<FloatArray>(neg.size + generic.size).apply { addAll(neg); addAll(generic) }
            if (negAll.isEmpty()) { Log.w(TAG, "Not training: no negatives"); return false }
            val t0 = System.currentTimeMillis()
            val model = fit(pos, negAll)
            val ownScores = pos.map { model.score(it) }.sorted()
            val p5 = ownScores[((ownScores.size - 1) * 0.05).toInt()]
            val threshold = p5.coerceIn(0.05f, 0.5f)
            val stopped = negAll.count { model.score(it) < threshold }
            val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())
            val json = JSONObject()
                .put("version", 3)
                .put("trained", "on this phone $stamp from ${pos.size} real clips vs ${neg.size} own false + ${generic.size} generic negatives")
                .put("threshold", threshold.toDouble())
                .put("mean", JSONArray(model.mean.map { it.toDouble() }))
                .put("scale", JSONArray(model.scale.map { it.toDouble() }))
                .put("coef", JSONArray(model.coef.map { it.toDouble() }))
                .put("intercept", model.intercept.toDouble())
                .put("owners", JSONArray())   // log-only until enforcement is switched on deliberately
                .put("note", "Trained on the phone from the owner's labelled wake clips (WakeVerifierTrainer). A noise/TV filter, not a voice lock.")
            profileFile(context).parentFile?.mkdirs()
            profileFile(context).writeText(json.toString())
            stateFile(context).writeText(JSONObject().put("real_clips", pos.size).put("false_clips", neg.size).put("trained", stamp).toString())
            Log.i(TAG, "Profile trained in ${System.currentTimeMillis() - t0} ms: ${pos.size} real vs ${negAll.size} negatives; " +
                "threshold %.3f keeps %d/%d own clips, stops %d/%d negatives (training set, not held out)".format(
                    Locale.US, threshold, ownScores.count { it >= threshold }, ownScores.size, stopped, negAll.size))
            return true
        } catch (e: Exception) {
            Log.w(TAG, "Training failed: ${e.message}")
            return false
        } finally { busy = false }
    }

    // ── features ─────────────────────────────────────────────────────────────

    private fun readFeat(f: File): FloatArray? = try {
        val v = f.readText().trim().split(Regex("\\s+")).map { it.toFloat() }
        if (v.size == FEATS) v.toFloatArray() else null
    } catch (_: Exception) { null }

    /** Clips saved before features were recorded: run the saved WAV through the same
     *  mel → embedding models and pool the last 16 embeddings, exactly what the service
     *  had at the fire point (the clip ends with the chunk that fired). */
    private fun backfillFeatures(dir: File, modelDir: File) {
        val wavs = dir.listFiles { f -> f.name.endsWith(".wav") } ?: return
        val todo = wavs.filter { wav ->
            val stem = wav.name.removeSuffix(".wav")
            !File(dir, "$stem.feat").exists() && (File(dir, "$stem.real").exists() || File(dir, "$stem.false").exists())
        }
        if (todo.isEmpty()) return
        // Own sessions, not the service's: the audio loop is running inference on those at
        // the same time, and the service treats its sessions as single-threaded.
        val env = OrtEnvironment.getEnvironment()
        val mel = env.createSession(File(modelDir, "melspectrogram.onnx").absolutePath)
        val emb = env.createSession(File(modelDir, "embedding_model.onnx").absolutePath)
        var done = 0
        try {
            for (wav in todo) {
                val stem = wav.name.removeSuffix(".wav")
                try {
                    val pooled = featuresFromWav(wav, env, mel, emb) ?: continue
                    File(dir, "$stem.feat").writeText(pooled.joinToString(" ") { "%.5f".format(Locale.US, it) })
                    done++
                } catch (e: Exception) { Log.w(TAG, "Features from ${wav.name} failed: ${e.message}") }
            }
        } finally { mel.close(); emb.close() }
        if (done > 0) Log.i(TAG, "Computed features for $done older clips")
    }

    private fun featuresFromWav(wav: File, env: OrtEnvironment, mel: OrtSession, emb: OrtSession): FloatArray? {
        val bytes = wav.readBytes()
        if (bytes.size <= 44) return null
        val bb = java.nio.ByteBuffer.wrap(bytes, 44, bytes.size - 44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val n = (bytes.size - 44) / 2
        val pcm = FloatArray(n) { bb.getShort(44 + it * 2) / 32768f }
        val melBuf = ArrayDeque<FloatArray>(); val embBuf = ArrayDeque<FloatArray>()
        var i = 0
        while (i + CHUNK <= n) {
            val chunk = pcm.copyOfRange(i, i + CHUNK)
            val melOut = run(env, mel, chunk, longArrayOf(1, CHUNK.toLong()))
            for (f in 0 until MEL_FRAMES_PER_CHUNK) melBuf.addLast(FloatArray(MEL_BINS) { melOut[f * MEL_BINS + it] })
            while (melBuf.size > MEL_WINDOW) melBuf.removeFirst()
            if (melBuf.size == MEL_WINDOW) {
                val flatMel = FloatArray(MEL_WINDOW * MEL_BINS)
                melBuf.forEachIndexed { k, fr -> fr.copyInto(flatMel, k * MEL_BINS) }
                embBuf.addLast(run(env, emb, flatMel, longArrayOf(1, MEL_WINDOW.toLong(), MEL_BINS.toLong(), 1L)))
                while (embBuf.size > WINDOW) embBuf.removeFirst()
            }
            i += CHUNK
        }
        if (embBuf.size < WINDOW) return null
        val flatEmb = FloatArray(WINDOW * DIM)
        embBuf.forEachIndexed { k, e -> e.copyInto(flatEmb, k * DIM) }
        return WakeVerifier.pooledFeatures(flatEmb, WINDOW, DIM)
    }

    private fun run(env: OrtEnvironment, session: OrtSession, data: FloatArray, shape: LongArray): FloatArray {
        val buf = FloatBuffer.allocate(data.size); buf.put(data); buf.rewind()
        val tensor = OnnxTensor.createTensor(env, buf, shape)
        val results = session.run(Collections.singletonMap(session.inputNames.iterator().next(), tensor))
        tensor.close()
        val out = (results[0] as OnnxTensor).floatBuffer.let { fb -> FloatArray(fb.remaining()).also { fb.get(it) } }
        results.close()
        return out
    }

    private fun loadGenericNegatives(context: Context): List<FloatArray> = try {
        val local = File(context.filesDir, "wake-word-models/generic_negatives.json")
        val text = if (local.isFile) local.readText() else context.assets.open("wake-word-models/generic_negatives.json").bufferedReader().readText()
        val rows = JSONObject(text).getJSONArray("rows")
        List(rows.length()) { r -> val a = rows.getJSONArray(r); FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            .filter { it.size == FEATS }
    } catch (e: Exception) { Log.w(TAG, "No generic negatives: ${e.message}"); emptyList() }

    // ── logistic regression ──────────────────────────────────────────────────

    class Model(val mean: FloatArray, val scale: FloatArray, val coef: FloatArray, val intercept: Float) {
        fun score(x: FloatArray): Float {
            var z = intercept
            for (i in coef.indices) z += coef[i] * ((x[i] - mean[i]) / scale[i])
            return (1.0 / (1.0 + exp(-z.toDouble()))).toFloat()
        }
    }

    /** Same recipe as labs/wake_verifier/train.py (StandardScaler + balanced
     *  LogisticRegression, C = 0.05), solved by full-batch gradient descent. Checked
     *  against it on the desktop 2026-10-19 (labs/wake_verifier/verifier_sim.py). */
    fun fit(pos: List<FloatArray>, neg: List<FloatArray>, iters: Int = 1500, lr: Float = 0.05f, c: Float = 0.05f): Model {
        val n = pos.size + neg.size
        val x = ArrayList<FloatArray>(n).apply { addAll(pos); addAll(neg) }
        val y = FloatArray(n) { if (it < pos.size) 1f else 0f }
        val mean = FloatArray(FEATS); val scale = FloatArray(FEATS)
        for (d in 0 until FEATS) {
            var s = 0.0; for (r in x) s += r[d]; mean[d] = (s / n).toFloat()
            var v = 0.0; for (r in x) { val t = r[d] - mean[d]; v += t * t }
            scale[d] = sqrt(v / n).toFloat().let { if (it == 0f) 1f else it }
        }
        val z = Array(n) { r -> FloatArray(FEATS) { d -> (x[r][d] - mean[d]) / scale[d] } }
        val wPos = n / (2f * pos.size); val wNeg = n / (2f * neg.size)
        val lam = 1f / (c * n)
        val w = FloatArray(FEATS); var b = 0f
        val grad = FloatArray(FEATS)
        repeat(iters) {
            java.util.Arrays.fill(grad, 0f); var gb = 0f
            for (r in 0 until n) {
                var dot = b; val zr = z[r]
                for (d in 0 until FEATS) dot += w[d] * zr[d]
                val p = (1.0 / (1.0 + exp(-dot.toDouble()))).toFloat()
                val g = (p - y[r]) * (if (y[r] == 1f) wPos else wNeg)
                for (d in 0 until FEATS) grad[d] += g * zr[d]
                gb += g
            }
            for (d in 0 until FEATS) w[d] -= lr * (grad[d] / n + lam * w[d])
            b -= lr * (gb / n)
        }
        return Model(mean, scale, w, b)
    }
}
