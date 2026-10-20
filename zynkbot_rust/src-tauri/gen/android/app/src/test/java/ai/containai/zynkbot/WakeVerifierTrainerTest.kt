package ai.containai.zynkbot

import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/** The pure parts of the on-phone trainer: the fit and the held-out estimate that
 *  decides whether a retrained profile is promoted. Synthetic 192-number rows. */
class WakeVerifierTrainerTest {
    private val dim = 192

    /** Rows around a centre with Gaussian spread `sd`. */
    private fun cloud(n: Int, centre: Float, sd: Float, seed: Int): List<FloatArray> {
        val r = Random(seed)
        return List(n) { FloatArray(dim) { centre + (r.nextFloat() + r.nextFloat() + r.nextFloat() - 1.5f) * sd } }
    }

    @Test fun `separable owner and noise clips score well held out`() {
        val pos = cloud(40, 1f, 0.3f, 1)
        val neg = cloud(120, -1f, 0.3f, 2)
        val (keep, stop) = WakeVerifierTrainer.heldOut(pos, neg)
        assertTrue("keep $keep", keep >= 0.9f)
        assertTrue("stop $stop", stop >= 0.9f)
    }

    @Test fun `indistinguishable clips do not score well held out`() {
        val pos = cloud(40, 0f, 1f, 3)
        val neg = cloud(120, 0f, 1f, 4)
        val (keep, stop) = WakeVerifierTrainer.heldOut(pos, neg)
        // With no signal the model only fits noise: whatever it stops, it loses the
        // owner's own held-out clips in return. The pair is clearly worse than the
        // separable case, which is what the promotion rule compares.
        assertTrue("keep $keep stop $stop", keep + stop < 1.5f)
    }

    @Test fun `too few clips for folds gives zero rather than a crash`() {
        val (keep, stop) = WakeVerifierTrainer.heldOut(cloud(3, 1f, 0.1f, 5), cloud(3, -1f, 0.1f, 6))
        assertTrue(keep == 0f && stop == 0f)
    }
}
