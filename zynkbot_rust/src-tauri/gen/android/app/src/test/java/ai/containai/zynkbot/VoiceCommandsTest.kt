package ai.containai.zynkbot

import ai.containai.zynkbot.VoiceCommands.Cmd
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.util.Calendar

/** Alarm phrases as dictation actually produces them (Matt, Pixel, 2026-10-26). */
class VoiceCommandsTest {
    private fun at(h: Int, m: Int): Calendar = Calendar.getInstance().apply { set(2026, 9, 26, h, m, 0) }

    @Test fun misheardForIsStillAnAlarm() =
        assertEquals(Cmd.Alarm(20, 15), VoiceCommands.parse("set an alarm four eight fifteen pm", at(19, 26)))

    @Test fun aRealFourIsNotAFiller() =
        assertEquals(Cmd.Alarm(16, 15), VoiceCommands.parse("set an alarm 4 15 pm", at(9, 0)))

    @Test fun noAmPmMeansTheNextOneToCome() {
        assertEquals(Cmd.Alarm(20, 15), VoiceCommands.parse("set an alarm for 8:15", at(19, 0)))
        assertEquals(Cmd.Alarm(8, 15), VoiceCommands.parse("set an alarm for 8:15", at(21, 0)))
        assertEquals(Cmd.Alarm(18, 0), VoiceCommands.parse("wake me up at 6", at(7, 0)))
    }

    @Test fun spokenAmPmOrTwentyFourHourIsTakenAsSaid() {
        assertEquals(Cmd.Alarm(8, 15), VoiceCommands.parse("set an alarm for 8:15 am", at(19, 0)))
        assertEquals(Cmd.Alarm(20, 15), VoiceCommands.parse("set an alarm for 20:15", at(21, 0)))
    }

    @Test fun setItForAClockTimeIsAnAlarm() {
        assertEquals(Cmd.Alarm(20, 15), VoiceCommands.parse("set it for 8:15", at(19, 0)))
        assertEquals(Cmd.Alarm(20, 15), VoiceCommands.parse("set it for eight fifteen pm", at(9, 0)))
        assertNull(VoiceCommands.parse("set it for 8", at(19, 0)))
    }
}
