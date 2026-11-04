#!/usr/bin/env bash
# Tier-one Android checks: what a script can verify about a phone running Zynkbot,
# over ADB from the desktop, without a voice or a finger. Written 2026-11-04 as the
# baseline before the Android code moves into a Tauri plugin; run it again after.
#
#   labs/android_baseline/android_checks.sh <adb serial> <device name on the desktop> [report.md] [label]
#
# The device name is what the desktop's peer list calls the phone (Pixel-FB93, OnePlus),
# used for the sync round-trip check against the desktop's own database. The report is
# Markdown; the exit code is 1 if any check FAILED. Checks that cannot run on this phone
# (a release build hides some things) are marked SKIP with the reason, never silently passed.
#
# What it touches on the phone: opens and closes the assistant session once (as a long
# press of Home would), rings the timer sound for a few seconds and stops it, brings the
# app to the front and sends Back, turns the screen off and on. Nothing is stored.
set -u

SERIAL="${1:?adb serial}"
DEVNAME="${2:?device name as the desktop lists it}"
REPORT="${3:-/dev/stdout}"
LABEL="${4:-}"
PKG=ai.containai.zynkbot
PORT=57963
DESKTOP_DB="${DESKTOP_DB:-$HOME/.local/share/zynkbot/zynkbot.db}"

A() { adb -s "$SERIAL" shell "$@" 2>/dev/null; }
# Carriage returns come back from some Android shells.
clean() { tr -d '\r'; }

pass=0; fail=0; skip=0
rows=()
record() { # status name evidence
  local st="$1" name="$2" ev="$3"
  case "$st" in PASS) pass=$((pass+1));; FAIL) fail=$((fail+1));; SKIP) skip=$((skip+1));; esac
  ev="${ev//|/\\|}"; ev="${ev//$'\n'/ · }"
  rows+=("| $st | $name | $ev |")
  echo "[$st] $name — $ev" >&2
}

svc_dump() { A dumpsys activity services "$PKG" | clean; }
# Prints the block of one service record (the lines up to the next record).
svc_block() { svc_dump | awk -v n="$1" '
  /\* ServiceRecord\{/ { inblk = index($0, n) > 0 }
  inblk { print }'; }
svc_present() { svc_block "$1" | grep -q 'ServiceRecord'; }
svc_foreground() { svc_block "$1" | grep -q 'isForeground=true'; }
port_listening() { A "ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null" | clean | grep -q ":$PORT "; }
app_pid() { A pidof "$PKG" | clean | awk '{print $1}'; }

started_at=$(date -u +%s)
adb -s "$SERIAL" logcat -c 2>/dev/null

# 0. Reachable, installed, which build
if ! adb -s "$SERIAL" get-state 2>/dev/null | clean | grep -q device; then
  echo "Phone $SERIAL is not reachable over ADB." >&2; exit 2
fi
ver=$(A dumpsys package "$PKG" | clean | grep -m1 versionName | sed 's/.*versionName=//')
upd=$(A dumpsys package "$PKG" | clean | grep -m1 lastUpdateTime | sed 's/.*lastUpdateTime=//')
model=$(A getprop ro.product.model | clean)
android=$(A getprop ro.build.version.release | clean)
if [ -n "$ver" ]; then record PASS "App installed" "version $ver, installed $upd, $model, Android $android"
else record FAIL "App installed" "package $PKG not found"; fi

# Warm-up: the sync service and the Rust side start with the activity, so a phone that
# was just installed or rebooted has only the wake service up. Open the app and wait for
# the sync port (up to 45 s); the Back check later sends it to the background again.
A am start -n "$PKG/.MainActivity" >/dev/null
for i in $(seq 1 45); do port_listening && break; sleep 1; done
if port_listening; then record PASS "App opens and the Rust side starts" "sync port $PORT up ${i}s after the activity started"
else record FAIL "App opens and the Rust side starts" "sync port $PORT not up 45 s after the activity started"; fi

# 1. Assistant role: the phone hands "Hey Zynk" turns to Zynkbot's assistant service.
asst=$(A settings get secure assistant | clean)
vis=$(A settings get secure voice_interaction_service | clean)
if [ "$asst" = "$PKG/.ZynkAssistantService" ] && [ "$vis" = "$PKG/.ZynkAssistantService" ]; then
  record PASS "Assistant role held" "assistant and voice_interaction_service both name ZynkAssistantService"
else
  record FAIL "Assistant role held" "assistant='$asst' voice_interaction_service='$vis'"
fi

# 2. The long-running services
if svc_foreground ".WakeWordService"; then
  t=$(svc_block ".WakeWordService" | grep -o 'types=0x[0-9a-f]*' | head -1)
  record PASS "Wake-word service in the foreground" "$t (0x80 = microphone)"
else record FAIL "Wake-word service in the foreground" "not running or not foreground"; fi
if svc_foreground ".SyncForegroundService"; then
  t=$(svc_block ".SyncForegroundService" | grep -o 'types=0x[0-9a-f]*' | head -1)
  record PASS "Sync service in the foreground" "$t (0x1 = dataSync)"
else record FAIL "Sync service in the foreground" "not running or not foreground"; fi
if svc_present ".ZynkAssistantService" && svc_block ".ZynkAssistantService" | grep -q 'c:android'; then
  record PASS "Assistant service bound by the system" "ServiceRecord present, client android"
else record FAIL "Assistant service bound by the system" "no system-bound record (KI-051 shape)"; fi
if svc_present ".ZynkAssistantSessionService"; then
  record PASS "Assistant session service bound" "ServiceRecord present"
else record FAIL "Assistant session service bound" "no record"; fi

# 3. The sync server listens
if port_listening; then record PASS "Sync server listening" "port $PORT open on the phone"
else record FAIL "Sync server listening" "nothing on port $PORT"; fi

# 4. Sync round trip with the desktop: the desktop's peer table records when the phone last
#    reached it. The auto-sync loop runs every 60 s on each side.
if [ -r "$DESKTOP_DB" ] && command -v sqlite3 >/dev/null; then
  seen=$(sqlite3 -readonly "$DESKTOP_DB" "select last_seen_at from zynk_devices where device_name='$DEVNAME' limit 1" 2>/dev/null | clean)
  ip=$(sqlite3 -readonly "$DESKTOP_DB" "select device_ip from zynk_devices where device_name='$DEVNAME' limit 1" 2>/dev/null | clean)
  if [ -n "$seen" ]; then
    seen_s=$(date -u -d "${seen/T/ }" +%s 2>/dev/null || echo 0)
    age=$(( $(date -u +%s) - seen_s ))
    if [ "$age" -le 150 ]; then record PASS "Sync round trip with the desktop" "$DEVNAME at $ip reached the desktop ${age}s ago"
    else record FAIL "Sync round trip with the desktop" "$DEVNAME last reached the desktop ${age}s ago (at $ip)"; fi
  else record SKIP "Sync round trip with the desktop" "no peer named $DEVNAME in the desktop's table"; fi
else record SKIP "Sync round trip with the desktop" "desktop database not readable here"; fi

# 5. The share provider (ZynkLink's files, as the Files app sees them)
root=$(A content query --uri "content://$PKG.share/root" | clean)
if echo "$root" | grep -q 'root_id=zynkbot'; then
  kids=$(A content query --uri "content://$PKG.share/document/root/children" | clean | grep -c '^Row:')
  record PASS "Share provider answers" "root 'Zynkbot' listed, $kids entries under it"
else record FAIL "Share provider answers" "${root:-no rows}"; fi

# 6. The assistant session opens and the wake service hands over and takes back.
#    KEYCODE_ASSIST is what a long press of Home sends.
adb -s "$SERIAL" logcat -c 2>/dev/null
A input keyevent KEYCODE_ASSIST
t0=$(date +%s); shown=""; back=""
for i in $(seq 1 45); do
  sleep 1
  lc=$(adb -s "$SERIAL" logcat -d 2>/dev/null | clean | grep -E 'ZynkAssistantSession|WakeWordService')
  [ -z "$shown" ] && echo "$lc" | grep -q 'Session shown' && shown=$(( $(date +%s) - t0 ))
  [ -n "$shown" ] && echo "$lc" | grep -q '→ LISTENING' && back=$(( $(date +%s) - t0 )) && break
done
if [ -n "$shown" ] && [ -n "$back" ]; then
  eng=$(echo "$lc" | grep -o 'engine=[a-z]*' | head -1)
  record PASS "Assistant session opens and returns" "session shown ($eng), wake service back to LISTENING ${back}s after the assist key"
elif [ -n "$shown" ]; then
  record FAIL "Assistant session opens and returns" "session shown but the wake service did not return to LISTENING within 45 s"
else
  record FAIL "Assistant session opens and returns" "no 'Session shown' line within 45 s of KEYCODE_ASSIST"
fi
A input keyevent KEYCODE_BACK >/dev/null

# 7. The timer ring service: speaks and rings as a short foreground service, stops on request.
adb -s "$SERIAL" logcat -c 2>/dev/null
out=$(adb -s "$SERIAL" shell am start-foreground-service -n "$PKG/.TimerRingService" --es line "Zynkbot baseline check" 2>&1 | clean)
sleep 1
if echo "$out" | grep -qiE 'denial|not allowed|SecurityException' || adb -s "$SERIAL" logcat -d 2>/dev/null | grep -q "Permission Denial: Accessing service $PKG/.TimerRingService"; then
  record SKIP "Timer ring service" "not exported, so the shell may not start it on a release build; timers are a tier-two hand check"
else
  sleep 4
  if svc_foreground ".TimerRingService"; then
    A am startservice -n "$PKG/.TimerRingService" -a "$PKG.TIMER_STOP_RING" >/dev/null
    sleep 3
    if svc_present ".TimerRingService"; then record FAIL "Timer ring service" "rang, but did not stop on TIMER_STOP_RING"
    else record PASS "Timer ring service" "rang in the foreground, stopped on TIMER_STOP_RING"; fi
  else
    err=$(adb -s "$SERIAL" logcat -d 2>/dev/null | clean | grep -E 'TimerRingService' | tail -2)
    record FAIL "Timer ring service" "did not reach the foreground: ${err:-no log}"
  fi
fi

# 8. Back sends the app to the background; everything keeps running.
A am start -n "$PKG/.MainActivity" >/dev/null; sleep 4
pid_before=$(app_pid)
A input keyevent KEYCODE_BACK; sleep 5
pid_after=$(app_pid)
if [ -n "$pid_after" ] && [ "$pid_after" = "$pid_before" ] && svc_foreground ".WakeWordService" && svc_foreground ".SyncForegroundService" && port_listening; then
  record PASS "Survives Back" "same process ($pid_after), both services foreground, port open"
else
  record FAIL "Survives Back" "pid before=$pid_before after=$pid_after; wake fg=$(svc_foreground .WakeWordService && echo y || echo n) sync fg=$(svc_foreground .SyncForegroundService && echo y || echo n) port=$(port_listening && echo open || echo closed)"
fi

# 9. Screen off: the wake service keeps listening, the sync server stays up.
adb -s "$SERIAL" logcat -c 2>/dev/null
A input keyevent KEYCODE_SLEEP; sleep 12
pid_dark=$(app_pid)
dark_ok=0
[ -n "$pid_dark" ] && [ "$pid_dark" = "$pid_before" ] && svc_foreground ".WakeWordService" && port_listening && dark_ok=1
scr=$(adb -s "$SERIAL" logcat -d 2>/dev/null | clean | grep -E 'WakeWordService' | grep -iE 'screen|off|doze' | head -1 | sed 's/.*WakeWordService: //')
A input keyevent KEYCODE_WAKEUP; sleep 2
if [ "$dark_ok" = 1 ]; then record PASS "Survives screen off" "same process, wake service foreground, port open${scr:+; log: $scr}"
else record FAIL "Survives screen off" "pid=$pid_dark (was $pid_before), wake fg=$(svc_foreground .WakeWordService && echo y || echo n), port=$(port_listening && echo open || echo closed)"; fi

# 10. Nothing crashed during the run
crash=$(adb -s "$SERIAL" logcat -d 2>/dev/null | clean | grep -E 'FATAL EXCEPTION|AndroidRuntime.*'"$PKG"'|Process '"$PKG"'.*has died' | head -3)
if [ -z "$crash" ]; then record PASS "No crash during the run" "no FATAL EXCEPTION or process death for $PKG"
else record FAIL "No crash during the run" "$crash"; fi

# 11. The Rust side is logging (its stdout reaches logcat on Android)
rust=$(adb -s "$SERIAL" logcat -d -s RustStdoutStderr:* 2>/dev/null | clean | grep -c '\[')
if [ "$rust" -ge 1 ]; then record PASS "Rust core logging" "$rust Rust lines in logcat since the run started"
else record SKIP "Rust core logging" "no RustStdoutStderr lines in this run's window (quiet, or a release build without them)"; fi

# Report
{
  echo "# Android tier-one checks — $DEVNAME${LABEL:+ — $LABEL}"
  echo
  echo "Run $(date '+%Y-%m-%d %H:%M %Z') from the desktop over ADB ($SERIAL). App $ver installed $upd on $model, Android $android."
  echo
  echo "| result | check | evidence |"
  echo "|---|---|---|"
  printf '%s\n' "${rows[@]}"
  echo
  echo "**$pass passed, $fail failed, $skip skipped.** Hand checks (tier two) are in labs/android_baseline/README.md."
} > "$REPORT"
[ "$REPORT" != /dev/stdout ] && echo "Report: $REPORT" >&2
[ "$fail" -eq 0 ]
