#!/usr/bin/env bash
# plan.sh - turn a manifest into storage, money and time.
#
#   ./plan.sh frontier.manifest [usd_per_raw_tb] [parity_factor] [mbytes_per_sec]
set -uo pipefail
MAN="${1:?usage: plan.sh MANIFEST [usd/TB] [parity] [MB/s]}"
USD="${2:-30}"; PAR="${3:-1.4}"; MBS="${4:-100}"
awk -F'\t' -v usd="$USD" -v par="$PAR" -v mbs="$MBS" '
  { b+=$3; n++; repo[$1]=1 }
  END {
    r=0; for (k in repo) r++
    tb=b/1e12; raw=tb*par; days=(b/(mbs*1e6))/86400
    printf "repos           %d\n", r
    printf "files           %d\n", n
    printf "payload         %.2f TB\n", tb
    printf "raw w/ parity   %.2f TB  (x%.1f)\n", raw, par
    printf "drive cost      $%.0f  (at $%.0f/raw TB)\n", raw*usd, usd
    printf "pull time       %.1f days  (at %d MB/s sustained, 24/7)\n", days, mbs
  }' "$MAN"
