#!/usr/bin/env bash
# verify.sh - re-verify an existing archive against its manifest (bit-rot scrub).
#
#   ./verify.sh frontier.manifest /srv/models
#
# Run this on a schedule. Silent corruption is the failure mode that actually
# destroys long-term archives; ZFS/btrfs scrubs catch media rot, this catches
# everything else, including a truncated download you never noticed.
set -uo pipefail
MAN="${1:?usage: verify.sh MANIFEST DEST}"; DEST="${2:?usage: verify.sh MANIFEST DEST}"
missing=0; bad=0; good=0
while IFS=$'\t' read -r repo path bytes sum; do
  [ -z "${repo:-}" ] && continue
  f="$DEST/$repo/$path"
  [ -f "$f" ] || { echo "MISSING  $repo/$path"; missing=$((missing+1)); continue; }
  case "$sum" in
    sha1:*) [ "$(stat -c%s "$f")" = "$bytes" ] || { echo "SIZE     $repo/$path"; bad=$((bad+1)); continue; } ;;
    *) [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sum" ] || { echo "CORRUPT  $repo/$path"; bad=$((bad+1)); continue; } ;;
  esac
  good=$((good+1))
done < "$MAN"
printf 'ok=%d missing=%d corrupt=%d\n' "$good" "$missing" "$bad"
[ "$missing" -eq 0 ] && [ "$bad" -eq 0 ]
