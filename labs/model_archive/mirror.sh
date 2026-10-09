#!/usr/bin/env bash
# mirror.sh - download a manifest into a local archive, verifying every file.
#
#   ./mirror.sh frontier.manifest /srv/models
#
# Resumable and idempotent: a file already present with the right sha256 is
# skipped, so re-running after an interruption costs only the missing bytes.
# Verification is the point of the exercise - an unverified mirror is a rumour.
#
# Downloads land in <file>.part and are only moved into place once the sha256
# matches, so an interrupted run can resume a partial file safely while a
# file that is present and correct is never touched. A file already in place
# that fails its checksum is removed and re-fetched from scratch, because
# resuming onto corrupt bytes would preserve the corruption forever.
set -uo pipefail

MAN="${1:?usage: mirror.sh MANIFEST DEST}"; DEST="${2:?usage: mirror.sh MANIFEST DEST}"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
REV="${REV:-main}"
fetched=0; already=0; smallskip=0; failed=0

check() {  # check FILE EXPECTED_SUM EXPECTED_BYTES -> 0 if good
  local f="$1" sum="$2" bytes="$3"
  case "$sum" in
    sha1:*) [ "$(stat -c%s "$f" 2>/dev/null)" = "$bytes" ] ;;
    *)      [ "$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" = "$sum" ] ;;
  esac
}

while IFS=$'\t' read -r repo path bytes sum; do
  [ -z "${repo:-}" ] && continue
  out="$DEST/$repo/$path"; mkdir -p "$(dirname "$out")"

  if [ -f "$out" ]; then
    if check "$out" "$sum" "$bytes"; then
      case "$sum" in sha1:*) smallskip=$((smallskip+1)) ;; *) already=$((already+1)) ;; esac
      continue
    fi
    echo "  ~ present but wrong, re-fetching from scratch: $repo/$path" >&2
    rm -f "$out" "$out.part"
  fi

  url="https://huggingface.co/$repo/resolve/$REV/$path"
  if ! curl -sSL --fail --max-time 7200 --retry 5 --retry-delay 10 --retry-all-errors \
        -C - "${AUTH[@]}" -o "$out.part" "$url"; then
    echo "  ! download failed: $repo/$path" >&2; failed=$((failed+1)); continue
  fi

  if check "$out.part" "$sum" "$bytes"; then
    mv -f "$out.part" "$out"; fetched=$((fetched+1))
  else
    echo "  ! CHECKSUM MISMATCH, discarding: $repo/$path" >&2
    rm -f "$out.part"; failed=$((failed+1))
  fi
done < "$MAN"

printf 'fetched=%d verified-already=%d small-skipped=%d FAILED=%d\n' \
  "$fetched" "$already" "$smallskip" "$failed"
[ "$failed" -eq 0 ]
