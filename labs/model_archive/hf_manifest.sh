#!/usr/bin/env bash
# hf_manifest.sh - build a verifiable file manifest for Hugging Face model repos.
#
#   ./hf_manifest.sh tiers/frontier.txt > frontier.manifest
#   REV=main EXCLUDE='^original/' ./hf_manifest.sh tiers/frontier.txt > filtered.manifest
#
# Output (TSV): repo <TAB> path <TAB> bytes <TAB> sha256
# sha256 is the LFS object id for weight files; small non-LFS files carry a
# git blob sha1 instead and are emitted with a "sha1:" prefix.
#
# Honours $HF_TOKEN (needed for gated repos such as Meta's Llama) and backs off
# on HTTP 429, which the Hub returns per 5-minute window.
set -uo pipefail

REV="${REV:-main}"
EXCLUDE="${EXCLUDE:-}"          # regex of paths to skip, e.g. '^original/'
INCLUDE="${INCLUDE:-}"          # regex of paths to keep, applied after EXCLUDE
API="https://huggingface.co/api/models"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")

fetch() {   # fetch URL -> body on stdout, next-url on fd 3; retries on 429/5xx
  local url="$1" try=0 delay=5 hdr body code
  while [ "$try" -lt 6 ]; do
    hdr=$(mktemp); body=$(curl -sS --max-time 120 -w '%{http_code}' -D "$hdr" "${AUTH[@]}" "$url")
    code="${body: -3}"; body="${body%???}"
    case "$code" in
      200) sed -n 's/.*<\([^>]*\)>; *rel="next".*/\1/p' "$hdr" | head -1 >&3; rm -f "$hdr"
           printf '%s' "$body"; return 0 ;;
      429|5??) rm -f "$hdr"; sleep "$delay"; delay=$((delay*2)); try=$((try+1)) ;;
      *)   rm -f "$hdr"; echo "  ! HTTP $code for $url" >&2; return 1 ;;
    esac
  done
  echo "  ! gave up after retries: $url" >&2; return 1
}

while read -r repo; do
  [ -z "$repo" ] && continue
  case "$repo" in \#*) continue ;; esac
  url="$API/$repo/tree/$REV?recursive=true&expand=true&limit=100"
  n=0; bytes=0
  while [ -n "$url" ]; do
    next=$(mktemp)
    body=$(fetch "$url" 3>"$next") || { rm -f "$next"; break; }
    while IFS=$'\t' read -r path sz oid lfsoid; do
      [ -z "$path" ] && continue
      [ -n "$EXCLUDE" ] && [[ "$path" =~ $EXCLUDE ]] && continue
      [ -n "$INCLUDE" ] && ! [[ "$path" =~ $INCLUDE ]] && continue
      if [ "$lfsoid" != "null" ] && [ -n "$lfsoid" ]; then sum="$lfsoid"; else sum="sha1:$oid"; fi
      printf '%s\t%s\t%s\t%s\n' "$repo" "$path" "$sz" "$sum"
      n=$((n+1)); bytes=$((bytes+sz))
    done < <(printf '%s' "$body" | jq -r '
        .[] | select(.type=="file")
        | [.path, (.lfs.size // .size // 0), (.oid // "null"), (.lfs.oid // "null")] | @tsv' 2>/dev/null)
    url=$(cat "$next"); rm -f "$next"
  done
  printf '  %-58s %5d files %9.1f GB\n' "$repo" "$n" "$(echo "$bytes" | awk '{print $1/1e9}')" >&2
done < "${1:-/dev/stdin}"
