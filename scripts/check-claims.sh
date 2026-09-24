#!/usr/bin/env bash
# Re-runs every row of docs/claims.tsv and fails if a public number drifted or
# a control stopped being silent. Also fails if a withdrawn claim reappears in
# README.md, src/, or any command output.
#
# Usage:
#   scripts/check-claims.sh          # fast rows only (every push)
#   scripts/check-claims.sh --all    # fast + slow rows (weekly)
#
# Uses $STRUKTURA_BIN if set, otherwise `struktura` on PATH. Set it: a stale
# install on PATH gives stale numbers.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

BIN="${STRUKTURA_BIN:-struktura}"
ALL=0
[[ "${1:-}" == "--all" ]] && ALL=1

strip() { sed -e 's/\x1b\[[0-9;]*m//g' -e 's/\r$//'; }

fail=0
ran=0
all_out=""

while IFS=$'\t' read -r id speed want_exit cmd expect; do
  [[ -z "$id" || "$id" == \#* || "$id" == "id" ]] && continue
  [[ "$speed" == "slow" && $ALL -eq 0 ]] && continue
  ran=$((ran + 1))

  out="$(bash -c "${cmd//struktura /\"$BIN\" }" 2>&1 | strip; exit "${PIPESTATUS[0]}")"
  got_exit=$?
  all_out+="$out"$'\n'

  missing=()
  IFS=$'\n' read -r -d '' -a needles < <(printf '%s' "$expect" | sed 's/ && /\n/g'; printf '\0')
  for n in "${needles[@]}"; do
    LC_ALL=C grep -qF -- "$n" <<<"$out" || missing+=("$n")
  done

  if [[ "$got_exit" != "$want_exit" || ${#missing[@]} -gt 0 ]]; then
    fail=1
    echo "FAIL $id: exit $got_exit (want $want_exit)"
    for m in "${missing[@]}"; do echo "     missing: $m"; done
  else
    echo "ok   $id"
  fi
done < docs/claims.tsv

# Claims withdrawn after failing their controls. They must not come back.
withdrawn=(
  "323 samples"
  "early warning"
  "before failure"
  "detectable from public NASA data"
  "81-100%"
)
# Match on lowercased text with plain `grep -F`: `grep -i` aborts in Git Bash
# on this input (even with LC_ALL=C), and a grep that dies silently would let
# every withdrawn claim through. The phrases are lowercase ASCII.
lower() { LC_ALL=C tr 'A-Z' 'a-z'; }
out_lower="$(printf '%s' "$all_out" | lower)"
for w in "${withdrawn[@]}"; do
  hits=""
  while IFS= read -r file; do
    h="$(lower < "$file" | LC_ALL=C grep -nF -- "$w" | cut -c1-120)"
    [[ -n "$h" ]] && hits+="$file: $h"$'\n'
  done < <(find README.md src -type f)
  LC_ALL=C grep -qF -- "$w" <<<"$out_lower" && hits+="(in command output)"
  if [[ -n "${hits//$'\n'/}" ]]; then
    fail=1
    echo "FAIL withdrawn claim is back: \"$w\""
    echo "$hits" | sed 's/^/     /'
  fi
done

echo "check-claims: $ran rows, $([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit $fail
