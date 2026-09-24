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
  # A CRLF checkout (git autocrlf on Windows) leaves "\r" on the last field;
  # it would make the last expected string of every row unmatchable.
  expect="${expect%$'\r'}"
  [[ -z "$id" || "$id" == \#* || "$id" == "id" ]] && continue
  [[ "$speed" == "slow" && $ALL -eq 0 ]] && continue
  ran=$((ran + 1))

  out="$(bash -c "${cmd//struktura /\"$BIN\" }" 2>&1 | strip; exit "${PIPESTATUS[0]}")"
  got_exit=$?
  all_out+="$out"$'\n'

  missing=()
  IFS=$'\n' read -r -d '' -a needles < <(printf '%s' "$expect" | sed 's/ && /\n/g'; printf '\0')
  for n in "${needles[@]}"; do
    LC_ALL=C grep -qF -- "$n" <<<"$out"
    s=$?
    (( s >= 2 )) && { fail=1; echo "ERROR grep exited $s checking $id"; }
    (( s == 0 )) || missing+=("$n")
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
  "detectable from public nasa data"
  "81-100%"
  "predict failure"
  "before it happens"
  "85x faster"
  "dfa detected it"
  "threshold monitors would not have"
)
# Lines containing one of these markers are exempt: they quote someone else's
# work (e.g. a cited paper title), not a struktura claim. Lowercase.
allowed=(
  "mdpi.com/2076-3417/10/23/8489"
)
# Phrases are matched against lowercased text, so they must be lowercase.
for w in "${withdrawn[@]}"; do
  [[ "$w" == "$(LC_ALL=C tr 'A-Z' 'a-z' <<<"$w")" ]] || { echo "ERROR withdrawn phrase not lowercase: $w"; fail=1; }
done
# Everything a reader or a package index shows.
scanned() {
  find README.md REPRODUCIBILITY.md USE_CASES.md llms.txt Cargo.toml src docs/src examples \
    ogma-template assets/terminal-demo.svg -type f \
    \( -name '*.md' -o -name '*.rs' -o -name '*.toml' -o -name '*.py' -o -name '*.txt' -o -name '*.svg' \)
}
# Match on lowercased text with plain `grep -F`: `grep -i` aborts in Git Bash
# on this input (even with LC_ALL=C), and a grep that dies silently would let
# every withdrawn claim through. The phrases are lowercase ASCII.
# grep exit 0 = match, 1 = no match, >= 2 = grep itself failed. A failure must
# fail the gate, never read as "no match".
lower() { LC_ALL=C tr 'A-Z' 'a-z'; }
work="$(mktemp -d)"
printf '%s' "$all_out" | lower > "$work/output"
while IFS= read -r file; do
  mkdir -p "$work/files/$(dirname "$file")"
  lower < "$file" > "$work/files/$file"
done < <(scanned)
for w in "${withdrawn[@]}"; do
  hits=""
  # One recursive grep per phrase over the lowercased copies.
  LC_ALL=C grep -rnF -- "$w" "$work/files" > "$work/hit"
  s=$?
  if (( s >= 2 )); then fail=1; echo "ERROR grep exited $s on the scanned files"; fi
  for a in "${allowed[@]}"; do
    LC_ALL=C grep -vF -- "$a" "$work/hit" > "$work/hit2"; mv "$work/hit2" "$work/hit"
  done
  [[ -s "$work/hit" ]] && hits+="$(sed "s|^$work/files/||" "$work/hit" | cut -c1-140)"$'\n'
  LC_ALL=C grep -qF -- "$w" "$work/output"
  s=$?
  if (( s >= 2 )); then fail=1; echo "ERROR grep exited $s on command output"; fi
  (( s == 0 )) && hits+="(in command output)"
  if [[ -n "${hits//$'\n'/}" ]]; then
    fail=1
    echo "FAIL withdrawn claim is back: \"$w\""
    echo "$hits" | sed 's/^/     /'
  fi
done
rm -rf "$work"

echo "check-claims: $ran rows, $([[ $fail -eq 0 ]] && echo PASS || echo FAIL)"
exit $fail
