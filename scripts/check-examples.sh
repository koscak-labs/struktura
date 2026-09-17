#!/usr/bin/env bash
# Runs every docs/examples/<name>.cmd for real, normalizes its output, and
# diffs it against docs/examples/<name>.out. Exits 1 if any example drifted.
#
# Usage:
#   scripts/check-examples.sh            # verify all fixtures
#   scripts/check-examples.sh --update   # regenerate all .out files from real runs
#
# The struktura binary used is $STRUKTURA_BIN if set, otherwise whatever
# `struktura` resolves to on PATH. Works in Git Bash on Windows and on Linux.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/docs/examples"
cd "$REPO_ROOT" || exit 2

UPDATE=0
if [[ "${1:-}" == "--update" ]]; then
  UPDATE=1
fi

# Strip ANSI escapes, CRLF, and trailing whitespace so runs on different
# terminals/OSes compare equal.
normalize() {
  sed -e 's/\x1b\[[0-9;]*m//g' -e 's/\r$//' -e 's/[ \t]*$//'
}

shopt -s nullglob
cmdfiles=("$EXAMPLES_DIR"/*.cmd)
shopt -u nullglob

if [[ ${#cmdfiles[@]} -eq 0 ]]; then
  echo "check-examples: no fixtures found in $EXAMPLES_DIR"
  exit 2
fi

fail=0
drifted=()
count=0

for cmdfile in "${cmdfiles[@]}"; do
  name="$(basename "$cmdfile" .cmd)"
  outfile="$EXAMPLES_DIR/$name.out"
  count=$((count + 1))

  actual="$(bash "$cmdfile" 2>&1 | normalize)"

  if [[ $UPDATE -eq 1 ]]; then
    printf '%s\n' "$actual" > "$outfile"
    echo "updated: $name"
    continue
  fi

  if [[ ! -f "$outfile" ]]; then
    echo "MISSING fixture output: $outfile"
    fail=1
    drifted+=("$name")
    continue
  fi

  expected="$(normalize < "$outfile")"

  if [[ "$actual" != "$expected" ]]; then
    echo "DRIFT: $name"
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") || true
    fail=1
    drifted+=("$name")
  fi
done

if [[ $UPDATE -eq 1 ]]; then
  echo "check-examples: updated $count fixture(s)"
  exit 0
fi

if [[ $fail -ne 0 ]]; then
  echo "check-examples: FAIL (${#drifted[@]}/$count drifted: ${drifted[*]})"
  exit 1
fi

echo "check-examples: PASS ($count examples)"
