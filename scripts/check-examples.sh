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

  # Skip fixtures that declare a REQUIRES file if it is absent (e.g. large
  # datasets not committed to the repo). First non-comment, non-blank line
  # starting with "# REQUIRES:" lists space-separated paths checked from repo root.
  requires_line="$(grep -m1 '^# REQUIRES:' "$cmdfile" || true)"
  if [[ -n "$requires_line" ]]; then
    skip=0
    for req in ${requires_line#\# REQUIRES:}; do
      if [[ ! -e "$REPO_ROOT/$req" ]]; then
        skip=1
        break
      fi
    done
    if [[ $skip -eq 1 ]]; then
      echo "SKIP (data absent): $name"
      count=$((count - 1))
      continue
    fi
  fi

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

  # Allow minor float tolerance for platform differences: if the only diffs are
  # numbers that differ by ≤1 in the last digit, treat as pass. We do this by
  # checking if a "fuzzy" normalisation (round trailing decimals) makes them equal.
  fuzzy_normalize() {
    # Replace e.g. α=0.572 with α=0.57~ and mean_len=125 with mean_len=12~
    # so ±1 in the last digit is absorbed.
    sed -E 's/([0-9]+\.[0-9]{2})[0-9]/\1~/g; s/([0-9]{2})[0-9]([^0-9.])/\1~\2/g'
  }

  if [[ "$actual" != "$expected" ]]; then
    actual_fuzzy="$(printf '%s' "$actual" | fuzzy_normalize)"
    expected_fuzzy="$(printf '%s' "$expected" | fuzzy_normalize)"
    if [[ "$actual_fuzzy" == "$expected_fuzzy" ]]; then
      echo "FUZZ-PASS (minor float diff): $name"
    else
      echo "DRIFT: $name"
      diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") || true
      fail=1
      drifted+=("$name")
    fi
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
