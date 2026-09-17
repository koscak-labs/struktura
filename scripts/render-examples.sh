#!/usr/bin/env bash
# Replaces the content between each `<!-- example:<name> -->` /
# `<!-- /example -->` marker pair in README.md with the current content of
# docs/examples/<name>.out, so the README is byte-identical to the fixtures
# check-examples.sh verifies. Run scripts/check-examples.sh --update first
# if you changed a .cmd or the binary's output.
#
# Usage: scripts/render-examples.sh
# CI then runs: scripts/render-examples.sh && git diff --exit-code README.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/docs/examples"
README="$REPO_ROOT/README.md"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

awk -v examples_dir="$EXAMPLES_DIR" '
  BEGIN { skip = 0 }
  /^<!-- example:[A-Za-z0-9_-]+ -->$/ {
    print
    name = $0
    sub(/^<!-- example:/, "", name)
    sub(/ -->$/, "", name)
    outfile = examples_dir "/" name ".out"
    n = 0
    while ((getline line < outfile) > 0) { print line; n++ }
    close(outfile)
    if (n == 0) {
      print "render-examples: " outfile " missing or empty" > "/dev/stderr"
      exit 1
    }
    skip = 1
    next
  }
  /^<!-- \/example -->$/ {
    skip = 0
    print
    next
  }
  skip == 1 { next }
  { print }
' "$README" > "$TMP"

mv "$TMP" "$README"
echo "render-examples: README.md regenerated from docs/examples/*.out"
