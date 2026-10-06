#!/usr/bin/env bash
# Contract-gated install of the struktura CLI (the Fuxi judging strand).
#
#   scripts/struktura-install.sh --dest PATH [--ledger FUXI_LEDGER]
#   scripts/struktura-install.sh --dest PATH --rollback [--ledger FUXI_LEDGER]
#
# Install: builds release with CARGO_TARGET_DIR=<repo>/target, copies the binary to
# PATH.new and runs contract/run.sh on exactly those bytes. Only if every case holds:
# PATH is copied to PATH.prev, then PATH.new is renamed onto PATH (atomic). A broken
# contract installs nothing: PATH.new is removed, PATH is untouched, exit 1.
# --rollback: atomically restores PATH.prev onto PATH (PATH.prev is kept).
# --ledger: appends one JSON row per run:
#   {"v":1,"strand":"fuxi","kind":"contract","ts":..,"sha256":..,"cases":..,"pass":..,"xfail":..,"fail":..,"installed":..}
#   (pass counts every case that held, xfail = how many of those are known violations;
#   a build/io error row carries "error"; a rollback writes kind "rollback").
# Exit: 0 installed / rolled back, 1 contract broken, 2 usage, build or io error.
set -uo pipefail

DEST= ; LEDGER= ; ROLLBACK=0
while [ $# -gt 0 ]; do
  case $1 in
    --dest) shift; DEST=${1:-} ;;
    --ledger) shift; LEDGER=${1:-}; [ -n "$LEDGER" ] || { echo "struktura-install: --ledger needs a path" >&2; exit 2; } ;;
    --rollback) ROLLBACK=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "struktura-install: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$DEST" ] || { echo "struktura-install: --dest PATH is required" >&2; exit 2; }
[ -d "$DEST" ] && { echo "struktura-install: --dest is a directory; give the binary's path" >&2; exit 2; }
DIR=$(dirname -- "$DEST")
[ -d "$DIR" ] || { echo "struktura-install: $DIR does not exist" >&2; exit 2; }

sha() { sha256sum -- "$1" | cut -d' ' -f1; }
row() {  # append one ledger row (no-op without --ledger)
  [ -n "$LEDGER" ] || return 0
  printf '%s\n' "$1" >> "$LEDGER" || { echo "struktura-install: cannot append to $LEDGER" >&2; exit 2; }
}
now() { date +%s; }

if [ "$ROLLBACK" = 1 ]; then
  [ -f "$DEST.prev" ] || { echo "struktura-install: no $DEST.prev to roll back to" >&2; exit 2; }
  tmp="$DEST.rollback.$$"
  if ! { cp -p -- "$DEST.prev" "$tmp" && mv -f -- "$tmp" "$DEST"; }; then
    rm -f -- "$tmp"
    row "{\"v\":1,\"strand\":\"fuxi\",\"kind\":\"rollback\",\"ts\":$(now),\"sha256\":null,\"restored\":false,\"error\":\"io\"}"
    echo "struktura-install: rollback failed, $DEST unchanged" >&2; exit 2
  fi
  s=$(sha "$DEST")
  row "{\"v\":1,\"strand\":\"fuxi\",\"kind\":\"rollback\",\"ts\":$(now),\"sha256\":\"$s\",\"restored\":true}"
  echo "struktura-install: rolled back $DEST to $DEST.prev (sha256 $s)"
  exit 0
fi

REPO=$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export CARGO_TARGET_DIR="$REPO/target"
fail_row() { row "{\"v\":1,\"strand\":\"fuxi\",\"kind\":\"contract\",\"ts\":$(now),\"sha256\":${2:-null},\"cases\":0,\"pass\":0,\"xfail\":0,\"fail\":0,\"installed\":false,\"error\":\"$1\"}"; }

echo "struktura-install: building release (CARGO_TARGET_DIR=$CARGO_TARGET_DIR)"
if ! (cd "$REPO" && cargo build --release --bin struktura); then
  fail_row build; echo "struktura-install: build failed, nothing installed" >&2; exit 2
fi
BUILT="$CARGO_TARGET_DIR/release/struktura"
NEW="$DEST.new"
if ! { cp -- "$BUILT" "$NEW" && chmod 755 -- "$NEW"; }; then
  rm -f -- "$NEW"; fail_row stage; echo "struktura-install: cannot stage $NEW" >&2; exit 2
fi
S=$(sha "$NEW")

# The contract runs on the staged bytes, so what passed is exactly what gets installed.
LOG=$(mktemp "${TMPDIR:-/tmp}/struktura-install.XXXXXX") || { rm -f -- "$NEW"; exit 2; }
bash "$REPO/contract/run.sh" "$NEW" > "$LOG" 2>&1
crc=$?
cat "$LOG"
read -r CASES PASS XFAIL FAIL < <(sed -n 's/^SUMMARY cases=\([0-9]*\) pass=\([0-9]*\) xfail=\([0-9]*\) fail=\([0-9]*\)$/\1 \2 \3 \4/p' "$LOG" | tail -1)
rm -f -- "$LOG"
if [ -z "${CASES:-}" ]; then CASES=0; PASS=0; XFAIL=0; FAIL=0; fi
HELD=$((PASS + XFAIL))

if [ "$CASES" = 0 ]; then
  rm -f -- "$NEW"; fail_row contract-runner "\"$S\""
  echo "struktura-install: contract runner failed (exit $crc), nothing installed; $DEST unchanged" >&2; exit 1
fi
if [ "$crc" != 0 ] || [ "$FAIL" != 0 ]; then
  rm -f -- "$NEW"
  row "{\"v\":1,\"strand\":\"fuxi\",\"kind\":\"contract\",\"ts\":$(now),\"sha256\":\"$S\",\"cases\":$CASES,\"pass\":$HELD,\"xfail\":$XFAIL,\"fail\":$((CASES - HELD)),\"installed\":false}"
  echo "struktura-install: contract broken ($((CASES - HELD)) of $CASES cases), nothing installed; $DEST unchanged" >&2
  exit 1
fi

if [ -f "$DEST" ]; then
  if ! { cp -p -- "$DEST" "$DEST.prev.$$" && mv -f -- "$DEST.prev.$$" "$DEST.prev"; }; then
    rm -f -- "$DEST.prev.$$" "$NEW"; fail_row backup "\"$S\""; echo "struktura-install: cannot back up $DEST, nothing installed" >&2; exit 2
  fi
fi
if ! mv -f -- "$NEW" "$DEST"; then
  rm -f -- "$NEW"; fail_row install "\"$S\""; echo "struktura-install: rename onto $DEST failed" >&2; exit 2
fi
row "{\"v\":1,\"strand\":\"fuxi\",\"kind\":\"contract\",\"ts\":$(now),\"sha256\":\"$S\",\"cases\":$CASES,\"pass\":$HELD,\"xfail\":$XFAIL,\"fail\":0,\"installed\":true}"
echo "struktura-install: installed $DEST (sha256 $S; contract $HELD/$CASES held, $XFAIL known violation(s))"
