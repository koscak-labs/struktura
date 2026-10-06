#!/usr/bin/env bash
# Frozen decision contract for the struktura CLI.
#
#   contract/run.sh <struktura-binary> [--bless] [--only NAME]
#
# Runs every case of contract/cases.tsv against the binary, each in a fresh temp
# dir, normalizes the output and diffs it against contract/expected/. Prints
# PASS / FAIL / XFAIL per case and exits 0 iff every case holds.
#   --bless   regenerate contract/expected/ from this binary (assertions still checked)
#   --only N  run one case
#
# Normalization (nothing else is stripped; a second run must be byte-identical):
#   - JSON lines -> `jq -S -c` (key order is not part of the contract);
#   - the case's temp dir and the fixture dir -> @OUT@ / @FIX@ (they move every run).
# Recorded per case: stdout, stderr (if any), exit code, and every file left in @OUT@.
set -uo pipefail

usage() { echo "usage: $0 <struktura-binary> [--bless] [--only NAME]" >&2; exit 2; }
[ $# -ge 1 ] || usage
BIN=$1; shift
BLESS=0; ONLY=
while [ $# -gt 0 ]; do
  case $1 in
    --bless) BLESS=1 ;;
    --only) shift; ONLY=${1:-}; [ -n "$ONLY" ] || usage ;;
    *) usage ;;
  esac
  shift
done
case $BIN in /*) ;; *) BIN=$PWD/$BIN ;; esac
[ -f "$BIN" ] && [ -x "$BIN" ] || { echo "contract: not an executable file: $BIN" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "contract: jq not found" >&2; exit 2; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FIX=$HERE/fixtures
EXP=$HERE/expected
CASES=$HERE/cases.tsv
[ -f "$CASES" ] || { echo "contract: missing $CASES" >&2; exit 2; }
mkdir -p "$EXP"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/struktura-contract.XXXXXX") || exit 2
trap 'rm -rf "$TMP_ROOT"' EXIT
export LC_ALL=C TZ=UTC
TO=()
command -v timeout >/dev/null 2>&1 && TO=(timeout 120)

# sed-quote a literal for the left side of s|...|...|
sq() { printf '%s' "$1" | sed -e 's/[][\\.*^$|]/\\&/g'; }
# One line in, one line out: sorted compact JSON when the line is JSON, else unchanged.
norm() { jq -R -r -S -c '. as $l | try fromjson catch $l'; }

declare -A BLESSED=()
n_cases=0; n_pass=0; n_xfail=0; n_fail=0

run_case() {
  local name=$1 argv=$2 exp=$3 assert=$4 mode=$5 note=$6
  local out act raw rest cmd rc w f
  out=$(mktemp -d "$TMP_ROOT/case.XXXXXX") || return 2
  act=$TMP_ROOT/$name.actual
  raw=$TMP_ROOT/$name.stdout
  : > "$act"; : > "$raw"
  local subst=(-e "s|$(sq "$out")|@OUT@|g" -e "s|$(sq "$FIX")|@FIX@|g")
  rest=$argv
  while :; do
    cmd=${rest%% ;; *}
    local words=() args=()
    read -r -a words <<< "$cmd"
    for w in "${words[@]}"; do w=${w//@FIX@/$FIX}; w=${w//@OUT@/$out}; args+=("$w"); done
    ( cd "$out" && exec "${TO[@]}" "$BIN" "${args[@]}" ) > "$TMP_ROOT/so" 2> "$TMP_ROOT/se" < /dev/null
    rc=$?
    {
      printf '$ struktura %s\n' "$cmd"
      sed "${subst[@]}" "$TMP_ROOT/so" | norm
      if [ -s "$TMP_ROOT/se" ]; then echo '--- stderr'; sed "${subst[@]}" "$TMP_ROOT/se" | norm; fi
      echo "--- exit $rc"
    } >> "$act"
    cat "$TMP_ROOT/so" >> "$raw"
    [ "$cmd" = "$rest" ] && break
    rest=${rest#* ;; }
  done
  while IFS= read -r f; do
    { echo "--- file $f"; sed "${subst[@]}" "$out/$f" | norm; } >> "$act"
  done < <(cd "$out" && find . -type f | sed 's|^\./||' | sort)

  # 1. the frozen output
  local diff_ok=0
  if [ "$BLESS" = 1 ]; then
    if [ -n "${BLESSED[$exp]:-}" ] && ! cmp -s "$EXP/$exp" "$act"; then
      echo "FAIL  $name: shares expected/$exp with ${BLESSED[$exp]} but its output differs" ; return 1
    fi
    cp "$act" "$EXP/$exp"; BLESSED[$exp]=$name; diff_ok=1
  elif [ -f "$EXP/$exp" ] && cmp -s "$EXP/$exp" "$act"; then
    diff_ok=1
  fi
  # 2. the assertion (1 holds, 0 false, 2 error / none)
  local a=2 jrc
  if [ -n "$assert" ] && [ "$assert" != "-" ]; then
    jq -R -c 'try fromjson catch empty | objects' < "$raw" | jq -s -e "$assert" > /dev/null 2> "$TMP_ROOT/jqerr"
    jrc=$?
    case $jrc in 0) a=1 ;; 1) a=0 ;; *) a=3 ;; esac
  fi

  local why=
  [ "$diff_ok" = 1 ] || why="output differs from expected/$exp"
  if [ "$a" = 3 ]; then why="${why:+$why; }assertion error: $(head -c 300 "$TMP_ROOT/jqerr")"; fi
  if [ "$mode" = xfail ]; then
    if [ "$a" = 2 ]; then why="${why:+$why; }xfail case needs an assertion"
    elif [ "$a" = 1 ]; then why="${why:+$why; }XPASS: the known violation no longer occurs - flip mode to pass in cases.tsv"; fi
  elif [ "$a" = 0 ]; then
    why="${why:+$why; }assertion does not hold: $assert"
  fi
  if [ -n "$why" ]; then
    echo "FAIL  $name: $why"
    if [ "$diff_ok" != 1 ]; then
      if [ -f "$EXP/$exp" ]; then diff -u "$EXP/$exp" "$act" | sed -n '3,40p' | sed 's/^/      /'
      else echo "      (no expected/$exp: run with --bless)"; fi
    fi
    return 1
  fi
  if [ "$mode" = xfail ]; then echo "XFAIL $name: known violation: $note"; return 3; fi
  echo "PASS  $name"
  return 0
}

while IFS=$'\t' read -r name argv exp assert mode note || [ -n "${name:-}" ]; do
  case $name in ''|'#'*) continue ;; esac
  [ -z "$ONLY" ] || [ "$name" = "$ONLY" ] || continue
  if [ -z "${argv:-}" ] || [ -z "${exp:-}" ]; then echo "FAIL  $name: malformed cases.tsv row"; n_cases=$((n_cases + 1)); n_fail=$((n_fail + 1)); continue; fi
  mode=${mode:-pass}
  case $mode in pass|xfail) ;; *) echo "FAIL  $name: unknown mode $mode"; n_cases=$((n_cases + 1)); n_fail=$((n_fail + 1)); continue ;; esac
  n_cases=$((n_cases + 1))
  run_case "$name" "$argv" "$exp" "${assert:--}" "$mode" "${note:--}"
  case $? in 0) n_pass=$((n_pass + 1)) ;; 3) n_xfail=$((n_xfail + 1)) ;; *) n_fail=$((n_fail + 1)) ;; esac
done < "$CASES"

[ "$n_cases" -gt 0 ] || { echo "contract: no cases ran${ONLY:+ (--only $ONLY)}" >&2; exit 2; }
tag=; [ "$BLESS" = 1 ] && tag=' [blessed]'
echo "contract: $n_cases cases, $n_pass pass, $n_xfail xfail (known violations), $n_fail fail$tag"
echo "SUMMARY cases=$n_cases pass=$n_pass xfail=$n_xfail fail=$n_fail"
[ "$n_fail" = 0 ]
