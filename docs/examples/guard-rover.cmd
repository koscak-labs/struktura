BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura guard examples/rover.csv --baseline 1000'
"$BIN" guard examples/rover.csv --baseline 1000
