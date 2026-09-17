BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura guard data/sylv_spike.csv'
"$BIN" guard data/sylv_spike.csv
rc=$?
echo '$ echo $?'
echo "$rc"
