# REQUIRES: data/smap_msl/train_csv/T-1.csv data/smap_msl/test_csv/T-1.csv
BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura smap --ar 0 --dfa'
"$BIN" smap --ar 0 --dfa
