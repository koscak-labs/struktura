# REQUIRES: data/mechanical_text.txt
# NOTE: minor float differences (±1 in mean_len, ±0.001 in α) expected across platforms
BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura text data/austen_shuffled.txt data/mechanical_text.txt'
"$BIN" text data/austen_shuffled.txt data/mechanical_text.txt
