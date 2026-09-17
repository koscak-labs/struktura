BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura text data/austen_shuffled.txt data/mechanical_text.txt'
"$BIN" text data/austen_shuffled.txt data/mechanical_text.txt
