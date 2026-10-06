BIN="${STRUKTURA_BIN:-struktura}"
echo '$ struktura ab data/ab_a_synthetic.csv data/ab_b_synthetic.csv --col step_ms --lower-is-better'
"$BIN" ab data/ab_a_synthetic.csv data/ab_b_synthetic.csv --col step_ms --lower-is-better
