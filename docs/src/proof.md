# Cross-Domain Results

The same DFA runs on very different signals. These are α values measured on bundled data; only the bearing pair is a normal-vs-fault comparison.

| Domain | Signal | DFA α | R² |
|--------|--------|-------|----|
| Bearings | CWRU 12 kHz vibration, normal | 0.689 | |
| Bearings | CWRU 12 kHz vibration, inner-race fault | 0.183 | |
| Genome | Human chr1 GC% in 1 kb windows (`struktura genome`) | 0.967 | 0.980 |

## Shuffle control

To check that a measured structure is real and not an artifact, permute the signal and run DFA again. If shuffling moves α toward 0.5, the order of the values carried the structure. `struktura prove <file>` runs this test with a bootstrap interval on α.

The full list of checked numbers, with commands and controls, is [docs/claims.tsv](https://github.com/koscak-labs/struktura/blob/master/docs/claims.tsv).
