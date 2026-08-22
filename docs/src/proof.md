# Cross-Domain Proof

The same algorithm works on completely different signal types.

| Domain | Signal | N | DFA alpha | R-squared |
|--------|--------|---|-----------|-----------|
| Spacecraft | Queue depth | 500 | 0.593 | 0.789 |
| Bearings | CWRU 12kHz vibration | 243,938 | 0.389 | 0.872 |
| Genome | Human chr1 GC% | 8,000 | 0.909 | 0.991 |
| Cardiac | RR intervals | 2,048 | 0.695 | 0.985 |

## Genome: 8 chromosomes at R-squared > 0.99

| Chromosome | DFA alpha | R-squared |
|-----------|-----------|-----------|
| chr1 | 0.909 | 0.991 |
| chr2 | 0.699 | 0.991 |
| chr3 | 0.659 | 0.998 |
| chr4 | 0.894 | 0.997 |
| chr5 | 0.824 | 0.994 |
| chr6 | 0.822 | 0.998 |
| chr7 | 0.862 | 0.997 |
| chr8 | 0.816 | 0.995 |

## Shuffle control

To prove the structure is real and not an artifact, we permute each signal and re-run DFA. If shuffling destroys the alpha (moves it toward 0.5), the original structure was real.

This is the empirical standard: the crate never claims structure it cannot prove.
