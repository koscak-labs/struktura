# How DFA Works

Detrended Fluctuation Analysis measures long-range correlation in a time series.

## The algorithm in 4 steps

1. **Profile**: compute the cumulative sum of deviations from the mean
2. **Box**: divide the profile into non-overlapping boxes of size *s*
3. **Detrend**: fit and subtract a linear trend within each box
4. **Scale**: measure the root-mean-square residual *F(s)* at each box size

The scaling exponent alpha is the slope of log *F(s)* vs log *s*.

## What alpha means

| Alpha range | Interpretation |
|-------------|---------------|
| ~0.5 | White noise (uncorrelated) |
| 0.5 - 1.0 | Long-range correlated (healthy complexity) |
| ~1.0 | 1/f noise (pink noise) |
| > 1.0 | Non-stationary / trend-dominated |

## When it helps anomaly detection

Many healthy systems keep a characteristic alpha, and some faults change it. Whether alpha moves before an amplitude monitor fires depends on the fault. On the bundled CWRU bearing excerpts alpha drops from 0.689 to 0.183 while RMS amplitude rises 11% (`struktura demo`). On the IMS run-to-failure bearing a plain RMS threshold trips earlier than the alpha alarm (see REPRODUCIBILITY.md). So alpha complements amplitude checks; it does not replace them.

## References

1. Peng et al., "Mosaic organization of DNA nucleotide sequences," Physical Review E 49(2), 1994.
2. Peng et al., "Quantification of scaling exponents," Chaos 5(1), 1995.
3. Goldberger et al., "Fractal dynamics in physiology," PNAS 99(suppl 1), 2002.
