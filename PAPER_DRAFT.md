# Universal Structural Health Detection via Detrended Fluctuation Analysis

Phil Koščák

## Abstract

We demonstrate that a single algorithm — Detrended Fluctuation Analysis (DFA, Peng 1994) — detects structural changes across 10 domains spanning 24 orders of magnitude in scale, from mitochondrial DNA (16 kbp) to the cosmic microwave background (13.8 Gyr). The entire implementation is 98 lines of C with zero dependencies, zero training, and zero domain-specific configuration. We provide empirical results on public datasets including NASA Voyager 1 magnetometer data (heliopause crossing detection), NASA/IMS bearing run-to-failure prognostics (early warning 2+ hours before collapse), and human-chimpanzee genome comparison (10-20x structural divergence at Human Accelerated Regions vs genome-wide average). All results are reproducible via a single command-line tool.

## 1. Introduction

Complex systems maintain their function through long-range temporal or spatial correlations. When these correlations change, the system is transitioning — whether that system is a bearing wearing out, a spacecraft crossing a physical boundary, or a genome evolving under selection pressure. DFA measures the scaling exponent α of these correlations: α=0.5 indicates random (no memory), α>0.5 indicates persistent structure, and α>1.0 indicates strong long-range dependence.

## 2. Cross-Domain Results

All measurements performed with struktura v2.x (Rust, https://crates.io/crates/struktura).

### Table 1. Universal Structural Measurement

| Domain | Signal | Healthy/Ref α | Changed α | Δα | Detection |
|--------|--------|--------------|-----------|-----|-----------|
| CMB | WMAP TT power spectrum | 1.252 | — | — | Cosmic structure |
| Financial | S&P 500 150yr prices | 1.529 | — | — | Long memory |
| Heliosphere | Voyager 1 B-field (pre) | 1.137 | 1.056 (post) | -0.081 | Heliopause |
| Nuclear DNA | Human chr1 GC% 1kb | 0.987 | 0.936 (chimp) | -0.051 | Species divergence |
| Solar | Sunspot numbers 300yr | 0.666 | — | — | Solar cycle |
| Seismic | USGS earthquakes 2024 | 0.602 | — | — | Magnitude correlations |
| Mechanical | CWRU bearing vibration | 0.689 | 0.183 (fault) | -0.506 | Bearing failure |
| Cardiac | HRV (literature) | 0.695 | 0.483 (arrhythmic) | -0.212 | Disease detection |
| Mitochondrial | Human mito GC% 100bp | 0.515 | 0.532 (chimp) | +0.017 | Conserved |
| Prognostic | IMS run-to-failure | 0.167 | 0.472 (pre-fail) | +0.305 | 2h early warning |

### Table 2. Human vs Chimpanzee at Human Accelerated Regions

| Location | Human α | Chimp α | |Δα| | vs genome avg |
|----------|---------|---------|------|---------------|
| chr1:10.2M | 1.692 | 0.535 | 1.157 | 22.7x |
| chr1:181.2M | 0.124 | 0.916 | 0.792 | 15.5x |
| chr1:91.2M | 1.187 | 0.471 | 0.716 | 14.0x |
| chr1:2.1M | 1.141 | 0.741 | 0.400 | 7.8x |
| Genome avg | 0.987 | 0.936 | 0.051 | 1.0x |

## 3. Key Findings

1. **Three natural clusters**: cosmic/information systems (α≈1.0-1.5), geophysical/mechanical systems (α≈0.6-0.7), minimal-structure systems (α≈0.5).

2. **Heliopause detection**: α shifts from 1.137 (heliosphere) to 1.056 (interstellar) at the boundary. The magnetic field's correlation structure changes when Voyager leaves the solar system.

3. **Bearing prognostics**: α spikes from 0.167 to 0.472 (structure stiffens) 2+ hours before final collapse to 0.107. The bearing enters a new mechanical regime before it destroys itself.

4. **Genomic architecture**: DFA structural divergence at Human Accelerated Regions is 10-20x larger than the genome-wide average. The same function that detects bearing faults finds what makes human DNA different from chimpanzee DNA.

5. **Mitochondrial conservation**: Human vs chimp mitochondrial DNA shows α=0.515 vs 0.532 (Δ=+0.017, HEALTHY) — consistent with the endosymbiotic origin and strong purifying selection on mitochondrial genomes.

## 4. Implementation

The core algorithm is 98 lines of C (dfa_core.h), embedded in a Rust crate with zero dependencies. All demos are reproducible:

```
cargo install struktura
struktura demo           # bearing fault detection
struktura voyager        # spacecraft anomaly
struktura heliopause     # edge of the solar system
struktura ims            # run-to-failure early warning
struktura genome seq.fa  # genome structural profile
```

## 5. Limitations

- The IMS bearing result shows DFA and RMS detecting the fault at similar times — DFA does not consistently lead amplitude-based monitoring on this specific dataset.
- HAR coordinates used were approximate from literature review, not from a formal catalog with exact positions.
- CMB and financial results are single measurements without a structural change comparison.
- Cardiac results are from literature, not independently verified in this study.

## Data Availability

All datasets are public: NASA SPDF (Voyager), NASA DASHlink (IMS), CWRU (bearings), UCSC (genomes), WMAP (CMB), USGS (earthquakes), SIDC (sunspots).

## References

Peng C-K et al. (1994) Mosaic organization of DNA nucleotide sequences. Phys Rev E 49(2).
Gurnett DA et al. (2013) In Situ Observations of Interstellar Plasma with Voyager 1. Science 341(6153).
Pollard KS et al. (2006) An RNA gene expressed during cortical development evolved rapidly in humans. Nature 443(7108).
Goldberger AL et al. (1996) Non-linear dynamics, fractals, and chaos theory: Implications for neuroautonomic heart rate control. Comp Biochem Physiol.

## Detectability bound (analytic, with measured calibration)

The DFA exponent estimated over a window of n samples carries irreducible
statistical scatter. Modeling F²(s) as chi-square-like with (n/s)(s−2)
degrees of freedom gives var(ln F(s)) ≈ 1/(2(n/s)(s−2)), which propagated
through the OLS slope yields an analytic sd(α̂). Because the box scales share
one integrated profile, their fluctuations are positively correlated and the
analytic value is a LOWER bound; the measured inflation factor is ≈1.3x at
n = 96 rising to ≈4x at n = 384 (white noise, 800 windows per point — test
`analytic_alpha_sd_bounds_measured_scatter`).

Consequence: a structural shift is detectable only when |Δα| exceeds a few
multiples of the measured sd(α̂) at the observation scale. This is the
mechanism behind the empirical resolution law measured for
correlation-structure faults occupying part of a longer sequence (fault
duration → detection rate: 84 samples → 35%, 140 → 42%, 210 → 74%,
294 → 88% at 700-sample sequences, FPR ≤ 5%): a short structural fault
moves the whole-sequence α̂ by less than its own scatter, and becomes
detectable as its share of the analyzed window grows. The streaming
monitor's window (96) sets its resolution floor the same way: faults
briefer than the window's α scatter allows are caught, if at all, by the
value-domain legs — the two-axis complementarity measured in the
hybrid-monitor benchmark.
