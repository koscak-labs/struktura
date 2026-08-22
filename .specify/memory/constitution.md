# Struktura Constitution

## Principles

### I. Zero Dependencies
The crate MUST have zero external dependencies in its dependency tree.

### II. Exact-or-Abstain
The crate MUST never claim structure it cannot prove. If R-squared < threshold, quality MUST be Abstain.

### III. Deterministic Output
Given the same input, the crate MUST produce the same output. No randomness in the algorithm.

### IV. No Unsafe Code
The crate MUST NOT contain any unsafe blocks.

### V. Cross-Platform
The crate MUST compile and pass tests on Linux, macOS, and Windows.
