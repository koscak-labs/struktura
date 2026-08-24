"""
struktura Python demo — requires: pip install struktura

85x faster than nolds. same DFA algorithm. zero training.
"""
import struktura

# basic DFA on a signal
import random
random.seed(42)
noise = [random.gauss(0, 1) for _ in range(4096)]

result = struktura.py_dfa(noise)
print(f"white noise: alpha={result.alpha:.4f}, R²={result.r_squared:.4f}")
# expected: alpha ~0.5 (random), R² > 0.9

# full analysis
analysis = struktura.py_analyze(noise)
print(f"full analysis: {analysis}")
print(f"  quality: {analysis.quality}")
print(f"  hurst: {analysis.hurst:.4f}")

# compare two signals
brownian = []
s = 0.0
for x in noise:
    s += x
    brownian.append(s)

verdict = struktura.py_compare(noise, brownian)
print(f"\nnoise vs brownian: {verdict}")

# check if degraded (simple bool)
print(f"degraded? {struktura.py_is_degraded(noise, brownian)}")

# anomaly scores on a signal with a structural shift
signal = noise + brownian  # white noise then brownian = regime change
scores = struktura.py_anomaly_scores(signal, 256, 128, 0.05)
print(f"\nanomaly scores: {len(scores)} windows")
print(f"  baseline (first 5): {[f'{s:.2f}' for s in scores[:5]]}")
print(f"  shifted  (last 5):  {[f'{s:.2f}' for s in scores[-5:]]}")
