//! Telemanom-protocol evaluation with a closed-form predictor.
//!
//! JPL's telemanom pipeline (Hundman et al., KDD 2018) is: per-channel
//! LSTM one-step prediction → |residual| → EWMA smoothing → unsupervised
//! epsilon selection (no labels) → contiguous anomaly sequences → pruning.
//! This module reproduces that pipeline faithfully but replaces the LSTM
//! with ridge autoregression of order `p` fitted on the nominal train
//! split — closed-form least squares, no gradient training, no GPU.
//!
//! The experiment this enables: how much of the benchmark performance is
//! the deep network, and how much is the residual mathematics?

use crate::solve_ridge;

/// Fit y_t = b + Σ_{i=1..p} w_i · y_{t-i} on the train series (ridge).
pub struct ArPredictor {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub p: usize,
}

impl ArPredictor {
    pub fn fit(train: &[f64], p: usize, lambda: f64) -> Option<ArPredictor> {
        let n = train.len();
        if n < p * 3 + 10 {
            return None;
        }
        // Standardize for conditioning
        let mean = train.iter().sum::<f64>() / n as f64;
        let sd = {
            let v = train.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
            crate::sqrt(v).max(1e-9)
        };
        let z: Vec<f64> = train.iter().map(|x| (x - mean) / sd).collect();

        let rows = n - p;
        let dim = p; // centered → no bias column; bias recovered after
        let mut xtx = vec![0.0f64; dim * dim];
        let mut xty = vec![0.0f64; dim];
        for t in p..n {
            let y = z[t];
            for i in 0..p {
                let xi = z[t - 1 - i];
                xty[i] += xi * y;
                for j in 0..p {
                    xtx[i * dim + j] += xi * z[t - 1 - j];
                }
            }
        }
        let l = lambda * rows as f64;
        for i in 0..dim {
            xtx[i * dim + i] += l;
        }
        let mut w = xty.clone();
        if !solve_ridge(&mut xtx, &mut w, dim) {
            return None;
        }
        // Back-transform to raw scale: y = mean + sd*Σ w_i (x_{t-1-i}-mean)/sd
        //   = (mean - Σ w_i mean) + Σ w_i x_{t-1-i}
        let wsum: f64 = w.iter().sum();
        Some(ArPredictor { bias: mean * (1.0 - wsum), weights: w, p })
    }

    /// One-step-ahead |residual| stream over `series` (first p entries 0).
    pub fn residuals(&self, series: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; series.len()];
        for t in self.p..series.len() {
            let mut pred = self.bias;
            for i in 0..self.p {
                pred += self.weights[i] * series[t - 1 - i];
            }
            out[t] = (series[t] - pred).abs();
        }
        out
    }
}

/// EWMA smoothing (telemanom smooths |residual| before thresholding).
pub fn ewma(x: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut out = Vec::with_capacity(x.len());
    let mut s = x.first().copied().unwrap_or(0.0);
    for &v in x {
        s = alpha * v + (1.0 - alpha) * s;
        out.push(s);
    }
    out
}

/// Telemanom's unsupervised epsilon selection: try ε = μ + z·σ for a grid
/// of z; removing the points above ε shrinks the residual mean/sd — pick
/// the z that maximizes that shrink per anomalous point/sequence.
pub fn find_epsilon(errors: &[f64]) -> f64 {
    find_epsilon_from(errors, 2.5)
}

/// Same as [`find_epsilon`] with a configurable lower bound on the z grid —
/// a lower floor trades precision for recall.
pub fn find_epsilon_from(errors: &[f64], z_min: f64) -> f64 {
    let n = errors.len() as f64;
    let mean = errors.iter().sum::<f64>() / n;
    let sd = crate::sqrt(errors.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n)
        .max(1e-12);
    let mut best_eps = mean + 12.0 * sd;
    let mut best_score = f64::MIN;
    let mut z = z_min;
    while z <= 12.0 {
        let eps = mean + z * sd;
        let below: Vec<f64> = errors.iter().cloned().filter(|&e| e < eps).collect();
        let n_above = errors.len() - below.len();
        if n_above == 0 {
            z += 0.5;
            continue;
        }
        let bm = below.iter().sum::<f64>() / below.len() as f64;
        let bsd = crate::sqrt(
            below.iter().map(|x| (x - bm) * (x - bm)).sum::<f64>() / below.len() as f64,
        );
        // sequence count among the above-points
        let mut seqs = 0usize;
        let mut prev_above = false;
        for &e in errors {
            let above = e >= eps;
            if above && !prev_above {
                seqs += 1;
            }
            prev_above = above;
        }
        let score = ((mean - bm) / mean + (sd - bsd) / sd)
            / (n_above as f64 + (seqs * seqs) as f64);
        if score > best_score {
            best_score = score;
            best_eps = eps;
        }
        z += 0.5;
    }
    best_eps
}

/// Contiguous above-epsilon regions, expanded by `buffer`, merged.
pub fn anomaly_sequences(errors: &[f64], eps: f64, buffer: usize) -> Vec<(usize, usize)> {
    let n = errors.len();
    let mut seqs: Vec<(usize, usize)> = Vec::new();
    let mut start: Option<usize> = None;
    for t in 0..n {
        if errors[t] >= eps {
            if start.is_none() {
                start = Some(t);
            }
        } else if let Some(s) = start.take() {
            seqs.push((s.saturating_sub(buffer), (t - 1 + buffer).min(n - 1)));
        }
    }
    if let Some(s) = start {
        seqs.push((s.saturating_sub(buffer), n - 1));
    }
    // merge overlaps
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in seqs {
        if let Some(last) = merged.last_mut() {
            if s <= last.1 + 1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    merged
}

/// Telemanom's pruning: rank sequences by their max smoothed error; walk
/// down the ranking and drop every sequence below the first relative drop
/// smaller than `p_prune` (default 0.13) — weak stragglers are noise.
pub fn prune_sequences(
    errors: &[f64],
    seqs: &[(usize, usize)],
    p_prune: f64,
) -> Vec<(usize, usize)> {
    if seqs.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(f64, (usize, usize))> = seqs
        .iter()
        .map(|&(s, e)| {
            let m = errors[s..=e].iter().cloned().fold(0.0f64, f64::max);
            (m, (s, e))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    // baseline: the max error NOT in any sequence
    let mut in_seq = vec![false; errors.len()];
    for &(s, e) in seqs {
        for slot in in_seq.iter_mut().take(e + 1).skip(s) {
            *slot = true;
        }
    }
    let floor = errors
        .iter()
        .zip(in_seq.iter())
        .filter(|(_, &m)| !m)
        .map(|(&e, _)| e)
        .fold(0.0f64, f64::max);

    let mut keep = scored.len();
    for i in 0..scored.len() {
        let next = if i + 1 < scored.len() { scored[i + 1].0 } else { floor };
        let drop = if scored[i].0 > 1e-12 {
            (scored[i].0 - next) / scored[i].0
        } else {
            0.0
        };
        if drop < p_prune {
            keep = i;
            break;
        }
    }
    scored.truncate(keep.max(0));
    let mut out: Vec<(usize, usize)> = scored.into_iter().map(|(_, se)| se).collect();
    out.sort_unstable();
    out
}

/// Full pipeline for one channel: residuals → smooth → epsilon → sequences
/// → prune. Returns predicted anomaly sequences on the test split.
pub fn detect_channel(train: &[f64], test: &[f64], p: usize) -> Vec<(usize, usize)> {
    detect_channel_tuned(train, test, p, 2.5, 0.13, 50)
}

/// The tunable pipeline: `z_min` (epsilon grid floor), `p_prune`
/// (pruning strength), `buffer` (sequence expansion) trade the measured
/// precision surplus for recall.
pub fn detect_channel_tuned(
    train: &[f64],
    test: &[f64],
    p: usize,
    z_min: f64,
    p_prune: f64,
    buffer: usize,
) -> Vec<(usize, usize)> {
    let pred = match ArPredictor::fit(train, p, 1e-4) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let res = pred.residuals(test);
    let span = (test.len() / 30).clamp(10, 300);
    let sm = ewma(&res, span);
    let eps = find_epsilon_from(&sm[p..], z_min);
    let seqs = anomaly_sequences(&sm, eps, buffer);
    prune_sequences(&sm, &seqs, p_prune)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar_predictor_learns_a_sine() {
        let train: Vec<f64> = (0..2000).map(|i| crate::sin(i as f64 * 0.1)).collect();
        let m = ArPredictor::fit(&train, 10, 1e-4).expect("fit");
        let test: Vec<f64> = (2000..3000).map(|i| crate::sin(i as f64 * 0.1)).collect();
        let res = m.residuals(&test);
        let mean_res = res[10..].iter().sum::<f64>() / (res.len() - 10) as f64;
        assert!(mean_res < 1e-3, "sine must be nearly perfectly predicted, got {}", mean_res);
    }

    #[test]
    fn injected_burst_is_detected_and_isolated() {
        let mut rng = 987654321u64;
        let mut noise = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        };
        let train: Vec<f64> = (0..3000)
            .map(|i| crate::sin(i as f64 * 0.05) + 0.05 * noise())
            .collect();
        let mut test: Vec<f64> = (0..3000)
            .map(|i| crate::sin(i as f64 * 0.05) + 0.05 * noise())
            .collect();
        for item in test.iter_mut().skip(1500).take(120) {
            *item += 1.5;
        }
        let seqs = detect_channel(&train, &test, 20);
        assert!(!seqs.is_empty(), "burst must be detected");
        let hit = seqs.iter().any(|&(s, e)| s <= 1620 && e >= 1500);
        assert!(hit, "detected sequences {:?} must overlap the burst", seqs);
        assert!(seqs.len() <= 3, "pruning must keep it tight, got {:?}", seqs);
    }
}
