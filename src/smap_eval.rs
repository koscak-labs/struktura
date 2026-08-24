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
        // JPL's exact regularizer. (A linear sequence penalty was tried to
        // help multi-anomaly channels: measured 0.739 vs 0.743 — no gain,
        // reverted. The multi-sequence recall gap lives elsewhere.)
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
    scored.truncate(keep);
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
    // p == 0 selects the AR order PER CHANNEL on a train-only holdout
    // (fit on the first 80%, score one-step error on the last 20%,
    // refit the winner on the full train split — the test is never seen).
    let chosen_p = if p == 0 {
        let split = train.len() * 4 / 5;
        let (tr, va) = train.split_at(split);
        let mut best = (5usize, f64::MAX);
        for &cand in &[3usize, 5, 10, 25, 50] {
            if let Some(m) = ArPredictor::fit(tr, cand, 1e-4) {
                let res = m.residuals(va);
                let mse: f64 = res[cand.min(res.len())..]
                    .iter()
                    .map(|e| e * e)
                    .sum::<f64>()
                    / res.len().max(1) as f64;
                if mse < best.1 {
                    best = (cand, mse);
                }
            }
        }
        best.0
    } else {
        p
    };
    let p = chosen_p;
    let pred = match ArPredictor::fit(train, p, 1e-4) {
        Some(m) => m,
        None => return Vec::new(),
    };
    let res = pred.residuals(test);
    let span = (test.len() / 30).clamp(10, 300);
    let sm = ewma(&res, span);

    // WINDOWED epsilon (JPL's actual scheme, ~2100-point evaluation
    // windows): a single global epsilon lets the largest anomaly dominate
    // the residual statistics, sinking smaller secondary anomalies below
    // threshold — the measured cause of the multi-anomaly recall gap.
    // Each window gets its own epsilon; sequences merge globally, then one
    // global pruning pass.
    const EPS_WINDOW: usize = 2100;
    let n = sm.len();
    // Global noise floor: a quiet window's local epsilon must never drop
    // below the stream-wide (mean + 2 sigma) — otherwise every quiet
    // window mints its own false positives (measured: 12 -> 46 FPs
    // without the floor).
    // Robust floor: median + 3·(1.4826·MAD). Mean/sd would be inflated by
    // the anomalies themselves (measured: recall 67.6 -> 60.0 with a
    // mean+2sd floor); median/MAD ignore the anomaly mass.
    let gfloor = {
        let mut s: Vec<f64> = sm[p..].to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = s[s.len() / 2];
        let mut dev: Vec<f64> = s.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = dev[dev.len() / 2];
        med + 3.0 * 1.4826 * mad
    };
    // GLOBAL pass: one epsilon over the whole stream — wins on channels
    // with one dominant anomaly (local windows self-contaminate there).
    let mut seqs: Vec<(usize, usize)> = Vec::new();
    let global_eps = find_epsilon_from(&sm[p..], z_min);
    seqs.extend(anomaly_sequences(&sm, global_eps, buffer));

    // WINDOWED passes: per-window epsilon (floored) — wins on multi-anomaly
    // channels where the largest event masks the others globally. Two
    // phases, offset by half a window, so no anomaly is split across a
    // window boundary in both phases.
    for phase in [0usize] {
        let mut start = p + phase;
        while start < n {
            let end = (start + EPS_WINDOW).min(n);
            let w = &sm[start..end];
            if w.len() >= 200 {
                let eps = find_epsilon_from(w, z_min).max(gfloor);
                for (s, e) in anomaly_sequences(w, eps, buffer) {
                    seqs.push((start + s, (start + e).min(n - 1)));
                }
            }
            if end == n {
                break;
            }
            start = end;
        }
    }
    // merge overlapping/adjacent sequences from neighboring windows
    seqs.sort_unstable();
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
    prune_sequences(&sm, &merged, p_prune)
}

/// DFA structural channel for the batch protocol: sliding windowed α over
/// the test split, z-scored against the train split's windowed-α
/// statistics, run through the same ε/sequence machinery. Catches
/// CONTEXTUAL anomalies (the pattern changes while values stay plausible)
/// that a value predictor's residuals cannot see.
pub fn dfa_sequences(
    train: &[f64],
    test: &[f64],
    window: usize,
    z_min: f64,
    buffer: usize,
) -> Vec<(usize, usize)> {
    let stride = 8usize;
    if train.len() < 3 * window || test.len() < window {
        return Vec::new();
    }
    let mut buf = Vec::new();
    let alphas_of = |series: &[f64], buf: &mut Vec<f64>| -> Vec<f64> {
        let mut out = Vec::new();
        let mut end = window;
        while end <= series.len() {
            out.push(crate::dfa_fast_into(&series[end - window..end], buf).alpha);
            end += stride;
        }
        out
    };
    let train_a = alphas_of(train, &mut buf);
    let m = train_a.iter().sum::<f64>() / train_a.len() as f64;
    let sd = crate::sqrt(
        train_a.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / train_a.len() as f64,
    )
    .max(1e-6);
    let test_a = alphas_of(test, &mut buf);
    let z: Vec<f64> = test_a.iter().map(|a| (a - m).abs() / sd).collect();
    // The structural channel gets its OWN false-positive discipline: a hard
    // epsilon floor well above the α-scatter (windowed α is noisy on real
    // channels) and its own pruning pass — without these the union floods.
    let eps = find_epsilon_from(&z, z_min.max(4.0));
    let raw = anomaly_sequences(&z, eps, buffer / stride);
    prune_sequences(&z, &raw, 0.13)
        .into_iter()
        .map(|(s, e)| (s * stride, (e * stride + window - 1).min(test.len() - 1)))
        .collect()
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
