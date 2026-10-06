//! Power: how much evidence an experiment needs before its verdict means
//! anything. Two designs used by the lab:
//!
//! - **Runs** (benchmark A/B): each run's throughput has a coefficient of
//!   variation `cv` (the calibration arm's noise). Comparing two arms of `n`
//!   runs each, a relative effect `e` is detected with the requested power
//!   when `n >= 2 (z_{1-a/2} + z_{power})^2 cv^2 / e^2` (normal approximation).
//! - **Tasks** (workbench, paired by task): on each task the two configs
//!   disagree with probability `d`; when they disagree, the better config wins
//!   with probability `q`. The verdict is an exact two-sided sign test on the
//!   disagreements. Power is computed exactly by summing over the binomial
//!   outcomes, and the smallest task count reaching the requested power is
//!   found by search.

#![cfg(feature = "std")]

use crate::arms::sign_test_p;

/// Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9).
pub fn norm_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0);
    let a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    let b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    let c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    let d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        -norm_quantile(1.0 - p)
    }
}

/// Runs per arm needed to detect a relative effect `effect_pct` given per-run
/// noise `cv_pct` (two-sided `alpha`, requested `power`). At least 2.
pub fn runs_needed(cv_pct: f64, effect_pct: f64, alpha: f64, power: f64) -> usize {
    let z = norm_quantile(1.0 - alpha / 2.0) + norm_quantile(power);
    let n = 2.0 * z * z * cv_pct * cv_pct / (effect_pct * effect_pct);
    (n.ceil() as usize).max(2)
}

/// Smallest relative effect (percent) detectable with `n` runs per arm.
pub fn min_detectable(cv_pct: f64, n: usize, alpha: f64, power: f64) -> f64 {
    let z = norm_quantile(1.0 - alpha / 2.0) + norm_quantile(power);
    z * cv_pct * (2.0 / n.max(1) as f64).sqrt()
}

fn binom_pmf_row(n: usize, p: f64) -> Vec<f64> {
    // pmf[k] for k = 0..=n, computed in log space for stability.
    let lp = p.max(1e-300).ln();
    let lq = (1.0 - p).max(1e-300).ln();
    let mut lfact = vec![0.0f64; n + 1];
    for i in 1..=n { lfact[i] = lfact[i - 1] + (i as f64).ln(); }
    (0..=n).map(|k| {
        if (p == 0.0 && k > 0) || (p == 1.0 && k < n) { return 0.0; }
        (lfact[n] - lfact[k] - lfact[n - k] + k as f64 * if p > 0.0 { lp } else { 0.0 } + (n - k) as f64 * if p < 1.0 { lq } else { 0.0 }).exp()
    }).collect()
}

/// Exact power of the paired sign test with `tasks` tasks: P(p < alpha AND the
/// better config is the one declared better).
pub fn sign_test_power(tasks: usize, discordant: f64, favour: f64, alpha: f64) -> f64 {
    let pn = binom_pmf_row(tasks, discordant);
    let mut power = 0.0;
    for (n, &w) in pn.iter().enumerate() {
        if w < 1e-15 || n == 0 { continue; }
        let pk = binom_pmf_row(n, favour);
        for (k, &v) in pk.iter().enumerate() {
            // k = disagreements won by the truly better config.
            if k * 2 > n && sign_test_p(n - k, n) < alpha { power += w * v; }
        }
    }
    power
}

/// Smallest task count (<= `max_tasks`) whose sign-test power reaches `power`.
pub fn tasks_needed(discordant: f64, favour: f64, alpha: f64, power: f64, max_tasks: usize) -> Option<usize> {
    (1..=max_tasks).find(|&t| sign_test_power(t, discordant, favour, alpha) >= power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_match_tables() {
        assert!((norm_quantile(0.975) - 1.959964).abs() < 1e-5);
        assert!((norm_quantile(0.8) - 0.841621).abs() < 1e-5);
        assert!((norm_quantile(0.01) + 2.326348).abs() < 1e-5);
    }

    #[test]
    fn runs_for_the_lab_noise_floor() {
        // Calibration CV 0.578%: 2 runs per arm detect ~1.6%; a 1.16% effect needs 4.
        assert_eq!(runs_needed(0.578, 1.16, 0.05, 0.8), 4);
        assert_eq!(runs_needed(0.578, 5.0, 0.05, 0.8), 2);
        let mde = min_detectable(0.578, 4, 0.05, 0.8);
        assert!(mde < 1.16 && mde > 1.0, "mde={}", mde);
    }

    #[test]
    fn sign_test_power_is_zero_below_six_tasks_and_grows() {
        // Even if every task is discordant and the better config always wins,
        // 5 tasks give min p = 2^(1-5) = 0.0625 > 0.05.
        assert_eq!(sign_test_power(5, 1.0, 1.0, 0.05), 0.0);
        assert!((sign_test_power(6, 1.0, 1.0, 0.05) - 1.0).abs() < 1e-12);
        let a = sign_test_power(20, 0.4, 0.85, 0.05);
        let b = sign_test_power(40, 0.4, 0.85, 0.05);
        assert!(b > a && a > 0.0, "{} {}", a, b);
        // Under no difference (favour 0.5) the false-verdict rate stays <= alpha/2 per direction.
        assert!(sign_test_power(40, 0.5, 0.5, 0.05) <= 0.025 + 1e-12);
    }

    #[test]
    fn tasks_needed_search() {
        assert_eq!(tasks_needed(1.0, 1.0, 0.05, 0.8, 100), Some(6));
        let t = tasks_needed(0.3, 0.8, 0.05, 0.8, 500).unwrap();
        assert!(sign_test_power(t, 0.3, 0.8, 0.05) >= 0.8);
        assert!(sign_test_power(t - 1, 0.3, 0.8, 0.05) < 0.8 || t <= 6);
    }
}
