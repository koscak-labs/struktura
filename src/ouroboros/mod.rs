//! Ouroboros: an experiment brain that learns from its own lab, with no model
//! in the loop.
//!
//! observe -> learn -> decide -> design -> remember -> (lab runs the job) -> observe ...
//!
//! - **observe**: the lab ledger (`lab::analyze`), job out logs (`arms`), and
//!   optionally production requests (`pulse`) for a deploy gate.
//! - **learn** ([`learn`]): how far measured effects land from what the lab
//!   pre-registered (calibration factor), plus fragile / easy verdicts.
//! - **decide** ([`agenda`]): a ranked agenda. Fragile load-bearing results are
//!   re-measured first; then challengers for each knob, scored by how much
//!   their next test would teach; infeasible values and knobs without an
//!   instrument are listed, never run.
//! - **design** ([`design`]): runs per arm from `power`, a threshold inside one
//!   noise band of the calibrated predicted effect, a job that measures the
//!   metric the knob actually moves, and a pre-registration that cannot be
//!   tautological.
//! - **remember** ([`memory`]): append-only, idempotent `knowledge.jsonl`.
//! - **backtest** ([`backtest`]): replays the ledger to check its own signals.
//!
//! The loop stops at the queue: jobs land in an outbox; submitting them to the
//! GPU is the queue owner's decision (`--submit` exists, nothing calls it).

#![cfg(feature = "std")]

pub mod knobs;
pub mod learn;
pub mod agenda;
pub mod design;
pub mod memory;
pub mod backtest;

use crate::arms::{rank, ArmData, ArmsReport, Direction};
use crate::lab::{analyze, LabReport};

/// Everything the brain looks at before deciding.
#[derive(Clone, Debug)]
pub struct Observation {
    pub lab: LabReport,
    /// One arms report per job log: (log name, report, raw data).
    pub logs: Vec<(String, ArmsReport, ArmData)>,
    /// Smallest believable single-run difference (percent).
    pub band_pct: f64,
}

/// Observe a ledger and job logs. `logs` are (name, text) pairs.
pub fn observe(ledger: &str, logs: &[(String, String)]) -> Observation {
    let lab = analyze(ledger);
    let band_pct = if lab.pair_band_pct.is_finite() { lab.pair_band_pct.max(lab.floor_pct) } else { lab.floor_pct };
    let logs = logs.iter().map(|(name, text)| {
        let mut d = ArmData::default();
        d.ingest(text);
        let r = rank(&d, band_pct, Direction::Auto, 2000, 42);
        (name.clone(), r, d)
    }).collect();
    Observation { lab, logs, band_pct }
}

/// Median of the samples of (group, arm, metric) across all observed logs.
/// Ungrouped arms also collect the `rep` group, where jobs emitted by the loop
/// print their raw repeats (their `arm` group holds medians and is not counted
/// again).
pub fn arm_value(obs: &Observation, group: &str, arm: &str, metric: &str) -> Option<(f64, usize)> {
    let mut all: Vec<f64> = Vec::new();
    let groups: &[&str] = if group.is_empty() { &["", "rep"] } else { &[group] };
    for (_, _, d) in &obs.logs {
        for g in groups {
            if let Some(v) = d.values.get(&(g.to_string(), arm.to_string())).and_then(|m| m.get(metric)) {
                all.extend_from_slice(v);
            }
        }
    }
    if all.is_empty() { return None; }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = all.len();
    let m = if n % 2 == 1 { all[n / 2] } else { 0.5 * (all[n / 2 - 1] + all[n / 2]) };
    Some((m, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_reads_ledger_and_logs() {
        let ledger = "{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":202.0}}\n";
        let log = "ub256\td0=2816 d100000=1397\nub512\td0=3008 d100000=1484\n".to_string();
        let o = observe(ledger, &[("lab-115.out".into(), log)]);
        assert_eq!(o.lab.cal_n, 2);
        assert!(o.band_pct > 1.9 && o.band_pct < 2.0, "band={}", o.band_pct);
        assert_eq!(arm_value(&o, "", "ub512", "d100000"), Some((1484.0, 1)));
        assert_eq!(arm_value(&o, "", "ub999", "d100000"), None);
    }
}
