//! Arms: rank several configurations of one experiment from labelled log lines.
//!
//! Lab jobs print one line per configuration ("arm"), e.g.
//!
//! ```text
//! ub512\td0=3008 d32768=2279 d100000=1484  (prefill t/s ...)
//! pp256\tgqa2\td0=2648tps d65536=1658tps
//! ```
//!
//! Fields before the first `key=value` field name the arm; with two or more
//! such fields the first is a group and arms are only compared inside their
//! group. Every `key=<number>[unit]` token is a metric. Lines from repeated
//! runs (several logs, or repeats in one log) become samples of the same
//! (group, arm, metric).
//!
//! Each pair of arms is compared on each metric they share:
//! - **bootstrap**: both arms have at least [`MIN_BOOT`] samples: 95% CI of
//!   the % difference of medians; a winner needs the whole CI beyond the
//!   minimum believable effect.
//! - **floor**: fewer samples: a single run carries no spread of its own, so
//!   the difference must exceed the caller's noise floor (the lab's
//!   calibration noise, 2 × CV) or it is a tie.
//!
//! A pairwise-only pre-registration ("X vs incumbent") cannot rank X against
//! Y; this does, from the same raw numbers.

#![cfg(feature = "std")]

use std::collections::BTreeMap;

/// Samples per arm needed before a bootstrap interval is used.
pub const MIN_BOOT: usize = 5;

#[derive(Clone, Debug, PartialEq)]
pub enum Evidence { Bootstrap, Floor }

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome { AWins, BWins, Tie, Inconclusive }

#[derive(Clone, Debug)]
pub struct Pair {
    pub group: String,
    pub metric: String,
    pub a: String,
    pub b: String,
    pub a_median: f64,
    pub b_median: f64,
    /// (b - a) / a, percent.
    pub delta_pct: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub evidence: Evidence,
    pub outcome: Outcome,
}

#[derive(Clone, Debug)]
pub struct ArmScore {
    pub group: String,
    pub arm: String,
    pub wins: usize,
    pub ties: usize,
    pub losses: usize,
    pub inconclusive: usize,
    pub samples: usize,
}

#[derive(Clone, Debug, Default)]
pub struct ArmsReport {
    pub lines: usize,
    pub pairs: Vec<Pair>,
    pub scores: Vec<ArmScore>,
    /// Per group: an arm that wins or ties every comparison and wins at least one.
    pub dominant: Vec<(String, String)>,
}

/// (group, arm) -> metric -> samples, in first-seen order of arms.
#[derive(Clone, Debug, Default)]
pub struct ArmData {
    pub order: Vec<(String, String)>,
    pub values: BTreeMap<(String, String), BTreeMap<String, Vec<f64>>>,
    pub lines: usize,
}

/// Parse `key=<number>[unit]`; the number is the leading numeric prefix.
fn metric_token(tok: &str) -> Option<(String, f64)> {
    let (k, v) = tok.split_once('=')?;
    if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') { return None; }
    let num: String = v.chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e' || *c == 'E').collect();
    let x: f64 = num.trim_end_matches(|c| c == 'e' || c == 'E').parse().ok()?;
    if x.is_finite() { Some((k.to_string(), x)) } else { None }
}

impl ArmData {
    /// Add every labelled line of `text`. Lines without a tab-separated label
    /// field followed by at least one `key=value` token are ignored.
    pub fn ingest(&mut self, text: &str) {
        for line in text.lines() {
            let fields: Vec<&str> = line.split('\t').map(|f| f.trim()).collect();
            let first_metric = fields.iter().position(|f| f.split_whitespace().any(|t| metric_token(t).is_some()));
            let Some(fm) = first_metric else { continue };
            if fm == 0 { continue; }
            let labels: Vec<&str> = fields[..fm].iter().copied().filter(|f| !f.is_empty()).collect();
            if labels.is_empty() || labels.iter().any(|l| l.contains(' ')) { continue; }
            let (group, arm) = if labels.len() >= 2 { (labels[0].to_string(), labels[1..].join("/")) } else { (String::new(), labels[0].to_string()) };
            let mut metrics = Vec::new();
            for f in &fields[fm..] {
                for t in f.split_whitespace() {
                    if let Some(m) = metric_token(t) { metrics.push(m); } else if t.starts_with('(') { break; }
                }
            }
            if metrics.is_empty() { continue; }
            self.lines += 1;
            let key = (group, arm);
            if !self.values.contains_key(&key) { self.order.push(key.clone()); }
            let slot = self.values.entry(key).or_default();
            for (k, x) in metrics { slot.entry(k).or_default().push(x); }
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

/// Compare every pair of arms inside each group on every shared metric.
/// `higher_is_better` decides who "wins"; `min_effect_pct` is the smallest
/// believable effect (noise floor).
pub fn rank(data: &ArmData, min_effect_pct: f64, higher_is_better: bool, n_boot: usize, seed: u64) -> ArmsReport {
    let mut rng = Rng(seed | 1);
    let mut pairs = Vec::new();
    for (i, ka) in data.order.iter().enumerate() {
        for kb in data.order.iter().skip(i + 1) {
            if ka.0 != kb.0 { continue; }
            let (ma, mb) = (&data.values[ka], &data.values[kb]);
            for (metric, va) in ma {
                let Some(vb) = mb.get(metric) else { continue };
                let (a_med, b_med) = (median(&mut va.clone()), median(&mut vb.clone()));
                if a_med == 0.0 { continue; }
                let delta = 100.0 * (b_med - a_med) / a_med;
                let (lo, hi, evidence) = if va.len() >= MIN_BOOT && vb.len() >= MIN_BOOT {
                    let mut buf = Vec::new();
                    let mut ds: Vec<f64> = (0..n_boot.max(100)).map(|_| {
                        buf.clear(); for _ in 0..va.len() { buf.push(va[rng.below(va.len())]); }
                        let a = median(&mut buf);
                        buf.clear(); for _ in 0..vb.len() { buf.push(vb[rng.below(vb.len())]); }
                        let b = median(&mut buf);
                        100.0 * (b - a) / a
                    }).collect();
                    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    (ds[(ds.len() as f64 * 0.025) as usize], ds[((ds.len() as f64 * 0.975) as usize).min(ds.len() - 1)], Evidence::Bootstrap)
                } else {
                    (delta, delta, Evidence::Floor)
                };
                // "B better" in the caller's direction.
                let (better_lo, better_hi) = if higher_is_better { (lo, hi) } else { (-hi, -lo) };
                let outcome = if better_lo >= min_effect_pct { Outcome::BWins }
                    else if better_hi <= -min_effect_pct { Outcome::AWins }
                    else if better_lo > -min_effect_pct && better_hi < min_effect_pct { Outcome::Tie }
                    else { Outcome::Inconclusive };
                pairs.push(Pair { group: ka.0.clone(), metric: metric.clone(), a: ka.1.clone(), b: kb.1.clone(),
                    a_median: a_med, b_median: b_med, delta_pct: delta, ci_low: lo, ci_high: hi, evidence, outcome });
            }
        }
    }
    let mut scores: Vec<ArmScore> = data.order.iter().map(|(g, a)| ArmScore {
        group: g.clone(), arm: a.clone(), wins: 0, ties: 0, losses: 0, inconclusive: 0,
        samples: data.values[&(g.clone(), a.clone())].values().map(|v| v.len()).max().unwrap_or(0),
    }).collect();
    let keys: Vec<(String, String)> = scores.iter().map(|s| (s.group.clone(), s.arm.clone())).collect();
    let idx = |g: &str, a: &str| keys.iter().position(|(kg, ka)| kg == g && ka == a).unwrap();
    for p in &pairs {
        let (ia, ib) = (idx(&p.group, &p.a), idx(&p.group, &p.b));
        match p.outcome {
            Outcome::AWins => { scores[ia].wins += 1; scores[ib].losses += 1; }
            Outcome::BWins => { scores[ib].wins += 1; scores[ia].losses += 1; }
            Outcome::Tie => { scores[ia].ties += 1; scores[ib].ties += 1; }
            Outcome::Inconclusive => { scores[ia].inconclusive += 1; scores[ib].inconclusive += 1; }
        }
    }
    let dominant = scores.iter()
        .filter(|s| s.losses == 0 && s.inconclusive == 0 && s.wins > 0)
        .map(|s| (s.group.clone(), s.arm.clone())).collect();
    ArmsReport { lines: data.lines, pairs, scores, dominant }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAB115: &str = "ub256\td0=2816 d32768=2163 d100000=1397  (prefill t/s for a 2048-token chunk at depth)\n\
ub512\td0=3008 d32768=2279 d100000=1484  (prefill t/s for a 2048-token chunk at depth)\n\
ub1024\td0=2914 d32768=2256 d100000=1469  (prefill t/s for a 2048-token chunk at depth)\n\
some free text = not a metric line\n";

    #[test]
    fn parses_two_and_three_field_lines_with_units() {
        let mut d = ArmData::default();
        d.ingest(LAB115);
        d.ingest("correct\tallpass\t12/12 tests passed\npp256\tstock\td0=2632tps d65536=1568tps \npp256\tgqa2\td0=2648tps d65536=1658tps\n");
        assert_eq!(d.lines, 5);
        assert_eq!(d.values[&(String::new(), "ub512".into())]["d100000"], vec![1484.0]);
        assert_eq!(d.values[&("pp256".into(), "gqa2".into())]["d65536"], vec![1658.0]);
        assert!(!d.values.contains_key(&("correct".into(), "allpass".into())));
    }

    #[test]
    fn single_run_differences_below_the_floor_are_ties() {
        let mut d = ArmData::default();
        d.ingest(LAB115);
        let r = rank(&d, 1.16, true, 500, 1);
        let p = r.pairs.iter().find(|p| p.a == "ub512" && p.b == "ub1024" && p.metric == "d100000").unwrap();
        assert_eq!(p.evidence, Evidence::Floor);
        assert_eq!(p.outcome, Outcome::Tie, "{:?}", p); // 1484 vs 1469 = -1.0%
        let p0 = r.pairs.iter().find(|p| p.a == "ub512" && p.b == "ub1024" && p.metric == "d0").unwrap();
        assert_eq!(p0.outcome, Outcome::AWins, "{:?}", p0); // 3008 vs 2914 = -3.1%
        let s512 = r.scores.iter().find(|s| s.arm == "ub512").unwrap();
        assert_eq!(s512.losses, 0);
        assert_eq!(r.dominant, vec![(String::new(), "ub512".to_string())]);
    }

    #[test]
    fn groups_are_never_compared_across() {
        let mut d = ArmData::default();
        d.ingest("g1\ta\tx=100\ng1\tb\tx=200\ng2\ta\tx=50\n");
        let r = rank(&d, 1.0, true, 200, 2);
        assert_eq!(r.pairs.len(), 1);
        assert_eq!(r.pairs[0].group, "g1");
    }

    #[test]
    fn repeated_runs_use_bootstrap_and_respect_direction() {
        let mut d = ArmData::default();
        for i in 0..8 {
            let j = (i % 3) as f64;
            d.ingest(&format!("lat\tfast\tms={}\nlat\tslow\tms={}\n", 40.0 + j, 50.0 + j));
        }
        let hi = rank(&d, 1.16, true, 1000, 3);
        let lo = rank(&d, 1.16, false, 1000, 3);
        assert_eq!(hi.pairs[0].evidence, Evidence::Bootstrap);
        assert_eq!(hi.pairs[0].outcome, Outcome::BWins); // higher ms "better" -> slow wins
        assert_eq!(lo.pairs[0].outcome, Outcome::AWins); // lower is better -> fast wins
        assert_eq!(lo.dominant, vec![("lat".to_string(), "fast".to_string())]);
    }

    #[test]
    fn deterministic_for_a_seed() {
        let mut d = ArmData::default();
        for i in 0..6 { d.ingest(&format!("a\tx={}\nb\tx={}\n", 10 + i, 11 + i)); }
        let (r1, r2) = (rank(&d, 1.0, true, 500, 9), rank(&d, 1.0, true, 500, 9));
        assert_eq!(r1.pairs[0].ci_low, r2.pairs[0].ci_low);
    }
}
