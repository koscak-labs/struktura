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
pub enum Evidence { Bootstrap, Floor, Paired }

/// Which way is better. `Auto`: time-like metrics (keys containing secs, sec, ms,
/// time, lat, wall, dur) are lower-is-better, everything else higher-is-better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction { Higher, Lower, Auto }

pub fn lower_is_better(metric: &str, dir: Direction) -> bool {
    match dir {
        Direction::Higher => false,
        Direction::Lower => true,
        Direction::Auto => metric.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric())
            .any(|t| matches!(t, "secs" | "sec" | "seconds" | "ms" | "us" | "time" | "lat" | "latency" | "wall" | "dur" | "duration" | "elapsed" | "ttft" | "gen" | "tokens" | "tok" | "cost")),
    }
}

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
    /// Paired pass/fail only: tasks where exactly one arm passed, and the exact
    /// two-sided sign-test p-value on them (NaN for other evidence).
    pub n_discordant: usize,
    pub p_value: f64,
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
    /// Metrics reported as pass/fail words (tasks), compared paired by task.
    pub binary: std::collections::BTreeSet<String>,
    /// Numeric metrics recorded per task as `<metric>@<task>` (workbench verdicts):
    /// compared paired by task as ratios.
    pub paired: std::collections::BTreeSet<String>,
    pub lines: usize,
}

/// Parse `key=<number>[unit]`; the number is the leading numeric prefix.
fn metric_token(tok: &str) -> Option<(String, f64)> { metric_token_kind(tok).map(|(k, x, _)| (k, x)) }

/// Like `metric_token`, also accepting pass/fail words (pass ok true yes / fail false no error, any case) as 1/0.
fn metric_token_kind(tok: &str) -> Option<(String, f64, bool)> {
    let (k, v) = tok.split_once('=')?;
    if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.') { return None; }
    match v.to_ascii_lowercase().as_str() {
        "pass" | "ok" | "true" | "yes" => return Some((k.to_string(), 1.0, true)),
        "fail" | "false" | "no" | "error" => return Some((k.to_string(), 0.0, true)),
        _ => {}
    }
    let num: String = v.chars().take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e' || *c == 'E').collect();
    let x: f64 = num.trim_end_matches(|c| c == 'e' || c == 'E').parse().ok()?;
    if x.is_finite() { Some((k.to_string(), x, false)) } else { None }
}

impl ArmData {
    /// Add every labelled line of `text`. Lines without a tab-separated label
    /// field followed by at least one `key=value` token are ignored.
    pub fn ingest(&mut self, text: &str) {
        for line in text.lines() {
            let fields: Vec<&str> = line.split('\t').map(|f| f.trim()).collect();
            // Workbench verdicts: config<TAB>task<TAB>sample<TAB>pass|fail<TAB>k=v ...
            if fields.len() >= 4 && fields[2].parse::<u64>().is_ok() {
                if let Some((_, ok, true)) = metric_token_kind(&format!("v={}", fields[3])) {
                    let (cfg, task) = (fields[0], fields[1]);
                    if cfg.is_empty() || task.is_empty() { continue; }
                    self.lines += 1;
                    let key = (String::new(), cfg.to_string());
                    if !self.values.contains_key(&key) { self.order.push(key.clone()); }
                    self.binary.insert(task.to_string());
                    let slot = self.values.entry(key).or_default();
                    slot.entry(task.to_string()).or_default().push(ok);
                    if ok == 1.0 {
                        for t in fields[4..].iter().flat_map(|f| f.split_whitespace()) {
                            if let Some((k, x, false)) = metric_token_kind(t) {
                                self.paired.insert(k.clone());
                                slot.entry(format!("{}@{}", k, task)).or_default().push(x);
                            }
                        }
                    }
                    continue;
                }
            }
            let first_metric = fields.iter().position(|f| f.split_whitespace().any(|t| metric_token_kind(t).is_some()));
            let Some(fm) = first_metric else { continue };
            if fm == 0 { continue; }
            let labels: Vec<&str> = fields[..fm].iter().copied().filter(|f| !f.is_empty()).collect();
            if labels.is_empty() || labels.iter().any(|l| l.contains(' ')) { continue; }
            let (group, arm) = if labels.len() >= 2 { (labels[0].to_string(), labels[1..].join("/")) } else { (String::new(), labels[0].to_string()) };
            let mut metrics = Vec::new();
            for f in &fields[fm..] {
                for t in f.split_whitespace() {
                    if let Some(m) = metric_token_kind(t) { metrics.push(m); } else if t.starts_with('(') { break; }
                }
            }
            if metrics.is_empty() { continue; }
            self.lines += 1;
            let key = (group, arm);
            if !self.values.contains_key(&key) { self.order.push(key.clone()); }
            let slot = self.values.entry(key).or_default();
            for (k, x, bin) in metrics { if bin { self.binary.insert(k.clone()); } slot.entry(k).or_default().push(x); }
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

/// Exact two-sided sign test: P(at most `k` of `n` discordant tasks favour the
/// less successful arm | no difference), doubled, capped at 1.
pub fn sign_test_p(k: usize, n: usize) -> f64 {
    if n == 0 { return 1.0; }
    let mut c = 1.0f64; // C(n, 0)
    let mut tail = 0.0f64;
    for i in 0..=k { if i > 0 { c = c * (n - i + 1) as f64 / i as f64; } tail += c; }
    (2.0 * tail / 2f64.powi(n as i32)).min(1.0)
}

/// Compare every pair of arms inside each group on every shared metric.
/// `higher_is_better` decides who "wins"; `min_effect_pct` is the smallest
/// believable effect (noise floor).
pub fn rank(data: &ArmData, min_effect_pct: f64, dir: Direction, n_boot: usize, seed: u64) -> ArmsReport {
    let mut rng = Rng(seed | 1);
    let mut pairs = Vec::new();
    for (i, ka) in data.order.iter().enumerate() {
        for kb in data.order.iter().skip(i + 1) {
            if ka.0 != kb.0 { continue; }
            let (ma, mb) = (&data.values[ka], &data.values[kb]);
            for (metric, va) in ma {
                if data.binary.contains(metric) || metric.contains('@') { continue; }
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
                let (better_lo, better_hi) = if !lower_is_better(metric, dir) { (lo, hi) } else { (-hi, -lo) };
                let outcome = if better_lo >= min_effect_pct { Outcome::BWins }
                    else if better_hi <= -min_effect_pct { Outcome::AWins }
                    else if better_lo > -min_effect_pct && better_hi < min_effect_pct { Outcome::Tie }
                    else { Outcome::Inconclusive };
                pairs.push(Pair { group: ka.0.clone(), metric: metric.clone(), a: ka.1.clone(), b: kb.1.clone(),
                    a_median: a_med, b_median: b_med, delta_pct: delta, ci_low: lo, ci_high: hi, evidence, outcome,
                    n_discordant: 0, p_value: f64::NAN });
            }
            // Pass/fail tasks: paired by task (both arms ran the same tasks). Per task, an arm's
            // score is its pass rate over repeats; only tasks where the arms differ carry evidence.
            let tasks: Vec<&String> = ma.keys().filter(|k| data.binary.contains(*k) && mb.contains_key(*k)).collect();
            if !tasks.is_empty() {
                let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
                let (mut a_better, mut b_better, mut ra, mut rb) = (0usize, 0usize, 0.0, 0.0);
                for t in &tasks {
                    let (x, y) = (mean(&ma[*t]), mean(&mb[*t]));
                    ra += x; rb += y;
                    if x > y { a_better += 1 } else if y > x { b_better += 1 }
                }
                let n = a_better + b_better;
                let p = sign_test_p(a_better.min(b_better), n);
                let outcome = if n == 0 { Outcome::Tie } else if p < 0.05 { if b_better > a_better { Outcome::BWins } else { Outcome::AWins } } else { Outcome::Inconclusive };
                let (ra, rb) = (ra / tasks.len() as f64, rb / tasks.len() as f64);
                pairs.push(Pair { group: ka.0.clone(), metric: format!("pass_rate({} tasks)", tasks.len()), a: ka.1.clone(), b: kb.1.clone(),
                    a_median: ra, b_median: rb, delta_pct: 100.0 * (rb - ra), ci_low: f64::NAN, ci_high: f64::NAN,
                    evidence: Evidence::Paired, outcome, n_discordant: n, p_value: p });
            }
            // Paired numeric metrics (time-to-done etc. on passing samples): per task the ratio of
            // medians; a task favours an arm only if the ratio clears the noise floor.
            for base in &data.paired {
                let pre = format!("{}@", base);
                let mut logs = Vec::new();
                let (mut a_better, mut b_better) = (0usize, 0usize);
                for (k, va) in ma.iter().filter(|(k, _)| k.starts_with(&pre)) {
                    let Some(vb) = mb.get(k) else { continue };
                    let (x, y) = (median(&mut va.clone()), median(&mut vb.clone()));
                    if x <= 0.0 || y <= 0.0 { continue; }
                    let r = (y / x).ln();
                    logs.push(r);
                    let pct = 100.0 * (y / x - 1.0);
                    if pct.abs() < min_effect_pct { continue; }
                    let b_wins = if lower_is_better(base, dir) { pct < 0.0 } else { pct > 0.0 };
                    if b_wins { b_better += 1 } else { a_better += 1 }
                }
                if logs.is_empty() { continue; }
                let n = a_better + b_better;
                let p = sign_test_p(a_better.min(b_better), n);
                let outcome = if n == 0 { Outcome::Tie } else if p < 0.05 { if b_better > a_better { Outcome::BWins } else { Outcome::AWins } } else { Outcome::Inconclusive };
                let gm = 100.0 * ((logs.iter().sum::<f64>() / logs.len() as f64).exp() - 1.0);
                pairs.push(Pair { group: ka.0.clone(), metric: format!("{}({} tasks, paired)", base, logs.len()), a: ka.1.clone(), b: kb.1.clone(),
                    a_median: f64::NAN, b_median: f64::NAN, delta_pct: gm, ci_low: f64::NAN, ci_high: f64::NAN,
                    evidence: Evidence::Paired, outcome, n_discordant: n, p_value: p });
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
        let r = rank(&d, 1.16, Direction::Auto, 500, 1);
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
        let r = rank(&d, 1.0, Direction::Auto, 200, 2);
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
        let hi = rank(&d, 1.16, Direction::Higher, 1000, 3);
        let lo = rank(&d, 1.16, Direction::Lower, 1000, 3);
        assert_eq!(hi.pairs[0].evidence, Evidence::Bootstrap);
        assert_eq!(hi.pairs[0].outcome, Outcome::BWins); // higher ms "better" -> slow wins
        assert_eq!(lo.pairs[0].outcome, Outcome::AWins); // lower is better -> fast wins
        assert_eq!(lo.dominant, vec![("lat".to_string(), "fast".to_string())]);
    }

    #[test]
    fn sign_test_exact_values() {
        assert_eq!(sign_test_p(0, 0), 1.0);
        assert!((sign_test_p(0, 7) - 2.0 / 128.0).abs() < 1e-12);
        assert!((sign_test_p(1, 10) - 2.0 * 11.0 / 1024.0).abs() < 1e-12);
        assert_eq!(sign_test_p(5, 10), 1.0);
    }

    #[test]
    fn workbench_verdicts_paired_pass_rate_and_time() {
        let mut d = ArmData::default();
        let mut text = String::new();
        for t in 0..10 {
            for s in 0..2 {
                // "think" passes every task, slowly; "fast" passes only tasks 0-2, 40% quicker.
                text.push_str(&format!("think\tt{t}\t{s}\tpass\tttd_s={} gen=3000\n", 50 + t));
                let ok = if t < 3 { "pass" } else { "fail" };
                text.push_str(&format!("fast\tt{t}\t{s}\t{ok}\tttd_s={} gen=900\n", 30 + t));
            }
        }
        text.push_str("this line\tis not\tx\tpass\n");
        d.ingest(&text);
        assert_eq!(d.lines, 40);
        assert_eq!(d.binary.len(), 10);
        let r = rank(&d, 1.16, Direction::Auto, 200, 1);
        let pr = r.pairs.iter().find(|p| p.metric.starts_with("pass_rate")).unwrap();
        assert_eq!((pr.a.as_str(), pr.b.as_str()), ("think", "fast"));
        assert_eq!(pr.outcome, Outcome::AWins, "{:?}", pr);
        assert_eq!(pr.n_discordant, 7);
        assert!((pr.a_median - 1.0).abs() < 1e-12 && (pr.b_median - 0.3).abs() < 1e-12);
        // time-to-done is only compared on tasks both passed (3), lower is better -> fast wins those,
        // but 3 tasks cannot reach p < 0.05 (p = 0.25): inconclusive, not a false verdict.
        let tt = r.pairs.iter().find(|p| p.metric.starts_with("ttd_s")).unwrap();
        assert_eq!(tt.n_discordant, 3);
        assert_eq!(tt.outcome, Outcome::Inconclusive, "{:?}", tt);
        assert!(tt.delta_pct < -35.0);
        assert!(r.pairs.iter().all(|p| !p.metric.contains('@')), "per-task keys never compared as numbers");
    }

    #[test]
    fn deterministic_for_a_seed() {
        let mut d = ArmData::default();
        for i in 0..6 { d.ingest(&format!("a\tx={}\nb\tx={}\n", 10 + i, 11 + i)); }
        let (r1, r2) = (rank(&d, 1.0, Direction::Auto, 500, 9), rank(&d, 1.0, Direction::Auto, 500, 9));
        assert_eq!(r1.pairs[0].ci_low, r2.pairs[0].ci_low);
    }
}
