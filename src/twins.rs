//! Twins: is variant A as good as variant B, judged only on TWIN tasks.
//!
//! A workbench asks a model to solve tasks under several configs (arms). Some tasks
//! come in twins that differ only in one token of their name, e.g. `t11-helix-gcd`
//! and `t12-rust-gcd` (or `L01-helix-wake-classify` / `L02-rust-wake-classify`): the
//! same problem, asked in two languages. Comparing the pass rate of all Helix tasks
//! with all Rust tasks would mostly measure which problems happened to be harder;
//! comparing twins removes that confound. An explicit map (`family<TAB>a_task<TAB>b_task`
//! with a header naming the `<a>_task` / `<b>_task` columns) overrides name matching.
//!
//! The unit of inference is the task FAMILY, not the run. Within a family, each side's
//! pass rate is taken per arm and averaged over the arms BOTH sides ran (so a side run
//! mostly under an easier config gains nothing), with no need for shared sample ids
//! (an always-on feeder draws a fresh seed per pass). Families are compared with an
//! exact two-sided sign test; fewer than 6 one-sided families can never reach p < 0.05,
//! and the verdict says `underpowered` instead of pretending. Samples that do share an
//! (arm, run, sample) id are paired for the time-to-done ratio.
//!
//! `first_try` counts a pass only when it took one attempt (`attempts=1`; rows without
//! an attempts count were single-shot): it separates writing it right from repairing it.
//!
//! Tasks without a twin are listed separately as `solos`: informative, but confounded.

use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

/// One paired observation: the same arm, family, run and sample on both twins.
#[derive(Clone, Debug, PartialEq)]
pub struct Obs {
    pub arm: String,
    pub family: String,
    pub sample: String,
    pub a: bool,
    pub b: bool,
    pub ttd_a: Option<f64>,
    pub ttd_b: Option<f64>,
}

/// One family: per-arm pass rates averaged over the arms both sides ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Family {
    pub family: String,
    pub task_a: String,
    pub task_b: String,
    /// Arms both sides ran.
    pub arms: usize,
    pub n_a: usize,
    pub pass_a: usize,
    pub n_b: usize,
    pub pass_b: usize,
    pub rate_a: f64,
    pub rate_b: f64,
}

impl Family {
    /// +1 when A passed more often, -1 when B did, 0 on a tie.
    pub fn favours(&self) -> i8 {
        if self.rate_a > self.rate_b + 1e-12 { 1 } else if self.rate_b > self.rate_a + 1e-12 { -1 } else { 0 }
    }
}

/// A task with no twin on the other side (side `'a'` or `'b'`).
#[derive(Clone, Debug, PartialEq)]
pub struct Solo {
    pub task: String,
    pub side: char,
    pub n: usize,
    pub pass: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Opts {
    /// task -> (side 'a' | 'b', family), from an explicit twins map.
    pub map: BTreeMap<String, (char, String)>,
    pub first_try: bool,
}

#[derive(Clone, Debug)]
pub struct TwinsReport {
    pub a: String,
    pub b: String,
    pub families: Vec<Family>,
    /// Per arm over twin families: (arm, n_a, pass_a, n_b, pass_b).
    pub per_arm: Vec<(String, usize, usize, usize, usize)>,
    pub paired_obs: usize,
    pub favour_a: usize,
    pub favour_b: usize,
    pub ties: usize,
    /// Exact two-sided sign test over the families that lean one way.
    pub p_value: f64,
    /// Median of ttd_b / ttd_a over paired observations where BOTH passed.
    pub ttd_ratio: Option<f64>,
    pub ttd_pairs: usize,
    pub solos: Vec<Solo>,
    pub first_try: bool,
    /// `a_better` | `b_better` | `no_difference_detected` | `underpowered`
    pub verdict: &'static str,
}

/// Fewest one-sided families for which the exact two-sided sign test can reach p < 0.05.
pub const MIN_FAMILIES: usize = 6;

/// A leading id part: up to 3 letters then digits (`t11`, `L01`).
fn is_id(p: &str) -> bool {
    let l = p.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    (1..=3).contains(&l) && p.len() > l && p[l..].chars().all(|c| c.is_ascii_digit())
}

/// The family of `task` with respect to `token`: the task name without its leading id
/// and without the `token` part (`t11-helix-gcd`, `helix` -> `gcd`).
/// `None` when `token` is not one of the dash-separated parts.
pub fn family_of(task: &str, token: &str) -> Option<String> {
    let mut parts: Vec<&str> = task.split('-').collect();
    if parts.len() > 1 && is_id(parts[0]) { parts.remove(0); }
    let at = parts.iter().position(|p| p.eq_ignore_ascii_case(token))?;
    parts.remove(at);
    if parts.is_empty() { return None; }
    Some(parts.join("-"))
}

/// Parse an explicit twins map: a header line naming `family`, `<a>_task` and `<b>_task`
/// columns (any order, case-insensitive), then one family per line.
pub fn parse_map(text: &str, a: &str, b: &str) -> Result<BTreeMap<String, (char, String)>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let head: Vec<String> = lines.next().ok_or("empty twins map")?.split('\t').map(|h| h.trim().to_ascii_lowercase()).collect();
    let col = |name: &str| head.iter().position(|h| *h == name).ok_or_else(|| std::format!("twins map: no '{}' column", name));
    let (cf, ca, cb) = (col("family")?, col(&std::format!("{}_task", a.to_ascii_lowercase()))?, col(&std::format!("{}_task", b.to_ascii_lowercase()))?);
    let mut m = BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').map(|s| s.trim()).collect();
        let (Some(fam), Some(ta), Some(tb)) = (f.get(cf), f.get(ca), f.get(cb)) else { continue };
        if fam.is_empty() || ta.is_empty() || tb.is_empty() { continue; }
        m.insert(ta.to_string(), ('a', fam.to_string()));
        m.insert(tb.to_string(), ('b', fam.to_string()));
    }
    Ok(m)
}

fn meta_num(meta: &[&str], key: &str) -> Option<f64> {
    meta.iter().flat_map(|f| f.split_whitespace())
        .find_map(|t| t.strip_prefix(key)).and_then(|v| v.trim_end_matches('s').parse::<f64>().ok())
        .filter(|x| x.is_finite())
}

/// Per (family, arm): side a / side b (n, pass); plus every task's side, family and record.
pub struct Parsed {
    cells: BTreeMap<(String, String), [(usize, usize); 2]>,
    samples: BTreeMap<(String, String, String), [Option<(bool, Option<f64>)>; 2]>,
    tasks: BTreeMap<String, (char, String, usize, usize)>,
}

/// Parse workbench verdict files (`arm<TAB>task<TAB>sample<TAB>pass|fail<TAB>k=v ...`).
/// Each input text is one run; samples of different runs are never paired.
pub fn parse(texts: &[&str], a: &str, b: &str, o: &Opts) -> Parsed {
    let mut p = Parsed { cells: BTreeMap::new(), samples: BTreeMap::new(), tasks: BTreeMap::new() };
    for (run, text) in texts.iter().enumerate() {
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').map(|s| s.trim()).collect();
            if f.len() < 4 || f[2].parse::<u64>().is_err() || f[0].is_empty() { continue; }
            let mut pass = match f[3].to_ascii_lowercase().as_str() { "pass" | "ok" | "true" => true, "fail" | "false" | "error" => false, _ => continue };
            if o.first_try && pass && meta_num(&f[4..], "attempts=").unwrap_or(1.0) > 1.0 { pass = false; }
            let (side, fam) = match o.map.get(f[1]) {
                Some((s, fam)) => (*s, fam.clone()),
                None => match (family_of(f[1], a), family_of(f[1], b)) {
                    (Some(fa), None) => ('a', fa),
                    (None, Some(fb)) => ('b', fb),
                    _ => continue, // neither variant, or both tokens (ambiguous)
                },
            };
            let si = if side == 'a' { 0 } else { 1 };
            let t = p.tasks.entry(f[1].to_string()).or_insert((side, fam.clone(), 0, 0));
            t.2 += 1; t.3 += pass as usize;
            let c = p.cells.entry((fam.clone(), f[0].to_string())).or_insert([(0, 0); 2]);
            c[si].0 += 1; c[si].1 += pass as usize;
            let s = p.samples.entry((fam, f[0].to_string(), std::format!("{}:{}", run, f[2]))).or_insert([None, None]);
            s[si] = Some((pass, meta_num(&f[4..], "ttd_s=")));
        }
    }
    p
}

/// Pool per family over the arms both sides ran, and decide.
pub fn judge(p: &Parsed, a: &str, b: &str, alpha: f64, first_try: bool) -> TwinsReport {
    let mut fams: BTreeMap<String, Family> = BTreeMap::new();
    let mut arms: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for ((fam, arm), c) in &p.cells {
        if c[0].0 == 0 || c[1].0 == 0 { continue; } // this arm did not run both twins
        let f = fams.entry(fam.clone()).or_insert_with(|| Family { family: fam.clone(), task_a: String::new(), task_b: String::new(),
            arms: 0, n_a: 0, pass_a: 0, n_b: 0, pass_b: 0, rate_a: 0.0, rate_b: 0.0 });
        f.arms += 1; f.n_a += c[0].0; f.pass_a += c[0].1; f.n_b += c[1].0; f.pass_b += c[1].1;
        f.rate_a += c[0].1 as f64 / c[0].0 as f64; f.rate_b += c[1].1 as f64 / c[1].0 as f64;
        let e = arms.entry(arm.clone()).or_insert((0, 0, 0, 0));
        e.0 += c[0].0; e.1 += c[0].1; e.2 += c[1].0; e.3 += c[1].1;
    }
    for f in fams.values_mut() { f.rate_a /= f.arms as f64; f.rate_b /= f.arms as f64; }
    for (task, (side, fam, _, _)) in &p.tasks {
        if let Some(f) = fams.get_mut(fam) { if *side == 'a' { f.task_a = task.clone() } else { f.task_b = task.clone() } }
    }
    let mut ratios: Vec<f64> = Vec::new();
    let mut paired_obs = 0;
    for ((fam, _, _), s) in &p.samples {
        if !fams.contains_key(fam) { continue; }
        if let [Some((pa, da)), Some((pb, db))] = s {
            paired_obs += 1;
            if let (true, true, Some(x), Some(y)) = (pa, pb, da, db) { if *x > 0.0 { ratios.push(y / x); } }
        }
    }
    let families: Vec<Family> = fams.into_values().collect();
    let favour_a = families.iter().filter(|f| f.favours() > 0).count();
    let favour_b = families.iter().filter(|f| f.favours() < 0).count();
    let ties = families.len() - favour_a - favour_b;
    let p_value = crate::arms::sign_test_p(favour_a.min(favour_b), favour_a + favour_b);
    let ttd_pairs = ratios.len();
    let ttd_ratio = if ratios.is_empty() { None } else {
        ratios.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
        let n = ratios.len();
        Some(if n % 2 == 1 { ratios[n / 2] } else { 0.5 * (ratios[n / 2 - 1] + ratios[n / 2]) })
    };
    let verdict = if favour_a + favour_b < MIN_FAMILIES { "underpowered" }
        else if p_value < alpha { if favour_a > favour_b { "a_better" } else { "b_better" } }
        else { "no_difference_detected" };
    let solos = p.tasks.iter().filter(|(_, (_, fam, _, _))| !families.iter().any(|f| f.family == *fam))
        .map(|(task, (side, _, n, pass))| Solo { task: task.clone(), side: *side, n: *n, pass: *pass }).collect();
    TwinsReport { a: a.to_string(), b: b.to_string(), families,
        per_arm: arms.into_iter().map(|(k, v)| (k, v.0, v.1, v.2, v.3)).collect(),
        paired_obs, favour_a, favour_b, ties, p_value, ttd_ratio, ttd_pairs, solos, first_try, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(r: &[(&str, &str, u32, &str)]) -> String {
        r.iter().map(|(arm, task, s, v)| std::format!("{}\t{}\t{}\t{}\tttd_s=2.0\n", arm, task, s, v)).collect()
    }
    fn run(texts: &[&str], o: &Opts) -> TwinsReport { judge(&parse(texts, "helix", "rust", o), "helix", "rust", 0.05, o.first_try) }

    #[test]
    fn family_strips_bench_and_live_ids_and_token_only() {
        assert_eq!(family_of("t11-helix-gcd", "helix").as_deref(), Some("gcd"));
        assert_eq!(family_of("t12-rust-gcd", "rust").as_deref(), Some("gcd"));
        assert_eq!(family_of("L01-helix-wake-classify", "helix").as_deref(), Some("wake-classify"));
        assert_eq!(family_of("L02-rust-wake-classify", "rust").as_deref(), Some("wake-classify"));
        assert_eq!(family_of("t51-patch-rust-lru", "rust").as_deref(), Some("patch-lru"));
        assert_eq!(family_of("t12-rust-gcd", "helix"), None);
        assert_eq!(family_of("trusty-gcd", "rust"), None, "a token must be a whole part");
        assert_eq!(family_of("t9-rust", "rust"), None, "nothing left = no family");
    }

    #[test]
    fn unpaired_seeds_still_compare_and_twinned_tasks_are_not_solos() {
        // a feeder draws a fresh seed per pass: helix samples 1,2 / rust samples 7,8 never share an id
        let r = rows(&[("x", "t11-helix-gcd", 1, "pass"), ("x", "t11-helix-gcd", 2, "fail"),
                       ("x", "t12-rust-gcd", 7, "pass"), ("x", "t12-rust-gcd", 8, "pass"),
                       ("y", "t12-rust-gcd", 1, "pass"),                       // arm y never ran helix: ignored
                       ("x", "t55-helix-roman", 1, "pass"), ("x", "t31-rust-csv", 1, "fail")]);
        let t = run(&[&r], &Opts::default());
        assert_eq!(t.families.len(), 1);
        let f = &t.families[0];
        assert_eq!((f.arms, f.n_a, f.pass_a, f.n_b, f.pass_b), (1, 2, 1, 2, 2));
        assert_eq!((f.task_a.as_str(), f.task_b.as_str(), f.favours()), ("t11-helix-gcd", "t12-rust-gcd", -1));
        assert_eq!(t.paired_obs, 0);
        let s: Vec<(&str, char)> = t.solos.iter().map(|s| (s.task.as_str(), s.side)).collect();
        assert_eq!(s, std::vec![("t31-rust-csv", 'b'), ("t55-helix-roman", 'a')]);
    }

    #[test]
    fn arm_mix_cannot_fake_a_winner() {
        // helix mostly ran the easy arm, rust mostly the hard one: raw pooling says helix 9/12 vs rust 4/12,
        // yet rust is better WITHIN each arm (easy 1.0 vs 0.9, hard 0.2 vs 0.0) -> rust.
        let mut v = Vec::new();
        for s in 0..10 { v.push(("easy", "t1-helix-gcd", s, if s < 9 { "pass" } else { "fail" })); }
        for s in 0..2 { v.push(("hard", "t1-helix-gcd", s, "fail")); }
        for s in 0..2 { v.push(("easy", "t2-rust-gcd", s, if s < 2 { "pass" } else { "fail" })); }
        for s in 0..10 { v.push(("hard", "t2-rust-gcd", s, if s < 2 { "pass" } else { "fail" })); }
        let t = run(&[&rows(&v)], &Opts::default());
        let f = &t.families[0];
        assert!((f.rate_a - (0.9 + 0.0) / 2.0).abs() < 1e-12 && (f.rate_b - (1.0 + 0.2) / 2.0).abs() < 1e-12);
        assert_eq!(f.favours(), -1, "per-arm: easy 0.9 vs 1.0, hard 0.0 vs 0.2 -> rust, not the raw-pooled helix");
    }

    #[test]
    fn five_one_sided_families_are_underpowered_six_decide() {
        let mut r = String::new();
        for (i, f) in ["a", "b", "c", "d", "e", "f"].iter().enumerate() {
            r += &std::format!("x\tt{}-helix-{}\t1\tpass\nx\tt{}-rust-{}\t1\tfail\n", i, f, i + 50, f);
        }
        let five: String = r.lines().take(10).map(|l| std::format!("{l}\n")).collect();
        let t = run(&[&five], &Opts::default());
        assert_eq!((t.favour_a, t.favour_b, t.verdict), (5, 0, "underpowered"));
        assert!((t.p_value - 0.0625).abs() < 1e-12, "5/5 one way: p = 2/32");
        let t = run(&[&r], &Opts::default());
        assert_eq!((t.favour_a, t.verdict), (6, "a_better"));
        assert!((t.p_value - 0.03125).abs() < 1e-12);
    }

    #[test]
    fn many_samples_of_one_family_count_once() {
        let r: String = (1..=40).map(|s| std::format!("x\tt1-helix-gcd\t{s}\tpass\nx\tt2-rust-gcd\t{s}\tfail\n")).collect();
        let t = run(&[&r], &Opts::default());
        assert_eq!((t.families.len(), t.favour_a, t.verdict, t.paired_obs), (1, 1, "underpowered", 40));
        assert_eq!(t.per_arm, std::vec![("x".to_string(), 40, 40, 40, 0)]);
    }

    #[test]
    fn explicit_map_overrides_names_and_first_try_counts_one_attempt_only() {
        let map = parse_map("# twins\nfamily\trust_task\thelix_task\nroman\tt59-rust-roman\tt55-helix-roman\nodd\tq1\tq2\n", "helix", "rust").unwrap();
        assert_eq!(map.get("q2"), Some(&('a', "odd".to_string())), "names without a language token pair via the map");
        let r = "x\tt55-helix-roman\t1\tpass\tattempts=3 ttd_s=9\nx\tt59-rust-roman\t1\tpass\tattempts=1 ttd_s=3\n\
                 x\tq2\t1\tpass\nx\tq1\t1\tfail\n";
        let t = run(&[r], &Opts { map: map.clone(), first_try: false });
        assert_eq!(t.families.iter().map(|f| (f.family.as_str(), f.favours())).collect::<Vec<_>>(), std::vec![("odd", 1), ("roman", 0)]);
        assert_eq!((t.ttd_pairs, t.ttd_ratio), (1, Some(3.0 / 9.0)));
        let t = run(&[r], &Opts { map, first_try: true });
        let roman = t.families.iter().find(|f| f.family == "roman").unwrap();
        assert_eq!((roman.pass_a, roman.pass_b, roman.favours()), (0, 1, -1), "helix needed 3 attempts: not a first-try pass");
        assert!(parse_map("family\trust_task\n", "helix", "rust").is_err());
    }
}
