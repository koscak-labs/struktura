//! Twins: is variant A as good as variant B, judged only on TWIN tasks.
//!
//! A workbench asks a model to solve tasks under several configs (arms). Some tasks
//! come in twins that differ only in one token of their name, e.g. `t11-helix-gcd`
//! and `t12-rust-gcd`: the same problem, asked in two languages. Comparing the
//! pass rate of all Helix tasks with all Rust tasks would mostly measure which
//! problems happened to be harder; comparing twins removes that confound.
//!
//! The unit of inference is the task FAMILY (`gcd`), not the run: repeated samples
//! and arms of one family are correlated, so they are pooled into one family-level
//! pass rate per side, and families are compared with an exact two-sided sign test.
//! With fewer than 6 families that lean one way, no outcome can reach p < 0.05, and
//! the verdict says `underpowered` instead of pretending.
//!
//! Tasks without a twin are listed separately as `solos`: informative, but confounded.

use std::string::{String, ToString};
use std::vec::Vec;
use std::collections::BTreeMap;

/// One paired observation: the same arm, family and sample, run on both twins.
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

/// One family, pooled over arms and samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Family {
    pub family: String,
    pub task_a: String,
    pub task_b: String,
    pub n: usize,
    pub pass_a: usize,
    pub pass_b: usize,
}

impl Family {
    /// +1 when A passed more often, -1 when B did, 0 on a tie.
    pub fn favours(&self) -> i8 {
        if self.pass_a > self.pass_b { 1 } else if self.pass_b > self.pass_a { -1 } else { 0 }
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

#[derive(Clone, Debug)]
pub struct TwinsReport {
    pub a: String,
    pub b: String,
    pub families: Vec<Family>,
    /// Per arm: (arm, paired n, A passes, B passes).
    pub per_arm: Vec<(String, usize, usize, usize)>,
    pub favour_a: usize,
    pub favour_b: usize,
    pub ties: usize,
    /// Exact two-sided sign test over the families that lean one way.
    pub p_value: f64,
    /// Median of ttd_b / ttd_a over paired observations where BOTH passed.
    pub ttd_ratio: Option<f64>,
    pub ttd_pairs: usize,
    pub solos: Vec<Solo>,
    /// `a_better` | `b_better` | `no_difference_detected` | `underpowered`
    pub verdict: &'static str,
}

/// Fewest one-sided families for which the exact two-sided sign test can reach p < 0.05.
pub const MIN_FAMILIES: usize = 6;

/// The family of `task` with respect to `token`: the task name without its leading
/// `tNN-` id and without the `token` part (`t11-helix-gcd`, `helix` -> `gcd`).
/// `None` when `token` is not one of the dash-separated parts.
pub fn family_of(task: &str, token: &str) -> Option<String> {
    let mut parts: Vec<&str> = task.split('-').collect();
    if parts.len() > 1 {
        let p0 = parts[0];
        if p0.len() > 1 && p0.starts_with('t') && p0[1..].chars().all(|c| c.is_ascii_digit()) { parts.remove(0); }
    }
    let at = parts.iter().position(|p| p.eq_ignore_ascii_case(token))?;
    parts.remove(at);
    if parts.is_empty() { return None; }
    Some(parts.join("-"))
}

fn ttd_of(meta: &[&str]) -> Option<f64> {
    meta.iter().flat_map(|f| f.split_whitespace())
        .find_map(|t| t.strip_prefix("ttd_s=")).and_then(|v| v.trim_end_matches('s').parse::<f64>().ok())
        .filter(|x| x.is_finite())
}

/// Parse workbench verdict files (`arm<TAB>task<TAB>sample<TAB>pass|fail<TAB>k=v ...`)
/// into twin observations and solos. Each input text is one run; samples of different
/// runs are never paired with each other.
pub fn parse(texts: &[&str], a: &str, b: &str) -> (Vec<Obs>, Vec<Solo>, Vec<(String, String, String)>) {
    // (arm, family, run:sample) -> (side a: (task, pass, ttd), side b: ...)
    type Side = Option<(String, bool, Option<f64>)>;
    let mut cells: BTreeMap<(String, String, String), (Side, Side)> = BTreeMap::new();
    let mut tasks: BTreeMap<String, (char, String, usize, usize)> = BTreeMap::new(); // task -> (side, family, n, pass)
    for (run, text) in texts.iter().enumerate() {
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').map(|s| s.trim()).collect();
            if f.len() < 4 || f[2].parse::<u64>().is_err() { continue; }
            let pass = match f[3].to_ascii_lowercase().as_str() { "pass" | "ok" | "true" => true, "fail" | "false" | "error" => false, _ => continue };
            let (side, fam) = match (family_of(f[1], a), family_of(f[1], b)) {
                (Some(fa), None) => ('a', fa),
                (None, Some(fb)) => ('b', fb),
                _ => continue, // neither variant, or both tokens (ambiguous)
            };
            let t = tasks.entry(f[1].to_string()).or_insert((side, fam.clone(), 0, 0));
            t.2 += 1;
            if pass { t.3 += 1; }
            let cell = cells.entry((f[0].to_string(), fam, std::format!("{}:{}", run, f[2]))).or_insert((None, None));
            let v = Some((f[1].to_string(), pass, ttd_of(&f[4..])));
            if side == 'a' { cell.0 = v } else { cell.1 = v }
        }
    }
    let mut obs = Vec::new();
    let mut twin_names: Vec<(String, String, String)> = Vec::new(); // (family, task_a, task_b)
    for ((arm, family, sample), (sa, sb)) in cells {
        if let (Some((ta, pa, da)), Some((tb, pb, db))) = (sa, sb) {
            if !twin_names.iter().any(|(f, _, _)| *f == family) { twin_names.push((family.clone(), ta, tb)); }
            obs.push(Obs { arm, family, sample, a: pa, b: pb, ttd_a: da, ttd_b: db });
        }
    }
    let solos = tasks.into_iter()
        .filter(|(_, (_, fam, _, _))| !twin_names.iter().any(|(f, _, _)| f == fam))
        .map(|(task, (side, _, n, pass))| Solo { task, side, n, pass })
        .collect();
    (obs, solos, twin_names)
}

/// Pool the paired observations per family and per arm, and decide.
pub fn judge(obs: &[Obs], solos: Vec<Solo>, names: &[(String, String, String)], a: &str, b: &str, alpha: f64) -> TwinsReport {
    let mut fams: Vec<Family> = Vec::new();
    let mut arms: Vec<(String, usize, usize, usize)> = Vec::new();
    let mut ratios: Vec<f64> = Vec::new();
    for o in obs {
        let fi = match fams.iter().position(|f| f.family == o.family) {
            Some(i) => i,
            None => {
                let (ta, tb) = names.iter().find(|(f, _, _)| *f == o.family).map(|(_, x, y)| (x.clone(), y.clone())).unwrap_or_default();
                fams.push(Family { family: o.family.clone(), task_a: ta, task_b: tb, n: 0, pass_a: 0, pass_b: 0 });
                fams.len() - 1
            }
        };
        let f = &mut fams[fi];
        f.n += 1; f.pass_a += o.a as usize; f.pass_b += o.b as usize;
        let ai = match arms.iter().position(|x| x.0 == o.arm) { Some(i) => i, None => { arms.push((o.arm.clone(), 0, 0, 0)); arms.len() - 1 } };
        arms[ai].1 += 1; arms[ai].2 += o.a as usize; arms[ai].3 += o.b as usize;
        if let (true, true, Some(x), Some(y)) = (o.a, o.b, o.ttd_a, o.ttd_b) { if x > 0.0 { ratios.push(y / x); } }
    }
    let favour_a = fams.iter().filter(|f| f.favours() > 0).count();
    let favour_b = fams.iter().filter(|f| f.favours() < 0).count();
    let ties = fams.len() - favour_a - favour_b;
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
    TwinsReport { a: a.to_string(), b: b.to_string(), families: fams, per_arm: arms, favour_a, favour_b, ties,
        p_value, ttd_ratio, ttd_pairs, solos, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(rows: &[(&str, &str, u32, &str)]) -> String {
        rows.iter().map(|(arm, task, s, v)| std::format!("{}\t{}\t{}\t{}\tttd_s=2.0\n", arm, task, s, v)).collect()
    }

    #[test]
    fn family_strips_id_and_token_only() {
        assert_eq!(family_of("t11-helix-gcd", "helix").as_deref(), Some("gcd"));
        assert_eq!(family_of("t12-rust-gcd", "rust").as_deref(), Some("gcd"));
        assert_eq!(family_of("t51-patch-rust-lru", "rust").as_deref(), Some("patch-lru"));
        assert_eq!(family_of("t12-rust-gcd", "helix"), None);
        assert_eq!(family_of("trusty-gcd", "rust"), None, "a token must be a whole part");
        assert_eq!(family_of("t9-rust", "rust"), None, "nothing left = no family");
    }

    #[test]
    fn pairs_only_same_arm_family_sample_and_run() {
        let r1 = run(&[("x", "t11-helix-gcd", 1, "pass"), ("x", "t12-rust-gcd", 1, "fail"),
                       ("x", "t11-helix-gcd", 2, "pass"),                       // no rust twin for sample 2
                       ("y", "t12-rust-gcd", 1, "pass"),                        // no helix for arm y
                       ("x", "t55-helix-roman", 1, "pass"), ("x", "t31-rust-csv", 1, "fail")]);
        let r2 = run(&[("x", "t12-rust-gcd", 2, "pass")]);                      // other run: never pairs with r1
        let (obs, solos, names) = parse(&[&r1, &r2], "helix", "rust");
        assert_eq!(obs.len(), 1);
        assert_eq!((obs[0].a, obs[0].b), (true, false));
        assert_eq!(names, std::vec![("gcd".to_string(), "t11-helix-gcd".to_string(), "t12-rust-gcd".to_string())]);
        let s: Vec<(&str, char)> = solos.iter().map(|s| (s.task.as_str(), s.side)).collect();
        assert_eq!(s, std::vec![("t31-rust-csv", 'b'), ("t55-helix-roman", 'a')], "twinned tasks are not solos");
    }

    #[test]
    fn five_one_sided_families_are_underpowered_six_decide() {
        let mut rows = Vec::new();
        let fams = ["a", "b", "c", "d", "e", "f"];
        for (i, f) in fams.iter().enumerate() {
            rows.push(std::format!("x\tt{}-helix-{}\t1\tpass\n", i, f));
            rows.push(std::format!("x\tt{}-rust-{}\t1\tfail\n", i + 50, f));
        }
        let five: String = rows[..10].concat();
        let (o, s, n) = parse(&[&five], "helix", "rust");
        let r = judge(&o, s, &n, "helix", "rust", 0.05);
        assert_eq!((r.favour_a, r.favour_b, r.verdict), (5, 0, "underpowered"));
        assert!((r.p_value - 0.0625).abs() < 1e-12, "5/5 one way: p = 2/32");
        let six: String = rows.concat();
        let (o, s, n) = parse(&[&six], "helix", "rust");
        let r = judge(&o, s, &n, "helix", "rust", 0.05);
        assert_eq!((r.favour_a, r.verdict), (6, "a_better"));
        assert!((r.p_value - 0.03125).abs() < 1e-12);
    }

    #[test]
    fn many_samples_of_one_family_count_once() {
        // 40 samples, all helix pass / rust fail, but ONE family: still one vote, underpowered.
        let rows: String = (1..=40).map(|s| std::format!("x\tt1-helix-gcd\t{s}\tpass\nx\tt2-rust-gcd\t{s}\tfail\n")).collect();
        let (o, s, n) = parse(&[&rows], "helix", "rust");
        assert_eq!(o.len(), 40);
        let r = judge(&o, s, &n, "helix", "rust", 0.05);
        assert_eq!((r.families.len(), r.favour_a, r.verdict), (1, 1, "underpowered"));
        assert_eq!(r.per_arm, std::vec![("x".to_string(), 40, 40, 0)]);
    }

    #[test]
    fn ttd_ratio_uses_only_pairs_where_both_passed() {
        let rows = "x\tt1-helix-gcd\t1\tpass\tttd_s=2.0\nx\tt2-rust-gcd\t1\tpass\tttd_s=6.0\n\
                    x\tt1-helix-gcd\t2\tpass\tttd_s=1.0\nx\tt2-rust-gcd\t2\tfail\tttd_s=99\n";
        let (o, s, n) = parse(&[rows], "helix", "rust");
        let r = judge(&o, s, &n, "helix", "rust", 0.05);
        assert_eq!((r.ttd_pairs, r.ttd_ratio), (1, Some(3.0)));
    }
}
