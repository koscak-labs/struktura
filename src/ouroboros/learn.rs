//! What the lab teaches about itself.
//!
//! **Calibration factor.** Every scored relative prediction records the
//! measured difference next to the threshold the lab registered:
//! `"A vs B (d%, need OP p%)"`. `measured / registered` per prediction (in the
//! claimed direction) tells how the lab's claims compare with reality: well
//! above 1 means thresholds sit far below real effects (easy passes, little
//! information); below 1 means the lab over-claims. The median is used to
//! calibrate any effect the brain has to assume rather than measure.
//!
//! **Fragile results.** Confirmed or refuted predictions whose margin sits
//! inside the single-run noise band are load-bearing guesses: a re-run could
//! flip them. They are the first things to re-measure.

use super::Observation;

#[derive(Clone, Debug, PartialEq)]
pub struct Fragile { pub pred: String, pub name: String, pub verdict: String, pub margin_pct: f64 }

#[derive(Clone, Debug)]
pub struct Lessons {
    /// Median of measured / registered effect over scored relative predictions (1.0 if too few).
    pub calibration: f64,
    pub calibration_n: usize,
    pub fragile: Vec<Fragile>,
    pub easy: usize,
    pub band_pct: f64,
}

/// Parse lab-score's relative value text: "A vs B (d%, need OP p%)" -> (d, OP, p).
pub fn parse_relative(value: &str) -> Option<(f64, String, f64)> {
    let open = value.find('(')?;
    let inner = &value[open + 1..];
    let d: f64 = inner.split('%').next()?.trim().parse().ok()?;
    let need = inner.split("need").nth(1)?;
    let mut w = need.split_whitespace();
    let op = w.next()?.to_string();
    let p: f64 = w.next()?.trim_end_matches(|c| c == '%' || c == ')').parse().ok()?;
    Some((d, op, p))
}

/// Minimum number of scored relative predictions before the factor is trusted.
pub const MIN_CALIBRATION: usize = 3;

pub fn learn(obs: &Observation) -> Lessons {
    let mut ratios = Vec::new();
    let mut fragile = Vec::new();
    for p in &obs.lab.predictions {
        let scored = p.verdict == "pass" || p.verdict == "fail";
        if !scored { continue; }
        if let Some((d, op, reg)) = parse_relative(&p.value) {
            // Only directional claims carry an effect size; equivalence (~%) does not.
            let measured = match op.as_str() { ">%" => Some(d), "<%" => Some(-d), _ => None };
            if let Some(m) = measured { if reg > 0.0 { ratios.push(m / reg); } }
        }
        if let Some(m) = p.margin_pct {
            if m.abs() < obs.band_pct {
                fragile.push(Fragile { pred: p.pred.clone(), name: p.name.clone(), verdict: p.verdict.clone(), margin_pct: m });
            }
        }
    }
    fragile.sort_by(|a, b| a.margin_pct.abs().partial_cmp(&b.margin_pct.abs()).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.pred.cmp(&b.pred)).then(a.name.cmp(&b.name)));
    let n = ratios.len();
    let calibration = if n >= MIN_CALIBRATION {
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = if n % 2 == 1 { ratios[n / 2] } else { 0.5 * (ratios[n / 2 - 1] + ratios[n / 2]) };
        m.clamp(0.25, 4.0)
    } else { 1.0 };
    Lessons { calibration, calibration_n: n, fragile, easy: obs.lab.easy, band_pct: obs.band_pct }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ouroboros::observe;

    fn pred(name: &str, value: &str, verdict: &str, ts: u32) -> String {
        format!("{{\"kind\":\"prediction\",\"ts\":{},\"pred\":\"9.tsv\",\"name\":\"{}\",\"value\":\"{}\",\"op\":\">%\",\"threshold\":\"arm B\",\"verdict\":\"{}\"}}\n", ts, name, value, verdict)
    }

    #[test]
    fn parses_lab_score_relative_text() {
        assert_eq!(parse_relative("1484 vs 1397 (6.23%, need >% 1.60%)"), Some((6.23, ">%".into(), 1.6)));
        assert_eq!(parse_relative("42.6 vs 51.1 (-16.56%, need <% 15%)"), Some((-16.56, "<%".into(), 15.0)));
        assert_eq!(parse_relative("10"), None);
    }

    #[test]
    fn calibration_is_median_measured_over_registered() {
        let mut l = String::from("{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":201.0}}\n");
        l += &pred("a", "120 vs 100 (20.00%, need >% 5%)", "pass", 1);   // ratio 4.0
        l += &pred("b", "106 vs 100 (6.00%, need >% 5%)", "pass", 2);    // ratio 1.2
        l += &pred("c", "110 vs 100 (10.00%, need >% 5%)", "pass", 3);   // ratio 2.0
        l += &pred("d", "100.5 vs 100 (0.50%, need ~% 1%)", "pass", 4);  // equivalence: no effect claim
        let ls = learn(&observe(&l, &[]));
        assert_eq!(ls.calibration_n, 3);
        assert!((ls.calibration - 2.0).abs() < 1e-9, "{}", ls.calibration);
    }

    #[test]
    fn too_few_predictions_leave_calibration_at_one() {
        let l = pred("a", "120 vs 100 (20.00%, need >% 5%)", "pass", 1);
        let ls = learn(&observe(&l, &[]));
        assert_eq!((ls.calibration, ls.calibration_n), (1.0, 1));
    }

    #[test]
    fn fragile_sorted_by_smallest_margin() {
        let mut l = String::from("{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":202.0}}\n");
        l += &pred("wide", "130 vs 100 (30.00%, need >% 5%)", "pass", 1);
        l += &pred("thin", "105.5 vs 100 (5.50%, need >% 5%)", "pass", 2);
        l += &pred("thinner", "105.2 vs 100 (5.20%, need >% 5%)", "pass", 3);
        let ls = learn(&observe(&l, &[]));
        let names: Vec<&str> = ls.fragile.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["thinner", "thin"]);
    }
}
