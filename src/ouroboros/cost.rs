//! What a measurement costs: GPU minutes per knob, from the lab's own job rows.
//!
//! The lab only runs inside windows the production traffic leaves free, so a minute
//! of window is the scarce input. Two uses:
//! - **ranking**: between challengers the agenda values equally, the cheaper one goes
//!   first (more decisions per window);
//! - **fitting**: `p90_min` is published with every agenda row so a scheduler can pick
//!   the best item that fits the window it has, instead of the best item overall.
//!
//! A job is attributed to a knob by its name: designed jobs are `<id>-<knob><value>.sh`
//! (any `-`-separated segment that is the knob name followed by digits). Only clean
//! exits (rc 0) count; a knob with fewer than 2 jobs falls back to knobs on the same
//! metric (same workload), then to all attributed jobs.

use super::knobs::Knob;

#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    pub median_min: f64,
    pub p90_min: f64,
    pub n: usize,
    /// "job" (re-runs of one job), "knob" (its own jobs), "metric" (knobs on the same metric), "all", or "none".
    pub source: &'static str,
}

/// (job name, minutes) of every finished, clean job row.
pub fn job_minutes(ledger: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for line in ledger.lines() {
        let Some(j) = crate::lab::parse_json(line) else { continue };
        if j.str("kind") != Some("job") || j.num("rc").unwrap_or(-1.0) != 0.0 { continue; }
        if let (Some(name), Some(s), Some(e)) = (j.str("job"), j.num("start"), j.num("end")) {
            if e >= s { out.push((name.to_string(), (e - s) / 60.0)); }
        }
    }
    out
}

/// Does this job name measure `knob`? (`343-draft5.sh`, `404-auto-ub512.sh`; not `808-sweep-draftn-…`).
pub fn measures(job: &str, knob: &str) -> bool {
    job.trim_end_matches(".sh").split('-').any(|seg| {
        seg.len() > knob.len() && seg.starts_with(knob) && seg[knob.len()..].chars().all(|c| c.is_ascii_digit())
    })
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn summarise(mut v: Vec<f64>, source: &'static str) -> Cost {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    Cost { median_min: quantile(&v, 0.5), p90_min: quantile(&v, 0.9), n: v.len(), source }
}

/// Cost of one more measurement of `knob`.
pub fn estimate(jobs: &[(String, f64)], knobs: &[Knob], knob: &str) -> Cost {
    let own: Vec<f64> = jobs.iter().filter(|(j, _)| measures(j, knob)).map(|(_, m)| *m).collect();
    if own.len() >= 2 { return summarise(own, "knob"); }
    // the metric names the workload (b and ub both run the 100K prefill bench): same metric, similar minutes
    let metric = knobs.iter().find(|k| k.name == knob).map(|k| k.metric.as_str()).unwrap_or("");
    let kin: Vec<f64> = jobs.iter().filter(|(j, _)| knobs.iter().any(|k| k.metric == metric && measures(j, &k.name))).map(|(_, m)| *m).collect();
    if !metric.is_empty() && kin.len() >= 2 { return summarise(kin, "metric"); }
    let all: Vec<f64> = jobs.iter().filter(|(j, _)| knobs.iter().any(|k| measures(j, &k.name))).map(|(_, m)| *m).collect();
    if all.len() >= 2 { return summarise(all, "all"); }
    Cost { median_min: f64::NAN, p90_min: f64::NAN, n: 0, source: "none" }
}

/// Job id of a prediction file: `lab-344-draft5.tsv` / `344.tsv` / `101-retro.tsv` -> "344" / "344" / "101".
pub fn job_id_of_pred(pred: &str) -> Option<&str> {
    let s = pred.strip_prefix("lab-").unwrap_or(pred);
    let n = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if n == 0 { None } else { Some(&s[..n]) }
}

/// Cost of one agenda item. A re-measure re-runs the job behind its prediction file (source "job":
/// every clean run with that id); a challenger costs what its knob's jobs cost (see `estimate`).
pub fn for_item(jobs: &[(String, f64)], knobs: &[Knob], knob: &str, pred: Option<&str>) -> Cost {
    if let Some(id) = pred.and_then(job_id_of_pred) {
        let runs: Vec<f64> = jobs.iter().filter(|(j, _)| j.split(|c: char| !c.is_ascii_digit()).next() == Some(id)).map(|(_, m)| *m).collect();
        if !runs.is_empty() { return summarise(runs, "job"); }
    }
    if knob.is_empty() { return Cost { median_min: f64::NAN, p90_min: f64::NAN, n: 0, source: "none" }; }
    estimate(jobs, knobs, knob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::knobs::{parse_knobs, BUILTIN};

    const L: &str = "\
{\"kind\":\"job\",\"job\":\"343-draft5.sh\",\"start\":0,\"end\":36,\"rc\":0}
{\"kind\":\"job\",\"job\":\"344-draft5.sh\",\"start\":100,\"end\":136,\"rc\":0}
{\"kind\":\"job\",\"job\":\"346-draft9.sh\",\"start\":200,\"end\":224,\"rc\":0}
{\"kind\":\"job\",\"job\":\"336-ub128.sh\",\"start\":300,\"end\":642,\"rc\":0}
{\"kind\":\"job\",\"job\":\"404-auto-ub512.sh\",\"start\":700,\"end\":1204,\"rc\":0}
{\"kind\":\"job\",\"job\":\"405-ub256.sh\",\"start\":1300,\"end\":9999,\"rc\":1}
{\"kind\":\"job\",\"job\":\"808-sweep-draftn-c7-5-6-8.sh\",\"start\":0,\"end\":400,\"rc\":0}
{\"kind\":\"job\",\"job\":\"999-draftmin1.sh\",\"start\":0,\"end\":60,\"rc\":0}
";

    #[test]
    fn attributes_jobs_by_knob_segment() {
        assert!(measures("343-draft5.sh", "draft"));
        assert!(measures("404-auto-ub512.sh", "ub"));
        assert!(!measures("808-sweep-draftn-c7-5-6-8.sh", "draft"), "a sweep named draftn is not a draft job");
        assert!(!measures("999-draftmin1.sh", "draft"), "draftmin is its own knob");
        assert!(measures("999-draftmin1.sh", "draftmin"));
        assert!(!measures("190-wb-5a3cf82-10061554.sh", "b"), "a hex sha segment is not b<digits>");
    }

    #[test]
    fn per_knob_cost_with_fallback() {
        let k = parse_knobs(BUILTIN).unwrap();
        let j = job_minutes(L);
        assert_eq!(j.len(), 7, "rc!=0 rows are dropped");
        let d = estimate(&j, &k, "draft");
        assert_eq!((d.source, d.n), ("knob", 3));
        assert!((d.median_min - 0.6).abs() < 1e-9 && (d.p90_min - 0.6).abs() < 1e-9);
        let u = estimate(&j, &k, "ub");
        assert_eq!((u.source, u.n), ("knob", 2));
        assert!(u.p90_min > 8.0, "ub is the expensive knob: {:?}", u);
        // b has no jobs of its own: the ub jobs run the same 100K prefill workload (not the cheap draft jobs)
        let b = estimate(&j, &k, "b");
        assert_eq!((b.source, b.n), ("metric", 2));
        assert!(b.median_min > 5.0, "{:?}", b);
        // draftmin has one job of its own: pooled with draft (same code_tps workload)
        assert_eq!(estimate(&j, &k, "draftmin").source, "metric");
        // budget shares no metric with any job: every attributed job stands in
        assert_eq!((estimate(&j, &k, "budget").source, estimate(&j, &k, "budget").n), ("all", 6));
        assert_eq!(estimate(&[], &k, "draft").source, "none");
    }

    #[test]
    fn remeasures_cost_the_job_behind_their_prediction() {
        let k = parse_knobs(BUILTIN).unwrap();
        let j = job_minutes(L);
        assert_eq!(job_id_of_pred("lab-344-draft5.tsv"), Some("344"));
        assert_eq!(job_id_of_pred("404.tsv"), Some("404"));
        assert_eq!(job_id_of_pred("101-retro.tsv"), Some("101"));
        assert_eq!(job_id_of_pred("sweep.tsv"), None);
        let r = for_item(&j, &k, "", Some("404.tsv"));
        assert_eq!((r.source, r.n), ("job", 1));
        assert!((r.median_min - 8.4).abs() < 1e-9);
        // "34" must not match job 343/344
        assert_eq!(for_item(&j, &k, "", Some("lab-34.tsv")).source, "none");
        // a challenger falls through to its knob
        assert_eq!(for_item(&j, &k, "draft", None).source, "knob");
    }
}
