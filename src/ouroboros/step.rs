//! One turn of the loop: observe -> learn -> decide -> design -> remember.
//!
//! The design is written to the outbox (`<job>.sh`, `<job>.tsv`,
//! `<job>.json`) and one decision row to `knowledge.jsonl`. Re-running on
//! unchanged inputs recalls the decision and changes nothing. Submission to
//! the lab queue is a separate, explicit step ([`submit`]) the caller must ask
//! for; the GPU queue belongs to the lab owner.

use std::path::{Path, PathBuf};

use super::agenda::{agenda, Item, Kind};
use super::design::{design, Design};
use super::knobs::{constraints, Constraint, Knob};
use super::learn::{learn, Lessons};
use super::memory::{decision_id, esc, fnv64, load, next_job_id, recalled, remember};
use super::observe;

pub struct Config {
    pub knobs: Vec<Knob>,
    pub knobs_text: String,
    pub cli_constraints: Vec<Constraint>,
    /// Write the design + memory here; None = dry run.
    pub outbox: Option<PathBuf>,
    pub job_floor: u32,
}

pub struct Turn {
    pub lessons: Lessons,
    pub cv_pct: f64,
    pub constraints: Vec<Constraint>,
    pub agenda: Vec<Item>,
    pub design: Option<Design>,
    /// Why the top challengers could not be designed (no instrument, ...).
    pub skipped: Vec<String>,
    pub id: String,
    pub recalled: bool,
    pub written: Vec<PathBuf>,
}

pub fn turn(ledger: &str, logs: &[(String, String)], cfg: &Config) -> Turn {
    let obs = observe(ledger, logs);
    let lessons = learn(&obs);
    let cons = constraints(&cfg.cli_constraints, &obs.lab.constraints);
    let ag = agenda(&obs, &lessons, &cfg.knobs, &cons);
    let cv = obs.lab.cal_cv_pct;
    let logs_digest: String = logs.iter().map(|(n, t)| format!("{}:{:x}", n, fnv64(t))).collect::<Vec<_>>().join(",");
    let cons_digest: String = cons.iter().map(|c| format!("{}{}{}", c.knob, c.op, c.value)).collect::<Vec<_>>().join(",");

    // The first challenger that can be designed; a dry design (id 0) fixes the decision content.
    let mut skipped = Vec::new();
    let mut pick: Option<&Item> = None;
    for it in ag.iter().filter(|i| i.kind == Kind::Challenger) {
        let k = cfg.knobs.iter().find(|k| k.name == it.knob).unwrap();
        match design(it, k, &lessons, cv, 0) {
            Ok(_) => { pick = Some(it); break; }
            Err(e) => skipped.push(format!("{} {}: {}", it.knob, it.value, e)),
        }
    }
    let top = pick.map(|i| format!("{}={}", i.knob, i.value)).unwrap_or_else(|| "none".into());
    let id = decision_id(&[&format!("{:x}", fnv64(ledger)), &logs_digest, &cfg.knobs_text, &cons_digest, &top]);

    let mut t = Turn { lessons, cv_pct: cv, constraints: cons, agenda: ag.clone(), design: None, skipped, id: id.clone(), recalled: false, written: Vec::new() };
    let Some(item) = pick.cloned() else { return t };
    let k = cfg.knobs.iter().find(|k| k.name == item.knob).unwrap().clone();

    let Some(outbox) = &cfg.outbox else {
        t.design = design(&item, &k, &t.lessons, cv, cfg.job_floor).ok();
        return t;
    };
    let mem = outbox.join("knowledge.jsonl");
    let rows = load(&mem);
    let job_id = match recalled(&rows, &id) {
        Some(Some(job)) => { t.recalled = true; job.strip_prefix("lab-").and_then(|r| r.split('-').next()).and_then(|n| n.parse().ok()).unwrap_or(cfg.job_floor) }
        _ => next_job_id(outbox, &rows, cfg.job_floor),
    };
    let d = match design(&item, &k, &t.lessons, cv, job_id) { Ok(d) => d, Err(e) => { t.skipped.push(e); return t; } };
    if !t.recalled {
        let _ = std::fs::create_dir_all(outbox);
        for (ext, body) in [("sh", &d.job), ("tsv", &d.pred), ("json", &d.manifest)] {
            let p = outbox.join(format!("{}.{}", d.job_name, ext));
            if std::fs::write(&p, body).is_ok() { t.written.push(p); }
        }
        let fr: Vec<String> = t.lessons.fragile.iter().map(|f| format!("\"{}::{}\"", esc(&f.pred), esc(&f.name))).collect();
        let row = format!("{{\"kind\":\"decision\",\"id\":\"{}\",\"ledger_rows\":{},\"calibration\":{:.3},\"calibration_n\":{},\"band_pct\":{:.3},\"cv_pct\":{:.3},\"fragile\":[{}],\"emitted\":\"{}\",\"knob\":\"{}\",\"challenger\":\"{}\",\"incumbent\":\"{}\",\"metric\":\"{}\",\"predicted_effect_pct\":{:.3},\"threshold_pct\":{:.3},\"runs_per_arm\":{},\"why\":\"{}\"}}",
            id, obs.lab.rows, t.lessons.calibration, t.lessons.calibration_n, t.lessons.band_pct, cv, fr.join(","), d.job_name, d.knob, d.challenger,
            d.incumbent, d.metric, d.predicted_effect_pct, d.threshold_pct, d.runs_per_arm, esc(&item.why));
        if remember(&mem, &id, &row).unwrap_or(false) { t.written.push(mem); }
    }
    t.design = Some(d);
    t
}

/// Hand an emitted job to the lab queue (`~/.oura/lab-q.sh <job.sh> <NNN-name>`).
/// Only ever called on an explicit request; the queue owner runs the GPU.
pub fn submit(outbox: &Path, job_name: &str) -> Result<String, String> {
    let script = outbox.join(format!("{}.sh", job_name));
    let q = std::env::var("HOME").map(|h| format!("{}/.oura/lab-q.sh", h)).map_err(|e| e.to_string())?;
    let name = job_name.strip_prefix("lab-").unwrap_or(job_name);
    let out = std::process::Command::new(&q).arg(&script).arg(name).output().map_err(|e| format!("{}: {}", q, e))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if out.status.success() { Ok(text) } else { Err(text) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ouroboros::knobs::{parse_knobs, BUILTIN};

    const LEDGER: &str = "{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":200.0}}\n{\"kind\":\"cal\",\"ok\":true,\"code\":{\"tps\":202.4}}\n";

    fn cfg(outbox: Option<PathBuf>) -> Config {
        Config { knobs: parse_knobs(BUILTIN).unwrap(), knobs_text: BUILTIN.into(), cli_constraints: vec![], outbox, job_floor: 300 }
    }

    #[test]
    fn idempotent_turns_emit_once() {
        let d = std::env::temp_dir().join(format!("struktura-turn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let logs = vec![("lab-115.out".to_string(), "ub128\td100000=1390\nub256\td100000=1397\n".to_string())];
        let a = turn(LEDGER, &logs, &cfg(Some(d.clone())));
        let job = a.design.as_ref().unwrap().job_name.clone();
        assert!(!a.recalled && a.written.len() == 4, "{:?}", a.written);
        let b = turn(LEDGER, &logs, &cfg(Some(d.clone())));
        assert!(b.recalled && b.written.is_empty());
        assert_eq!(b.design.unwrap().job_name, job, "same decision, same job name");
        assert_eq!(load(&d.join("knowledge.jsonl")).len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn dry_run_writes_nothing_and_skips_uninstrumented() {
        let t = turn(LEDGER, &[], &cfg(None));
        assert!(t.written.is_empty());
        let d = t.design.unwrap();
        assert!(d.knob == "draft" || d.knob == "ub", "only instrumented knobs are designed: {}", d.knob);
    }
}
