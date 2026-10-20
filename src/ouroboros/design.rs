//! Design one experiment: power-sized, honestly thresholded, measuring the
//! metric the knob moves, with a pre-registration lab-score can score.
//!
//! - Predicted effect: the observed effect if the logs have one, else a 3%
//!   prior shrunk (never inflated) by the lab's calibration factor. Never
//!   below one noise band.
//! - Threshold: one band below the predicted effect (but at least one band),
//!   so a pass carries information and noise cannot pass it.
//! - Runs per arm: `power::runs_needed` for the predicted effect at the
//!   ledger's calibration CV, clamped to 2..8.
//! - Output: each repeat as `rep<TAB><arm><TAB><metric>=x`, then one summary
//!   line per arm `arm<TAB><arm><TAB>median <metric>=m`. lab-score reads the
//!   first matching line, so the pre-registration targets the summary lines;
//!   `struktura arms` bootstraps the repeats.

use super::agenda::Item;
use super::knobs::Knob;
use super::learn::Lessons;

#[derive(Clone, Debug)]
pub struct Design {
    pub job_name: String,
    pub knob: String,
    pub challenger: String,
    pub incumbent: String,
    pub metric: String,
    pub lower_is_better: bool,
    pub predicted_effect_pct: f64,
    pub effect_source: String,
    pub threshold_pct: f64,
    pub runs_per_arm: usize,
    pub est_min: u32,
    pub job: String,
    pub pred: String,
    pub manifest: String,
}

/// Prior effect (percent) assumed for a challenger never measured.
pub const PRIOR_EFFECT_PCT: f64 = 3.0;

const MED: &str = "med(){ sort -n | awk '{a[NR]=$1} END{if(NR==0){print \"VOID\"; exit} if(NR%2) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}'; }";

const UB: &str = r#"D=$(stage llama.cpp-alien2)
M=/home/phil/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
for v in @INC@ @CH@; do
  for r in $(seq 1 @R@); do
    x=$(LD_LIBRARY_PATH=$D timeout --foreground 900 $D/llama-bench -m $M -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -b 2048 -ub $v -p 2048 -n 0 -d 0,32768,100000 -r 1 -o csv 2>/dev/null \
        | awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/"/,"",$i); h[$i]=i}} NR>1{gsub(/"/,""); printf "d%s=%d ", $h["n_depth"], $h["avg_ts"]}')
    printf 'rep\t@P@%s\t%s\n' "$v" "${x:-FAIL}" | tee -a "$OUT"
  done
  m=$(grep -P "^rep\t@P@$v\t" "$OUT" | grep -oP '@METRIC@=\K[0-9.]+' | med)
  printf 'arm\t@P@%s\tmedian @METRIC@=%s\n' "$v" "$m" | tee -a "$OUT"
done
"#;

const DRAFT: &str = r#"B=$(stage llama.cpp-alien2)
M=/home/phil/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
DF=/home/phil/qmodels/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
q(){ curl -s -m 300 localhost:$LAB_P/v1/chat/completions -H 'Content-Type: application/json' \
     -d "$(jq -nc --arg c "$1" '{messages:[{role:"user",content:$c}],max_tokens:512,temperature:0.6}')" | jq -r '.timings.predicted_per_second // empty'; }
for v in @INC@ @CH@; do
  SRV=~/logs/@JOB@-srv-$v.log
  $B/llama-server -m $M -ngl 99 -fa on --jinja -np 2 -c 262144 --kv-unified -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 \
    -md $DF -ngld 99 --spec-type draft-dflash --spec-draft-n-max $v -ub 256 --port $LAB_P --host 127.0.0.1 > $SRV 2>&1 & pid=$!
  if ! up $pid; then printf 'arm\t@P@%s\tERROR server-died\n' "$v" | tee -a "$OUT"; continue; fi
  if ! smoke $SRV >/dev/null; then printf 'arm\t@P@%s\tERROR smoke\n' "$v" | tee -a "$OUT"; kill $pid; wait $pid 2>/dev/null; continue; fi
  for r in $(seq 1 @R@); do
    t=$(q "Write a C++ function split_csv(const std::string&) that splits one CSV line into fields, handling quoted commas. Output only code.")
    printf 'rep\t@P@%s\t@METRIC@=%s\n' "$v" "${t:-FAIL}" | tee -a "$OUT"
  done
  m=$(grep -P "^rep\t@P@$v\t" "$OUT" | grep -oP '@METRIC@=\K[0-9.]+' | med)
  printf 'arm\t@P@%s\tmedian @METRIC@=%s\n' "$v" "$m" | tee -a "$OUT"
  kill $pid; wait $pid 2>/dev/null
done
"#;

fn template(id: &str) -> Option<(&'static str, u32, u32)> {
    // (body, fixed minutes, minutes per run per arm)
    match id { "ub" => Some((UB, 2, 2)), "draft" => Some((DRAFT, 4, 1)), _ => None }
}

/// The pre-registration regex for one arm's summary line.
pub fn arm_regex(k: &Knob, value: &str) -> String {
    format!("^arm\\t{}\\tmedian {}=([\\d.]+)", k.arm_of(value).1, k.metric)
}

pub fn design(item: &Item, k: &Knob, lessons: &Lessons, cv_pct: f64, job_id: u32) -> Result<Design, String> {
    // External lab template (`template=ext:<file>` in the knob file): a wrapper hands the arms to
    // ~/lab-templates/<file> on raven, which must print the same contract as built-in templates:
    // one `rep<TAB><arm><TAB><metric>=x` line per run and one `arm<TAB><arm><TAB>median <metric>=m` per arm.
    let ext_body;
    let (body, fixed, per_run): (&str, u32, u32) = if let Some(file) = k.template.strip_prefix("ext:") {
        if file.is_empty() || file.contains('/') || file.contains("..") { return Err(format!("knob {}: bad external template name {:?}", k.name, file)); }
        ext_body = format!("TPL=~/lab-templates/{f}\n[ -f \"$TPL\" ] || {{ echo \"MISSING template $TPL\" | tee -a \"$OUT\"; exit 2; }}\n\
            ARMS=\"@CH@ @INC@\" RUNS=@R@ OUT=\"$OUT\" PREFIX=\"@P@\" METRIC=\"@METRIC@\" JOB=\"@JOB@\" bash \"$TPL\"\n", f = file);
        (ext_body.as_str(), 10, 4)
    } else {
        let t = template(&k.template).ok_or(format!("knob {} has no job instrument for {}", k.name, k.metric))?;
        if !k.arm_of(&item.value).0.is_empty() { return Err(format!("knob {} uses grouped arms; built-in templates need prefix arms", k.name)); }
        t
    };
    let band = lessons.band_pct;
    let (pred, src) = match item.observed_effect_pct {
        Some(e) if e > 0.0 => (e, format!("observed {:+.2}% in {} run(s)", e, item.samples)),
        // The factor only shrinks a guess: >1 means the lab registers thresholds below what it then
        // measures, which says nothing about how large an unmeasured effect is.
        _ => { let f = lessons.calibration.min(1.0); (PRIOR_EFFECT_PCT * f, format!("prior {:.1}% x calibration min(1, {:.2})", PRIOR_EFFECT_PCT, lessons.calibration)) }
    };
    let pred = pred.max(band);
    let threshold = (pred - band).max(band);
    let runs = if cv_pct.is_finite() && cv_pct > 0.0 { crate::power::runs_needed(cv_pct, pred, 0.05, 0.8).clamp(2, 8) } else { 3 };
    let est = fixed + per_run * runs as u32 * 2;
    let job_name = format!("lab-{}-{}{}", job_id, k.name, item.value);
    let prefix = match k.matcher.split_once(':') { Some(("prefix", p)) => p, Some(("group", _)) => "", _ => &k.name };
    let value_meta = ((item.score * 5.0).round() as i64).clamp(1, 10);
    let mut job = format!("#!/usr/bin/env bash\n# LAB-META: est_min={} value={} out=logs/{n}.out pred=lab-pred/{n}.tsv\n\
# struktura loop {n}: {} {} (arm A) vs incumbent {} (arm B) on {} ({}), {} runs per arm\n\
# why: {}\nset -uo pipefail\nsource ~/lab-lib.sh\nOUT=~/logs/{n}.out; : > \"$OUT\"\n{}\n",
        est, value_meta, k.name, item.value, k.current, k.metric, k.what, runs, item.why, MED, n = job_name);
    job.push_str(&body.replace("@INC@", &k.current).replace("@CH@", &item.value).replace("@R@", &runs.to_string())
        .replace("@P@", prefix).replace("@METRIC@", &k.metric).replace("@JOB@", &job_name));
    job.push_str(&format!("echo \"### {}-DONE $(date +%T)\" | tee -a \"$OUT\"\n", job_name));
    let (a, b) = (arm_regex(k, &item.value), arm_regex(k, &k.current));
    let op = if k.lower_is_better { "<%" } else { ">%" };
    let pred_tsv = format!("# struktura loop {}: predicted {:.2}% ({}), threshold {:.2}% = predicted - one {:.2}% band, {} runs/arm\n\
{}{}-beats-{}{}-{}\t{}\t{}\t{}@@{:.2}\n",
        job_name, pred, src, threshold, band, runs, k.name, item.value, k.name, k.current, k.metric, a, op, b, threshold);
    check_pred(&pred_tsv)?;
    let manifest = format!("{{\"job\":\"{}\",\"knob\":\"{}\",\"challenger\":\"{}\",\"incumbent\":\"{}\",\"metric\":\"{}\",\"lower_is_better\":{},\"predicted_effect_pct\":{:.3},\"effect_source\":\"{}\",\"threshold_pct\":{:.3},\"band_pct\":{:.3},\"cv_pct\":{:.3},\"runs_per_arm\":{},\"est_min\":{},\"calibration\":{:.3},\"why\":\"{}\"}}",
        job_name, k.name, item.value, k.current, k.metric, k.lower_is_better, pred, src, threshold, band, cv_pct, runs, est, lessons.calibration, item.why.replace('"', "'"));
    Ok(Design { job_name, knob: k.name.clone(), challenger: item.value.clone(), incumbent: k.current.clone(), metric: k.metric.clone(),
        lower_is_better: k.lower_is_better, predicted_effect_pct: pred, effect_source: src, threshold_pct: threshold, runs_per_arm: runs,
        est_min: est, job, pred: pred_tsv, manifest })
}

/// Refuse a relative prediction whose two arms read the same line (it would compare a value with itself).
pub fn check_pred(tsv: &str) -> Result<(), String> {
    for line in tsv.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 4 { return Err(format!("malformed pred line: {}", line)); }
        if matches!(c[2], ">%" | "<%" | "~%") {
            let b = c[3].split("@@").next().unwrap_or("");
            if c[1] == b { return Err(format!("tautological prediction {}: arm A regex == arm B regex", c[0])); }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ouroboros::agenda::{Item, Kind};

    #[test]
    fn external_template_wrapper_for_grouped_knob() {
        let k = crate::ouroboros::knobs::parse_knobs("knob chunk 0 32 64 128 current=64 metric=sum_s dir=lower match=group:contend alias=0:stock template=ext:contend.sh").unwrap().remove(0);
        let it = Item { kind: Kind::Challenger, knob: "chunk".into(), value: "32".into(), pred: None, score: 0.6, observed_effect_pct: None, samples: 0, why: "test".into() };
        let ls = crate::ouroboros::learn::Lessons { calibration: 1.0, calibration_n: 0, fragile: Vec::new(), easy: 0, band_pct: 1.6 };
        let d = design(&it, &k, &ls, 0.578, 301).expect("designs");
        assert!(d.job.contains("TPL=~/lab-templates/contend.sh") && d.job.contains("MISSING template"), "{}", d.job);
        assert!(d.job.contains("ARMS=\"32 64\"") && d.job.contains("PREFIX=\"\""), "{}", d.job);
        assert!(d.pred.contains("^arm\\t32\\tmedian sum_s=") && d.pred.contains("<%"), "{}", d.pred);
        assert!(design(&it, &crate::ouroboros::knobs::parse_knobs("knob chunk 32 64 current=64 metric=sum_s template=ext:../x.sh").unwrap().remove(0), &ls, 0.578, 301).is_err(), "path traversal refused");
    }
    use crate::ouroboros::knobs::{parse_knobs, BUILTIN};

    fn lessons(cal: f64) -> Lessons { Lessons { calibration: cal, calibration_n: 5, fragile: vec![], easy: 0, band_pct: 1.60 } }
    fn item(knob: &str, v: &str, e: Option<f64>) -> Item {
        Item { kind: Kind::Challenger, knob: knob.into(), value: v.into(), pred: None, score: 0.8, observed_effect_pct: e, samples: 1, why: "test".into() }
    }
    fn knob(name: &str) -> Knob { parse_knobs(BUILTIN).unwrap().into_iter().find(|k| k.name == name).unwrap() }

    /// The literal a regex of the form ^...([\d.]+) requires at line start.
    fn literal(re: &str) -> String { re.trim_start_matches('^').trim_end_matches("([\\d.]+)").replace("\\t", "\t") }

    #[test]
    fn power_sized_threshold_inside_one_band() {
        let d = design(&item("ub", "128", Some(6.2)), &knob("ub"), &lessons(1.0), 0.578, 300).unwrap();
        assert!((d.predicted_effect_pct - 6.2).abs() < 1e-9);
        assert!((d.threshold_pct - 4.6).abs() < 1e-9, "{}", d.threshold_pct);
        assert_eq!(d.runs_per_arm, crate::power::runs_needed(0.578, 6.2, 0.05, 0.8).clamp(2, 8));
        // Unmeasured: prior 3% x calibration 0.5 = 1.5% -> raised to one band; threshold never below the band.
        let d = design(&item("ub", "128", None), &knob("ub"), &lessons(0.5), 0.578, 301).unwrap();
        assert!((d.predicted_effect_pct - 1.60).abs() < 1e-9 && (d.threshold_pct - 1.60).abs() < 1e-9);
        assert_eq!(d.runs_per_arm, 3, "1.6% at CV 0.578% needs 3 runs per arm");
    }

    #[test]
    fn pred_regexes_match_exactly_the_summary_lines() {
        let k = knob("ub");
        let d = design(&item("ub", "128", Some(3.0)), &k, &lessons(1.0), 0.578, 302).unwrap();
        let sample = "rep\tub256\td0=2816 d32768=2163 d100000=1397\nrep\tub256\td0=2820 d32768=2160 d100000=1401\narm\tub256\tmedian d100000=1399\n\
rep\tub128\td0=2700 d32768=2100 d100000=1350\nrep\tub128\td0=2710 d32768=2101 d100000=1352\narm\tub128\tmedian d100000=1351\n";
        let line = d.pred.lines().find(|l| !l.starts_with('#')).unwrap();
        let c: Vec<&str> = line.split('\t').collect();
        let (a, b) = (c[1], c[3].split("@@").next().unwrap());
        assert_ne!(a, b);
        for re in [a, b] {
            let lit = literal(re);
            assert_eq!(sample.lines().filter(|l| l.starts_with(&lit)).count(), 1, "regex {} must match one line", re);
            assert!(sample.lines().find(|l| l.starts_with(&lit)).unwrap().starts_with("arm\t"));
        }
        assert_eq!(c[2], ">%");
    }

    #[test]
    fn tautology_and_missing_instrument_refused() {
        assert!(check_pred("x\t^arm\\tub256\\tmedian d=([\\d.]+)\t>%\t^arm\\tub256\\tmedian d=([\\d.]+)@@2\n").is_err());
        assert!(design(&item("chunk", "32", Some(40.0)), &knob("chunk"), &lessons(1.0), 0.578, 303).is_err());
        assert!(design(&item("budget", "4000", None), &knob("budget"), &lessons(1.0), 0.578, 304).is_err());
    }

    #[test]
    fn emitted_job_is_lab_shaped() {
        let d = design(&item("draft", "5", None), &knob("draft"), &lessons(1.0), 0.578, 305).unwrap();
        assert!(d.job.starts_with("#!/usr/bin/env bash\n# LAB-META: est_min="));
        assert!(d.job.contains("source ~/lab-lib.sh") && d.job.contains("set -uo pipefail"));
        assert!(d.job.contains("up $pid") && d.job.contains("smoke $SRV"), "server jobs use the up/smoke gates");
        assert!(!d.job.lines().any(|l| l.trim() == "wait" || l.contains("wait;")), "no bare wait (lint R1)");
        assert!(!d.job.contains("2>&1 |") && !d.job.contains("build/bin/llama"), "lint R2/R3");
        assert!(d.job.contains("--spec-draft-n-max $v") && d.job.contains("for v in 7 5"));
        assert!(d.manifest.contains("\"metric\":\"code_tps\""));
    }
}

#[cfg(test)]
mod prior_tests {
    use super::*;
    use crate::ouroboros::agenda::{Item, Kind};
    use crate::ouroboros::knobs::{parse_knobs, BUILTIN};

    #[test]
    fn calibration_never_inflates_an_unmeasured_prior() {
        let k = parse_knobs(BUILTIN).unwrap().into_iter().find(|k| k.name == "draft").unwrap();
        let it = Item { kind: Kind::Challenger, knob: "draft".into(), value: "5".into(), pred: None, score: 0.5, observed_effect_pct: None, samples: 0, why: "t".into() };
        let ls = |c| Lessons { calibration: c, calibration_n: 19, fragile: vec![], easy: 0, band_pct: 1.6 };
        assert!((design(&it, &k, &ls(3.0), 0.578, 1).unwrap().predicted_effect_pct - 3.0).abs() < 1e-9, "factor 3 must not inflate");
        assert!((design(&it, &k, &ls(0.6), 0.578, 1).unwrap().predicted_effect_pct - 1.8).abs() < 1e-9, "factor 0.6 shrinks");
    }
}
