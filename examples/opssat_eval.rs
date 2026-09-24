//! OPS-SAT-AD (ESA OPS-SAT telemetry, KP Labs; Zenodo 12588359): does
//! struktura's own DFA alpha separate anomalous segments from nominal ones?
//!
//! Protocol: for each segment, alpha = struktura::dfa_short(values).alpha
//! (struktura::dfa is reported too; it has no alpha below 64 samples). Per
//! channel, take the median and std of alpha over the nominal TRAIN
//! segments; the test score is |alpha - median| / std. The train labels pick
//! that nominal reference, so this is not fully unsupervised; test labels
//! are used only for scoring. Report AUC-ROC and AUC-PR on the 529 test
//! segments. This is segment classification, not the streaming guard.
//!
//! Controls on the same segments and scoring:
//! - shuffled: values shuffled within each segment before DFA (keeps
//!   variance and length, destroys order). Structure-driven AUC should drop.
//! - length only, log-variance only: simpler features with the same
//!   per-channel |z| scoring, to see whether alpha only proxies them.
//!
//! Run: OPSSAT_DIR=path/to/OPS-SAT-AD/data cargo run --release --example opssat_eval

use std::collections::HashMap;

struct Seg {
    channel: String,
    anomaly: bool,
    train: bool,
    values: Vec<f64>,
}

fn load(path: &str) -> Vec<Seg> {
    let text = std::fs::read_to_string(path).expect("read segments.csv");
    let mut order: Vec<u64> = Vec::new();
    let mut segs: HashMap<u64, Seg> = HashMap::new();
    for line in text.lines().skip(1) {
        // channel,timestamp,value,label,sampling,anomaly,segment,train
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 8 {
            continue;
        }
        let id: u64 = f[6].trim().parse().expect("segment id");
        let v: f64 = match f[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let s = segs.entry(id).or_insert_with(|| {
            order.push(id);
            Seg {
                channel: f[0].to_string(),
                anomaly: f[5].trim() == "1",
                train: f[7].trim() == "1",
                values: Vec::new(),
            }
        });
        s.values.push(v);
    }
    order.into_iter().map(|id| segs.remove(&id).unwrap()).collect()
}

fn shuffled(v: &[f64], seed: u64) -> Vec<f64> {
    let mut out = v.to_vec();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..out.len()).rev() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let j = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN; // e.g. struktura::dfa has no finite alpha on any n < 64 segment
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn std(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// Per-channel |x - median| / std against nominal train segments; 0 when
/// the feature is not finite or the channel has no usable train stats.
fn z_scores(segs: &[Seg], feat: &[f64]) -> Vec<(f64, bool)> {
    let mut nom: HashMap<&str, Vec<f64>> = HashMap::new();
    for (s, &f) in segs.iter().zip(feat) {
        if s.train && !s.anomaly && f.is_finite() {
            nom.entry(&s.channel).or_default().push(f);
        }
    }
    let stats: HashMap<&str, (f64, f64)> = nom
        .into_iter()
        .filter(|(_, v)| v.len() >= 2)
        .map(|(k, mut v)| (k, (median(&mut v), std(&v))))
        .collect();
    segs.iter()
        .zip(feat)
        .filter(|(s, _)| !s.train)
        .map(|(s, &f)| {
            let z = match stats.get(s.channel.as_str()) {
                Some(&(m, sd)) if f.is_finite() && sd > 0.0 => (f - m).abs() / sd,
                _ => 0.0,
            };
            (z, s.anomaly)
        })
        .collect()
}

/// AUC-ROC via the rank-sum statistic (ties get average rank).
fn auc_roc(scored: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = scored.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut rank_sum, mut i) = (0.0, 0);
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        rank_sum += v[i..=j].iter().filter(|x| x.1).count() as f64 * avg;
        i = j + 1;
    }
    let pos = v.iter().filter(|x| x.1).count() as f64;
    let neg = v.len() as f64 - pos;
    (rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

/// Average precision (AUC-PR as sklearn's average_precision_score).
fn auc_pr(scored: &[(f64, bool)]) -> f64 {
    let mut v: Vec<(f64, bool)> = scored.to_vec();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let pos = v.iter().filter(|x| x.1).count() as f64;
    let (mut tp, mut ap, mut i) = (0.0, 0.0, 0);
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        let new_tp = v[i..=j].iter().filter(|x| x.1).count() as f64;
        tp += new_tp;
        ap += (new_tp / pos) * (tp / (j + 1) as f64);
        i = j + 1;
    }
    ap
}

fn main() {
    let dir = std::env::var("OPSSAT_DIR").expect("set OPSSAT_DIR to OPS-SAT-AD/data");
    let all = load(&format!("{dir}/segments.csv"));
    println!("== all segments ==");
    evaluate(&all);
    // struktura::dfa needs at least 64 samples; dfa_short goes lower.
    let (long, short): (Vec<Seg>, Vec<Seg>) = all.into_iter().partition(|s| s.values.len() >= 64);
    println!();
    println!("== segments with >= 64 samples ==");
    evaluate(&long);
    println!();
    println!("== segments with < 64 samples ==");
    evaluate(&short);
}

fn evaluate(segs: &[Seg]) {
    let test: Vec<&Seg> = segs.iter().filter(|s| !s.train).collect();
    let mut lens: Vec<f64> = segs.iter().map(|s| s.values.len() as f64).collect();
    println!(
        "segments={} test={} test anomalies={} median length={}",
        segs.len(),
        test.len(),
        test.iter().filter(|s| s.anomaly).count(),
        median(&mut lens)
    );

    let alpha = |v: &[f64]| {
        let r = struktura::dfa(v);
        // n < 64: struktura::dfa returns the placeholder alpha 0.5 with R² 0.
        if r.alpha.is_finite() && r.r_squared > 0.0 { r.alpha } else { f64::NAN }
    };
    let feats: Vec<(&str, Vec<f64>)> = vec![
        ("struktura dfa alpha", segs.iter().map(|s| alpha(&s.values)).collect()),
        (
            "control: alpha of shuffled values",
            segs.iter().enumerate().map(|(i, s)| alpha(&shuffled(&s.values, i as u64))).collect(),
        ),
        ("struktura dfa_short alpha", segs.iter().map(|s| struktura::dfa_short(&s.values).map_or(f64::NAN, |r| r.alpha)).collect()),
        (
            "control: dfa_short alpha of shuffled values",
            segs.iter().enumerate().map(|(i, s)| struktura::dfa_short(&shuffled(&s.values, i as u64)).map_or(f64::NAN, |r| r.alpha)).collect(),
        ),
        ("control: segment length", segs.iter().map(|s| s.values.len() as f64).collect()),
        (
            "control: log variance",
            segs.iter()
                .map(|s| if s.values.len() > 1 { std(&s.values).powi(2).max(1e-300).ln() } else { f64::NAN })
                .collect(),
        ),
    ];

    let nan = feats[0].1.iter().filter(|a| !a.is_finite()).count();
    println!("alpha not finite on {nan} segments (scored 0)");
    println!();
    println!("| score (per-channel |z| vs nominal train) | AUC-ROC | AUC-PR |");
    println!("|---|---|---|");
    for (name, f) in &feats {
        let z = z_scores(segs, f);
        println!("| {name} | {:.3} | {:.3} |", auc_roc(&z), auc_pr(&z));
    }
    let base = test.iter().filter(|s| s.anomaly).count() as f64 / test.len() as f64;
    println!("| random score (AUC-PR baseline = anomaly rate) | 0.500 | {base:.3} |");

    // Where alpha sits: nominal vs anomalous test segments.
    let mut a_nom: Vec<f64> = segs.iter().zip(&feats[0].1).filter(|(s, a)| !s.train && !s.anomaly && a.is_finite()).map(|(_, &a)| a).collect();
    let mut a_an: Vec<f64> = segs.iter().zip(&feats[0].1).filter(|(s, a)| !s.train && s.anomaly && a.is_finite()).map(|(_, &a)| a).collect();
    println!();
    println!("median alpha, test: nominal {:.3}, anomalous {:.3}", median(&mut a_nom), median(&mut a_an));
}
