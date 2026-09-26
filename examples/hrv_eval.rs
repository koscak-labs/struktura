//! Heart-rate variability: congestive heart failure (PhysioNet chf2db, 29 records) vs healthy
//! sinus rhythm (nsr2db, 54 records), about 24 h of Holter beat annotations each.
//!
//! Pre-registered question: is struktura's short-term DFA exponent (alpha1, `dfa_short` on
//! 64-beat segments) lower in CHF, as Peng et al. 1995 report? Also reported: alpha2 (`dfa`
//! on 2048-beat segments), a shuffled-beat control, SDNN, mean NN, RMSSD, and whether alpha1
//! adds anything to conventional HRV (leave-one-out logistic regression). The age and heart
//! rate checks at the end are exploratory (added after the first results).
//!
//! NN intervals: consecutive annotation pairs where both are normal beats (N), 0.3-2.0 s.
//! Reads the WFDB MIT annotation files (`.ecg`) and headers (`.hea`) directly.
//!
//! Data: https://physionet.org/content/nsr2db/ and https://physionet.org/content/chf2db/
//! Run: PHYSIONET_DIR=path/containing/nsr2db+chf2db cargo run --release --example hrv_eval

type Feature = fn(&Record) -> f64;

struct Record {
    chf: bool,
    age: f64,
    nyha: String,
    alpha1: f64,
    alpha2: f64,
    rr: Vec<f64>,
    mean_nn: f64,
    sdnn: f64,
    rmssd: f64,
}

/// (sample, annotation code) for every annotation in a WFDB MIT-format file.
/// Each 16-bit little-endian word holds code (top 6 bits) and sample increment (low 10).
/// Codes 59..63 are pseudo-annotations: SKIP carries a 32-bit increment (high word
/// first), NUM/SUB/CHN modify the annotation just read, AUX is followed by I bytes.
fn read_annotations(path: &str) -> Vec<(i64, u8)> {
    let b = std::fs::read(path).expect("read annotation file");
    let word = |k: usize| u16::from_le_bytes([b[k], b[k + 1]]);
    let (mut out, mut t, mut k) = (Vec::new(), 0i64, 0);
    while k + 1 < b.len() {
        let w = word(k);
        let (code, inc) = ((w >> 10) as u8, (w & 0x3ff) as usize);
        k += 2;
        match code {
            0 if inc == 0 => break,
            59 => {
                t += ((word(k) as u32) << 16 | word(k + 2) as u32) as i32 as i64;
                k += 4;
            }
            60..=62 => {}
            63 => k += inc + inc % 2,
            _ => {
                t += inc as i64;
                out.push((t, code));
            }
        }
    }
    out
}

fn nn_intervals(ann: &[(i64, u8)], fs: f64) -> Vec<f64> {
    ann.windows(2)
        .filter(|p| p[0].1 == 1 && p[1].1 == 1)
        .map(|p| (p[1].0 - p[0].0) as f64 / fs)
        .filter(|rr| (0.3..=2.0).contains(rr))
        .collect()
}

fn header_field(text: &str, key: &str) -> Option<String> {
    let rest = &text[text.find(key)? + key.len()..];
    rest.split_whitespace().next().map(str::to_string)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn seg_median(x: &[f64], size: usize, f: impl Fn(&[f64]) -> Option<f64>) -> f64 {
    let mut v: Vec<f64> = x
        .chunks_exact(size)
        .filter_map(&f)
        .filter(|a| a.is_finite())
        .collect();
    median(&mut v)
}

fn short_alpha(s: &[f64]) -> Option<f64> {
    struktura::dfa_short(s).map(|r| r.alpha)
}

/// Control: alpha1 after shuffling the beats inside each 64-beat segment, which keeps
/// the NN distribution and destroys the beat-to-beat correlation.
fn shuffled_alpha1(rr: &[f64], rng: &mut Rng) -> f64 {
    let mut shuf = rr.to_vec();
    for seg in shuf.chunks_exact_mut(64) {
        for i in (1..seg.len()).rev() {
            seg.swap(i, rng.below(i + 1));
        }
    }
    seg_median(&shuf, 64, short_alpha)
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

fn record(dir: &str, db: &str, rec: &str) -> Record {
    let hea = std::fs::read_to_string(format!("{dir}/{db}/{rec}.hea")).expect("read header");
    let fs: f64 = hea
        .split_whitespace()
        .nth(2)
        .and_then(|f| f.split(['/', '(']).next()?.parse().ok())
        .expect("fs");
    let rr = nn_intervals(&read_annotations(&format!("{dir}/{db}/{rec}.ecg")), fs);
    let n = rr.len() as f64;
    let mean_nn = rr.iter().sum::<f64>() / n;
    let d2: f64 = rr.windows(2).map(|p| (p[1] - p[0]).powi(2)).sum();
    Record {
        chf: db == "chf2db",
        age: header_field(&hea, "Age:")
            .and_then(|a| a.parse().ok())
            .unwrap_or(f64::NAN),
        nyha: header_field(&hea, "NYHA class:").unwrap_or_default(),
        alpha1: seg_median(&rr, 64, short_alpha),
        alpha2: seg_median(&rr, 2048, |s| Some(struktura::dfa(s).alpha)),
        mean_nn,
        sdnn: (rr.iter().map(|v| (v - mean_nn).powi(2)).sum::<f64>() / n).sqrt(),
        rmssd: (d2 / (n - 1.0)).sqrt(),
        rr,
    }
}

/// AUC-ROC via the rank-sum statistic (ties get average rank); higher score = CHF.
fn auc(score: &[f64], chf: &[bool]) -> f64 {
    let mut v: Vec<(f64, bool)> = score.iter().copied().zip(chf.iter().copied()).collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (mut rank_sum, mut i) = (0.0, 0);
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1].0 == v[i].0 {
            j += 1;
        }
        rank_sum += v[i..=j].iter().filter(|x| x.1).count() as f64 * ((i + j) as f64 / 2.0 + 1.0);
        i = j + 1;
    }
    let pos = v.iter().filter(|x| x.1).count() as f64;
    (rank_sum - pos * (pos + 1.0) / 2.0) / (pos * (v.len() as f64 - pos))
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// numpy's default (linear) percentile of an unsorted sample.
fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = p / 100.0 * (v.len() - 1) as f64;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}

/// Bootstrap resamples (with replacement) that contain both classes.
fn boot_indices(chf: &[bool], n: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    while out.len() < n {
        let b: Vec<usize> = (0..chf.len()).map(|_| rng.below(chf.len())).collect();
        if b.iter().any(|&i| chf[i]) && b.iter().any(|&i| !chf[i]) {
            out.push(b);
        }
    }
    out
}

/// L2-penalised logistic regression, penalty 0.5 |w|^2 on weights only, loss weight C = 1
/// (scikit-learn's LogisticRegression defaults), solved by Newton's method.
fn logistic_fit(x: &[Vec<f64>], y: &[bool]) -> Vec<f64> {
    let d = x[0].len() + 1; // weights, then intercept
    let mut beta = vec![0.0; d];
    for _ in 0..100 {
        let mut g: Vec<f64> = (0..d)
            .map(|j| if j + 1 < d { beta[j] } else { 0.0 })
            .collect();
        let mut h: Vec<Vec<f64>> = (0..d)
            .map(|j| {
                (0..d)
                    .map(|k| if j == k && j + 1 < d { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        for (xi, &yi) in x.iter().zip(y) {
            let xa: Vec<f64> = xi.iter().copied().chain([1.0]).collect();
            let p = 1.0 / (1.0 + (-xa.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).exp());
            for j in 0..d {
                g[j] += (p - yi as u8 as f64) * xa[j];
                for k in 0..d {
                    h[j][k] += p * (1.0 - p) * xa[j] * xa[k];
                }
            }
        }
        // Solve h * step = g (Gaussian elimination, partial pivoting).
        for c in 0..d {
            let piv = (c..d)
                .max_by(|&a, &b| h[a][c].abs().partial_cmp(&h[b][c].abs()).unwrap())
                .unwrap();
            h.swap(c, piv);
            g.swap(c, piv);
            let pivot = h[c].clone();
            for r in c + 1..d {
                let f = h[r][c] / pivot[c];
                h[r][c..]
                    .iter_mut()
                    .zip(&pivot[c..])
                    .for_each(|(a, b)| *a -= f * b);
                g[r] -= f * g[c];
            }
        }
        let mut step = vec![0.0; d];
        for c in (0..d).rev() {
            step[c] = (g[c] - (c + 1..d).map(|k| h[c][k] * step[k]).sum::<f64>()) / h[c][c];
        }
        beta.iter_mut().zip(&step).for_each(|(b, s)| *b -= s);
        if step.iter().all(|s| s.abs() < 1e-12) {
            break;
        }
    }
    beta
}

/// Leave-one-out AUC of a standardised logistic model on the chosen features.
fn loo(recs: &[&Record], feats: &[Feature]) -> f64 {
    let x: Vec<Vec<f64>> = recs
        .iter()
        .map(|r| feats.iter().map(|f| f(r)).collect())
        .collect();
    let y: Vec<bool> = recs.iter().map(|r| r.chf).collect();
    let scores: Vec<f64> = (0..recs.len())
        .map(|i| {
            let train: Vec<usize> = (0..recs.len()).filter(|&j| j != i).collect();
            let n = train.len() as f64;
            let stats: Vec<(f64, f64)> = (0..feats.len())
                .map(|c| {
                    let m = train.iter().map(|&j| x[j][c]).sum::<f64>() / n;
                    (
                        m,
                        (train.iter().map(|&j| (x[j][c] - m).powi(2)).sum::<f64>() / n).sqrt(),
                    )
                })
                .collect();
            let z = |row: &[f64]| {
                row.iter()
                    .zip(&stats)
                    .map(|(v, (m, s))| (v - m) / s)
                    .collect::<Vec<f64>>()
            };
            let beta = logistic_fit(
                &train.iter().map(|&j| z(&x[j])).collect::<Vec<_>>(),
                &train.iter().map(|&j| y[j]).collect::<Vec<_>>(),
            );
            z(&x[i])
                .iter()
                .chain([1.0].iter())
                .zip(&beta)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    auc(&scores, &y)
}

fn main() {
    let dir = std::env::var("PHYSIONET_DIR")
        .expect("set PHYSIONET_DIR to the folder holding nsr2db and chf2db");
    let mut recs = Vec::new();
    for db in ["nsr2db", "chf2db"] {
        let list = std::fs::read_to_string(format!("{dir}/{db}/RECORDS")).expect("read RECORDS");
        for rec in list.split_whitespace() {
            recs.push(record(&dir, db, rec));
        }
    }
    let all: Vec<&Record> = recs.iter().collect();
    let chf: Vec<bool> = recs.iter().map(|r| r.chf).collect();
    let col = |f: fn(&Record) -> f64| recs.iter().map(f).collect::<Vec<f64>>();
    let group_median = |f: fn(&Record) -> f64, c: bool| {
        median(
            &mut recs
                .iter()
                .filter(|r| r.chf == c)
                .map(f)
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>(),
        )
    };
    let neg = |v: Vec<f64>| v.into_iter().map(|x| -x).collect::<Vec<f64>>();
    println!(
        "records: healthy {}, CHF {}",
        chf.iter().filter(|c| !**c).count(),
        chf.iter().filter(|c| **c).count()
    );

    let rows: [(&str, Feature); 5] = [
        ("alpha1 (dfa_short, 64-beat segments)", |r| r.alpha1),
        ("alpha2 (dfa, 2048-beat segments)", |r| r.alpha2),
        ("SDNN (s)", |r| r.sdnn),
        ("mean NN (s)", |r| r.mean_nn),
        ("RMSSD (s)", |r| r.rmssd),
    ];
    println!();
    println!("| measure | healthy median | CHF median | AUC, lower = CHF |");
    println!("|---|---|---|---|");
    for (name, f) in rows {
        println!(
            "| {name} | {:.3} | {:.3} | {:.3} |",
            group_median(f, false),
            group_median(f, true),
            auc(&neg(col(f)), &chf)
        );
    }

    let (mut ctrl, mut ctrl_med) = (Vec::new(), [Vec::new(), Vec::new()]);
    for seed in 1..=20u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let a: Vec<f64> = recs
            .iter()
            .map(|r| shuffled_alpha1(&r.rr, &mut rng))
            .collect();
        for (g, c) in [(0, false), (1, true)] {
            ctrl_med[g].push(median(
                &mut a
                    .iter()
                    .zip(&chf)
                    .filter(|p| *p.1 == c)
                    .map(|p| *p.0)
                    .collect::<Vec<_>>(),
            ));
        }
        ctrl.push(auc(&neg(a), &chf));
    }
    let (clo, chi) = (
        ctrl.iter().cloned().fold(f64::MAX, f64::min),
        ctrl.iter().cloned().fold(f64::MIN, f64::max),
    );
    println!(
        "| control: alpha1, beats shuffled in each segment (median of 20 shuffles; AUC range {clo:.3}-{chi:.3}) | {:.3} | {:.3} | {:.3} |",
        median(&mut ctrl_med[0]),
        median(&mut ctrl_med[1]),
        median(&mut ctrl)
    );

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let boots = boot_indices(&chf, 2000, &mut rng);
    let (a1, sd) = (neg(col(|r| r.alpha1)), neg(col(|r| r.sdnn)));
    let pick = |v: &[f64], b: &[usize]| b.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let (mut ci, mut diff): (Vec<f64>, Vec<f64>) = boots
        .iter()
        .map(|b| {
            let yb: Vec<bool> = b.iter().map(|&i| chf[i]).collect();
            let a = auc(&pick(&a1, b), &yb);
            (a, auc(&pick(&sd, b), &yb) - a)
        })
        .unzip();
    println!();
    println!(
        "alpha1 AUC 95% bootstrap CI [{:.3}, {:.3}] (2000 resamples)",
        percentile(&mut ci, 2.5),
        percentile(&mut ci, 97.5)
    );
    println!(
        "AUC(SDNN) - AUC(alpha1), paired bootstrap 95% CI [{:.3}, {:.3}]",
        percentile(&mut diff, 2.5),
        percentile(&mut diff, 97.5)
    );
    println!(
        "leave-one-out logistic AUC: [mean NN, SDNN] {:.3}; [mean NN, SDNN, alpha1] {:.3}",
        loo(&all, &[|r| r.mean_nn, |r| r.sdnn]),
        loo(&all, &[|r| r.mean_nn, |r| r.sdnn, |r| r.alpha1])
    );

    println!();
    println!("exploratory: age (from the record headers)");
    let known: Vec<&Record> = recs.iter().filter(|r| r.age.is_finite()).collect();
    for (c, name) in [(false, "healthy"), (true, "CHF")] {
        let g: Vec<&&Record> = known.iter().filter(|r| r.chf == c).collect();
        let ages: Vec<f64> = g.iter().map(|r| r.age).collect();
        println!(
            "  {name}: n {}, age median {:.0}, range {:.0}-{:.0}, corr(alpha1, age) {:.2}",
            g.len(),
            median(&mut ages.clone()),
            ages.iter().cloned().fold(f64::MAX, f64::min),
            ages.iter().cloned().fold(f64::MIN, f64::max),
            corr(&g.iter().map(|r| r.alpha1).collect::<Vec<_>>(), &ages)
        );
    }
    let mut nyha: Vec<&str> = recs
        .iter()
        .filter(|r| r.chf)
        .map(|r| r.nyha.as_str())
        .collect();
    nyha.sort();
    nyha.dedup();
    let counts: Vec<String> = nyha
        .iter()
        .map(|k| {
            format!(
                "{k}: {}",
                recs.iter().filter(|r| r.chf && r.nyha == *k).count()
            )
        })
        .collect();
    println!("  NYHA class in CHF: {}", counts.join(", "));
    let kchf: Vec<bool> = known.iter().map(|r| r.chf).collect();
    println!(
        "  AUC of age alone, higher = CHF: {:.3}",
        auc(&known.iter().map(|r| r.age).collect::<Vec<_>>(), &kchf)
    );
    println!(
        "  leave-one-out logistic AUC: [age] {:.3}; [age, alpha1] {:.3}",
        loo(&known, &[|r| r.age]),
        loo(&known, &[|r| r.age, |r| r.alpha1])
    );
    let bound = |c: bool, f: fn(f64, f64) -> f64, init: f64| {
        known
            .iter()
            .filter(|r| r.chf == c)
            .map(|r| r.age)
            .fold(init, f)
    };
    let lo = bound(false, f64::min, f64::MAX).max(bound(true, f64::min, f64::MAX));
    let hi = bound(false, f64::max, f64::MIN).min(bound(true, f64::max, f64::MIN));
    let band: Vec<&&Record> = known
        .iter()
        .filter(|r| r.age >= lo && r.age <= hi)
        .collect();
    let bchf: Vec<bool> = band.iter().map(|r| r.chf).collect();
    println!(
        "  overlapping age band {lo:.0}-{hi:.0}: healthy {}, CHF {}; alpha1 AUC {:.3}",
        bchf.iter().filter(|c| !**c).count(),
        bchf.iter().filter(|c| **c).count(),
        auc(&band.iter().map(|r| -r.alpha1).collect::<Vec<_>>(), &bchf)
    );

    println!();
    println!("exploratory: heart rate (box sizes are counted in beats)");
    println!(
        "  corr(alpha1, mean NN) over all records {:.2}",
        corr(&col(|r| r.alpha1), &col(|r| r.mean_nn))
    );
    println!(
        "  leave-one-out logistic AUC: [mean NN] {:.3}; [mean NN, alpha1] {:.3}",
        loo(&all, &[|r| r.mean_nn]),
        loo(&all, &[|r| r.mean_nn, |r| r.alpha1])
    );
}
