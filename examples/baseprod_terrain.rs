//! BASEPROD terrain separation — does one untrained DFA number per block
//! separate ESA's four operator-labelled terrain classes?
//!
//! Labels: the `labelled_traverses` list from spaceuma/fts-assessment
//! (preprocessing/baseprod_helpers.py, MIT, Levin Gerdes), which backs
//! "Field Assessment of Force Torque Sensors for Planetary Rover Navigation"
//! (Gerdes et al., J Intell Robot Syst 111, 122, 2025). Their SVM on 180
//! windowed statistics reaches 95.6% on these segments.
//!
//! usage:
//!   cargo run --release --example baseprod_terrain -- path/to/rover_sensors/
//! where the directory holds one sub-directory per traverse containing
//! FTS_{FL,FR,CL,CR,BL,BR}_CORRECTED.csv. Traverses that are not present
//! are skipped and reported.
//!
//! For every labelled segment: cut the samples inside [start, end] from each
//! leg, split into BLOCK-sample blocks, DFA α per block. Then per class and
//! per leg: mean ± sd of α; pairwise Cohen's d between classes; and a
//! leave-one-out nearest-centroid accuracy on the six-leg α vector, which is
//! the zero-training baseline to set against the paper's trained classifiers.

use std::{collections::BTreeMap, fs, path::Path};
use struktura::changepoint::BLOCK;
use struktura::dfa;

const LEGS: [&str; 6] = ["FL", "FR", "CL", "CR", "BL", "BR"];

/// (traverse, start ns, end ns, comment, class)
const LABELS: &[(&str, u64, u64, &str, &str)] = &[
    ("2023-07-20_18-12-05", 1689869949240820000, 1689869979196410000, "compressed sand", "COMPRESSED_SAND"),
    ("2023-07-20_18-12-05", 1689870081457050000, 1689870127052580000, "compressed sand", "COMPRESSED_SAND"),
    ("2023-07-21_17-34-18", 1689953867491410000, 1689953904046950000, "compressed sand", "COMPRESSED_SAND"),
    ("2023-07-21_17-34-18", 1689954054664820000, 1689954098683840000, "compressed sand", "COMPRESSED_SAND"),
    ("2023-07-20_18-12-05", 1689870278000020000, 1689870324792390000, "right side in compressed riverbed with pebbles", "PEBBLES"),
    ("2023-07-20_18-12-05", 1689870617825620000, 1689870675467330000, "shaky riverbed, mainly right", "PEBBLES"),
    ("2023-07-20_19-12-27", 1689873200913700000, 1689873283070410000, "loose soil", "LOOSE_SOIL"),
    ("2023-07-21_17-34-18", 1689953714661230000, 1689953790316480000, "loose soil", "LOOSE_SOIL"),
    ("2023-07-21_12-38-15", 1689936306126180000, 1689936337894780000, "rock with turn", "ROCK"),
    ("2023-07-21_12-58-11", 1689937232387100000, 1689937258936820000, "rock straight", "ROCK"),
    ("2023-07-21_14-08-29", 1689943035867130000, 1689943073473720000, "rock straight", "ROCK"),
    ("2023-07-21_12-58-11", 1689937474564900000, 1689937482721210000, "rock uphill", "ROCK"),
    ("2023-07-21_12-58-11", 1689937701162690000, 1689937755383280000, "pebbles downhill", "PEBBLES"),
    ("2023-07-21_12-58-11", 1689939448613310000, 1689939477579240000, "rock uphill. diagonal.", "ROCK"),
    ("2023-07-21_14-08-29", 1689942160028760000, 1689942184442090000, "rock uphill. diagonal.", "ROCK"),
];

/// Rows of (timestamp ns, Force_Z) for one leg file.
fn load_leg(path: &Path) -> Vec<(u64, f64)> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    content.lines().skip(1).filter_map(|l| {
        let mut it = l.split(',');
        let ts = it.next()?.trim().parse::<u64>().ok()?;
        let fz = it.nth(2)?.trim().parse::<f64>().ok()?; // Force_Z is column 3
        Some((ts, fz))
    }).collect()
}

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = if xs.len() > 1 { xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, v.sqrt())
}

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: baseprod_terrain <dir with one sub-dir per traverse>");
        std::process::exit(1);
    });
    let root = Path::new(&root);

    // class -> leg -> block alphas ; and per-block 6-leg vectors for LOO
    let mut per_class_leg: BTreeMap<&str, BTreeMap<&str, Vec<f64>>> = BTreeMap::new();
    let mut vectors: Vec<(&str, [f64; 6])> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();
    let mut leg_cache: BTreeMap<(String, &str), Vec<(u64, f64)>> = BTreeMap::new();

    println!();
    println!("  BASEPROD terrain separation — block-α (block = {} samples ≈ 10 s @ 99 Hz)", BLOCK);
    println!("  ====================================================================");
    for (trav, t0, t1, comment, class) in LABELS {
        let dir = root.join(trav);
        if !dir.exists() {
            if !missing.contains(trav) { missing.push(trav); }
            continue;
        }
        // Per leg: samples in window -> blocks -> alphas. Keep block index alignment
        // across legs so a 6-leg vector can be formed per block.
        let mut leg_alphas: Vec<Vec<f64>> = Vec::new();
        for leg in LEGS {
            let key = (trav.to_string(), leg);
            let rows = leg_cache.entry(key).or_insert_with(|| load_leg(&dir.join(format!("FTS_{}_CORRECTED.csv", leg))));
            let seg: Vec<f64> = rows.iter().filter(|(ts, _)| ts >= t0 && ts <= t1).map(|(_, fz)| *fz).collect();
            let alphas: Vec<f64> = seg.chunks_exact(BLOCK).map(dfa).filter(|r| r.r_squared > 0.3).map(|r| r.alpha).collect();
            per_class_leg.entry(class).or_default().entry(leg).or_default().extend(&alphas);
            leg_alphas.push(alphas);
        }
        let nblocks = leg_alphas.iter().map(|v| v.len()).min().unwrap_or(0);
        for b in 0..nblocks {
            let mut v = [0.0; 6];
            for (i, la) in leg_alphas.iter().enumerate() { v[i] = la[b]; }
            vectors.push((class, v));
        }
        let secs = (t1 - t0) as f64 / 1e9;
        println!("  {:19} {:15} {:5.0} s  {:2} blocks  {}", trav, class, secs, nblocks, comment);
    }
    if !missing.is_empty() {
        println!();
        println!("  not present (skipped): {}", missing.join(", "));
    }

    // Per class / per leg table
    println!();
    println!("  mean ± sd of block-α per class and leg (n blocks):");
    print!("  {:16}", "class");
    for leg in LEGS { print!("  {:>14}", leg); }
    println!("  {:>14}", "all legs");
    for (class, legs) in &per_class_leg {
        print!("  {:16}", class);
        let mut all: Vec<f64> = Vec::new();
        for leg in LEGS {
            let xs = legs.get(leg).map(|v| v.as_slice()).unwrap_or(&[]);
            all.extend_from_slice(xs);
            if xs.is_empty() { print!("  {:>14}", "-"); } else {
                let (m, s) = mean_sd(xs); print!("  {:5.3}±{:5.3}({:2})", m, s, xs.len());
            }
        }
        if all.is_empty() { println!("  {:>14}", "-"); } else { let (m, s) = mean_sd(&all); println!("  {:5.3}±{:5.3}({:3})", m, s, all.len()); }
    }

    // Pairwise Cohen's d on pooled (all legs) alpha
    let classes: Vec<&str> = per_class_leg.keys().copied().collect();
    println!();
    println!("  pairwise Cohen's d (pooled legs): |d| > 0.8 is a large separation");
    for i in 0..classes.len() {
        for j in (i + 1)..classes.len() {
            let a: Vec<f64> = per_class_leg[classes[i]].values().flatten().copied().collect();
            let b: Vec<f64> = per_class_leg[classes[j]].values().flatten().copied().collect();
            if a.len() < 2 || b.len() < 2 { continue; }
            let (ma, sa) = mean_sd(&a); let (mb, sb) = mean_sd(&b);
            let sp = ((sa * sa + sb * sb) / 2.0).sqrt();
            let d = if sp > 0.0 { (ma - mb) / sp } else { 0.0 };
            println!("    {:15} vs {:15}  d = {:+.2}", classes[i], classes[j], d);
        }
    }

    // Leave-one-out nearest-centroid on the 6-leg alpha vector
    if vectors.len() >= 4 {
        let mut correct = 0usize;
        let mut confusion: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for (i, (truth, v)) in vectors.iter().enumerate() {
            let mut best: Option<(&str, f64)> = None;
            for class in &classes {
                let others: Vec<&[f64; 6]> = vectors.iter().enumerate()
                    .filter(|(j, (c, _))| *j != i && c == class).map(|(_, (_, w))| w).collect();
                if others.is_empty() { continue; }
                let mut centroid = [0.0; 6];
                for w in &others { for k in 0..6 { centroid[k] += w[k]; } }
                for c in centroid.iter_mut() { *c /= others.len() as f64; }
                let dist: f64 = (0..6).map(|k| (v[k] - centroid[k]).powi(2)).sum::<f64>().sqrt();
                if best.map_or(true, |(_, bd)| dist < bd) { best = Some((class, dist)); }
            }
            if let Some((pred, _)) = best {
                if pred == *truth { correct += 1; }
                *confusion.entry((truth, pred)).or_default() += 1;
            }
        }
        println!();
        println!("  leave-one-out nearest-centroid on the 6-leg α vector: {}/{} = {:.1}%  (zero training, one feature per leg)",
            correct, vectors.len(), 100.0 * correct as f64 / vectors.len() as f64);
        println!("  confusion (truth -> predicted):");
        for ((t, p), n) in &confusion { println!("    {:15} -> {:15} {}", t, p, n); }
        println!();
        println!("  reference: Gerdes et al. 2025, SVM on 180 F/T statistics, 1 s windows: 95.6%;");
        println!("  IMU-only SVM: 85.8%. Not the same windows or splits; a baseline, not a head-to-head.");
    }
    println!();
}
