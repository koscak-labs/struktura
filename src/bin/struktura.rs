use std::env;
use std::fs;
use std::process;
use struktura::{analyze, health_check, HealthVerdict, LawQuality, StructuralLaw};

fn read_csv(path: &str) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    });
    content
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let s = if let Some(pos) = l.rfind(',') {
                &l[pos + 1..]
            } else {
                l
            };
            s.trim().parse::<f64>().ok()
        })
        .collect()
}

fn quality_str(q: LawQuality) -> &'static str {
    match q {
        LawQuality::Exact => "EXACT",
        LawQuality::Strong => "STRONG",
        LawQuality::Good => "GOOD",
        LawQuality::Approx => "APPROX",
        LawQuality::Abstain => "ABSTAIN",
        LawQuality::Insufficient => "INSUFFICIENT",
    }
}

fn verdict_str(v: HealthVerdict) -> &'static str {
    match v {
        HealthVerdict::Healthy => "HEALTHY",
        HealthVerdict::Watch => "WATCH",
        HealthVerdict::Warning => "WARNING",
        HealthVerdict::Critical => "CRITICAL",
    }
}

fn print_law(law: &StructuralLaw) {
    println!("  Samples:    {}", law.n);
    println!("  DFA alpha:  {:.3}  (R2 = {:.4})", law.dfa.alpha, law.dfa.r_squared);
    println!("  Hurst H:    {:.3}", law.hurst);
    println!("  Kurtosis:   {:.1}", law.kurtosis);
    println!("  Mean:       {:.3}  Std: {:.3}", law.mean, law.std_dev);
    println!("  Quality:    {}", quality_str(law.quality));
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("struktura - predict failure before it happens");
        println!();
        println!("USAGE:");
        println!("  struktura check <file.csv>                 Analyze a signal");
        println!("  struktura check <file.csv> --baseline 0.39 Compare against baseline");
        println!("  struktura compare <file1.csv> <file2.csv>  Compare two signals");
        println!();
        println!("INPUT: CSV or one-value-per-line. Uses last column if CSV.");
        println!();
        println!("EXAMPLES:");
        println!("  struktura check vibration.csv");
        println!("  struktura check bearing_data.csv --baseline 0.389");
        println!("  struktura compare normal.csv faulted.csv");
        process::exit(0);
    }

    match args[1].as_str() {
        "check" => {
            if args.len() < 3 {
                eprintln!("Usage: struktura check <file.csv> [--baseline N]");
                process::exit(1);
            }
            let path = &args[2];
            let data = read_csv(path);

            if data.len() < 20 {
                eprintln!("Error: need at least 20 data points, got {}", data.len());
                process::exit(1);
            }

            let law = analyze(&data);

            println!();
            println!("  STRUKTURA - structural health analysis");
            println!("  File: {}", path);
            println!("  ----------------------------------------");
            print_law(&law);

            let mut baseline: Option<f64> = None;
            for i in 0..args.len() {
                if args[i] == "--baseline" && i + 1 < args.len() {
                    baseline = args[i + 1].parse().ok();
                }
            }

            if let Some(b) = baseline {
                let verdict = health_check(&law, b);
                let shift = law.dfa.alpha - b;
                println!("  ----------------------------------------");
                println!("  Baseline:   {:.3}", b);
                println!("  Shift:      {:+.3}", shift);
                println!("  Verdict:    {}", verdict_str(verdict));

                match verdict {
                    HealthVerdict::Critical => {
                        println!();
                        println!("  >>> CRITICAL: major structural departure detected");
                    }
                    HealthVerdict::Warning => {
                        println!();
                        println!("  >>> WARNING: significant structural change");
                    }
                    _ => {}
                }
            }
            println!();
        }

        "compare" => {
            if args.len() < 4 {
                eprintln!("Usage: struktura compare <baseline.csv> <current.csv>");
                process::exit(1);
            }
            let path_a = &args[2];
            let path_b = &args[3];
            let data_a = read_csv(path_a);
            let data_b = read_csv(path_b);

            let law_a = analyze(&data_a);
            let law_b = analyze(&data_b);
            let verdict = health_check(&law_b, law_a.dfa.alpha);
            let shift = law_b.dfa.alpha - law_a.dfa.alpha;

            println!();
            println!("  STRUKTURA - structural comparison");
            println!("  ========================================");
            println!("  BASELINE: {}", path_a);
            print_law(&law_a);
            println!("  ----------------------------------------");
            println!("  CURRENT:  {}", path_b);
            print_law(&law_b);
            println!("  ========================================");
            println!("  Shift:    {:+.3}", shift);
            println!("  Verdict:  {}", verdict_str(verdict));

            match verdict {
                HealthVerdict::Critical => {
                    println!();
                    println!("  >>> CRITICAL: major structural departure detected");
                }
                HealthVerdict::Warning => {
                    println!();
                    println!("  >>> WARNING: significant structural change");
                }
                _ => {}
            }
            println!();
        }

        other => {
            eprintln!("Unknown command: {}", other);
            eprintln!("Try: struktura check <file.csv>");
            process::exit(1);
        }
    }
}
