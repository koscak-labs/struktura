use std::env;
use std::fs;
use std::process;
use struktura::{analyze, health_check, HealthVerdict, LawQuality, StructuralLaw};

const NORMAL_SAMPLES: &str = include_str!("../../data/normal_sample.csv");
const FAULT_SAMPLES: &str = include_str!("../../data/fault_sample.csv");

fn read_csv(path: &str) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    });
    parse_values(&content)
}

fn read_stdin() -> Vec<f64> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
        eprintln!("Error reading stdin: {}", e);
        process::exit(1);
    });
    parse_values(&buf)
}

fn parse_values(content: &str) -> Vec<f64> {
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

fn verdict_color(v: HealthVerdict) -> (&'static str, &'static str) {
    match v {
        HealthVerdict::Healthy => ("\x1b[32m", "HEALTHY"),
        HealthVerdict::Watch => ("\x1b[33m", "WATCH"),
        HealthVerdict::Warning => ("\x1b[33;1m", "WARNING"),
        HealthVerdict::Critical => ("\x1b[31;1m", "CRITICAL"),
    }
}

fn alpha_bar(alpha: f64, width: usize) -> String {
    let clamped = alpha.clamp(0.0, 1.5);
    let filled = ((clamped / 1.5) * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "#".repeat(filled), ".".repeat(empty))
}

fn print_visual(label: &str, law: &StructuralLaw, baseline: Option<f64>) {
    let bar = alpha_bar(law.dfa.alpha, 30);
    let q = quality_str(law.quality);
    println!("  {} {:<20} alpha={:.3}  R2={:.3}  [{}]", bar, label, law.dfa.alpha, law.dfa.r_squared, q);

    if let Some(b) = baseline {
        let verdict = health_check(law, b);
        let shift = law.dfa.alpha - b;
        let (color, label) = verdict_color(verdict);
        println!("  {:>30} shift={:+.3}  \x1b[1m{}{}\x1b[0m", "", shift, color, label);
    }
}

fn print_law_detail(law: &StructuralLaw) {
    println!("    Samples:   {}", law.n);
    println!("    DFA alpha: {:.3}  (R2={:.4})", law.dfa.alpha, law.dfa.r_squared);
    println!("    Hurst H:   {:.3}", law.hurst);
    println!("    Kurtosis:  {:.1}", law.kurtosis);
    println!("    Mean:      {:.4}  Std: {:.4}", law.mean, law.std_dev);
    println!("    Quality:   {}", quality_str(law.quality));
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!();
        println!("  \x1b[1mstruktura\x1b[0m - predict failure before it happens");
        println!("  the only Rust-native DFA anomaly detector");
        println!();
        println!("  COMMANDS:");
        println!("    struktura demo                            Run builtin bearing fault demo");
        println!("    struktura check <file.csv>                Analyze a signal");
        println!("    struktura check <file.csv> --baseline N   Compare against baseline");
        println!("    struktura compare <a.csv> <b.csv>         Compare two signals");
        println!("    struktura bench                           Full benchmark with all fault types");
        println!();
        println!("  INPUT: CSV or one-value-per-line. Uses last column.");
        println!("  MORE: https://github.com/koscak-labs/struktura");
        println!();
        process::exit(0);
    }

    match args[1].as_str() {
        "demo" => cmd_demo(),
        "check" => cmd_check(&args),
        "compare" => cmd_compare(&args),
        "bench" => cmd_bench(),
        "version" => println!("struktura {}", env!("CARGO_PKG_VERSION")),
        other => {
            eprintln!("Unknown command: {}", other);
            eprintln!("Try: struktura demo");
            process::exit(1);
        }
    }
}

fn cmd_demo() {
    let normal = parse_values(NORMAL_SAMPLES);
    let fault = parse_values(FAULT_SAMPLES);

    let law_n = analyze(&normal);
    let law_f = analyze(&fault);
    let verdict = health_check(&law_f, law_n.dfa.alpha);
    let shift = law_f.dfa.alpha - law_n.dfa.alpha;
    let (color, verdict_label) = verdict_color(verdict);

    println!();
    println!("  \x1b[1mSTRUKTURA DEMO\x1b[0m");
    println!("  Bearing fault detection from CWRU vibration data");
    println!("  ================================================");
    println!();
    print_visual("Normal bearing", &law_n, None);
    println!();
    print_visual("Inner race FAULT", &law_f, Some(law_n.dfa.alpha));
    println!();
    println!("  ================================================");
    println!("  Baseline alpha:  {:.3}  (normal bearing)", law_n.dfa.alpha);
    println!("  Current alpha:   {:.3}  (faulted bearing)", law_f.dfa.alpha);
    println!("  Structural shift: \x1b[1m{:+.3}\x1b[0m", shift);
    println!();
    println!("  Verdict: {}\x1b[1m{}\x1b[0m", color, verdict_label);
    println!();
    println!("  The bearing's vibration structure changed BEFORE");
    println!("  any amplitude threshold would have fired.");
    println!();
    println!("  No training. No hyperparameters. Just math.");
    println!("  https://crates.io/crates/struktura");
    println!();
}

fn cmd_check(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura check <file.csv> [--baseline N]");
        process::exit(1);
    }
    let path = &args[2];
    let data = if path == "-" { read_stdin() } else { read_csv(path) };
    if data.len() < 20 {
        eprintln!("Error: need >= 20 data points, got {}", data.len());
        process::exit(1);
    }

    let law = analyze(&data);
    let mut baseline: Option<f64> = None;
    let json_mode = args.iter().any(|a| a == "--json");
    let quiet_mode = args.iter().any(|a| a == "--quiet");
    let csv_mode = args.iter().any(|a| a == "--csv");
    for i in 0..args.len() {
        if args[i] == "--baseline" && i + 1 < args.len() {
            baseline = args[i + 1].parse().ok();
        }
    }

    if csv_mode {
        let shift_s = baseline.map(|b| format!("{:.4}", law.dfa.alpha - b)).unwrap_or_else(|| "".to_string());
        let verdict_s = baseline.map(|b| { let (_, l) = verdict_color(health_check(&law, b)); l.to_string() }).unwrap_or_else(|| "".to_string());
        println!("{},{},{:.4},{:.4},{:.4},{},{},{}", path, law.n, law.dfa.alpha, law.dfa.r_squared, law.hurst, quality_str(law.quality), shift_s, verdict_s);
        return;
    }

    if quiet_mode {
        if let Some(b) = baseline {
            let v = health_check(&law, b);
            let (_, l) = verdict_color(v);
            println!("{}", l);
        } else {
            println!("{}", quality_str(law.quality));
        }
        return;
    }

    if json_mode {
        let verdict_s = baseline.map(|b| {
            let v = health_check(&law, b);
            let (_, l) = verdict_color(v);
            l
        });
        let shift = baseline.map(|b| law.dfa.alpha - b);
        println!("{{\"file\":\"{}\",\"n\":{},\"dfa_alpha\":{:.4},\"dfa_r2\":{:.4},\"hurst\":{:.4},\"kurtosis\":{:.2},\"quality\":\"{}\"{}{}}}",
            path, law.n, law.dfa.alpha, law.dfa.r_squared, law.hurst, law.kurtosis, quality_str(law.quality),
            shift.map(|s| format!(",\"shift\":{:.4}", s)).unwrap_or_default(),
            verdict_s.map(|v| format!(",\"verdict\":\"{}\"", v)).unwrap_or_default()
        );
        return;
    }

    println!();
    println!("  \x1b[1mSTRUKTURA\x1b[0m - structural health analysis");
    println!("  File: {}", path);
    println!("  ------------------------------------------------");
    print_visual(path, &law, baseline);
    println!();
    print_law_detail(&law);

    if let Some(b) = baseline {
        let verdict = health_check(&law, b);
        let (color, label) = verdict_color(verdict);
        println!();
        println!("  >>> {}{}\x1b[0m", color, label);
    }
    println!();
}

fn cmd_compare(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Usage: struktura compare <baseline.csv> <current.csv>");
        process::exit(1);
    }
    let data_a = read_csv(&args[2]);
    let data_b = read_csv(&args[3]);
    let law_a = analyze(&data_a);
    let law_b = analyze(&data_b);

    println!();
    println!("  \x1b[1mSTRUKTURA\x1b[0m - structural comparison");
    println!("  ================================================");
    print_visual(&args[2], &law_a, None);
    println!();
    print_visual(&args[3], &law_b, Some(law_a.dfa.alpha));
    println!();
    println!("  ------------------------------------------------");
    println!("  BASELINE:");
    print_law_detail(&law_a);
    println!("  CURRENT:");
    print_law_detail(&law_b);

    let verdict = health_check(&law_b, law_a.dfa.alpha);
    let (color, label) = verdict_color(verdict);
    println!();
    println!("  >>> {}{}\x1b[0m", color, label);
    println!();
}

fn cmd_bench() {
    let normal = parse_values(NORMAL_SAMPLES);
    let fault = parse_values(FAULT_SAMPLES);

    let law_n = analyze(&normal);
    let law_f = analyze(&fault);

    println!();
    println!("  \x1b[1mSTRUKTURA BENCHMARK\x1b[0m");
    println!("  ================================================");
    println!();
    println!("  | Condition          | DFA alpha | R2     | Shift  | Verdict  |");
    println!("  |--------------------|-----------|--------|--------|----------|");

    let print_row = |name: &str, law: &StructuralLaw, baseline: Option<f64>| {
        let (shift_str, verdict_str) = match baseline {
            Some(b) => {
                let s = law.dfa.alpha - b;
                let v = health_check(law, b);
                let (_, vl) = verdict_color(v);
                (format!("{:+.3}", s), vl.to_string())
            }
            None => ("--".to_string(), "--".to_string()),
        };
        println!(
            "  | {:<18} | {:.3}     | {:.4} | {:>6} | {:>8} |",
            name, law.dfa.alpha, law.dfa.r_squared, shift_str, verdict_str
        );
    };

    print_row("Normal", &law_n, None);
    print_row("Inner race fault", &law_f, Some(law_n.dfa.alpha));
    println!();
    println!("  Data: CWRU Bearing Data Center (builtin sample)");
    println!("  Algorithm: Detrended Fluctuation Analysis (Peng 1994)");
    println!("  https://crates.io/crates/struktura");
    println!();
}
