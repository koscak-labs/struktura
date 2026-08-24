use std::env;
use std::fs;
use std::process;
use struktura::{analyze, health_check, prove_structure, HealthVerdict, LawQuality, StructuralLaw};

const NORMAL_SAMPLES: &str = include_str!("../../data/normal_sample.csv");
const FAULT_SAMPLES: &str = include_str!("../../data/fault_sample.csv");

fn read_input(path: &str) -> Vec<f64> {
    if path == "-" {
        return read_stdin();
    }
    if !std::path::Path::new(path).exists() {
        eprintln!("Error: file not found: {}", path);
        process::exit(1);
    }
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    });
    parse_values(&content)
}

fn read_csv(path: &str) -> Vec<f64> { read_input(path) }

fn read_stdin() -> Vec<f64> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
        eprintln!("Error reading stdin: {}", e);
        process::exit(1);
    });
    parse_values(&buf)
}

fn read_text_input(path: &str) -> String {
    if path == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        });
        return buf;
    }
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    })
}

fn parse_values(content: &str) -> Vec<f64> {
    parse_values_col(content, None)
}

fn parse_values_col(content: &str, column: Option<usize>) -> Vec<f64> {
    content
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let parts: Vec<&str> = l.split(',').collect();
            let s = if let Some(col) = column {
                parts.get(col).copied().unwrap_or("")
            } else if parts.len() > 1 {
                parts.last().copied().unwrap_or("")
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
        _ => "UNKNOWN",
    }
}

fn verdict_color(v: HealthVerdict) -> (&'static str, &'static str) {
    match v {
        HealthVerdict::Healthy => ("\x1b[32m", "HEALTHY"),
        HealthVerdict::Watch => ("\x1b[33m", "WATCH"),
        HealthVerdict::Warning => ("\x1b[33;1m", "WARNING"),
        HealthVerdict::Critical => ("\x1b[31;1m", "CRITICAL"),
        _ => ("", "UNKNOWN"),
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
        println!("    struktura voyager                         Voyager 1 AACS anomaly analysis");
        println!("    struktura heliopause                      Detect the edge of the solar system");
        println!("    struktura spacecraft                      Multi-channel spacecraft health demo");
        println!("    struktura scan <file_or_-> [--baseline f]  Auto-analyze any signal (pipe with -)");
        println!("    struktura mf <file_or_->                 Multifractal spectrum (multi-scale complexity)");
        println!("    struktura text <file.txt>                  Writing rhythm analysis");
        println!("    struktura market <prices.csv>              Financial regime detection");
        println!("    struktura rhythm <timestamps.csv>          Event rhythm (git/heartbeat/keystrokes)");
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
        "report" => cmd_report(&args),
        "validate" => cmd_validate(&args),
        "self-test" => cmd_self_test(),
        "codegen" => cmd_codegen(&args),
        "generate" => cmd_generate(&args),
        "voyager" => cmd_voyager(),
        "heliopause" => cmd_heliopause(),
        "spacecraft" => cmd_spacecraft(),
        "text" => cmd_text(&args),
        "market" => cmd_market(&args),
        "rhythm" => cmd_rhythm(&args),
        "scan" => cmd_scan(&args),
        "classify" | "what" => cmd_classify(&args),
        "multifractal" | "mf" => cmd_multifractal(&args),
        "health" => cmd_health(&args),
        "batch" => cmd_batch(&args),
        "fingerprint" | "fp" | "dna" => cmd_fingerprint(&args),
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
        eprintln!("Usage: struktura check <file.csv> [file2.csv ...] [--baseline N]");
        process::exit(1);
    }

    let files: Vec<&str> = args[2..].iter()
        .map(|s| s.as_str())
        .take_while(|s| !s.starts_with("--"))
        .collect();

    if files.len() > 1 {
        cmd_check_multi(&files, args);
        return;
    }

    let path = &args[2];
    let data = if path == "-" { read_stdin() } else { read_csv(path) };
    if data.len() < 20 {
        eprintln!("Error: need >= 20 data points, got {}", data.len());
        process::exit(1);
    }

    let law = analyze(&data);
    let mut baseline: Option<f64> = None;
    let mut threshold: Option<f64> = None;
    let json_mode = args.iter().any(|a| a == "--json");
    let quiet_mode = args.iter().any(|a| a == "--quiet");
    let csv_mode = args.iter().any(|a| a == "--csv");
    for i in 0..args.len() {
        if args[i] == "--baseline" && i + 1 < args.len() {
            baseline = args[i + 1].parse().ok();
        }
        if args[i] == "--threshold" && i + 1 < args.len() {
            threshold = args[i + 1].parse().ok();
        }
    }
    let _ = threshold;

    if csv_mode {
        let shift_s = baseline.map(|b| format!("{:.4}", law.dfa.alpha - b)).unwrap_or_default();
        let verdict_s = baseline.map(|b| { let (_, l) = verdict_color(health_check(&law, b)); l.to_string() }).unwrap_or_default();
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

    if args.iter().any(|a| a == "--shuffle") {
        let proof = prove_structure(&data);
        println!("  ------------------------------------------------");
        println!("  \x1b[1mShuffle proof:\x1b[0m");
        println!("    Real alpha:     {:.3}  (R2={:.4})", proof.real_alpha, proof.real_r2);
        println!("    Shuffled alpha: {:.3}  (R2={:.4})", proof.shuffled_alpha, proof.shuffled_r2);
        if proof.structure_confirmed {
            println!("    Result: \x1b[32mCONFIRMED\x1b[0m — shuffling destroyed the structure");
        } else {
            println!("    Result: \x1b[33mINCONCLUSIVE\x1b[0m — signal may lack exploitable structure");
        }
    }

    println!();
}

fn cmd_check_multi(files: &[&str], args: &[String]) {
    let mut baseline: Option<f64> = None;
    let mut threshold: Option<f64> = None;
    for i in 0..args.len() {
        if args[i] == "--baseline" && i + 1 < args.len() {
            baseline = args[i + 1].parse().ok();
        }
        if args[i] == "--threshold" && i + 1 < args.len() {
            threshold = args[i + 1].parse().ok();
        }
    }
    let _ = threshold;

    println!();
    println!("  \x1b[1mSTRUKTURA\x1b[0m — batch analysis ({} files)", files.len());
    println!("  ================================================");
    println!();
    println!("  | {:<30} | {:>6} | {:>7} | {:>6} | {:>8} | {:>8} |", "File", "N", "Alpha", "R2", "Quality", "Verdict");
    println!("  |{:-<32}|{:-<8}|{:-<9}|{:-<8}|{:-<10}|{:-<10}|", "", "", "", "", "", "");

    for &path in files {
        let data = read_csv(path);
        if data.len() < 20 {
            println!("  | {:<30} | {:>6} | {:>7} | {:>6} | {:>8} | {:>8} |",
                path, data.len(), "--", "--", "TOO FEW", "--");
            continue;
        }
        let law = analyze(&data);
        let verdict_s = baseline.map(|b| {
            let (_, l) = verdict_color(health_check(&law, b));
            l.to_string()
        }).unwrap_or_else(|| "--".to_string());
        let short_path: &str = path.rsplit('/').next().unwrap_or(path);
        let short_path: &str = short_path.rsplit('\\').next().unwrap_or(short_path);
        println!("  | {:<30} | {:>6} | {:>7.3} | {:>6.4} | {:>8} | {:>8} |",
            short_path, law.n, law.dfa.alpha, law.dfa.r_squared, quality_str(law.quality), verdict_s);
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

fn cmd_report(args: &[String]) {
    let files: Vec<&str> = args[2..].iter()
        .map(|s| s.as_str())
        .take_while(|s| !s.starts_with("--"))
        .collect();
    if files.is_empty() {
        eprintln!("Usage: struktura report <file1.csv> [file2.csv ...]");
        process::exit(1);
    }
    let mut baseline: Option<f64> = None;
    for i in 0..args.len() {
        if args[i] == "--baseline" && i + 1 < args.len() {
            baseline = args[i + 1].parse().ok();
        }
    }
    println!("# Struktura Report\n");
    println!("Generated by struktura v{}\n", env!("CARGO_PKG_VERSION"));
    println!("| File | N | DFA alpha | R2 | Quality | Verdict |");
    println!("|------|---|-----------|-----|---------|---------|");
    for &path in &files {
        let data = read_csv(path);
        if data.len() < 20 { println!("| {} | {} | -- | -- | TOO FEW | -- |", path, data.len()); continue; }
        let law = analyze(&data);
        let vs = baseline.map(|b| { let (_, l) = verdict_color(health_check(&law, b)); l.to_string() }).unwrap_or_else(|| "--".to_string());
        let short: &str = path.rsplit('/').next().unwrap_or(path);
        let short: &str = short.rsplit('\\').next().unwrap_or(short);
        println!("| {} | {} | {:.3} | {:.4} | {} | {} |", short, law.n, law.dfa.alpha, law.dfa.r_squared, quality_str(law.quality), vs);
    }
    println!("\nAlgorithm: Detrended Fluctuation Analysis (Peng et al., 1994)");
}

fn cmd_validate(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura validate <file.csv>");
        process::exit(1);
    }
    let path = &args[2];
    if !std::path::Path::new(path).exists() {
        println!("FAIL: file not found: {}", path);
        process::exit(1);
    }
    let data = read_csv(path);
    if data.is_empty() {
        println!("FAIL: no numeric values found in {}", path);
        process::exit(1);
    }
    if data.len() < 20 {
        println!("WARN: only {} values (need >= 20 for analysis, >= 64 for DFA)", data.len());
    } else if data.len() < 64 {
        println!("WARN: {} values (DFA needs >= 64, ACR works with >= 20)", data.len());
    } else {
        println!("OK: {} numeric values, ready for analysis", data.len());
    }
}

fn cmd_self_test() {
    println!("  STRUKTURA SELF-TEST v{}", env!("CARGO_PKG_VERSION"));
    println!("  ========================================");
    let mut pass = 0;
    let mut fail = 0;

    // Test 1: Demo data detects fault
    let normal = parse_values(NORMAL_SAMPLES);
    let fault = parse_values(FAULT_SAMPLES);
    let law_n = analyze(&normal);
    let law_f = analyze(&fault);
    let verdict = health_check(&law_f, law_n.dfa.alpha);
    if verdict == HealthVerdict::Critical {
        println!("  [PASS] Demo data: fault detected as CRITICAL");
        pass += 1;
    } else {
        println!("  [FAIL] Demo data: expected CRITICAL, got {:?}", verdict);
        fail += 1;
    }

    // Test 2: R2 > 0.9 on both signals
    if law_n.dfa.r_squared > 0.9 && law_f.dfa.r_squared > 0.9 {
        println!("  [PASS] Both signals R2 > 0.9");
        pass += 1;
    } else {
        println!("  [FAIL] R2 too low: normal={:.3} fault={:.3}", law_n.dfa.r_squared, law_f.dfa.r_squared);
        fail += 1;
    }

    // Test 3: Shuffle proof on normal bearing
    let proof = prove_structure(&normal);
    if proof.structure_confirmed {
        println!("  [PASS] Shuffle proof: structure CONFIRMED (real={:.3} shuffled={:.3})", proof.real_alpha, proof.shuffled_alpha);
        pass += 1;
    } else {
        println!("  [FAIL] Shuffle proof: INCONCLUSIVE");
        fail += 1;
    }

    // Test 4: Constant signal returns Abstain
    let constant: Vec<f64> = vec![5.0; 100];
    let law_c = analyze(&constant);
    if law_c.quality == LawQuality::Abstain {
        println!("  [PASS] Constant signal: Abstain (no panic)");
        pass += 1;
    } else {
        println!("  [FAIL] Constant signal: expected Abstain, got {:?}", law_c.quality);
        fail += 1;
    }

    // Test 5: NaN handling
    let mut with_nan = normal.clone();
    with_nan[0] = f64::NAN;
    with_nan[10] = f64::INFINITY;
    let law_nan = analyze(&with_nan);
    if law_nan.n > 0 && law_nan.quality != LawQuality::Insufficient {
        println!("  [PASS] NaN/Inf filtered: {} samples analyzed", law_nan.n);
        pass += 1;
    } else {
        println!("  [FAIL] NaN handling failed");
        fail += 1;
    }

    println!("  ========================================");
    println!("  {}/{} passed", pass, pass + fail);
    if fail > 0 {
        process::exit(1);
    }
}

fn cmd_codegen(args: &[String]) {
    let mut window: usize = 256;
    let mut threshold: f64 = 0.08;

    for i in 0..args.len() {
        if args[i] == "--window" && i + 1 < args.len() {
            window = args[i + 1].parse().unwrap_or(256);
        }
        if args[i] == "--threshold" && i + 1 < args.len() {
            threshold = args[i + 1].parse().unwrap_or(0.08);
        }
    }

    let name = args.iter().position(|a| a == "--name")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("DfaHealthMonitor");

    if args.iter().any(|a| a == "--fprime") {
        print!("{}", struktura::codegen::generate_fprime_component(name, window));
    } else if args.iter().any(|a| a == "--cfs") {
        print!("{}", struktura::codegen::generate_cfs_app(name, window));
    } else {
        print!("{}", struktura::codegen::generate_c_monitor(window, threshold));
    }
}

fn cmd_voyager() {
    use struktura::space::voyager_demo;
    let result = voyager_demo();

    println!();
    println!("  \x1b[1mVOYAGER 1 STRUCTURAL HEALTH ANALYSIS\x1b[0m");
    println!("  DFA scaling analysis of magnetometer data across");
    println!("  the May 2022 AACS anomaly (embedded data — works after cargo install)");
    println!("  ====================================================================");
    println!();
    println!("  Data: NASA SPDF, L.F. Burlaga, VIM 48-second averages");
    println!("  https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/");
    println!();

    let bar_h = make_bar(result.healthy_alpha);
    let bar_a = make_bar(result.anomaly_alpha);

    println!("  {} 2021 (healthy)    alpha={:.3}  R²={:.4}", bar_h, result.healthy_alpha, result.healthy_r2);
    println!("  {} 2022 (anomaly)    alpha={:.3}  R²={:.4}", bar_a, result.anomaly_alpha, result.anomaly_r2);

    let v_str = match result.verdict {
        HealthVerdict::Healthy => "\x1b[32mHEALTHY\x1b[0m",
        HealthVerdict::Watch => "\x1b[33mWATCH\x1b[0m",
        HealthVerdict::Warning => "\x1b[31mWARNING\x1b[0m",
        HealthVerdict::Critical => "\x1b[31;1mCRITICAL\x1b[0m",
        _ => "UNKNOWN",
    };
    println!("  {:30} shift={:+.3}  {}", "", result.shift, v_str);

    println!();
    println!("  ====================================================================");
    println!("  Baseline alpha:  {:.3}  (2021 healthy operation)", result.healthy_alpha);
    println!("  Anomaly alpha:   {:.3}  (May-Jul 2022 AACS anomaly)", result.anomaly_alpha);
    println!();
    println!("  The magnetic field's fractal structure changed during the anomaly —");
    println!("  detectable from public NASA data, zero training, zero ML.");
    println!();
}

fn cmd_heliopause() {
    use struktura::space::heliopause_demo;
    let result = heliopause_demo();

    println!();
    println!("  \x1b[1mVOYAGER 1 HELIOPAUSE CROSSING\x1b[0m");
    println!("  DFA structural analysis across the boundary of our solar system");
    println!("  August 25, 2012 — first human-made object to enter interstellar space");
    println!("  ====================================================================");
    println!();
    println!("  Data: NASA SPDF, L.F. Burlaga, VIM 48-second magnetometer averages");
    println!("  https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/");
    println!();

    let bar_h = make_bar(result.helio_alpha);
    let bar_i = make_bar(result.interstellar_alpha);

    println!("  {} Heliosphere (days 1-200)     alpha={:.3}  R²={:.4}", bar_h, result.helio_alpha, result.helio_r2);
    println!("  {} Interstellar (days 260-331)  alpha={:.3}  R²={:.4}", bar_i, result.interstellar_alpha, result.interstellar_r2);

    let v_str = match result.verdict {
        HealthVerdict::Healthy => "\x1b[32mHEALTHY\x1b[0m",
        HealthVerdict::Watch => "\x1b[33mWATCH\x1b[0m",
        HealthVerdict::Warning => "\x1b[31mWARNING\x1b[0m",
        HealthVerdict::Critical => "\x1b[31;1mCRITICAL\x1b[0m",
        _ => "UNKNOWN",
    };
    println!("  {:30} shift={:+.3}  {}", "", result.shift, v_str);

    println!();
    println!("  ====================================================================");
    println!("  Heliosphere:  alpha={:.3}  (sun's magnetic field — strong persistence)", result.helio_alpha);
    println!("  Interstellar: alpha={:.3}  (galactic field — different structure)", result.interstellar_alpha);
    println!();
    println!("  The magnetic field's long-range correlation structure changed");
    println!("  when Voyager crossed from solar wind into interstellar medium.");
    println!("  Same 98 lines of C that detect bearing faults.");
    println!();
}

fn cmd_batch(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura batch <file1.csv> <file2.csv> ... [--baseline <file>] [--json]");
        eprintln!("  Analyze multiple signals at once. Perfect for CI/CD monitoring.");
        process::exit(1);
    }
    use struktura::classify::classify;

    let mut files = Vec::new();
    let mut baseline_file: Option<String> = None;
    let mut json_mode = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" if i + 1 < args.len() => { baseline_file = Some(args[i + 1].clone()); i += 2; }
            "--json" => { json_mode = true; i += 1; }
            f => { files.push(f.to_string()); i += 1; }
        }
    }

    let baseline = baseline_file.as_ref().map(|f| read_input(f));
    let baseline_alpha = baseline.as_ref().map(|b| analyze(b).dfa.alpha);

    if json_mode {
        print!("[");
        for (idx, f) in files.iter().enumerate() {
            let values = read_input(f);
            let law = analyze(&values);
            let cls = classify(&values);
            print!("{{\"file\":\"{}\",\"alpha\":{:.4},\"r_squared\":{:.4},\"quality\":\"{}\",\"type\":\"{}\",\"n\":{}",
                f.replace('\\', "/"), law.dfa.alpha, law.dfa.r_squared, law.quality, cls.signal_type, law.n);
            if let Some(ba) = baseline_alpha {
                let shift = law.dfa.alpha - ba;
                let verdict = health_check(&law, ba);
                print!(",\"shift\":{:.4},\"verdict\":\"{}\"", shift, verdict);
            }
            print!("}}");
            if idx < files.len() - 1 { print!(","); }
        }
        println!("]");
        return;
    }

    println!();
    println!("  \x1b[1mBATCH ANALYSIS\x1b[0m");
    println!("  ====================================================================");
    println!();
    println!("  {:<40} {:>7} {:>7} {:>7} {}", "FILE", "α", "R²", "n", "TYPE");
    println!("  {}", "─".repeat(80));

    let mut any_alert = false;
    for f in &files {
        let values = read_input(f);
        let law = analyze(&values);
        let cls = classify(&values);
        let short_name: String = f.chars().rev().take(38).collect::<String>().chars().rev().collect();

        if let Some(ba) = baseline_alpha {
            let shift = law.dfa.alpha - ba;
            let verdict = health_check(&law, ba);
            let (color, label) = verdict_color(verdict);
            if verdict != HealthVerdict::Healthy { any_alert = true; }
            println!("  {:<40} {:>7.3} {:>7.3} {:>7} {}  {}{}\x1b[0m",
                short_name, law.dfa.alpha, law.dfa.r_squared, law.n, cls.signal_type, color, label);
        } else {
            println!("  {:<40} {:>7.3} {:>7.3} {:>7} {}",
                short_name, law.dfa.alpha, law.dfa.r_squared, law.n, cls.signal_type);
        }
    }

    println!();
    if any_alert {
        println!("  \x1b[31;1m⚠ STRUCTURAL CHANGES DETECTED — see verdicts above\x1b[0m");
    } else if baseline_alpha.is_some() {
        println!("  \x1b[32m✓ All signals within baseline tolerance\x1b[0m");
    }
    println!();
}

fn cmd_fingerprint(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura dna <file1> [file2] ...  (aliases: fingerprint, fp)");
        eprintln!("  Structural DNA — the unique identity of each signal.");
        process::exit(1);
    }
    use struktura::fingerprint::fingerprint;

    println!();
    println!("  \x1b[1mSTRUCTURAL DNA\x1b[0m");
    println!("  ====================================================================");
    println!();

    let fps: Vec<_> = args[2..].iter().map(|f| {
        let values = read_input(f);
        let fp = fingerprint(&values);
        (f.clone(), fp)
    }).collect();

    for (f, fp) in &fps {
        println!("  {}", f);
        println!("    {fp}");
        println!();
    }

    if fps.len() >= 2 {
        println!("  \x1b[1mSimilarity Matrix\x1b[0m");
        for i in 0..fps.len() {
            for j in (i+1)..fps.len() {
                let d = fps[i].1.distance(&fps[j].1);
                let sim = if d < 0.05 { "\x1b[32mNEAR-IDENTICAL\x1b[0m" }
                    else if d < 0.2 { "\x1b[33mSIMILAR\x1b[0m" }
                    else { "\x1b[36mDIFFERENT\x1b[0m" };
                let short_i: String = fps[i].0.chars().rev().take(20).collect::<String>().chars().rev().collect();
                let short_j: String = fps[j].0.chars().rev().take(20).collect::<String>().chars().rev().collect();
                println!("    {} ↔ {}  distance={:.3} {sim}", short_i, short_j, d);
            }
        }
        println!();
    }
}

fn cmd_classify(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura classify <file_or_->  (alias: what)");
        eprintln!("  What kind of signal is this? White noise? Brownian? 1/f?");
        process::exit(1);
    }
    use struktura::classify::classify;

    let values = read_input(&args[2]);
    let result = classify(&values);

    println!();
    println!("  \x1b[1mSIGNAL CLASSIFICATION\x1b[0m");
    println!("  ====================================================================");
    println!();
    let bar = alpha_bar(result.alpha, 30);
    println!("  {} α={:.3}  R²={:.4}", bar, result.alpha, result.r_squared);
    println!();
    println!("  Type: \x1b[1m{}\x1b[0m", result.signal_type);
    println!("  {}", result.description);
    println!();
    println!("  ┌─────────────────────────────────────────────────────┐");
    println!("  │  α<0.3    0.5    0.6-0.85   0.85-1.15  1.15-1.65  │");
    println!("  │  anti    white   correlated   1/f      brownian   │");
    println!("  │  ─────────────────▲─────────────────────────────── │");
    let pos = ((result.alpha.clamp(0.0, 2.0) / 2.0) * 50.0) as usize;
    let indicator = format!("  │  {}▲", " ".repeat(pos.min(50)));
    println!("{}{}│", indicator, " ".repeat(53usize.saturating_sub(indicator.len() - 5)));
    println!("  └─────────────────────────────────────────────────────┘");
    println!();
}

fn cmd_health(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura health <current> --baseline <healthy>");
        eprintln!("  Full diagnostic: DFA + MFDFA + statistics + verdict.");
        process::exit(1);
    }
    use struktura::mfdfa::mfdfa;

    let mut baseline_file: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--baseline" && i + 1 < args.len() {
            baseline_file = Some(args[i + 1].clone());
            i += 2;
        } else { i += 1; }
    }

    let current = read_input(&args[2]);
    if current.len() < 64 {
        eprintln!("Need at least 64 data points, got {}", current.len());
        process::exit(1);
    }

    let law = analyze(&current);
    let qs = [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0];
    let spectrum = mfdfa(&current, &qs);
    let proof = struktura::prove_structure(&current);

    println!();
    println!("  \x1b[1mFULL HEALTH DIAGNOSTIC\x1b[0m");
    println!("  ====================================================================");
    println!();
    println!("  Signal:  {} samples", current.len());
    println!("  Mean:    {:.4}   Std: {:.4}   Kurtosis: {:.2}", law.mean, law.std_dev, law.kurtosis);
    println!();
    println!("  \x1b[1mStructural Analysis (DFA)\x1b[0m");
    let bar = make_bar(law.dfa.alpha);
    println!("  {} α={:.3}  R²={:.4}  quality={}", bar, law.dfa.alpha, law.dfa.r_squared, law.quality);
    println!("  Hurst exponent: {:.3}", law.hurst);
    println!("  Structure proof: {} (real α={:.3} vs shuffled α={:.3})",
        if proof.structure_confirmed { "\x1b[32mCONFIRMED\x1b[0m" } else { "\x1b[33mINCONCLUSIVE\x1b[0m" },
        proof.real_alpha, proof.shuffled_alpha);
    println!();
    println!("  \x1b[1mMultifractal Spectrum (MFDFA)\x1b[0m");
    println!("  h(2)={:.3}  width={:.3}  {}",
        spectrum.h2, spectrum.width,
        if spectrum.is_multifractal { "\x1b[35mMULTIFRACTAL\x1b[0m" } else { "monofractal" });
    print!("  h(q): ");
    for p in &spectrum.points { print!("{:.2} ", p.h_q); }
    println!();

    // Trend analysis — is α getting worse over time within this signal?
    if current.len() >= 512 {
        let trend = struktura::trend::alpha_trend(&current, 256, 64);
        let dir_str = match trend.direction {
            struktura::trend::TrendDirection::Improving => "\x1b[32mIMPROVING\x1b[0m",
            struktura::trend::TrendDirection::Stable => "\x1b[32mSTABLE\x1b[0m",
            struktura::trend::TrendDirection::Degrading => "\x1b[31mDEGRADING\x1b[0m",
        };
        println!();
        println!("  \x1b[1mTrend (is it getting worse?)\x1b[0m");
        println!("  α {:.3} → {:.3}  slope={:.6}/sample  {dir_str}",
            trend.alpha_start, trend.alpha_end, trend.slope);
    }

    if let Some(ref bf) = baseline_file {
        let baseline = read_input(bf);
        let result = struktura::compare(&baseline, &current);
        let baseline_spectrum = mfdfa(&baseline, &qs);
        let v_str = match result.verdict {
            HealthVerdict::Healthy => "\x1b[32mHEALTHY\x1b[0m",
            HealthVerdict::Watch => "\x1b[33mWATCH\x1b[0m",
            HealthVerdict::Warning => "\x1b[31mWARNING\x1b[0m",
            HealthVerdict::Critical => "\x1b[31;1mCRITICAL\x1b[0m",
            _ => "UNKNOWN",
        };
        println!();
        println!("  \x1b[1mvs Baseline\x1b[0m");
        println!("  DFA shift:    {:+.3}  {v_str}", result.shift);
        println!("  MFDFA width:  {:.3} → {:.3} (Δ={:+.3})",
            baseline_spectrum.width, spectrum.width, spectrum.width - baseline_spectrum.width);
    }

    println!();
    println!("  ====================================================================");
    println!();
}

fn cmd_multifractal(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura multifractal <file_or_->  (alias: mf)");
        eprintln!("  Reveals multi-scale structure — is the signal simple or complex?");
        process::exit(1);
    }
    use struktura::mfdfa::mfdfa;

    let values = read_input(&args[2]);
    if values.len() < 64 {
        eprintln!("Need at least 64 data points, got {}", values.len());
        process::exit(1);
    }

    let qs = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0];
    let spectrum = mfdfa(&values, &qs);

    println!();
    println!("  \x1b[1mMULTIFRACTAL SPECTRUM\x1b[0m");
    println!("  How does structure vary across scales?");
    println!("  ====================================================================");
    println!();
    println!("  h(2) = {:.3}  (equivalent to standard DFA alpha)", spectrum.h2);
    println!("  Spectral width = {:.3}", spectrum.width);
    println!();

    // ASCII spectrum visualization
    println!("  q       h(q)     R²");
    println!("  ─────────────────────────────────────────");
    for p in &spectrum.points {
        let bar_len = ((p.h_q * 40.0).round() as usize).min(40);
        let bar = "#".repeat(bar_len);
        let pad = ".".repeat(40 - bar_len);
        println!("  {:+5.1}   {bar}{pad}  {:.3}  {:.3}", p.q, p.h_q, p.r_squared);
    }
    println!();

    let mf_str = if spectrum.is_multifractal {
        "\x1b[35mMULTIFRACTAL\x1b[0m — different scales have different structure"
    } else {
        "\x1b[32mMONOFRACTAL\x1b[0m — uniform structure across all scales"
    };
    println!("  Verdict: {mf_str}");
    println!("  Width {:.3} (wider = more complex multi-scale behavior)", spectrum.width);
    println!();
}

fn cmd_scan(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura scan <file_or_->  [--baseline <file>] [--json]");
        eprintln!("  Auto-analyzes any signal. Pipe from stdin with -");
        eprintln!("  Examples:");
        eprintln!("    struktura scan sensor.csv");
        eprintln!("    cat data.csv | struktura scan -");
        eprintln!("    struktura scan current.csv --baseline healthy.csv");
        eprintln!("    struktura scan data.csv --json  # machine-readable output");
        process::exit(1);
    }
    let values = read_input(&args[2]);
    if values.is_empty() {
        eprintln!("No numeric values found in input");
        process::exit(1);
    }

    let mut baseline_file: Option<String> = None;
    let mut json_mode = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline" if i + 1 < args.len() => { baseline_file = Some(args[i + 1].clone()); i += 2; }
            "--json" => { json_mode = true; i += 1; }
            _ => { i += 1; }
        }
    }

    if json_mode {
        let law = analyze(&values);
        let baseline_info = baseline_file.as_ref().map(|bf| {
            let baseline = read_input(bf);
            struktura::compare(&baseline, &values)
        });
        print!("{{\"alpha\":{:.4},\"r_squared\":{:.4},\"quality\":\"{}\",\"n\":{},\"mean\":{:.4},\"std\":{:.4},\"kurtosis\":{:.2},\"hurst\":{:.3}",
            law.dfa.alpha, law.dfa.r_squared, law.quality, law.n, law.mean, law.std_dev, law.kurtosis, law.hurst);
        if let Some(ref result) = baseline_info {
            print!(",\"shift\":{:.4},\"verdict\":\"{}\"", result.shift, result.verdict);
        }
        println!("}}");
        return;
    }

    println!();
    println!("  \x1b[1mSTRUKTURA SCAN\x1b[0m");
    println!("  ====================================================================");
    println!();

    let law = analyze(&values);
    let bar = make_bar(law.dfa.alpha);
    let q = quality_str(law.quality);
    println!("  {} alpha={:.3}  R²={:.4}  [{q}]  n={}", bar, law.dfa.alpha, law.dfa.r_squared, law.n);
    println!("  {:30} mean={:.2}  std={:.2}  kurtosis={:.2}", "", law.mean, law.std_dev, law.kurtosis);

    if let Some(ref bf) = baseline_file {
        let baseline = read_input(bf);
        let result = struktura::compare(&baseline, &values);
        let v_str = match result.verdict {
            HealthVerdict::Healthy => "\x1b[32mHEALTHY\x1b[0m",
            HealthVerdict::Watch => "\x1b[33mWATCH\x1b[0m",
            HealthVerdict::Warning => "\x1b[31mWARNING\x1b[0m",
            HealthVerdict::Critical => "\x1b[31;1mCRITICAL\x1b[0m",
            _ => "UNKNOWN",
        };
        println!();
        println!("  vs baseline ({bf}):");
        println!("  {:30} shift={:+.3}  {v_str}", "", result.shift);
    }

    println!();
    println!("  ====================================================================");
    println!("  α ≈ 0.5 = random noise | 0.5-1.0 = structured | shift from baseline = degradation");
    println!();
}

fn cmd_market(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura market <prices.csv>");
        eprintln!("  CSV with one price per line (close prices). Need 200+ prices.");
        process::exit(1);
    }
    use struktura::market::{returns_from_prices, regime_detect};

    let prices = read_csv(&args[2]);
    if prices.len() < 64 {
        eprintln!("Need at least 64 prices, got {}", prices.len());
        process::exit(1);
    }
    let returns = returns_from_prices(&prices);
    let result = regime_detect(&returns);

    println!();
    println!("  \x1b[1mMARKET REGIME DETECTION\x1b[0m");
    println!("  DFA on log-returns — what strategy works NOW");
    println!("  ====================================================================");
    println!();
    let bar = make_bar(result.alpha);
    let regime_str = match result.regime {
        struktura::market::MarketRegime::Trending => "\x1b[32mTRENDING\x1b[0m (momentum works)",
        struktura::market::MarketRegime::RandomWalk => "\x1b[33mRANDOM WALK\x1b[0m (no reliable edge)",
        struktura::market::MarketRegime::MeanReverting => "\x1b[36mMEAN-REVERTING\x1b[0m (reversals work)",
    };
    println!("  {} α={:.3}  R²={:.4}  n={}", bar, result.alpha, result.r_squared, result.n_returns);
    println!("  {:30} regime: {}", "", regime_str);
    println!();
    println!("  ====================================================================");
    println!("  α > 0.6 = trending (momentum)");
    println!("  α ≈ 0.5 = random walk (efficient market)");
    println!("  α < 0.45 = mean-reverting (reversal strategies)");
    println!();
}

fn cmd_rhythm(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura rhythm <timestamps.csv>");
        eprintln!("  One unix timestamp per line. Try: git log --format=%at | struktura rhythm -");
        process::exit(1);
    }
    use struktura::rhythm::{intervals_from_timestamps, rhythm_analyze};

    let timestamps = read_csv(&args[2]);
    let intervals = intervals_from_timestamps(&timestamps);
    if intervals.len() < 64 {
        eprintln!("Need at least 64 intervals (65 timestamps), got {}", intervals.len());
        process::exit(1);
    }
    let result = rhythm_analyze(&intervals);

    println!();
    println!("  \x1b[1mEVENT RHYTHM ANALYSIS\x1b[0m");
    println!("  DFA on inter-event intervals");
    println!("  ====================================================================");
    println!();
    let bar = make_bar(result.alpha);
    let rhythm_str = match result.rhythm {
        struktura::rhythm::RhythmType::Bursty => "\x1b[35mBURSTY\x1b[0m (creative/flow state bursts)",
        struktura::rhythm::RhythmType::Natural => "\x1b[32mNATURAL\x1b[0m (human work rhythm)",
        struktura::rhythm::RhythmType::Random => "\x1b[33mRANDOM\x1b[0m (uncorrelated events)",
        struktura::rhythm::RhythmType::Metronomic => "\x1b[36mMETRONOMIC\x1b[0m (automated/scheduled)",
    };
    println!("  {} α={:.3}  R²={:.4}  n={}", bar, result.alpha, result.r_squared, result.n_intervals);
    println!("  {:30} rhythm: {}", "", rhythm_str);
    println!("  {:30} mean interval: {:.1}s", "", result.mean_interval);
    println!();
    println!("  ====================================================================");
    println!("  α > 0.7 = bursty (long correlated work sessions)");
    println!("  α 0.55-0.7 = natural human rhythm");
    println!("  α ≈ 0.5 = random/uncorrelated");
    println!("  α < 0.45 = metronomic (bot/cron/automated)");
    println!();
}

fn cmd_text(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: struktura text <file.txt> [file2.txt ...]");
        process::exit(1);
    }
    use struktura::text::text_structure;

    println!();
    println!("  \x1b[1mTEXT STRUCTURE ANALYSIS\x1b[0m");
    println!("  DFA on sentence-length sequences — measures writing rhythm");
    println!("  ====================================================================");
    println!();
    println!("  Reference: human prose α≈0.7-0.8 | shuffled/mechanical α≈0.5");
    println!();

    for path in &args[2..] {
        let text = read_text_input(path);
        let result = text_structure(&text);
        let bar = make_bar(result.dfa.alpha);
        println!("  {} {}", bar, path);
        println!("  {:30} sentences={}  mean_len={:.0}  α={:.3}  R²={:.4}",
            "", result.sentence_count, result.mean_sentence_len,
            result.dfa.alpha, result.dfa.r_squared);
        if result.sentence_count >= 64 && result.dfa.r_squared > 0.5 {
            let tag = if result.dfa.alpha > 0.7 {
                "\x1b[32mSTRONG RHYTHM\x1b[0m (human literary)"
            } else if result.dfa.alpha > 0.55 {
                "\x1b[33mMODERATE RHYTHM\x1b[0m"
            } else {
                "\x1b[36mUNIFORM/MECHANICAL\x1b[0m"
            };
            println!("  {:30} {}", "", tag);
        } else if result.sentence_count < 64 {
            println!("  {:30} \x1b[90m(need 200+ sentences for reliable DFA)\x1b[0m", "");
        }
        println!();
    }
    println!("  ====================================================================");
    println!("  α > 0.6 = long-range correlations in sentence rhythm (human writing)");
    println!("  α ≈ 0.5 = random/shuffled/uniform sentence lengths");
    println!();
}

fn cmd_spacecraft() {
    use struktura::space::spacecraft_demo;

    println!();
    println!("  \x1b[1mMULTI-CHANNEL SPACECRAFT HEALTH MONITOR\x1b[0m");
    println!("  DFA structural analysis across 4 telemetry channels");
    println!("  (RWA/BAT/THM synthetic; MAG = real Voyager 1 data)");
    println!("  ====================================================================");
    println!();

    let results = spacecraft_demo();
    for r in &results {
        let bar = make_bar(r.current_alpha);
        let v_str = match r.verdict {
            HealthVerdict::Healthy => "\x1b[32mHEALTHY\x1b[0m",
            HealthVerdict::Watch => "\x1b[33mWATCH\x1b[0m",
            HealthVerdict::Warning => "\x1b[31mWARNING\x1b[0m",
            HealthVerdict::Critical => "\x1b[31;1mCRITICAL\x1b[0m",
            _ => "UNKNOWN",
        };
        println!("  {} [{}:{}]", bar, r.subsystem, r.channel_name);
        println!("  {:30} alpha={:.3} baseline={:.3} shift={:+.3} R²={:.4}  {}",
            "", r.current_alpha, r.baseline_alpha, r.shift, r.r_squared, v_str);
        println!();
    }

    println!("  ====================================================================");
    println!("  Each channel: first half = baseline, second half = current period.");
    println!("  Structural (DFA) monitoring catches degradation threshold-based");
    println!("  monitors miss — the mean can look normal while the structure shifts.");
    println!();
}

fn make_bar(alpha: f64) -> String {
    let filled = (alpha * 30.0).round() as usize;
    let filled = if filled > 30 { 30 } else { filled };
    let empty = 30 - filled;
    format!("{}{}", "#".repeat(filled), ".".repeat(empty))
}

fn cmd_generate(args: &[String]) {
    let mut db_path: Option<String> = None;
    let mut target = "cfs";
    let mut output_dir = "dfa_monitor_app".to_string();
    let mut window: usize = 256;
    let mut threshold: f64 = 0.08;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--db" if i + 1 < args.len() => { db_path = Some(args[i + 1].clone()); i += 2; }
            "--cfs" => { target = "cfs"; i += 1; }
            "--fprime" => { target = "fprime"; i += 1; }
            "--ros" | "--ros2" => { target = "ros"; i += 1; }
            "--window" if i + 1 < args.len() => { window = args[i + 1].parse().unwrap_or(256); i += 2; }
            "--threshold" if i + 1 < args.len() => { threshold = args[i + 1].parse().unwrap_or(0.08); i += 2; }
            "-o" | "--output" if i + 1 < args.len() => { output_dir = args[i + 1].clone(); i += 2; }
            "--help" => {
                println!("struktura generate — generate complete cFS/F Prime DFA monitor apps");
                println!();
                println!("  Compatible with nasa/ogma db.json format. No Haskell required.");
                println!();
                println!("  OPTIONS:");
                println!("    --db <db.json>         Variable database (ogma-compatible)");
                println!("    --cfs                  Generate cFS application (default)");
                println!("    --fprime               Generate F Prime component");
                println!("    --ros                  Generate ROS 2 node");
                println!("    --window <N>           DFA window size (default: 256)");
                println!("    --threshold <F>        Shift threshold (default: 0.08)");
                println!("    -o, --output <dir>     Output directory (default: dfa_monitor_app)");
                println!();
                println!("  EXAMPLE:");
                println!("    struktura generate --cfs --db channels.json -o my_monitor/");
                println!();
                println!("  The db.json format is compatible with nasa/ogma's variable database.");
                println!("  See: https://github.com/nasa/ogma");
                process::exit(0);
            }
            _ => { i += 1; }
        }
    }

    let channels = if let Some(ref path) = db_path {
        parse_db_json(path)
    } else {
        vec![Channel { name: "input_value".into(), c_type: "double".into(), topic: "SAMPLE_MID".into(), field: "payload".into(), msg_type: "sample_msg_t".into() }]
    };

    let out = std::path::Path::new(&output_dir);
    fs::create_dir_all(out).unwrap_or_else(|e| {
        eprintln!("Error creating {}: {}", output_dir, e);
        process::exit(1);
    });

    match target {
        "cfs" => generate_cfs_app_dir(out, &channels, window, threshold),
        "fprime" => generate_fprime_dir(out, &channels, window, threshold),
        "ros" => generate_ros_dir(out, &channels, window, threshold),
        _ => { eprintln!("Unknown target: {}", target); process::exit(1); }
    }

    println!("Generated {} DFA monitor app in {}/", target, output_dir);
    println!("  Channels: {}", channels.len());
    println!("  Window:   {}", window);
    println!("  Threshold: {:.3}", threshold);
    if let Some(ref path) = db_path {
        println!("  Source:   {} (ogma-compatible db.json)", path);
    }
}

#[allow(dead_code)]
struct Channel {
    name: String,
    c_type: String,
    topic: String,
    field: String,
    msg_type: String,
}

fn parse_db_json(path: &str) -> Vec<Channel> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        process::exit(1);
    });

    let mut channels = Vec::new();
    let mut in_inputs = false;
    let mut cur_name = String::new();
    let mut cur_type = String::new();
    let mut cur_topic = String::new();
    let mut cur_field = String::new();

    for line in content.lines() {
        let t = line.trim();
        if t.contains("\"inputs\"") { in_inputs = true; }
        if !in_inputs { continue; }

        if t.contains("\"name\"") {
            if let Some(v) = extract_json_string(t) { cur_name = v; }
        }
        if t.contains("\"type\"") && !t.contains("\"fromType\"") && !t.contains("\"toType\"") {
            if let Some(v) = extract_json_string_after(t, "\"type\"") { cur_type = v; }
        }
        if t.contains("\"topic\"") {
            if let Some(v) = extract_json_string(t) { cur_topic = v; }
        }
        if t.contains("\"field\"") && !t.contains("\"fromField\"") {
            if let Some(v) = extract_json_string(t) { cur_field = v; }
        }

        if (t == "}" || t == "},") && !cur_name.is_empty() && !cur_topic.is_empty() {
            let clean_topic = cur_topic.trim_end_matches("_MID").to_string();
            channels.push(Channel {
                name: cur_name.clone(),
                c_type: if cur_type.is_empty() { "double".into() } else { cur_type.clone() },
                topic: clean_topic.clone(),
                field: if cur_field.is_empty() { "payload".into() } else { cur_field.clone() },
                msg_type: format!("{}_msg_t", clean_topic.to_lowercase()),
            });
            cur_name.clear();
            cur_type.clear();
            cur_topic.clear();
            cur_field.clear();
        }

        if t.contains("\"topics\"") { in_inputs = false; }
    }
    channels
}

fn extract_json_string(line: &str) -> Option<String> {
    let parts: Vec<&str> = line.split('"').collect();
    if parts.len() >= 4 { Some(parts[3].to_string()) } else { None }
}

fn extract_json_string_after(line: &str, key: &str) -> Option<String> {
    if let Some(pos) = line.find(key) {
        let rest = &line[pos + key.len()..];
        let parts: Vec<&str> = rest.split('"').collect();
        if parts.len() >= 2 { return Some(parts[1].to_string()); }
    }
    None
}

fn generate_cfs_app_dir(out: &std::path::Path, channels: &[Channel], window: usize, threshold: f64) {
    let src = out.join("fsw").join("src");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(out.join("fsw").join("platform_inc")).unwrap();
    fs::create_dir_all(out.join("fsw").join("mission_inc")).unwrap();

    fs::write(src.join("dfa_core.h"), generate_dfa_core_h()).unwrap();
    fs::write(src.join("dfa_monitor_cfs.c"), generate_cfs_main(channels, window, threshold)).unwrap();
    fs::write(src.join("dfa_monitor_cfs.h"), generate_cfs_header(channels)).unwrap();
    fs::write(src.join("dfa_monitor_cfs_events.h"), DFA_EVENTS_H).unwrap();
    fs::write(out.join("fsw").join("platform_inc").join("dfa_monitor_cfs_msgids.h"), generate_cfs_msgids(channels)).unwrap();
    fs::write(out.join("fsw").join("mission_inc").join("dfa_monitor_cfs_perfids.h"), DFA_PERFIDS_H).unwrap();
    fs::write(out.join("CMakeLists.txt"), CFS_CMAKE).unwrap();
}

fn generate_fprime_dir(out: &std::path::Path, channels: &[Channel], window: usize, threshold: f64) {
    fs::create_dir_all(out).unwrap();
    fs::write(out.join("dfa_core.h"), generate_dfa_core_h()).unwrap();
    fs::write(out.join("DfaMonitor.fpp"), generate_fprime_fpp(channels)).unwrap();
    fs::write(out.join("DfaMonitor.cpp"), generate_fprime_cpp(channels, window, threshold)).unwrap();
    fs::write(out.join("DfaMonitor.hpp"), generate_fprime_hpp(channels, window)).unwrap();
    fs::write(out.join("CMakeLists.txt"), FPRIME_CMAKE).unwrap();
}

fn generate_dfa_core_h() -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("/* dfa_core.h -- DFA scaling analysis (compute only)\n");
    s.push_str(" * Generated by struktura. Zero dependencies beyond <math.h>.\n");
    s.push_str(" * Peng et al., Physical Review E 49(2), 1994.\n");
    s.push_str(" * https://github.com/koscak-labs/struktura\n */\n\n");
    s.push_str("#ifndef DFA_CORE_H\n#define DFA_CORE_H\n\n");
    s.push_str("#include <math.h>\n\n");
    s.push_str("#define DFA_NUM_BOXES 6\n");
    s.push_str("static const int DFA_BOXES[DFA_NUM_BOXES] = {16, 24, 36, 54, 81, 121};\n\n");
    s.push_str("typedef struct { double alpha; double r_squared; } dfa_result_t;\n\n");
    s.push_str("static dfa_result_t dfa_compute(const double *values, int n) {\n");
    s.push_str("    dfa_result_t result = {0.5, 0.0};\n");
    s.push_str("    if (n < 64) return result;\n");
    s.push_str("    double mean = 0.0;\n    int i, seg, b;\n");
    s.push_str("    for (i = 0; i < n; i++) mean += values[i];\n");
    s.push_str("    mean /= (double)n;\n");
    s.push_str("    double y[512];\n    double cum = 0.0;\n");
    s.push_str("    int nn = n > 512 ? 512 : n;\n");
    s.push_str("    for (i = 0; i < nn; i++) { cum += values[i] - mean; y[i] = cum; }\n");
    s.push_str("    double log_s[DFA_NUM_BOXES], log_f[DFA_NUM_BOXES];\n    int pts = 0;\n");
    s.push_str("    for (b = 0; b < DFA_NUM_BOXES && DFA_BOXES[b] <= nn / 4; b++) {\n");
    s.push_str("        int s = DFA_BOXES[b], num_segs = nn / s;\n");
    s.push_str("        if (num_segs == 0) continue;\n        double f2_sum = 0.0;\n");
    s.push_str("        for (seg = 0; seg < num_segs; seg++) {\n");
    s.push_str("            int start = seg * s;\n");
    s.push_str("            double sx=0,sy=0,sxy=0,sx2=0;\n");
    s.push_str("            for (i = 0; i < s; i++) {\n");
    s.push_str("                double xi = (double)i;\n");
    s.push_str("                sx+=xi; sy+=y[start+i]; sxy+=xi*y[start+i]; sx2+=xi*xi;\n");
    s.push_str("            }\n");
    s.push_str("            double k=(double)s, det=k*sx2-sx*sx;\n");
    s.push_str("            if (fabs(det)<1e-15) continue;\n");
    s.push_str("            double a0=(sx2*sy-sx*sxy)/det, a1=(k*sxy-sx*sy)/det;\n");
    s.push_str("            double resid=0.0;\n");
    s.push_str("            for (i=0;i<s;i++){double d=y[start+i]-(a0+a1*(double)i);resid+=d*d;}\n");
    s.push_str("            f2_sum += resid / k;\n        }\n");
    s.push_str("        double f = sqrt(f2_sum / (double)num_segs);\n");
    s.push_str("        if (f > 0.0) { log_s[pts]=log((double)s); log_f[pts]=log(f); pts++; }\n");
    s.push_str("    }\n    if (pts < 3) return result;\n");
    s.push_str("    double k=(double)pts, sx=0,sy=0,sxy=0,sx2=0;\n");
    s.push_str("    for (i=0;i<pts;i++){sx+=log_s[i];sy+=log_f[i];sxy+=log_s[i]*log_f[i];sx2+=log_s[i]*log_s[i];}\n");
    s.push_str("    double slope=(k*sxy-sx*sy)/(k*sx2-sx*sx);\n");
    s.push_str("    double ic=(sy-slope*sx)/k, ym=sy/k, sst=0,ssr=0;\n");
    s.push_str("    for (i=0;i<pts;i++){sst+=(log_f[i]-ym)*(log_f[i]-ym);ssr+=(log_f[i]-slope*log_s[i]-ic)*(log_f[i]-slope*log_s[i]-ic);}\n");
    s.push_str("    result.alpha=slope; result.r_squared=1.0-ssr/(sst>1e-15?sst:1e-15);\n");
    s.push_str("    return result;\n}\n\n#endif /* DFA_CORE_H */\n");
    s
}

fn generate_cfs_main(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("/* dfa_monitor_cfs.c -- Generated by struktura generate --cfs\n");
    s.push_str(" * DFA structural health monitor for NASA cFS.\n");
    s.push_str(" * https://github.com/koscak-labs/struktura\n */\n\n");
    s.push_str("#include \"dfa_monitor_cfs.h\"\n");
    s.push_str("#include \"dfa_monitor_cfs_events.h\"\n");
    s.push_str("#include \"dfa_monitor_cfs_msgids.h\"\n");
    s.push_str("#include \"dfa_core.h\"\n\n");
    s.push_str(&format!("#define DFA_WINDOW_SIZE {}\n", window));
    s.push_str(&format!("#define DFA_THRESHOLD   {:.4}\n", threshold));
    s.push_str("#define DFA_LEARN_WINDOWS 10\n");
    s.push_str("#define DFA_R2_MIN 0.7\n\n");

    s.push_str("typedef struct {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str("    uint32 pos;\n    uint32 filled;\n");
    s.push_str("    double baseline_alpha;\n    uint8 baseline_set;\n    uint32 window_count;\n");
    s.push_str("} dfa_channel_t;\n\n");

    for ch in channels {
        s.push_str(&format!("static dfa_channel_t dfa_ch_{};\n", ch.name));
    }
    s.push_str("\nstatic CFE_SB_PipeId_t DFA_MONITOR_Pipe;\n");
    s.push_str("static CFE_SB_Buffer_t *SBBufPtr;\n\n");

    s.push_str("void DFA_MONITOR_AppMain(void) {\n");
    s.push_str("    CFE_Status_t status;\n    uint32 RunStatus = CFE_ES_RunStatus_APP_RUN;\n");
    s.push_str("    status = DFA_MONITOR_Init();\n");
    s.push_str("    if (status != CFE_SUCCESS) RunStatus = CFE_ES_RunStatus_APP_ERROR;\n");
    s.push_str("    while (CFE_ES_RunLoop(&RunStatus) == true) {\n");
    s.push_str("        status = CFE_SB_ReceiveBuffer(&SBBufPtr, DFA_MONITOR_Pipe, 500);\n");
    s.push_str("        if (status == CFE_SUCCESS) DFA_MONITOR_ProcessPkt();\n");
    s.push_str("    }\n    CFE_ES_ExitApp(RunStatus);\n}\n\n");

    s.push_str("CFE_Status_t DFA_MONITOR_Init(void) {\n");
    s.push_str("    CFE_Status_t status;\n");
    s.push_str("    CFE_EVS_Register(NULL, 0, CFE_EVS_EventFilter_BINARY);\n");
    s.push_str("    status = CFE_SB_CreatePipe(&DFA_MONITOR_Pipe, 32, \"DFA_MON_PIPE\");\n");
    s.push_str("    if (status != CFE_SUCCESS) return status;\n\n");
    for ch in channels {
        s.push_str(&format!("    CFE_SB_Subscribe(CFE_SB_ValueToMsgId({}_MID), DFA_MONITOR_Pipe);\n", ch.topic.to_uppercase()));
        s.push_str(&format!("    memset(&dfa_ch_{}, 0, sizeof(dfa_ch_{}));\n", ch.name, ch.name));
    }
    s.push_str("\n    CFE_EVS_SendEvent(DFA_MON_INIT_EID, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str(&format!("        \"DFA Monitor: {} channels, window={}, threshold={:.3}\");\n", channels.len(), window, threshold));
    s.push_str("    return CFE_SUCCESS;\n}\n\n");

    s.push_str("static void dfa_push(dfa_channel_t *ch, double value, const char *name) {\n");
    s.push_str(&format!("    ch->buffer[ch->pos] = value;\n    ch->pos = (ch->pos + 1) % {};\n", window));
    s.push_str("    if (ch->pos == 0) ch->filled = 1;\n    if (!ch->filled) return;\n");
    s.push_str("    ch->window_count++;\n");
    s.push_str(&format!("    dfa_result_t r = dfa_compute(ch->buffer, {});\n", window));
    s.push_str("    if (!ch->baseline_set && ch->window_count >= DFA_LEARN_WINDOWS && r.r_squared > DFA_R2_MIN) {\n");
    s.push_str("        ch->baseline_alpha = r.alpha;\n        ch->baseline_set = 1;\n");
    s.push_str("        CFE_EVS_SendEvent(DFA_MON_BASELINE_EID, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str("            \"DFA baseline %s: alpha=%.3f R2=%.4f\", name, r.alpha, r.r_squared);\n");
    s.push_str("        return;\n    }\n    if (!ch->baseline_set) return;\n");
    s.push_str("    if (r.r_squared < DFA_R2_MIN) return;\n");
    s.push_str("    double shift = r.alpha - ch->baseline_alpha;\n");
    s.push_str("    if (shift < 0) shift = -shift;\n");
    s.push_str("    if (shift >= DFA_THRESHOLD) {\n");
    s.push_str("        CFE_EVS_SendEvent(DFA_MON_SHIFT_EID, CFE_EVS_EventType_ERROR,\n");
    s.push_str("            \"DFA SHIFT %s: alpha=%.3f baseline=%.3f delta=%.3f\",\n");
    s.push_str("            name, r.alpha, ch->baseline_alpha, r.alpha - ch->baseline_alpha);\n");
    s.push_str("    }\n}\n\n");

    s.push_str("void DFA_MONITOR_ProcessPkt(void) {\n");
    s.push_str("    CFE_SB_MsgId_t MsgId = CFE_SB_INVALID_MSG_ID;\n");
    s.push_str("    CFE_MSG_GetMsgId(&SBBufPtr->Msg, &MsgId);\n\n");
    for (i, ch) in channels.iter().enumerate() {
        let kw = if i == 0 { "if" } else { "else if" };
        s.push_str(&format!("    {} (CFE_SB_MsgId_Equal(MsgId, CFE_SB_ValueToMsgId({}_MID))) {{\n", kw, ch.topic.to_uppercase()));
        s.push_str(&format!("        {} *msg = ({}*)&SBBufPtr->Msg;\n", ch.msg_type, ch.msg_type));
        s.push_str(&format!("        dfa_push(&dfa_ch_{}, (double)msg->{}, \"{}\");\n", ch.name, ch.field, ch.name));
        s.push_str("    }\n");
    }
    s.push_str("}\n");
    s
}

fn generate_cfs_header(channels: &[Channel]) -> String {
    let mut s = String::new();
    s.push_str("#ifndef DFA_MONITOR_CFS_H\n#define DFA_MONITOR_CFS_H\n\n");
    s.push_str("#include \"cfe.h\"\n#include <string.h>\n#include <math.h>\n\n");
    s.push_str("void DFA_MONITOR_AppMain(void);\n");
    s.push_str("CFE_Status_t DFA_MONITOR_Init(void);\n");
    s.push_str("void DFA_MONITOR_ProcessPkt(void);\n\n");
    let _ = channels;
    s.push_str("#endif\n");
    s
}

fn generate_cfs_msgids(channels: &[Channel]) -> String {
    let mut s = String::new();
    s.push_str("#ifndef DFA_MONITOR_CFS_MSGIDS_H\n#define DFA_MONITOR_CFS_MSGIDS_H\n\n");
    for (i, ch) in channels.iter().enumerate() {
        s.push_str(&format!("#define {}_MID 0x{:04X}\n", ch.topic.to_uppercase(), 0x1900 + i));
    }
    s.push_str("\n#endif\n");
    s
}

fn generate_fprime_fpp(channels: &[Channel]) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("module Svc {\n");
    s.push_str("  @ DFA structural health monitor\n");
    s.push_str("  @ Generated by: struktura generate --fprime\n");
    s.push_str("  @ https://github.com/koscak-labs/struktura\n");
    s.push_str("  passive component DfaMonitor {\n\n");
    s.push_str("    sync input port schedIn: Svc.Sched\n\n");
    for ch in channels {
        s.push_str(&format!("    guarded input port {}In: Fw.Tlm\n", ch.name));
    }
    s.push_str("\n    event StructuralShift(\n");
    s.push_str("      channelName: string size 32\n");
    s.push_str("      baseline_alpha: F64\n      current_alpha: F64\n      delta: F64\n");
    s.push_str("    ) severity warning high\n\n");
    s.push_str("    event BaselineEstablished(\n");
    s.push_str("      channelName: string size 32\n      alpha: F64\n      r_squared: F64\n");
    s.push_str("    ) severity activity high\n\n");
    s.push_str("    telemetry DfaAlpha: F64\n    telemetry DfaR2: F64\n\n");
    s.push_str("    time get port timeCaller\n    event port logOut\n    telemetry port tlmOut\n");
    s.push_str("  }\n}\n");
    s
}

fn generate_fprime_cpp(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("// DfaMonitor.cpp -- Generated by struktura generate --fprime\n");
    s.push_str("// https://github.com/koscak-labs/struktura\n\n");
    s.push_str("#include \"DfaMonitor.hpp\"\n#include \"dfa_core.h\"\n\n");
    s.push_str("namespace Svc {\n\n");
    s.push_str("DfaMonitor::DfaMonitor(const char* name) : DfaMonitorComponentBase(name) {\n");
    for ch in channels {
        s.push_str(&format!("    memset(&m_ch_{}, 0, sizeof(m_ch_{}));\n", ch.name, ch.name));
    }
    s.push_str("}\n\n");
    for ch in channels {
        s.push_str(&format!("void DfaMonitor::{}In_handler(NATIVE_INT_TYPE portNum, FwTlmBuffer& val) {{\n", ch.name));
        s.push_str("    F64 v; val.deserialize(v);\n");
        s.push_str(&format!("    pushSample(m_ch_{}, v, \"{}\");\n", ch.name, ch.name));
        s.push_str("}\n\n");
    }
    s.push_str("void DfaMonitor::pushSample(DfaChannel& ch, F64 value, const char* name) {\n");
    s.push_str(&format!("    ch.buffer[ch.pos] = value;\n    ch.pos = (ch.pos + 1) % {};\n", window));
    s.push_str("    if (ch.pos == 0) ch.filled = true;\n    if (!ch.filled) return;\n");
    s.push_str("    ch.windowCount++;\n");
    s.push_str(&format!("    dfa_result_t r = dfa_compute(ch.buffer, {});\n", window));
    s.push_str("    if (!ch.baselineSet && ch.windowCount >= 10 && r.r_squared > 0.7) {\n");
    s.push_str("        ch.baselineAlpha = r.alpha; ch.baselineSet = true;\n");
    s.push_str("        this->log_ACTIVITY_HI_BaselineEstablished(name, r.alpha, r.r_squared);\n");
    s.push_str("        return;\n    }\n    if (!ch.baselineSet || r.r_squared < 0.7) return;\n");
    s.push_str("    F64 shift = (r.alpha > ch.baselineAlpha) ? r.alpha - ch.baselineAlpha : ch.baselineAlpha - r.alpha;\n");
    s.push_str(&format!("    if (shift >= {:.4}) {{\n", threshold));
    s.push_str("        this->log_WARNING_HI_StructuralShift(name, ch.baselineAlpha, r.alpha, r.alpha - ch.baselineAlpha);\n");
    s.push_str("    }\n    this->tlmWrite_DfaAlpha(r.alpha);\n    this->tlmWrite_DfaR2(r.r_squared);\n");
    s.push_str("}\n\n} // namespace Svc\n");
    s
}

fn generate_fprime_hpp(channels: &[Channel], window: usize) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str("#ifndef DFA_MONITOR_HPP\n#define DFA_MONITOR_HPP\n\n");
    s.push_str("#include \"DfaMonitorComponentAc.hpp\"\n\n");
    s.push_str("namespace Svc {\n\n");
    s.push_str("struct DfaChannel {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str("    U32 pos; bool filled; double baselineAlpha;\n");
    s.push_str("    bool baselineSet; U32 windowCount;\n};\n\n");
    s.push_str("class DfaMonitor : public DfaMonitorComponentBase {\n");
    s.push_str("  public:\n    DfaMonitor(const char* name);\n");
    s.push_str("  private:\n");
    for ch in channels {
        s.push_str(&format!("    void {}In_handler(NATIVE_INT_TYPE portNum, FwTlmBuffer& val);\n", ch.name));
    }
    s.push_str("    void pushSample(DfaChannel& ch, F64 value, const char* name);\n");
    for ch in channels {
        s.push_str(&format!("    DfaChannel m_ch_{};\n", ch.name));
    }
    s.push_str("};\n\n} // namespace Svc\n\n#endif\n");
    s
}

const DFA_EVENTS_H: &str = "\
#ifndef DFA_MONITOR_CFS_EVENTS_H
#define DFA_MONITOR_CFS_EVENTS_H
#define DFA_MON_INIT_EID     1
#define DFA_MON_BASELINE_EID 2
#define DFA_MON_SHIFT_EID    3
#endif
";

const DFA_PERFIDS_H: &str = "\
#ifndef DFA_MONITOR_CFS_PERFIDS_H
#define DFA_MONITOR_CFS_PERFIDS_H
#define DFA_MON_PERF_ID 91
#endif
";

const CFS_CMAKE: &str = "\
cmake_minimum_required(VERSION 2.6.4)
project(CFE_DFA_MONITOR_APP C)
include_directories(fsw/mission_inc)
include_directories(fsw/platform_inc)
aux_source_directory(fsw/src APP_SRC_FILES)
add_cfe_app(dfa_monitor_cfs ${APP_SRC_FILES})
";

const FPRIME_CMAKE: &str = "\
set(SOURCE_FILES DfaMonitor.cpp)
set(MOD_DEPS Fw/Tlm Svc/Sched)
register_fprime_module()
";

fn generate_ros_dir(out: &std::path::Path, channels: &[Channel], window: usize, threshold: f64) {
    let src = out.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("dfa_core.h"), generate_dfa_core_h()).unwrap();
    fs::write(src.join("dfa_monitor_node.cpp"), generate_ros_monitor(channels, window, threshold)).unwrap();
    fs::write(out.join("CMakeLists.txt"), generate_ros_cmake()).unwrap();
    fs::write(out.join("package.xml"), generate_ros_package()).unwrap();
}

fn generate_ros_monitor(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("/* dfa_monitor_node.cpp -- Generated by struktura generate --ros\n");
    s.push_str(" * DFA structural health monitor for ROS 2.\n");
    s.push_str(" * https://github.com/koscak-labs/struktura\n */\n\n");
    s.push_str("#include <functional>\n#include <memory>\n#include <cmath>\n#include <cstring>\n\n");
    s.push_str("#include \"rclcpp/rclcpp.hpp\"\n");
    s.push_str("#include \"std_msgs/msg/float64.hpp\"\n");
    s.push_str("#include \"std_msgs/msg/string.hpp\"\n\n");
    s.push_str("#include \"dfa_core.h\"\n\n");
    s.push_str(&format!("#define DFA_WINDOW_SIZE {}\n", window));
    s.push_str(&format!("#define DFA_THRESHOLD   {:.4}\n", threshold));
    s.push_str("#define DFA_LEARN_WINDOWS 10\n#define DFA_R2_MIN 0.7\n\n");
    s.push_str("using std::placeholders::_1;\n\n");

    s.push_str("struct DfaChannel {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str("    uint32_t pos = 0;\n    bool filled = false;\n");
    s.push_str("    double baseline_alpha = 0.0;\n    bool baseline_set = false;\n");
    s.push_str("    uint32_t window_count = 0;\n};\n\n");

    s.push_str("class DfaMonitorNode : public rclcpp::Node {\n");
    s.push_str("public:\n");
    s.push_str("    DfaMonitorNode() : Node(\"dfa_monitor\") {\n");
    for ch in channels {
        s.push_str(&format!("        {}_sub_ = this->create_subscription<std_msgs::msg::Float64>(\n", ch.name));
        s.push_str(&format!("            \"/{}\", 10,\n", ch.name));
        s.push_str(&format!("            std::bind(&DfaMonitorNode::{}_callback, this, _1));\n", ch.name));
        s.push_str(&format!("        std::memset(&ch_{}, 0, sizeof(ch_{}));\n\n", ch.name, ch.name));
    }
    s.push_str("        shift_pub_ = this->create_publisher<std_msgs::msg::String>(\"dfa/shift\", 10);\n");
    s.push_str(&format!("        RCLCPP_INFO(this->get_logger(), \"DFA Monitor: {} channels, window={}, threshold={:.3}\");\n", channels.len(), window, threshold));
    s.push_str("    }\n\nprivate:\n");

    s.push_str("    void push_sample(DfaChannel &ch, double value, const char *name) {\n");
    s.push_str(&format!("        ch.buffer[ch.pos] = value;\n        ch.pos = (ch.pos + 1) % {};\n", window));
    s.push_str("        if (ch.pos == 0) ch.filled = true;\n        if (!ch.filled) return;\n");
    s.push_str("        ch.window_count++;\n");
    s.push_str(&format!("        dfa_result_t r = dfa_compute(ch.buffer, {});\n", window));
    s.push_str("        if (!ch.baseline_set && ch.window_count >= DFA_LEARN_WINDOWS && r.r_squared > DFA_R2_MIN) {\n");
    s.push_str("            ch.baseline_alpha = r.alpha; ch.baseline_set = true;\n");
    s.push_str("            RCLCPP_INFO(this->get_logger(), \"DFA baseline %s: alpha=%.3f R2=%.4f\", name, r.alpha, r.r_squared);\n");
    s.push_str("            return;\n        }\n        if (!ch.baseline_set || r.r_squared < DFA_R2_MIN) return;\n");
    s.push_str("        double shift = std::abs(r.alpha - ch.baseline_alpha);\n");
    s.push_str("        if (shift >= DFA_THRESHOLD) {\n");
    s.push_str("            auto msg = std_msgs::msg::String();\n");
    s.push_str("            char buf[128];\n");
    s.push_str("            std::snprintf(buf, sizeof(buf), \"SHIFT %s: alpha=%.3f baseline=%.3f delta=%.3f\",\n");
    s.push_str("                name, r.alpha, ch.baseline_alpha, r.alpha - ch.baseline_alpha);\n");
    s.push_str("            msg.data = buf;\n            shift_pub_->publish(msg);\n");
    s.push_str("            RCLCPP_WARN(this->get_logger(), \"%s\", buf);\n");
    s.push_str("        }\n    }\n\n");

    for ch in channels {
        s.push_str(&format!("    void {}_callback(const std_msgs::msg::Float64::SharedPtr msg) {{\n", ch.name));
        s.push_str(&format!("        push_sample(ch_{}, msg->data, \"{}\");\n", ch.name, ch.name));
        s.push_str("    }\n\n");
    }

    for ch in channels {
        s.push_str(&format!("    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr {}_sub_;\n", ch.name));
        s.push_str(&format!("    DfaChannel ch_{};\n", ch.name));
    }
    s.push_str("    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr shift_pub_;\n");
    s.push_str("};\n\n");

    s.push_str("int main(int argc, char *argv[]) {\n");
    s.push_str("    rclcpp::init(argc, argv);\n");
    s.push_str("    rclcpp::spin(std::make_shared<DfaMonitorNode>());\n");
    s.push_str("    rclcpp::shutdown();\n    return 0;\n}\n");
    s
}

fn generate_ros_cmake() -> String {
    let mut s = String::new();
    s.push_str("cmake_minimum_required(VERSION 3.8)\nproject(dfa_monitor)\n\n");
    s.push_str("find_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\n");
    s.push_str("find_package(std_msgs REQUIRED)\n\n");
    s.push_str("add_executable(dfa_monitor_node src/dfa_monitor_node.cpp)\n");
    s.push_str("ament_target_dependencies(dfa_monitor_node rclcpp std_msgs)\n\n");
    s.push_str("install(TARGETS dfa_monitor_node DESTINATION lib/${PROJECT_NAME})\n");
    s.push_str("ament_package()\n");
    s
}

fn generate_ros_package() -> String {
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<package format=\"3\">\n");
    s.push_str("  <name>dfa_monitor</name>\n  <version>1.0.0</version>\n");
    s.push_str("  <description>DFA structural health monitor for ROS 2</description>\n");
    s.push_str("  <maintainer email=\"phil@koscak.ai\">Phil Koscak</maintainer>\n");
    s.push_str("  <license>MIT</license>\n\n");
    s.push_str("  <buildtool_depend>ament_cmake</buildtool_depend>\n");
    s.push_str("  <depend>rclcpp</depend>\n  <depend>std_msgs</depend>\n\n");
    s.push_str("  <export>\n    <build_type>ament_cmake</build_type>\n  </export>\n");
    s.push_str("</package>\n");
    s
}
