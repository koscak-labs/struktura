use struktura::text::text_structure;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: text_real <file.txt> [file2.txt ...]");
        std::process::exit(1);
    }

    println!("STRUKTURA TEXT STRUCTURE — REAL FILE ANALYSIS");
    println!("=============================================\n");

    for path in &args[1..] {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => { eprintln!("  {}: {}", path, e); continue; }
        };
        let result = text_structure(&text);
        let _lens = &result.sentence_lengths;

        println!("  FILE: {}", path);
        println!("  sentences: {}  mean_len: {:.1} chars  total chars: {}", 
            result.sentence_count, result.mean_sentence_len, text.len());
        println!("  DFA alpha: {:.3}  R²: {:.4}  quality: {}", 
            result.dfa.alpha, result.dfa.r_squared, result.law.quality);
        println!("  Hurst: {:.3}  kurtosis: {:.2}", result.law.hurst, result.law.kurtosis);
        
        if result.sentence_count >= 64 {
            let interpretation = if result.dfa.r_squared < 0.5 {
                "insufficient structure for classification"
            } else if result.dfa.alpha > 0.7 {
                "strong long-range correlations (persistent rhythm)"
            } else if result.dfa.alpha > 0.55 {
                "moderate correlations (structured writing)"
            } else if result.dfa.alpha > 0.45 {
                "near-random sentence lengths (uniform or mechanical)"
            } else {
                "anti-correlated (alternating pattern)"
            };
            println!("  interpretation: {}", interpretation);
        }
        println!();
    }
}
