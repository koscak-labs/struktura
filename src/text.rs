//! Text structural analysis via DFA on sentence-length sequences.
//!
//! Human writing has long-range correlations in sentence length — short
//! sentences cluster with short, long with long, creating a fractal rhythm.
//! DFA measures this as α > 0.5. Random/shuffled text has α ≈ 0.5.
//!
//! ```
//! use struktura::text::text_structure;
//! let report = text_structure("The quick brown fox. It jumped. Over the lazy dog sleeping in the sun. A very long sentence that goes on and on and on to demonstrate variation in writing style and rhythm.");
//! assert!(report.sentence_count >= 3);
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::format;

use crate::{dfa, analyze, DfaResult, StructuralLaw};
use core::fmt;

/// Structural analysis of a text's sentence-length rhythm.
#[derive(Debug, Clone)]
pub struct TextStructure {
    pub sentence_count: usize,
    pub mean_sentence_len: f64,
    pub dfa: DfaResult,
    pub law: StructuralLaw,
    pub sentence_lengths: Vec<usize>,
}

impl fmt::Display for TextStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sentences={} mean_len={:.1} α={:.3} R²={:.4} quality={}",
            self.sentence_count, self.mean_sentence_len,
            self.dfa.alpha, self.dfa.r_squared, self.law.quality)
    }
}

/// Extract sentence lengths from text.
///
/// Splits on sentence-ending punctuation (.!?) followed by whitespace
/// or end-of-string. Filters out very short fragments (< 3 chars).
pub fn sentence_lengths(text: &str) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut current_len = 0;

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    for i in 0..n {
        current_len += 1;
        let is_terminal = chars[i] == '.' || chars[i] == '!' || chars[i] == '?';
        let followed_by_space_or_end = if i + 1 >= n {
            true
        } else {
            chars[i + 1].is_whitespace() || chars[i + 1] == '"' || chars[i + 1] == '\''
        };

        if is_terminal && followed_by_space_or_end && current_len >= 3 {
            lengths.push(current_len);
            current_len = 0;
        }
    }
    if current_len >= 10 {
        lengths.push(current_len);
    }
    lengths
}

/// Analyze the structural rhythm of a text.
///
/// Requires at least 64 sentences for reliable DFA. Fewer sentences
/// produce a result with low R² (the quality field reflects this).
pub fn text_structure(text: &str) -> TextStructure {
    let lens = sentence_lengths(text);
    let values: Vec<f64> = lens.iter().map(|&l| l as f64).collect();
    let n = lens.len();

    let mean = if n > 0 {
        values.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let dfa_result = dfa(&values);
    let law = analyze(&values);

    TextStructure {
        sentence_count: n,
        mean_sentence_len: mean,
        dfa: dfa_result,
        law,
        sentence_lengths: lens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_splitting() {
        let text = "Hello world. This is a test! Is it working? Yes it is.";
        let lens = sentence_lengths(text);
        assert_eq!(lens.len(), 4);
    }

    #[test]
    fn short_text_still_works() {
        let text = "Short text. More text. And here.";
        let result = text_structure(text);
        assert!(result.sentence_count >= 2);
    }

    #[test]
    fn builtin_text_demo() {
        // Use the CWRU bearing data description as a human-written text sample.
        // This is too short for reliable DFA but should not panic.
        let text = "The bearing data was collected at Case Western Reserve University. \
                     Vibration data was recorded using accelerometers. \
                     Normal bearings show a random vibration pattern. \
                     Inner race faults produce characteristic frequencies. \
                     Ball faults are harder to detect due to signal modulation. \
                     Outer race faults are typically the clearest. \
                     Each fault type changes the DFA scaling exponent. \
                     The structural signature shifts before performance degrades. \
                     This is the principle behind struktura.";
        let result = text_structure(text);
        assert!(result.sentence_count >= 8, "got {} sentences", result.sentence_count);
    }
}
