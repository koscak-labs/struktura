//! Error classes: turn raw compiler / checker messages into stable SIGNATURES and count them.
//!
//! The same mistake never prints the same bytes twice: line numbers, temp paths, identifiers
//! and payloads differ. A signature keeps what identifies the mistake and drops the rest:
//! - rustc: `error[E0308]: mismatched types` -> `rustc E0308 mismatched types`
//! - hxc:   `hxc check: line 15: frameshift — TTT closes a block that was never opened (check …)`
//!          -> `hxc: frameshift — TTT closes a block that was never opened`
//!          and `'ACA # brain_gpu <result_var>' needs 1 arg, got 4` -> `hxc: 'ACA' needs N arg, got N`
//! - anything else: the first line that says error/panic/fail, with numbers, paths and
//!   quoted text masked.
//! Zero LLM: the same input always gives the same classes, so a primer A/B can say WHICH
//! mistakes a change removed, reproducibly.

use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

/// Mask digits runs as `N`.
fn mask_numbers(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_num = false;
    for c in s.chars() {
        if c.is_ascii_digit() { if !in_num { out.push('N'); in_num = true; } } else { in_num = false; out.push(c); }
    }
    // a LIST of numbers is one value: "line(s) [3, 7, 12]" and "line(s) [5]" are the same mistake
    for sep in [", N", ",N", " N"] {
        let pair = std::format!("N{}", sep);
        while out.contains(&pair) { out = out.replace(&pair, "N"); }
    }
    out
}

/// Replace every '...' / "..." / `...` span: keep a leading codon (3 ACGT letters) if there is one.
fn mask_quotes(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' || c == '`' {
            let mut inner = String::new();
            let mut closed = false;
            for d in chars.by_ref() { if d == c { closed = true; break; } inner.push(d); }
            if !closed { out.push(c); out.push_str(&inner); break; }
            let codon: String = inner.chars().take(3).collect();
            if codon.len() == 3 && codon.chars().all(|x| "ACGT".contains(x)) { out.push(c); out.push_str(&codon); out.push(c); }
            else { out.push(c); out.push('…'); out.push(c); }
        } else { out.push(c); }
    }
    out
}

/// Drop `/tmp/...`-style paths and trailing parenthesised hints.
fn strip_noise(s: &str) -> String {
    let mut t: String = s.split_whitespace().filter(|w| !w.contains('/') || w.len() < 3).collect::<Vec<_>>().join(" ");
    if let Some(i) = t.find(" (check") { t.truncate(i); }
    if let Some(i) = t.find(" (hint") { t.truncate(i); }
    t.trim().trim_end_matches(|c: char| c == '.' || c == ':').to_string()
}

/// The stable signature of one error text (possibly multi-line). Empty text -> "none".
pub fn signature(text: &str) -> String {
    let lines: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() { return "none".into(); }
    // rustc: the first `error[E....]: message` line
    for l in &lines {
        if let Some(rest) = l.strip_prefix("error[E") {
            if let Some((code, msg)) = rest.split_once("]:") {
                return std::format!("rustc E{} {}", code, mask_numbers(&mask_quotes(msg.trim())));
            }
        }
    }
    // rustc without a code (before hxc: `hxc build` wraps rustc's real error in a generic hxc line)
    if let Some(l) = lines.iter().find(|l| l.starts_with("error:") && !l.starts_with("error: aborting")) {
        return std::format!("rustc {}", strip_noise(&mask_numbers(&mask_quotes(l))));
    }
    // hxc: `hxc check: line N: msg` / `hxc compile error: line N: msg` / `hxc check: msg`
    for l in &lines {
        if l.starts_with("hxc") {
            let msg = match l.split_once("line ").and_then(|(_, r)| r.split_once(':')) {
                Some((n, m)) if n.trim().chars().all(|c| c.is_ascii_digit()) => m.trim(),
                _ => {
                    let mut m: &str = l;
                    for p in ["hxc check:", "hxc compile error:", "hxc:"] { if let Some(r) = m.strip_prefix(p) { m = r.trim(); } }
                    m
                }
            };
            return std::format!("hxc: {}", strip_noise(&mask_numbers(&mask_quotes(msg))));
        }
    }
    let l = lines.iter().find(|l| { let x = l.to_ascii_lowercase(); x.contains("error") || x.contains("panic") || x.contains("fail") })
        .unwrap_or(&lines[0]);
    strip_noise(&mask_numbers(&mask_quotes(l)))
}

/// Counts per (group, class), with the first example key seen for each.
#[derive(Clone, Debug, Default)]
pub struct Classes {
    /// group -> class -> (count, example)
    pub by_group: BTreeMap<String, BTreeMap<String, (usize, String)>>,
}

impl Classes {
    pub fn add(&mut self, group: &str, key: &str, error: &str) {
        let e = self.by_group.entry(group.to_string()).or_default().entry(signature(error)).or_insert((0, key.to_string()));
        e.0 += 1;
    }
    /// One group's classes, most frequent first (ties by name): (class, count, example).
    pub fn ranked(&self, group: &str) -> Vec<(String, usize, String)> {
        let mut v: Vec<(String, usize, String)> = self.by_group.get(group).map(|m| m.iter().map(|(c, (n, ex))| (c.clone(), *n, ex.clone())).collect()).unwrap_or_default();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }
    pub fn total(&self, group: &str) -> usize { self.by_group.get(group).map(|m| m.values().map(|(n, _)| n).sum()).unwrap_or(0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_mistake_same_signature() {
        let a = "hxc check: line 15: frameshift — TTT closes a block that was never opened (check for missing AAA/AGA/CAA above this line)";
        let b = "hxc compile error: line 9: frameshift — TTT closes a block that was never opened (check for missing AAA/AGA/CAA above this line)";
        assert_eq!(signature(a), signature(b));
        assert_eq!(signature(a), "hxc: frameshift — TTT closes a block that was never opened");
        assert_eq!(signature("hxc check: line 4: 'ACA # brain_gpu <result_var>' needs 1 arg, got 4"), "hxc: 'ACA' needs N arg, got N");
        assert_eq!(signature("hxc check: line 12: 'CAC # else' with no open Plain block"), "hxc: 'CAC' with no open Plain block");
        // a variable-length list of line numbers is one class
        let a = signature("hxc check: NON-ASCII on line(s) [3, 7, 12] -- break rustc+shells");
        assert_eq!(a, signature("hxc check: NON-ASCII on line(s) [5] -- break rustc+shells"));
        assert_eq!(a, "hxc: NON-ASCII on line(s) [N] -- break rustc+shells");
        assert_eq!(signature("hxc check: frameshift: 2 block(s) opened but never closed — add 2 'TTT' line(s)"), "hxc: frameshift: N block(s) opened but never closed — add N 'TTT' line(s)");
        // hxc build wraps rustc: the rustc line wins over the generic hxc wrapper
        assert_eq!(signature("hxc: compile failed — check .hlx lines marked [hlx:7] above\nerror: expected one of `;` or `}`, found `x`\nerror: aborting due to 1 previous error"), "rustc error: expected one of `…` or `…`, found `…`");
    }

    #[test]
    fn rustc_codes_and_fallbacks() {
        let r = "error[E0308]: mismatched types\n  --> /tmp/tmp.5NsqQ9H8FU/c.rs:56:17\n   |";
        assert_eq!(signature(r), "rustc E0308 mismatched types");
        assert_eq!(signature("error[E0425]: cannot find value `cnt` in this scope"), "rustc E0425 cannot find value `…` in this scope");
        assert_eq!(signature("thread 'main' panicked at src/x.rs:3:5:\nindex out of bounds"), "thread '…' panicked at");
        assert_eq!(signature(""), "none");
        assert_eq!(signature("wrong output: expected 42 got 41"), "wrong output: expected N got N");
    }

    #[test]
    fn classes_rank_by_count() {
        let mut c = Classes::default();
        c.add("helix", "t1", "hxc check: line 3: frameshift — TTT closes a block that was never opened");
        c.add("helix", "t2", "hxc check: line 9: frameshift — TTT closes a block that was never opened");
        c.add("helix", "t3", "error[E0308]: mismatched types");
        let r = c.ranked("helix");
        assert_eq!((r[0].0.as_str(), r[0].1, r[0].2.as_str()), ("hxc: frameshift — TTT closes a block that was never opened", 2, "t1"));
        assert_eq!(c.total("helix"), 3);
        assert!(c.ranked("rust").is_empty());
    }
}
