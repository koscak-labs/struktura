//! Recall: a small BM25 index so an agent (or the brain) gets the few cited records that answer
//! a question instead of reading the whole history. Inspired by Leviathan (elstongun/leviathan,
//! Apache-2.0: SQLite FTS5 + BM25 + group scoping + capped cards); this is an independent, std-only
//! implementation with no SQLite, so it ships wherever struktura does.
//!
//! - Records have an id, a group, a time, a title and text. Title terms count twice.
//! - A search can be scoped to a group; when the group has no match, other groups are searched
//!   and their hits are labelled `other`, never silently mixed in.
//! - Results are ranked by BM25 (k1 1.2, b 0.75) and rendered as short cards capped in size, and
//!   every answer says `shown N of M`, so "no match" is never confused with "no data".

use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

const K1: f64 = 1.2;
const B: f64 = 0.75;
const STOP: &[&str] = &["the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "for", "with", "vs", "by", "it", "as", "be"];

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub id: String,
    pub group: String,
    pub ts: f64,
    pub title: String,
    pub text: String,
}

/// Lowercase alphanumeric terms, stopwords dropped; `ub512` stays one term and also yields `ub`.
pub fn terms(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split(|c: char| !c.is_ascii_alphanumeric()) {
        if raw.is_empty() { continue; }
        let t = raw.to_ascii_lowercase();
        if STOP.contains(&t.as_str()) { continue; }
        // a knob value like ub512 / draft9 also indexes its knob name
        let alpha: String = t.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
        if alpha.len() >= 2 && alpha.len() < t.len() && t[alpha.len()..].chars().all(|c| c.is_ascii_digit()) { out.push(alpha); }
        out.push(t);
    }
    out
}

#[derive(Clone, Debug, Default)]
pub struct Index {
    pub records: Vec<Record>,
    /// term -> (record index, weighted term frequency)
    post: BTreeMap<String, Vec<(usize, f64)>>,
    len: Vec<f64>,
    avg: f64,
}

impl Index {
    pub fn build(records: Vec<Record>) -> Self {
        let mut ix = Index { records, ..Default::default() };
        for (i, r) in ix.records.iter().enumerate() {
            let mut tf: BTreeMap<String, f64> = BTreeMap::new();
            for t in terms(&r.title) { *tf.entry(t).or_insert(0.0) += 2.0; }
            for t in terms(&r.text) { *tf.entry(t).or_insert(0.0) += 1.0; }
            ix.len.push(tf.values().sum());
            for (t, f) in tf { ix.post.entry(t).or_default().push((i, f)); }
        }
        let n = ix.len.len().max(1) as f64;
        ix.avg = ix.len.iter().sum::<f64>() / n;
        ix
    }

    /// BM25 scores of every record matching any query term, restricted by `keep`, best first.
    pub fn scores(&self, query: &str, keep: impl Fn(&Record) -> bool) -> Vec<(usize, f64)> {
        let n = self.records.len() as f64;
        let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
        let mut q = terms(query); q.sort(); q.dedup();
        for t in &q {
            let Some(p) = self.post.get(t) else { continue };
            let df = p.len() as f64;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(i, f) in p {
                if !keep(&self.records[i]) { continue; }
                let norm = f * (K1 + 1.0) / (f + K1 * (1.0 - B + B * self.len[i] / self.avg.max(1e-9)));
                *acc.entry(i).or_insert(0.0) += idf * norm;
            }
        }
        let mut v: Vec<(usize, f64)> = acc.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal).then(b.0.cmp(&a.0)));
        v
    }

    /// Group-scoped search with a labelled fallback: (hits, total matches, in_group).
    pub fn search(&self, query: &str, group: Option<&str>, n: usize) -> (Vec<(usize, f64)>, usize, bool) {
        match group {
            Some(g) => {
                let inside = self.scores(query, |r| r.group == g);
                if !inside.is_empty() { let m = inside.len(); return (inside.into_iter().take(n).collect(), m, true); }
                let other = self.scores(query, |r| r.group != g);
                let m = other.len();
                (other.into_iter().take(n).collect(), m, false)
            }
            None => { let all = self.scores(query, |_| true); let m = all.len(); (all.into_iter().take(n).collect(), m, true) }
        }
    }
}

/// Records from a lab ledger (predictions, jobs, deploys, daemon decisions) and an optional
/// instinct table (`id<TAB>lesson<TAB>found_by<TAB>enforced_by<TAB>status`).
pub fn records_from(ledger: &str, instincts: Option<&str>) -> Vec<Record> {
    let mut out = Vec::new();
    for line in ledger.lines() {
        let Some(j) = crate::lab::parse_json(line) else { continue };
        let s = |k: &str| j.str(k).unwrap_or("").to_string();
        let ts = j.num("ts").or_else(|| j.num("start")).unwrap_or(0.0);
        match j.str("kind") {
            Some("prediction") => {
                let name = s("name");
                let knob: String = name.chars().take_while(|c| c.is_ascii_alphabetic()).collect();
                out.push(Record { id: std::format!("{}::{}", s("pred"), name), group: if knob.is_empty() { "prediction".into() } else { knob },
                    ts, title: name.replace('-', " "), text: std::format!("{} verdict {} {} {}", s("pred"), s("verdict"), s("value"), s("note")) });
            }
            Some("deploy") => out.push(Record { id: std::format!("deploy@{}", ts as u64), group: "deploy".into(), ts,
                title: std::format!("deploy {}", s("binary")), text: std::format!("{} {}", s("env"), s("flags")) }),
            Some("job") => out.push(Record { id: s("job"), group: "job".into(), ts, title: s("job").replace(['-', '.'], " "),
                text: std::format!("rc {}", j.num("rc").unwrap_or(-1.0)) }),
            Some("daemon") => out.push(Record { id: std::format!("daemon@{}", ts as u64), group: "daemon".into(), ts,
                title: std::format!("{} {}", s("phase"), s("outcome")), text: s("detail") }),
            _ => {}
        }
    }
    if let Some(t) = instincts {
        for l in t.lines().filter(|l| l.starts_with('F')) {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 2 { continue; }
            out.push(Record { id: f[0].to_string(), group: "instinct".into(), ts: 0.0, title: f[1].to_string(),
                text: f.get(2..).map(|r| r.join(" ")).unwrap_or_default() });
        }
    }
    out
}

/// A capped card for one hit.
pub fn card(r: &Record, rank: usize, score: f64, max_chars: usize) -> String {
    let mut c = std::format!("[{}] {} · {} · rel {:.1}\n  {}\n  {}", rank, r.id, r.group, score, r.title, r.text.trim());
    if c.chars().count() > max_chars { c = c.chars().take(max_chars.saturating_sub(1)).collect::<String>() + "…"; }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, g: &str, title: &str, text: &str) -> Record { Record { id: id.into(), group: g.into(), ts: 0.0, title: title.into(), text: text.into() } }

    #[test]
    fn terms_index_knob_values_and_their_knob() {
        assert_eq!(terms("ub512 beats ub256 at d100000"), std::vec!["ub", "ub512", "beats", "ub", "ub256", "d100000"], "a one-letter prefix (d) is too common to index");
        assert!(terms("The and of").is_empty());
    }

    #[test]
    fn bm25_ranks_the_specific_record_first_and_scopes_groups() {
        let ix = Index::build(std::vec![
            rec("p1", "ub", "ub512 decode at 32K regressed", "decode -11% at 32768, accept 41 to 34"),
            rec("p2", "ub", "ub128 prefill", "prefill slower at d100000"),
            rec("p3", "draft", "draft9 failed to load", "MTP context creation failed, server exited"),
            rec("p4", "draft", "draft5 noisy", "two passes two fails"),
        ]);
        let (h, m, inside) = ix.search("why did draft9 not load", None, 3);
        assert_eq!((ix.records[h[0].0].id.as_str(), inside), ("p3", true));
        assert!(m >= 1);
        // scoped: the group has the answer
        let (h, _, inside) = ix.search("decode regression at 32K", Some("ub"), 3);
        assert_eq!((ix.records[h[0].0].id.as_str(), inside), ("p1", true));
        // scoped to a group without a match: falls back, labelled as other
        let (h, _, inside) = ix.search("MTP context", Some("ub"), 3);
        assert_eq!((ix.records[h[0].0].id.as_str(), inside), ("p3", false));
        // nothing anywhere
        let (h, m, _) = ix.search("zebra", None, 3);
        assert!(h.is_empty() && m == 0);
    }

    #[test]
    fn cards_are_capped() {
        let r = rec("x", "g", "title", &"long text ".repeat(100));
        let c = card(&r, 1, 3.2, 120);
        assert!(c.chars().count() <= 120 && c.ends_with('…'));
    }
}
