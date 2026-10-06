//! Ledger schema: one table of every row kind the two lab strands append to
//! the shared JSON-lines ledger, and a validator for that ledger. No model in
//! the loop.
//!
//! - **Nüwa** (the inference lab) writes `job`, `prediction`, `work`,
//!   `daemon`, `deploy`, `window`, `cal`, `workbench`, `deploy_failure`,
//!   `autogate` and `constraint` rows.
//! - **Fuxi** (the judging / learning strand) writes `brain`, `forecast`,
//!   `score`, `judge`, `contract`, `heartbeat` and `gym` rows, every one
//!   inside the envelope `{"v":1,"strand":"fuxi","kind":..,"ts":..}`.
//!
//! [`SCHEMA`] is the single source of truth: [`check`] validates against it
//! and [`schema_tsv`] exports it so a shell-side validator can share it.
//! A required field must be present and non-null; an optional field that is
//! absent or null is not type-checked. Fields not in the table are allowed.

#![cfg(feature = "std")]

use crate::lab::{parse_json, Json};
use std::collections::BTreeMap;
use std::fmt::Write as _;

// ------------------------------------------------------------- table ----

/// Which strand writes a kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strand { Nuwa, Fuxi }

impl Strand {
    pub fn name(self) -> &'static str { match self { Strand::Nuwa => "nuwa", Strand::Fuxi => "fuxi" } }
}

/// Light type check on a field's JSON value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// Any non-null value.
    Any,
    Num,
    /// A number with no fractional part.
    Int,
    Str,
    Bool,
    Obj,
    Arr,
    /// Exactly this JSON literal, e.g. `1` or `"fuxi"`.
    Const(&'static str),
}

impl Ty {
    pub fn name(self) -> String {
        match self {
            Ty::Any => "any".into(), Ty::Num => "num".into(), Ty::Int => "int".into(), Ty::Str => "str".into(),
            Ty::Bool => "bool".into(), Ty::Obj => "obj".into(), Ty::Arr => "arr".into(),
            Ty::Const(lit) => format!("const:{}", lit),
        }
    }
    pub fn accepts(self, v: &Json) -> bool {
        match (self, v) {
            (_, Json::Null) => false,
            (Ty::Any, _) => true,
            (Ty::Num, Json::Num(x)) => x.is_finite(),
            (Ty::Int, Json::Num(x)) => x.is_finite() && x.fract() == 0.0,
            (Ty::Str, Json::Str(_)) | (Ty::Bool, Json::Bool(_)) | (Ty::Obj, Json::Obj(_)) | (Ty::Arr, Json::Arr(_)) => true,
            (Ty::Const(lit), v) => parse_json(lit).as_ref() == Some(v),
            _ => false,
        }
    }
}

/// Whether a field must be present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Req {
    Yes,
    No,
    /// At least one field of the group must be present; the group is named
    /// by its members, e.g. `"env|chunk"`.
    AnyOf(&'static str),
}

impl Req {
    pub fn name(self) -> String {
        match self { Req::Yes => "yes".into(), Req::No => "no".into(), Req::AnyOf(g) => format!("anyof:{}", g) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: &'static str,
    pub ty: Ty,
    pub req: Req,
    /// Values seen in practice. Advisory only: anything else is a warning,
    /// never an error (the set is open).
    pub known: &'static [&'static str],
}

const fn req(name: &'static str, ty: Ty) -> Field { Field { name, ty, req: Req::Yes, known: &[] } }
const fn opt(name: &'static str, ty: Ty) -> Field { Field { name, ty, req: Req::No, known: &[] } }
const fn any_of(name: &'static str, ty: Ty, group: &'static str) -> Field { Field { name, ty, req: Req::AnyOf(group), known: &[] } }
const fn known(f: Field, values: &'static [&'static str]) -> Field { Field { known: values, ..f } }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KindSchema {
    pub kind: &'static str,
    pub strand: Strand,
    /// The kind's own fields. Fuxi kinds also carry [`FUXI_ENVELOPE`].
    pub fields: &'static [Field],
}

const fn nuwa(kind: &'static str, fields: &'static [Field]) -> KindSchema { KindSchema { kind, strand: Strand::Nuwa, fields } }
const fn fuxi(kind: &'static str, fields: &'static [Field]) -> KindSchema { KindSchema { kind, strand: Strand::Fuxi, fields } }

/// Fields every Fuxi row carries (besides `kind`).
pub const FUXI_ENVELOPE: &[Field] = &[req("v", Ty::Const("1")), req("strand", Ty::Const("\"fuxi\"")), req("ts", Ty::Num)];

/// Daemon phases seen so far. Open set: an unknown phase is a warning.
pub const DAEMON_PHASES: &[&str] = &["autogate", "ship", "protect", "feed", "run", "tick", "janitor"];

use Ty::*;

/// Every known row kind. Kind names are unique across strands.
pub const SCHEMA: &[KindSchema] = &[
    // ---- Nüwa: inference lab ----
    nuwa("job", &[req("job", Str), req("start", Num), req("end", Num), req("rc", Int)]),
    nuwa("prediction", &[req("name", Str), req("pred", Str), req("value", Any), req("verdict", Str), req("ts", Num),
        opt("op", Str), opt("threshold", Str), opt("note", Str)]),
    nuwa("work", &[req("task", Str), req("config", Str), req("run", Str), req("verdict", Str), req("ts", Num),
        opt("sample", Str), opt("meta", Any)]),
    nuwa("daemon", &[known(req("phase", Str), DAEMON_PHASES), req("outcome", Str), req("ts", Num), opt("detail", Any)]),
    nuwa("deploy", &[req("binary", Str), req("ts", Num), any_of("env", Str, "env|chunk"), any_of("chunk", Num, "env|chunk"),
        opt("verified", Arr)]),
    nuwa("window", &[req("event", Str), req("ts", Num)]),
    nuwa("cal", &[req("ts", Num), req("ok", Bool), opt("code", Obj), opt("prose", Obj), opt("dur_s", Num), opt("load_s", Num)]),
    nuwa("workbench", &[req("ts", Num), req("run", Str), req("outcome", Str), opt("live", Obj)]),
    nuwa("deploy_failure", &[req("ts", Num)]),
    nuwa("autogate", &[req("ts", Num)]),
    // Read by `lab` and the ouroboros knobs.
    nuwa("constraint", &[req("knob", Str), req("op", Str), req("value", Any)]),
    // lab-q.sh, at submit (the leadership arbiter's source of truth for what was queued, and by whom).
    nuwa("queued", &[req("job", Str), req("ts", Num), req("names", Arr), opt("pred_file", Str), opt("designed_by", Str),
        opt("src", Str)]), // src:"daemon" = queued from inside a daemon tick (never wakes the daemon)
    // ---- Fuxi: judging / learning (envelope only for now) ----
    fuxi("brain", &[]),
    fuxi("forecast", &[]),
    fuxi("score", &[]),
    fuxi("judge", &[]),
    fuxi("contract", &[]),
    fuxi("heartbeat", &[]),
    fuxi("gym", &[]),
];

impl KindSchema {
    /// Envelope (Fuxi kinds) followed by the kind's own fields.
    pub fn all_fields(&self) -> impl Iterator<Item = &'static Field> {
        let env: &'static [Field] = if self.strand == Strand::Fuxi { FUXI_ENVELOPE } else { &[] };
        env.iter().chain(self.fields.iter())
    }
}

pub fn lookup(kind: &str) -> Option<&'static KindSchema> { SCHEMA.iter().find(|s| s.kind == kind) }

/// The table as TSV, one row per (kind, field), after a header line:
/// `kind  field  type  required  known`. `required` is `yes`, `no` or
/// `anyof:a|b` (at least one of the group); `known` lists advisory values
/// (`a|b|c`) or `-`. No column is ever empty, so `read` with IFS=tab works.
pub fn schema_tsv() -> String {
    let mut s = String::from("kind\tfield\ttype\trequired\tknown\n");
    for k in SCHEMA {
        for f in k.all_fields() {
            let kn = if f.known.is_empty() { "-".to_string() } else { f.known.join("|") };
            let _ = writeln!(s, "{}\t{}\t{}\t{}\t{}", k.kind, f.name, f.ty.name(), f.req.name(), kn);
        }
    }
    s
}

// --------------------------------------------------------- validator ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity { Warn, Error }

impl Severity {
    pub fn name(self) -> &'static str { match self { Severity::Warn => "WARN", Severity::Error => "ERROR" } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Issue {
    /// 1-based line number in the file.
    pub line: usize,
    pub severity: Severity,
    pub kind: Option<String>,
    pub field: Option<String>,
    pub msg: String,
}

/// Rows of one kind, and the issues attributed to them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KindCount {
    /// `nuwa`, `fuxi`, or `unknown`.
    pub strand: &'static str,
    pub rows: usize,
    pub errors: usize,
    pub warnings: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FileCheck {
    /// Non-blank lines.
    pub lines: usize,
    /// Lines that parsed as JSON.
    pub rows: usize,
    pub unparseable: usize,
    /// The last line is incomplete JSON with no trailing newline.
    pub torn_tail: bool,
    pub kinds: BTreeMap<String, KindCount>,
    pub issues: Vec<Issue>,
}

impl FileCheck {
    pub fn errors(&self) -> usize { self.issues.iter().filter(|i| i.severity == Severity::Error).count() }
    pub fn warnings(&self) -> usize { self.issues.iter().filter(|i| i.severity == Severity::Warn).count() }

    fn push(&mut self, line: usize, severity: Severity, kind: Option<&str>, field: Option<&str>, msg: String) {
        if let Some(k) = kind.and_then(|k| self.kinds.get_mut(k)) {
            if severity == Severity::Error { k.errors += 1 } else { k.warnings += 1 }
        }
        self.issues.push(Issue { line, severity, kind: kind.map(String::from), field: field.map(String::from), msg });
    }
}

fn type_of(v: &Json) -> &'static str {
    match v { Json::Null => "null", Json::Bool(_) => "bool", Json::Num(_) => "num", Json::Str(_) => "str", Json::Arr(_) => "arr", Json::Obj(_) => "obj" }
}

/// Short rendering of a value for messages.
fn show(v: &Json) -> String {
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Num(x) => x.to_string(),
        Json::Str(s) if s.chars().count() > 32 => format!("{:?}...", s.chars().take(32).collect::<String>()),
        Json::Str(s) => format!("{:?}", s),
        Json::Arr(_) => "[..]".into(),
        Json::Obj(_) => "{..}".into(),
    }
}

/// Validate a ledger's text. `strict` turns unknown kinds into errors.
/// Blank lines are skipped. A last line that is not JSON and has no trailing
/// newline is an append still in flight (warning); any other line that is
/// not JSON is an error.
pub fn check(text: &str, strict: bool) -> FileCheck {
    let mut r = FileCheck::default();
    let ends_nl = text.ends_with('\n');
    let segs: Vec<&str> = text.split('\n').collect();
    let last = segs.len() - 1;
    for (i, raw) in segs.iter().enumerate() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.trim().is_empty() { continue; }
        r.lines += 1;
        match parse_json(line) {
            Some(j) => { r.rows += 1; check_row(&j, i + 1, strict, &mut r); }
            None if i == last && !ends_nl => {
                r.unparseable += 1;
                r.torn_tail = true;
                r.push(i + 1, Severity::Warn, None, None,
                    format!("torn tail: last line is incomplete JSON ({} bytes, no trailing newline; an append in flight?)", line.len()));
            }
            None => {
                r.unparseable += 1;
                r.push(i + 1, Severity::Error, None, None, format!("unparseable JSON ({} bytes)", line.len()));
            }
        }
    }
    r
}

fn check_row(j: &Json, line: usize, strict: bool, r: &mut FileCheck) {
    if !matches!(j, Json::Obj(_)) {
        r.push(line, Severity::Error, None, None, format!("row is not a JSON object (got {})", type_of(j)));
        return;
    }
    let kind = match j.get("kind") {
        Some(Json::Str(k)) => k.as_str(),
        Some(v) => { r.push(line, Severity::Error, None, Some("kind"), format!("field kind: expected str, got {} {}", type_of(v), show(v))); return; }
        None => { r.push(line, Severity::Error, None, Some("kind"), "missing required field kind (str)".into()); return; }
    };
    let schema = lookup(kind);
    let strand = schema.map(|s| s.strand.name()).unwrap_or("unknown");
    r.kinds.entry(kind.to_string()).or_insert_with(|| KindCount { strand, ..Default::default() }).rows += 1;
    match schema {
        Some(s) => {
            if s.strand == Strand::Nuwa {
                if let Some(st) = j.str("strand").filter(|st| *st != "nuwa") {
                    r.push(line, Severity::Error, Some(kind), Some("strand"),
                        format!("kind {} belongs to strand nuwa, row says {:?}", kind, st));
                }
            }
            check_fields(j, s.all_fields(), kind, line, r);
        }
        None => {
            let sev = if strict { Severity::Error } else { Severity::Warn };
            r.push(line, sev, Some(kind), None, format!("unknown kind {:?}", kind));
            // Every Fuxi row carries the envelope, known kind or not.
            if j.str("strand") == Some("fuxi") { check_fields(j, FUXI_ENVELOPE.iter(), kind, line, r); }
        }
    }
}

fn check_fields<'a>(j: &Json, fields: impl Iterator<Item = &'a Field>, kind: &str, line: usize, r: &mut FileCheck) {
    let mut groups: Vec<(&str, bool)> = Vec::new();
    for f in fields {
        let raw = j.get(f.name);
        let v = raw.filter(|v| **v != Json::Null);
        if let Req::AnyOf(g) = f.req {
            match groups.iter_mut().find(|(n, _)| *n == g) { Some(e) => e.1 |= v.is_some(), None => groups.push((g, v.is_some())) }
        }
        let Some(v) = v else {
            if f.req == Req::Yes {
                let msg = if raw.is_some() { format!("required field {} is null", f.name) }
                    else { format!("missing required field {} ({})", f.name, f.ty.name()) };
                r.push(line, Severity::Error, Some(kind), Some(f.name), msg);
            }
            continue;
        };
        if !f.ty.accepts(v) {
            r.push(line, Severity::Error, Some(kind), Some(f.name),
                format!("field {}: expected {}, got {} {}", f.name, f.ty.name(), type_of(v), show(v)));
        } else if let (false, Json::Str(s)) = (f.known.is_empty(), v) {
            if !f.known.contains(&s.as_str()) {
                r.push(line, Severity::Warn, Some(kind), Some(f.name),
                    format!("field {}: unrecognised value {:?} (known: {})", f.name, s, f.known.join("|")));
            }
        }
    }
    for (g, present) in groups {
        if !present { r.push(line, Severity::Error, Some(kind), Some(g), format!("missing one of {}", g)); }
    }
}

// ---------------------------------------------------------- reports ----

/// `"error"` if any file has an error, `"warn"` if any has a warning, else `"ok"`.
pub fn status(files: &[(String, FileCheck)]) -> &'static str {
    if files.iter().any(|(_, c)| c.errors() > 0) { "error" }
    else if files.iter().any(|(_, c)| c.warnings() > 0) { "warn" } else { "ok" }
}

/// Per-kind counts over all files, most rows first.
pub fn merged_kinds(files: &[(String, FileCheck)]) -> Vec<(String, KindCount)> {
    let mut m: BTreeMap<String, KindCount> = BTreeMap::new();
    for (_, c) in files {
        for (k, n) in &c.kinds {
            let e = m.entry(k.clone()).or_insert_with(|| KindCount { strand: n.strand, ..Default::default() });
            e.rows += n.rows; e.errors += n.errors; e.warnings += n.warnings;
        }
    }
    let mut v: Vec<(String, KindCount)> = m.into_iter().collect();
    v.sort_by(|a, b| b.1.rows.cmp(&a.1.rows).then_with(|| a.0.cmp(&b.0)));
    v
}

fn totals(files: &[(String, FileCheck)]) -> [usize; 5] {
    let mut t = [0usize; 5];
    for (_, c) in files { t[0] += c.lines; t[1] += c.rows; t[2] += c.unparseable; t[3] += c.errors(); t[4] += c.warnings(); }
    t
}

/// Text report: per-kind counts, the first `max_issues` issues, one line
/// per file, and a summary line.
pub fn render_text(files: &[(String, FileCheck)], strict: bool, max_issues: usize) -> String {
    let mut s = String::new();
    let kinds = merged_kinds(files);
    let _ = writeln!(s, "  {:<16} {:<8} {:>7} {:>7} {:>7}", "kind", "strand", "rows", "errors", "warns");
    for (k, n) in &kinds {
        let _ = writeln!(s, "  {:<16} {:<8} {:>7} {:>7} {:>7}", k, n.strand, n.rows, n.errors, n.warnings);
    }
    let all: Vec<(&str, &Issue)> = files.iter().flat_map(|(p, c)| c.issues.iter().map(move |i| (p.as_str(), i))).collect();
    if all.is_empty() {
        s.push_str("  issues: none\n");
    } else {
        let _ = writeln!(s, "  issues (first {} of {}):", all.len().min(max_issues), all.len());
        for (p, i) in all.iter().take(max_issues) {
            let k = i.kind.as_deref().map(|k| format!(" [{}]", k)).unwrap_or_default();
            let _ = writeln!(s, "    {}:{}: {}{} {}", p, i.line, i.severity.name(), k, i.msg);
        }
    }
    for (p, c) in files {
        let _ = writeln!(s, "  file {}: {} lines, {} rows, {} unparseable, {} errors, {} warnings{}", p, c.lines, c.rows,
            c.unparseable, c.errors(), c.warnings(), if c.torn_tail { ", torn tail" } else { "" });
    }
    let [lines, rows, bad, errors, warnings] = totals(files);
    let verdict = match status(files) { "error" => "FAIL", "warn" => "OK (warnings)", _ => "OK" };
    let _ = writeln!(s, "summary: {} file(s), {} lines, {} rows, {} kinds, {} unparseable, {} errors, {} warnings{} -> {}",
        files.len(), lines, rows, kinds.len(), bad, errors, warnings, if strict { " (strict)" } else { "" }, verdict);
    s
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => { let _ = write!(o, "\\u{:04x}", c as u32); }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn opt_esc(s: Option<&str>) -> String { s.map(esc).unwrap_or_else(|| "null".into()) }

/// The same report as one JSON object (single line).
pub fn render_json(files: &[(String, FileCheck)], strict: bool, max_issues: usize) -> String {
    let mut s = String::new();
    let st = status(files);
    let [lines, rows, bad, errors, warnings] = totals(files);
    let _ = write!(s, "{{\"ok\":{},\"status\":\"{}\",\"strict\":{},\"lines\":{},\"rows\":{},\"unparseable\":{},\"errors\":{},\"warnings\":{},\"files\":[",
        st != "error", st, strict, lines, rows, bad, errors, warnings);
    for (n, (p, c)) in files.iter().enumerate() {
        let _ = write!(s, "{}{{\"path\":{},\"lines\":{},\"rows\":{},\"unparseable\":{},\"torn_tail\":{},\"errors\":{},\"warnings\":{}}}",
            if n > 0 { "," } else { "" }, esc(p), c.lines, c.rows, c.unparseable, c.torn_tail, c.errors(), c.warnings());
    }
    s.push_str("],\"kinds\":{");
    for (n, (k, c)) in merged_kinds(files).iter().enumerate() {
        let _ = write!(s, "{}{}:{{\"strand\":\"{}\",\"rows\":{},\"errors\":{},\"warnings\":{}}}",
            if n > 0 { "," } else { "" }, esc(k), c.strand, c.rows, c.errors, c.warnings);
    }
    let total: usize = files.iter().map(|(_, c)| c.issues.len()).sum();
    let _ = write!(s, "}},\"issues_total\":{},\"issues\":[", total);
    let all = files.iter().flat_map(|(p, c)| c.issues.iter().map(move |i| (p, i)));
    for (n, (p, i)) in all.take(max_issues).enumerate() {
        let _ = write!(s, "{}{{\"file\":{},\"line\":{},\"severity\":\"{}\",\"kind\":{},\"field\":{},\"msg\":{}}}",
            if n > 0 { "," } else { "" }, esc(p), i.line, i.severity.name().to_ascii_lowercase(),
            opt_esc(i.kind.as_deref()), opt_esc(i.field.as_deref()), esc(&i.msg));
    }
    s.push_str("]}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(c: &FileCheck) -> Vec<String> {
        c.issues.iter().map(|i| format!("{}:{} {}", i.line, i.severity.name(), i.msg)).collect()
    }

    #[test]
    fn table_is_consistent() {
        for (i, k) in SCHEMA.iter().enumerate() {
            assert!(SCHEMA[i + 1..].iter().all(|o| o.kind != k.kind), "duplicate kind {}", k.kind);
            let names: Vec<&str> = k.all_fields().map(|f| f.name).collect();
            for (j, n) in names.iter().enumerate() { assert!(!names[j + 1..].contains(n), "{}: duplicate field {}", k.kind, n); }
            for f in k.all_fields() {
                if let Req::AnyOf(g) = f.req {
                    assert!(g.split('|').any(|m| m == f.name), "{}.{} not in its group {}", k.kind, f.name, g);
                    assert!(g.split('|').all(|m| names.contains(&m)), "{}: group {} names a missing field", k.kind, g);
                }
                if let Ty::Const(lit) = f.ty { assert!(parse_json(lit).is_some(), "bad literal {}", lit); }
            }
            if k.strand == Strand::Fuxi { assert_eq!(&names[..3], &["v", "strand", "ts"]); }
        }
    }

    #[test]
    fn tsv_has_five_nonempty_columns() {
        let t = schema_tsv();
        let mut lines = t.lines();
        assert_eq!(lines.next(), Some("kind\tfield\ttype\trequired\tknown"));
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
        assert!(rows.iter().all(|r| r.len() == 5 && r.iter().all(|c| !c.is_empty())));
        assert!(rows.contains(&vec!["job", "rc", "int", "yes", "-"]));
        assert!(rows.contains(&vec!["deploy", "chunk", "num", "anyof:env|chunk", "-"]));
        assert!(rows.contains(&vec!["gym", "v", "const:1", "yes", "-"]));
        assert!(rows.contains(&vec!["gym", "strand", "const:\"fuxi\"", "yes", "-"]));
        assert!(rows.contains(&vec!["daemon", "phase", "str", "yes", "autogate|ship|protect|feed|run|tick|janitor"]));
    }

    #[test]
    fn types() {
        let n = |x: f64| Json::Num(x);
        assert!(Ty::Int.accepts(&n(-9.0)) && !Ty::Int.accepts(&n(1.5)));
        assert!(Ty::Const("1").accepts(&n(1.0)) && !Ty::Const("1").accepts(&n(2.0)));
        assert!(Ty::Const("\"fuxi\"").accepts(&Json::Str("fuxi".into())) && !Ty::Const("\"fuxi\"").accepts(&Json::Str("nuwa".into())));
        assert!(!Ty::Any.accepts(&Json::Null) && !Ty::Num.accepts(&Json::Str("1".into())));
    }

    const GOOD: &str = r#"{"kind":"job","job":"a.sh","start":10,"end":20,"rc":0}
{"kind":"prediction","ts":1,"pred":"p.tsv","name":"x","value":"3","op":">=","threshold":"2","verdict":"pass"}
{"kind":"work","ts":2,"run":"r1","config":"c","task":"t","sample":"s","verdict":"ok","meta":"m"}
{"kind":"daemon","ts":3,"phase":"tick","outcome":"idle","detail":""}
{"kind":"deploy","ts":4,"binary":"b","chunk":64,"verified":[]}
{"kind":"deploy","ts":5,"binary":"b","env":"X=1"}
{"kind":"window","ts":6,"event":"up"}
{"kind":"cal","ts":7,"ok":true,"code":{"tps":1.0}}
{"kind":"workbench","ts":8,"run":"w","outcome":"pass","live":null}
{"kind":"workbench","ts":9,"run":"w","outcome":"pass","live":{"land":1,"repairs":0}}
{"kind":"deploy_failure","ts":10}

{"kind":"autogate","ts":11}
{"v":1,"strand":"fuxi","kind":"heartbeat","ts":12}
{"kind":"queued","ts":13,"job":"301-ub128","pred_file":"301.tsv","names":["ub128-beats-ub256-d100000"],"designed_by":"fuxi"}
"#;

    #[test]
    fn good_ledger_is_clean() {
        let c = check(GOOD, true);
        assert_eq!(msgs(&c), Vec::<String>::new());
        assert_eq!((c.lines, c.rows, c.unparseable, c.torn_tail), (14, 14, 0, false));
        assert_eq!(c.kinds["queued"].rows, 1);
        assert_eq!(c.kinds["deploy"].rows, 2);
        assert_eq!(c.kinds["heartbeat"].strand, "fuxi");
    }

    #[test]
    fn missing_and_ill_typed_fields_cite_lines() {
        let t = "{\"kind\":\"job\",\"job\":\"a\",\"start\":1,\"end\":2}\n{\"kind\":\"job\",\"job\":\"a\",\"start\":1,\"end\":2,\"rc\":0.5}\n\
                 {\"kind\":\"deploy\",\"ts\":1,\"binary\":\"b\"}\n{\"kind\":\"prediction\",\"ts\":1,\"pred\":\"p\",\"name\":\"n\",\"value\":\"v\",\"verdict\":null}\n\
                 {\"kind\":\"workbench\",\"ts\":1,\"run\":\"w\",\"outcome\":\"o\",\"live\":\"yes\"}\n{\"ts\":1}\n[1]\n";
        let c = check(t, false);
        assert_eq!(msgs(&c), vec![
            "1:ERROR missing required field rc (int)",
            "2:ERROR field rc: expected int, got num 0.5",
            "3:ERROR missing one of env|chunk",
            "4:ERROR required field verdict is null",
            "5:ERROR field live: expected obj, got str \"yes\"",
            "6:ERROR missing required field kind (str)",
            "7:ERROR row is not a JSON object (got arr)",
        ]);
        assert_eq!(c.kinds["job"].errors, 2);
    }

    #[test]
    fn torn_tail_warns_but_mid_file_corruption_errors() {
        let torn = check("{\"kind\":\"window\",\"ts\":1,\"event\":\"up\"}\n{\"kind\":\"win", false);
        assert!(torn.torn_tail && torn.errors() == 0 && torn.warnings() == 1);
        assert_eq!(torn.issues[0].line, 2);
        // The same bad line followed by a newline is complete, so corrupt.
        let ended = check("{\"kind\":\"window\",\"ts\":1,\"event\":\"up\"}\n{\"kind\":\"win\n", false);
        assert!(!ended.torn_tail && ended.errors() == 1);
        let mid = check("{\"kind\":\"win\n{\"kind\":\"window\",\"ts\":1,\"event\":\"up\"}", false);
        assert_eq!(msgs(&mid), vec!["1:ERROR unparseable JSON (12 bytes)"]);
    }

    #[test]
    fn unknown_kinds_strands_and_phases() {
        let t = "{\"kind\":\"mystery\",\"ts\":1}\n{\"v\":2,\"strand\":\"fuxi\",\"kind\":\"score\",\"ts\":1}\n\
                 {\"v\":1,\"strand\":\"fuxi\",\"kind\":\"newthing\"}\n{\"kind\":\"daemon\",\"ts\":1,\"phase\":\"dance\",\"outcome\":\"ok\"}\n\
                 {\"kind\":\"window\",\"ts\":1,\"event\":\"up\",\"strand\":\"fuxi\"}\n";
        let lax = check(t, false);
        assert_eq!(msgs(&lax), vec![
            "1:WARN unknown kind \"mystery\"",
            "2:ERROR field v: expected const:1, got num 2",
            "3:WARN unknown kind \"newthing\"",
            "3:ERROR missing required field ts (num)",
            "4:WARN field phase: unrecognised value \"dance\" (known: autogate|ship|protect|feed|run|tick|janitor)",
            "5:ERROR kind window belongs to strand nuwa, row says \"fuxi\"",
        ]);
        let strict = check(t, true);
        assert_eq!(strict.issues[0].severity, Severity::Error);
        // An unknown daemon phase stays a warning even when strict.
        assert_eq!(strict.issues.iter().find(|i| i.line == 4).unwrap().severity, Severity::Warn);
        assert_eq!(lax.kinds["mystery"].strand, "unknown");
    }

    #[test]
    fn json_report_is_valid_json() {
        let files = vec![("a \"q\".jsonl".to_string(), check("{\"kind\":\"job\"}\n{\"kind\":\"x\",\"ts\":1}\n{bad", false))];
        let out = render_json(&files, false, 1);
        let j = parse_json(out.trim()).expect("report parses");
        assert_eq!(j.str("status"), Some("error"));
        assert_eq!(j.boolean("ok"), Some(false));
        assert_eq!(j.num("issues_total"), Some(6.0));
        let Some(Json::Arr(issues)) = j.get("issues") else { panic!() };
        assert_eq!(issues.len(), 1);
        assert_eq!(j.get("kinds").and_then(|k| k.get("job")).and_then(|k| k.num("errors")), Some(4.0));
        assert!(render_text(&files, false, 20).contains("-> FAIL"));
    }
}
