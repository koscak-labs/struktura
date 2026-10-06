//! Knob space: what can be changed, which measurement each knob actually
//! moves, how its values appear in job logs, and which values are feasible.
//!
//! File format, one knob per line (`#` comments):
//!
//! ```text
//! knob ub 128 256 512 1024 current=256 metric=d100000 dir=higher match=prefix:ub template=ub
//! knob chunk 0 32 64 128 current=64 metric=sum_s dir=lower match=group:contend alias=0:stock template=none
//! ```
//!
//! `match=prefix:P` finds value V in log arms named `P<V>`; `match=group:G` in
//! arms named `<V>` inside group G. `template=none` means no job instrument
//! exists yet for this knob: it is observed but never emitted.
//!
//! The values are a seed, not the whole space: a PROVEN win grows integer neighbours in the winning
//! direction (see `agenda::grow`). `grow=N` caps that per turn (default 2, max 2, 0 = frozen) and
//! `range=LO:HI` bounds it (default 0..unbounded); constraints and FALSIFIED/INVALID walls still apply.

#[derive(Clone, Debug, PartialEq)]
pub struct Knob {
    pub name: String,
    pub values: Vec<String>,
    pub current: String,
    /// The measurement this knob moves (a metric key in job logs).
    pub metric: String,
    pub lower_is_better: bool,
    /// `prefix:<p>` or `group:<g>`.
    pub matcher: String,
    /// value -> arm label overrides (e.g. 0 -> stock).
    pub aliases: Vec<(String, String)>,
    /// Job template id, or "none".
    pub template: String,
    /// What the metric means, for humans.
    pub what: String,
    /// Adaptive catalog: at most this many values the agenda may grow per turn from a PROVEN win
    /// (`grow=N`, default 2, hard cap 2; `grow=0` freezes the catalog).
    pub grow: usize,
    /// Allowed range for grown values (`range=LO:HI`); default 0 .. unbounded. Constraints still apply.
    pub range: (i64, i64),
}

impl Knob {
    /// (group, arm) under which `value` appears in job logs.
    pub fn arm_of(&self, value: &str) -> (String, String) {
        let label = self.aliases.iter().find(|(v, _)| v == value).map(|(_, a)| a.clone()).unwrap_or_else(|| value.to_string());
        match self.matcher.split_once(':') {
            Some(("group", g)) => (g.to_string(), label),
            Some(("prefix", p)) => (String::new(), format!("{}{}", p, label)),
            _ => (String::new(), label),
        }
    }
    pub fn has_instrument(&self) -> bool { self.template != "none" && !self.template.is_empty() }
}

pub fn parse_knobs(text: &str) -> Result<Vec<Knob>, String> {
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() { continue; }
        let mut it = line.split_whitespace();
        if it.next() != Some("knob") { return Err(format!("line {}: expected `knob <name> ...`", ln + 1)); }
        let name = it.next().ok_or(format!("line {}: knob needs a name", ln + 1))?.to_string();
        let mut k = Knob { name: name.clone(), values: Vec::new(), current: String::new(), metric: String::new(), lower_is_better: false,
            matcher: format!("prefix:{}", name), aliases: Vec::new(), template: "none".into(), what: String::new(), grow: 2, range: (0, i64::MAX) };
        for t in it {
            match t.split_once('=') {
                None => k.values.push(t.to_string()),
                Some(("current", v)) => k.current = v.into(),
                Some(("metric", v)) => k.metric = v.into(),
                Some(("dir", v)) => k.lower_is_better = v == "lower",
                Some(("match", v)) => k.matcher = v.into(),
                Some(("template", v)) => k.template = v.into(),
                Some(("what", v)) => k.what = v.replace('_', " "),
                Some(("grow", v)) => k.grow = v.parse::<usize>().map_err(|_| format!("line {}: grow={} is not a count", ln + 1, v))?.min(2),
                Some(("range", v)) => {
                    let r = v.split_once(':').and_then(|(a, b)| Some((a.parse::<i64>().ok()?, b.parse::<i64>().ok()?)));
                    match r { Some((a, b)) if a <= b => k.range = (a, b), _ => return Err(format!("line {}: range={} must be LO:HI integers", ln + 1, v)) }
                }
                Some(("alias", v)) => { if let Some((a, b)) = v.split_once(':') { k.aliases.push((a.into(), b.into())); } }
                Some((o, _)) => return Err(format!("line {}: unknown field {}", ln + 1, o)),
            }
        }
        if k.metric.is_empty() || k.current.is_empty() || k.values.len() < 2 {
            return Err(format!("line {}: knob {} needs >= 2 values, current= and metric=", ln + 1, name));
        }
        if !k.values.contains(&k.current) { return Err(format!("line {}: current={} is not one of the values", ln + 1, k.current)); }
        // predictions are NAMED <knob><value>-beats-...; the prefix is how the agenda finds them again (proven /
        // falsified / invalid). A prefix other than the name would make every result of this knob invisible.
        if let Some(p) = k.matcher.strip_prefix("prefix:") { if p != k.name { return Err(format!("line {}: knob {}: match=prefix:{} must equal the knob name (results are named {}<value>-beats-...)", ln + 1, k.name, p, k.name)); } }
        out.push(k);
    }
    Ok(out)
}

/// Built-in knob space for the raven lab. Each knob names the measurement it moves.
pub const BUILTIN: &str = "\
knob ub 128 256 512 1024 current=256 metric=d100000 dir=higher match=prefix:ub template=ub what=prefill_t/s_of_a_2048-token_chunk_at_100K_depth
knob draft 3 4 5 6 7 8 9 current=7 metric=code_tps dir=higher match=prefix:draft template=draft what=decode_t/s_on_code_with_the_DFlash_drafter,_thinking_as_in_prod
knob draftmin 0 1 2 3 current=0 metric=code_tps dir=higher match=prefix:draftmin template=draftmin what=decode_t/s_on_code_with_a_minimum_DFlash_draft_length_(live_default_0)
knob b 512 1024 2048 4096 current=2048 metric=d100000 dir=higher match=prefix:b template=bbench what=prefill_t/s_at_100K_depth_with_logical_batch_b_(live_default_2048)
knob chunk 0 32 64 128 248 current=64 metric=sum_s dir=lower match=group:contend alias=0:stock template=none what=time-to-done_of_two_overlapping_streams_(read+decode)
knob budget 1500 4000 12000 current=12000 metric=pass_rate dir=higher match=prefix:budget template=none what=workbench_verified_pass_rate_x_time-to-done,_paired_by_task
";

#[derive(Clone, Debug, PartialEq)]
pub struct Constraint { pub knob: String, pub op: String, pub value: f64, pub source: String }

/// Parse `ub<=256` / `ub <= 256` style constraints.
pub fn parse_constraint(spec: &str, source: &str) -> Option<Constraint> {
    let s: String = spec.split_whitespace().collect();
    for op in ["<=", ">=", "<", ">", "=="] {
        if let Some((k, v)) = s.split_once(op) {
            return Some(Constraint { knob: k.into(), op: op.into(), value: v.parse().ok()?, source: source.into() });
        }
    }
    None
}

/// Constraints in force: explicit ones override the ledger's, the ledger's override built-ins.
pub fn constraints(cli: &[Constraint], ledger: &[String]) -> Vec<Constraint> {
    let mut out: Vec<Constraint> = vec![Constraint { knob: "ub".into(), op: "<=".into(), value: 256.0,
        source: "built-in: prod VRAM ~23.8 of 24 GiB at ub256".into() }];
    for l in ledger { if let Some(c) = parse_constraint(l, "ledger") { out.retain(|x| x.knob != c.knob); out.push(c); } }
    for c in cli { out.retain(|x| x.knob != c.knob); out.push(c.clone()); }
    out
}

pub fn violates(c: &Constraint, value: &str) -> bool {
    let Ok(v) = value.parse::<f64>() else { return false };
    match c.op.as_str() { "<=" => v > c.value, "<" => v >= c.value, ">=" => v < c.value, ">" => v <= c.value, "==" => v != c.value, _ => false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_parses_with_metric_per_knob() {
        let k = parse_knobs(BUILTIN).unwrap();
        assert_eq!(k.len(), 6, "ub draft draftmin b chunk budget");
        let ub = &k[0];
        assert_eq!((ub.metric.as_str(), ub.lower_is_better, ub.has_instrument()), ("d100000", false, true));
        assert_eq!(ub.arm_of("512"), (String::new(), "ub512".into()));
        let chunk = k.iter().find(|k| k.name == "chunk").unwrap();
        assert!(chunk.lower_is_better && !chunk.has_instrument());
        assert_eq!(chunk.arm_of("0"), ("contend".into(), "stock".into()));
        assert_eq!(chunk.arm_of("64"), ("contend".into(), "64".into()));
    }

    #[test]
    fn bad_knob_lines_are_rejected() {
        assert!(parse_knobs("knob x 1 2 current=3 metric=m").is_err());
        assert!(parse_knobs("knob x 1 current=1 metric=m").is_err());
        assert!(parse_knobs("lever x 1 2").is_err());
    }

    #[test]
    fn ledger_constraint_overrides_builtin_and_cli_overrides_ledger() {
        let c = constraints(&[], &["ub <= 512".to_string()]);
        let ub = c.iter().find(|c| c.knob == "ub").unwrap();
        assert_eq!((ub.value, ub.source.as_str()), (512.0, "ledger"));
        assert!(!violates(ub, "512") && violates(ub, "1024"));
        let c = constraints(&[parse_constraint("ub<=128", "cli").unwrap()], &["ub <= 512".to_string()]);
        assert!(violates(c.iter().find(|c| c.knob == "ub").unwrap(), "256"));
    }

    #[test]
    fn a_prefix_that_is_not_the_knob_name_is_refused() {
        let e = parse_knobs("knob draftmin 0 1 current=0 metric=code_tps match=prefix:dmin template=draftmin").unwrap_err();
        assert!(e.contains("must equal the knob name"), "{e}");
        assert!(parse_knobs("knob draftmin 0 1 current=0 metric=code_tps match=prefix:draftmin template=draftmin").is_ok());
        assert!(parse_knobs("knob chunk 32 64 current=64 metric=sum_s match=group:contend template=none").is_ok(), "group matchers are free");
    }
}
