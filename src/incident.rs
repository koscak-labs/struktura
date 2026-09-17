//! Incident records: groups temporally proximate alarms into inspectable
//! incidents with per-channel evidence, context, and reconstruction state.
//! [`IncidentBuilder`] folds [`AlarmReport`]s and [`Event`]s into
//! [`Incident`]s (alarms within `gap` ticks join; farther out starts a new
//! one), pulling in explanations via [`explain_alarm`] and context from a
//! [`ContextTimeline`].

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::autopilot::Event;
use crate::context::{ContextEvent, ContextTimeline};
use crate::monitor::{explain_alarm, AlarmReport, Leg};

/// Ticks to look back from an incident's start when pulling in context.
const CONTEXT_LOOKBACK: u64 = 20;

/// One piece of evidence: a single alarm's structured facts plus the
/// human-readable explanation of why it fired.
#[derive(Debug, Clone)]
pub struct Evidence {
    pub channel: usize,
    pub leg: Leg,
    pub tick: u64,
    pub observed: f64,
    pub threshold: f64,
    pub explanation: String,
}

impl Evidence {
    /// Convert a raw [`AlarmReport`] into evidence via [`explain_alarm`].
    pub fn from_alarm(report: &AlarmReport) -> Self {
        Self {
            channel: report.channel,
            leg: report.leg,
            tick: report.tick,
            observed: report.observed,
            threshold: report.threshold,
            explanation: String::from(explain_alarm(report)),
        }
    }
}

/// A group of temporally proximate alarms, with supporting evidence,
/// operating context, and (optionally) how it was resolved.
///
/// `Serialize` is hand-implemented for `Evidence`/`Incident` (below)
/// rather than derived: both embed [`Leg`], which does not derive
/// `Serialize` itself (`monitor.rs` is out of scope here), so a plain
/// `#[derive(Serialize)]` would not compile under the `serde` feature.
#[derive(Debug, Clone)]
pub struct Incident {
    pub id: u64,
    pub start_tick: u64,
    pub end_tick: u64,
    pub evidence: Vec<Evidence>,
    pub context: Vec<ContextEvent>,
    pub channels_involved: Vec<usize>,
    pub resolution: Option<String>,
    pub unresolved: Vec<String>,
}

impl Incident {
    fn new(id: u64, tick: u64) -> Self {
        Self {
            id,
            start_tick: tick,
            end_tick: tick,
            evidence: Vec::new(),
            context: Vec::new(),
            channels_involved: Vec::new(),
            resolution: None,
            unresolved: Vec::new(),
        }
    }

    /// Add evidence, extending the incident's tick span and channel set.
    pub fn add_evidence(&mut self, e: Evidence) {
        self.start_tick = self.start_tick.min(e.tick);
        self.end_tick = self.end_tick.max(e.tick);
        if !self.channels_involved.contains(&e.channel) {
            self.channels_involved.push(e.channel);
        }
        self.evidence.push(e);
    }

    /// Record a context annotation; does not affect the tick span.
    fn add_context(&mut self, e: ContextEvent) {
        self.context.push(e);
    }

    pub fn involves_channel(&self, ch: usize) -> bool {
        self.channels_involved.contains(&ch)
    }

    pub fn duration(&self) -> u64 {
        self.end_tick.saturating_sub(self.start_tick)
    }

    /// Hand-rolled JSON serialization (no serde dependency required).
    pub fn to_json(&self) -> String {
        let channels = json_arr(&self.channels_involved, |c| format!("{}", c));
        let evidence = json_arr(&self.evidence, |e| format!(
            "{{\"channel\":{},\"leg\":\"{:?}\",\"tick\":{},\"observed\":{},\"threshold\":{},\"explanation\":\"{}\"}}",
            e.channel, e.leg, e.tick, e.observed, e.threshold, json_escape(&e.explanation)));
        let context = json_arr(&self.context, |c| format!(
            "{{\"tick\":{},\"kind\":\"{}\",\"value\":\"{}\"}}",
            c.tick, json_escape(&c.kind), json_escape(&c.value)));
        let unresolved = json_arr(&self.unresolved, |u| format!("\"{}\"", json_escape(u)));
        let resolution = match &self.resolution {
            Some(r) => format!("\"{}\"", json_escape(r)),
            None => String::from("null"),
        };
        format!(
            "{{\"id\":{},\"start_tick\":{},\"end_tick\":{},\"channels_involved\":{},\"evidence\":{},\"context\":{},\"resolution\":{},\"unresolved\":{}}}",
            self.id, self.start_tick, self.end_tick, channels, evidence, context, resolution, unresolved)
    }
}

/// Render `items` as a JSON array, formatting each element with `f`.
fn json_arr<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&f(item));
    }
    out.push(']');
    out
}

/// Escape `"`, `\`, and control chars for a JSON string literal. Minimal
/// by design: incident text is internally generated, not untrusted input.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(feature = "serde")]
impl serde::Serialize for Evidence {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Evidence", 6)?;
        s.serialize_field("channel", &self.channel)?;
        s.serialize_field("leg", &format!("{:?}", self.leg))?;
        s.serialize_field("tick", &self.tick)?;
        s.serialize_field("observed", &self.observed)?;
        s.serialize_field("threshold", &self.threshold)?;
        s.serialize_field("explanation", &self.explanation)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Incident {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Incident", 8)?;
        s.serialize_field("id", &self.id)?;
        s.serialize_field("start_tick", &self.start_tick)?;
        s.serialize_field("end_tick", &self.end_tick)?;
        s.serialize_field("evidence", &self.evidence)?;
        s.serialize_field("context", &self.context)?;
        s.serialize_field("channels_involved", &self.channels_involved)?;
        s.serialize_field("resolution", &self.resolution)?;
        s.serialize_field("unresolved", &self.unresolved)?;
        s.end()
    }
}

/// Groups a stream of alarms (and related AutoPilot events) into
/// [`Incident`]s by temporal proximity.
#[derive(Debug, Clone)]
pub struct IncidentBuilder {
    gap: u64,
    next_id: u64,
    incidents: Vec<Incident>,
    /// Index into `incidents` of the incident new alarms/events attach to.
    active: Option<usize>,
}

impl Default for IncidentBuilder {
    fn default() -> Self {
        Self::new(10)
    }
}

impl IncidentBuilder {
    /// `gap`: max ticks between one alarm and the next in the same incident.
    pub fn new(gap: u64) -> Self {
        Self { gap, next_id: 0, incidents: Vec::new(), active: None }
    }

    fn starts_new_incident(&self, tick: u64) -> bool {
        match self.active.and_then(|idx| self.incidents.get(idx)) {
            Some(incident) => tick > incident.end_tick + self.gap,
            None => true,
        }
    }

    /// Join the active incident if `report` is within `gap` ticks, else
    /// start a new one. Returns the incident the alarm was added to.
    pub fn push_alarm(&mut self, report: &AlarmReport) -> &Incident {
        let tick = report.tick;
        if self.starts_new_incident(tick) {
            let id = self.next_id;
            self.next_id += 1;
            self.incidents.push(Incident::new(id, tick));
            self.active = Some(self.incidents.len() - 1);
        }
        let idx = self.active.expect("active incident set above");
        self.incidents[idx].add_evidence(Evidence::from_alarm(report));
        &self.incidents[idx]
    }

    /// Fold an AutoPilot [`Event`] into the active incident, if any.
    /// `RolledBack` carries a real [`AlarmReport`] and becomes genuine
    /// evidence; `Quarantined`/`Recalibrated` are AutoPilot decisions with
    /// no leg/observed/threshold, so they become context annotations.
    pub fn push_event(&mut self, event: &Event) {
        let Some(idx) = self.active else { return };
        let incident = &mut self.incidents[idx];
        match event {
            Event::Quarantined { tick, channel } => {
                incident.add_context(ContextEvent::new(*tick, "quarantined", format!("channel={}", channel)));
                incident.resolution = Some(format!("channel {} quarantined at tick {}", channel, tick));
            }
            Event::Recalibrated { tick } => {
                incident.add_context(ContextEvent::new(*tick, "recalibrated", ""));
                incident.resolution = Some(format!("recalibrated at tick {}", tick));
            }
            Event::RolledBack { tick, guard_report } => {
                incident.add_evidence(Evidence::from_alarm(guard_report));
                incident.add_context(ContextEvent::new(*tick, "rolled_back", ""));
                incident.unresolved.push(format!("adaptation rolled back at tick {}", tick));
            }
            Event::Alarm { .. } | Event::AdaptationStarted { .. } => {}
        }
    }

    /// Attach context events in `[start_tick - CONTEXT_LOOKBACK, end_tick]`
    /// from `timeline` to every incident built so far.
    pub fn attach_context(&mut self, timeline: &ContextTimeline) {
        for incident in &mut self.incidents {
            let lookback_start = incident.start_tick.saturating_sub(CONTEXT_LOOKBACK);
            for event in timeline.events_in(lookback_start, incident.end_tick) {
                incident.context.push(event.clone());
            }
        }
    }

    pub fn incidents(&self) -> &[Incident] {
        &self.incidents
    }

    /// Consume the builder, returning the completed incidents.
    pub fn finalize(self) -> Vec<Incident> {
        self.incidents
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(tick: u64, channel: usize) -> AlarmReport {
        AlarmReport { leg: Leg::LevelShift, channel, tick, observed: 4.2, threshold: 3.0, hit_gap: 0 }
    }

    #[test]
    fn from_alarm_converts_report_fields_and_fills_explanation() {
        let r = report(7, 2);
        let e = Evidence::from_alarm(&r);
        assert_eq!(e.channel, 2);
        assert_eq!(e.tick, 7);
        assert_eq!(e.leg, Leg::LevelShift);
        assert_eq!(e.observed, 4.2);
        assert_eq!(e.threshold, 3.0);
        assert!(!e.explanation.is_empty());
        assert_eq!(e.explanation, explain_alarm(&r));
    }

    #[test]
    fn two_alarms_within_gap_join_one_incident() {
        let mut b = IncidentBuilder::new(10);
        b.push_alarm(&report(100, 3));
        b.push_alarm(&report(105, 5));
        let incidents = b.finalize();
        assert_eq!(incidents.len(), 1);
        assert_eq!(incidents[0].evidence.len(), 2);
        assert_eq!(incidents[0].start_tick, 100);
        assert_eq!(incidents[0].end_tick, 105);
        assert_eq!(incidents[0].duration(), 5);
        assert!(incidents[0].involves_channel(3) && incidents[0].involves_channel(5));
        assert!(!incidents[0].involves_channel(9));
    }

    #[test]
    fn two_alarms_beyond_gap_form_two_incidents() {
        let mut b = IncidentBuilder::new(10);
        b.push_alarm(&report(100, 0));
        b.push_alarm(&report(200, 0));
        let incidents = b.finalize();
        assert_eq!(incidents.len(), 2);
        assert_eq!(incidents[0].evidence.len(), 1);
        assert_eq!(incidents[1].evidence.len(), 1);
        assert_eq!(incidents[0].id, 0);
        assert_eq!(incidents[1].id, 1);
    }

    #[test]
    fn push_event_folds_autopilot_events_into_active_incident() {
        let mut b = IncidentBuilder::new(10);
        b.push_event(&Event::Recalibrated { tick: 1 }); // no-op: no active incident
        assert!(b.incidents().is_empty());

        b.push_alarm(&report(100, 4));
        b.push_event(&Event::Quarantined { tick: 101, channel: 4 });
        b.push_event(&Event::RolledBack { tick: 104, guard_report: report(104, 4) });

        let incidents = b.finalize();
        assert_eq!(incidents[0].evidence.len(), 2); // original alarm + rollback guard
        assert_eq!(incidents[0].unresolved.len(), 1);
        assert!(incidents[0].resolution.is_some());
        assert_eq!(incidents[0].context.len(), 2); // quarantined + rolled_back notes
    }

    #[test]
    fn attach_context_pulls_events_in_lookback_window() {
        let mut b = IncidentBuilder::new(10);
        b.push_alarm(&report(100, 0));
        b.push_alarm(&report(105, 0));
        let mut timeline = ContextTimeline::new();
        timeline.push(ContextEvent::new(85, "mode", "before_window")); // 100-20=80<=85: in
        timeline.push(ContextEvent::new(70, "mode", "too_early")); // out of window
        timeline.push(ContextEvent::new(103, "mode", "mid_incident")); // in
        timeline.push(ContextEvent::new(200, "mode", "too_late")); // out
        b.attach_context(&timeline);
        let incidents = b.finalize();
        let values: Vec<&str> = incidents[0].context.iter().map(|c| c.value.as_str()).collect();
        assert!(values.contains(&"before_window"));
        assert!(values.contains(&"mid_incident"));
        assert!(!values.contains(&"too_early"));
        assert!(!values.contains(&"too_late"));
    }

    #[test]
    fn to_json_contains_expected_fields() {
        let mut b = IncidentBuilder::new(10);
        b.push_alarm(&report(100, 1));
        b.push_alarm(&report(102, 1));
        let mut timeline = ContextTimeline::new();
        timeline.push(ContextEvent::new(99, "mode", "cruise"));
        b.attach_context(&timeline);
        let mut incidents = b.finalize();
        let json = incidents[0].to_json();

        // Manual structural checks (no serde_json dependency): each
        // expected key appears with a plausible value, brackets balance.
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert_eq!(json.matches('[').count(), json.matches(']').count());
        assert!(json.contains("\"id\":0"));
        assert!(json.contains("\"start_tick\":100"));
        assert!(json.contains("\"end_tick\":102"));
        assert!(json.contains("\"channels_involved\":[1]"));
        assert!(json.contains("\"leg\":\"LevelShift\""));
        assert!(json.contains("\"explanation\":"));
        assert!(json.contains("\"context\":["));
        assert!(json.contains("cruise"));
        assert!(json.contains("\"resolution\":null"));
        assert!(json.contains("\"unresolved\":[]"));

        // json_escape: quotes/backslashes in generated text survive to_json.
        incidents[0].evidence[0].explanation = "say \"hi\"\\bye".into();
        assert!(incidents[0].to_json().contains("say \\\"hi\\\"\\\\bye"));
    }
}
