//! Operating context: column typing and a context timeline that pairs each
//! sample row with its active operating mode, command, and annotations.
//!
//! `ContextTimeline` holds mode/command/annotation events sorted by tick and
//! answers "what was active at tick T" in O(log n) via binary search — no
//! allocation on the lookup path. `ColumnSchema` classifies a CSV header into
//! measurement/command/mode/annotation/timestamp columns so callers can tell
//! signal from context without hardcoding column names per dataset.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// What kind of thing a column represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColumnRole {
    /// A sensed/physical quantity (e.g. `motor_current_A`).
    Measurement,
    /// A commanded setpoint or actuation (e.g. `cmd`, `command`).
    Command,
    /// An operating mode / state label (e.g. `mode`, `state`).
    Mode,
    /// Free-text annotation, not a measurement or a control signal.
    Annotation,
    /// The row's time axis.
    Timestamp,
}

/// One column's identity: name, role, and optional physical unit.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColumnSpec {
    pub name: String,
    pub role: ColumnRole,
    pub unit: Option<String>,
}

impl ColumnSpec {
    pub fn new(name: impl Into<String>, role: ColumnRole) -> Self {
        Self { name: name.into(), role, unit: None }
    }
}

/// A single context change: a mode switch, a command, a maintenance note —
/// anything that shifts how surrounding samples should be interpreted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextEvent {
    pub tick: u64,
    pub kind: String,
    pub value: String,
}

impl ContextEvent {
    pub fn new(tick: u64, kind: impl Into<String>, value: impl Into<String>) -> Self {
        Self { tick, kind: kind.into(), value: value.into() }
    }
}

/// A sorted-by-tick sequence of [`ContextEvent`]s with O(log n) point and
/// range lookups. Events sharing a tick keep insertion order among
/// themselves (stable sort on push).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextTimeline {
    events: Vec<ContextEvent>,
}

impl ContextTimeline {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Insert an event, keeping `events` sorted by tick (stable: ties keep
    /// relative insertion order, matching `Vec::sort_by_key`'s stability).
    pub fn push(&mut self, event: ContextEvent) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.tick);
    }

    /// The most recent event at or before `tick`, if any.
    ///
    /// Binary search over the sorted events for the partition point where
    /// `tick(event) <= tick` flips to false; the element just before that
    /// point (if any) is the answer.
    pub fn active_at(&self, tick: u64) -> Option<&ContextEvent> {
        let idx = self.events.partition_point(|e| e.tick <= tick);
        if idx == 0 {
            None
        } else {
            self.events.get(idx - 1)
        }
    }

    /// Events with `start <= tick <= end`, as a contiguous slice.
    pub fn events_in(&self, start: u64, end: u64) -> &[ContextEvent] {
        let lo = self.events.partition_point(|e| e.tick < start);
        let hi = self.events.partition_point(|e| e.tick <= end);
        if lo >= hi {
            &[]
        } else {
            &self.events[lo..hi]
        }
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn iter(&self) -> core::slice::Iter<'_, ContextEvent> {
        self.events.iter()
    }

    /// Parse a sidecar CSV of context events. Expects a header row (skipped)
    /// followed by rows where column `tick_col` parses as `u64`, `kind_col`
    /// and `value_col` are taken verbatim as strings.
    #[cfg(feature = "std")]
    pub fn from_csv(
        path: &str,
        tick_col: usize,
        kind_col: usize,
        value_col: usize,
    ) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {}", path, e))?;
        Self::from_csv_str(&text, tick_col, kind_col, value_col)
    }

    /// Same as [`Self::from_csv`] but parses an in-memory CSV string
    /// (header row included and skipped). Split out so tests don't touch
    /// the filesystem.
    #[cfg(feature = "std")]
    pub fn from_csv_str(
        text: &str,
        tick_col: usize,
        kind_col: usize,
        value_col: usize,
    ) -> Result<Self, String> {
        let mut timeline = Self::new();
        let need = tick_col.max(kind_col).max(value_col) + 1;
        for (lineno, line) in text.lines().enumerate().skip(1) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < need {
                return Err(format!(
                    "line {}: expected at least {} columns, got {}",
                    lineno + 1,
                    need,
                    cols.len()
                ));
            }
            let tick: u64 = cols[tick_col]
                .trim()
                .parse()
                .map_err(|e| format!("line {}: bad tick: {}", lineno + 1, e))?;
            timeline.push(ContextEvent::new(
                tick,
                cols[kind_col].trim(),
                cols[value_col].trim(),
            ));
        }
        Ok(timeline)
    }
}

/// A CSV header's column classification: which indices are measurements,
/// which carry mode/command context, which is the timestamp.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColumnSchema {
    pub columns: Vec<ColumnSpec>,
}

impl ColumnSchema {
    pub fn new(columns: Vec<ColumnSpec>) -> Self {
        Self { columns }
    }

    /// Auto-detect column roles from header names (case-insensitive):
    /// `time`/`timestamp` -> Timestamp; `mode`/`state` -> Mode;
    /// `cmd`/`command` -> Command; everything else -> Measurement.
    pub fn from_header(header: &[&str]) -> Self {
        let columns = header
            .iter()
            .map(|&name| {
                let lower = to_lower(name);
                let role = match lower.as_str() {
                    "time" | "timestamp" => ColumnRole::Timestamp,
                    "mode" | "state" => ColumnRole::Mode,
                    "cmd" | "command" => ColumnRole::Command,
                    _ => ColumnRole::Measurement,
                };
                ColumnSpec::new(name, role)
            })
            .collect();
        Self { columns }
    }

    pub fn indices_with_role(&self, role: ColumnRole) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.role == role)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn measurement_indices(&self) -> Vec<usize> {
        self.indices_with_role(ColumnRole::Measurement)
    }

    pub fn mode_indices(&self) -> Vec<usize> {
        self.indices_with_role(ColumnRole::Mode)
    }
}

/// Lowercase an ASCII-ish header name without pulling in std-only APIs
/// (works identically under `no_std` + alloc).
fn to_lower(s: &str) -> String {
    s.chars().flat_map(|c| c.to_lowercase()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_auto_detects_roles_from_header() {
        let header = ["timestamp", "motor_current_A", "wheel_rpm", "mode", "cmd"];
        let schema = ColumnSchema::from_header(&header);

        assert_eq!(schema.columns[0].role, ColumnRole::Timestamp);
        assert_eq!(schema.columns[1].role, ColumnRole::Measurement);
        assert_eq!(schema.columns[2].role, ColumnRole::Measurement);
        assert_eq!(schema.columns[3].role, ColumnRole::Mode);
        assert_eq!(schema.columns[4].role, ColumnRole::Command);

        assert_eq!(schema.measurement_indices(), vec![1, 2]);
        assert_eq!(schema.mode_indices(), vec![3]);
    }

    #[test]
    fn schema_case_insensitive_detection() {
        let header = ["Timestamp", "STATE", "Command"];
        let schema = ColumnSchema::from_header(&header);
        assert_eq!(schema.columns[0].role, ColumnRole::Timestamp);
        assert_eq!(schema.columns[1].role, ColumnRole::Mode);
        assert_eq!(schema.columns[2].role, ColumnRole::Command);
    }

    #[test]
    fn timeline_active_at_binary_search_correctness() {
        let mut tl = ContextTimeline::new();
        tl.push(ContextEvent::new(100, "mode", "cruise"));
        tl.push(ContextEvent::new(10, "mode", "boot"));
        tl.push(ContextEvent::new(50, "mode", "climb"));

        assert!(tl.active_at(5).is_none());
        assert_eq!(tl.active_at(10).unwrap().value, "boot");
        assert_eq!(tl.active_at(49).unwrap().value, "boot");
        assert_eq!(tl.active_at(50).unwrap().value, "climb");
        assert_eq!(tl.active_at(99).unwrap().value, "climb");
        assert_eq!(tl.active_at(100).unwrap().value, "cruise");
        assert_eq!(tl.active_at(1_000_000).unwrap().value, "cruise");
    }

    #[test]
    fn timeline_active_at_on_empty_timeline() {
        let tl = ContextTimeline::new();
        assert!(tl.active_at(0).is_none());
    }

    #[test]
    fn timeline_events_in_range() {
        let mut tl = ContextTimeline::new();
        for tick in [5u64, 15, 25, 35, 45] {
            tl.push(ContextEvent::new(tick, "cmd", "x"));
        }
        let mid = tl.events_in(10, 35);
        assert_eq!(mid.len(), 3);
        assert_eq!(mid[0].tick, 15);
        assert_eq!(mid[2].tick, 35);

        assert!(tl.events_in(1000, 2000).is_empty());
        assert_eq!(tl.events_in(0, 1000).len(), 5);
    }

    #[test]
    fn timeline_push_keeps_sorted_order_regardless_of_insertion_order() {
        let mut tl = ContextTimeline::new();
        tl.push(ContextEvent::new(3, "a", "3"));
        tl.push(ContextEvent::new(1, "a", "1"));
        tl.push(ContextEvent::new(2, "a", "2"));
        let ticks: Vec<u64> = tl.iter().map(|e| e.tick).collect();
        assert_eq!(ticks, vec![1, 2, 3]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_csv_str_parses_inline_csv() {
        let csv = "tick,kind,value\n0,mode,boot\n10,cmd,arm\n25,mode,cruise\n";
        let tl = ContextTimeline::from_csv_str(csv, 0, 1, 2).expect("parses");
        assert_eq!(tl.len(), 3);
        assert_eq!(tl.active_at(15).unwrap().value, "arm");
        assert_eq!(tl.active_at(25).unwrap().kind, "mode");
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_csv_str_rejects_short_rows() {
        let csv = "tick,kind,value\n0,mode\n";
        let err = ContextTimeline::from_csv_str(csv, 0, 1, 2).unwrap_err();
        assert!(err.contains("line 2"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_csv_reads_a_real_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("struktura_context_test.csv");
        std::fs::write(&path, "tick,kind,value\n0,mode,boot\n5,mode,cruise\n").unwrap();
        let tl = ContextTimeline::from_csv(path.to_str().unwrap(), 0, 1, 2).expect("parses");
        assert_eq!(tl.len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
