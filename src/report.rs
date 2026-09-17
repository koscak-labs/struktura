//! Human-readable investigation report from incident records.
//! Designed to be printed to a terminal or written to a Markdown file.
#![cfg(feature = "std")]

use crate::incident::Incident;
use crate::replay::ReplayDiff;

/// Render a list of incidents as a readable investigation report: one
/// section per incident (evidence, context, unresolved items), followed by
/// a summary line and, if anything is unresolved, a "Next checks" section.
pub fn investigation_report(
    incidents: &[Incident],
    channel_names: &[&str],
    recording_samples: usize,
) -> String {
    let mut out = String::new();

    for inc in incidents {
        out.push_str(&format!(
            "Incident #{} (ticks {}\u{2013}{}, duration {} samples):\n",
            inc.id,
            inc.start_tick,
            inc.end_tick,
            inc.duration()
        ));

        for e in &inc.evidence {
            let name = channel_names.get(e.channel).copied().unwrap_or("channel?");
            let ratio = if e.threshold != 0.0 {
                e.observed / e.threshold
            } else {
                f64::INFINITY
            };
            out.push_str(&format!(
                "  - {}: {:?} fired at tick {} (observed/threshold = {:.2}), {}\n",
                name, e.leg, e.tick, ratio, e.explanation
            ));
        }

        if !inc.context.is_empty() {
            out.push_str("  context:\n");
            for c in &inc.context {
                out.push_str(&format!("    tick {}: {}={}\n", c.tick, c.kind, c.value));
            }
        }

        if !inc.unresolved.is_empty() {
            out.push_str("  unresolved:\n");
            for u in &inc.unresolved {
                out.push_str(&format!("    - {}\n", u));
            }
        }

        if let Some(r) = &inc.resolution {
            out.push_str(&format!("  resolution: {}\n", r));
        }

        out.push('\n');
    }

    let mut channels_seen: Vec<usize> = Vec::new();
    for inc in incidents {
        for &c in &inc.channels_involved {
            if !channels_seen.contains(&c) {
                channels_seen.push(c);
            }
        }
    }
    let unresolved_total: usize = incidents.iter().map(|i| i.unresolved.len()).sum();

    out.push_str(&format!(
        "{} incidents, {} channels involved, {} items unresolved, {} samples analyzed\n",
        incidents.len(),
        channels_seen.len(),
        unresolved_total,
        recording_samples
    ));

    if unresolved_total > 0 {
        out.push_str("\nNext checks:\n");
        for inc in incidents {
            for u in &inc.unresolved {
                out.push_str(&format!("  - incident #{}: {}\n", inc.id, u));
            }
        }
    }

    out
}

/// Render a [`ReplayDiff`] as a readable comparison report: a summary line,
/// then details for every missed and new incident, then per-incident
/// timing deltas.
pub fn replay_report(
    diff: &ReplayDiff,
    old_incidents: &[Incident],
    new_incidents: &[Incident],
) -> String {
    let mut out = format!(
        "Replay comparison: {} matched, {} missed, {} new\n",
        diff.matched.len(),
        diff.missed.len(),
        diff.new_alarms.len()
    );

    if !diff.missed.is_empty() {
        out.push_str("\nMissed (saved incidents not reproduced):\n");
        for id in &diff.missed {
            if let Some(inc) = old_incidents.iter().find(|i| i.id == *id) {
                out.push_str(&incident_line(inc));
            }
        }
    }

    if !diff.new_alarms.is_empty() {
        out.push_str("\nNew (alarms not in the saved case):\n");
        for id in &diff.new_alarms {
            if let Some(inc) = new_incidents.iter().find(|i| i.id == *id) {
                out.push_str(&incident_line(inc));
            }
        }
    }

    if !diff.timing_deltas.is_empty() {
        out.push_str("\nTiming deltas:\n");
        for (id, delta) in &diff.timing_deltas {
            let sign = if *delta >= 0 { "+" } else { "" };
            out.push_str(&format!("  - incident #{}: {}{} ticks\n", id, sign, delta));
        }
    }

    out
}

fn incident_line(inc: &Incident) -> String {
    format!(
        "  - incident #{} (ticks {}\u{2013}{}, channels {:?})\n",
        inc.id, inc.start_tick, inc.end_tick, inc.channels_involved
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ContextEvent;
    use crate::incident::Evidence;
    use crate::monitor::Leg;

    fn incident_with_unresolved(id: u64) -> Incident {
        Incident {
            id,
            start_tick: 100,
            end_tick: 105,
            evidence: vec![Evidence {
                channel: 0,
                leg: Leg::LevelShift,
                tick: 100,
                observed: 6.0,
                threshold: 3.0,
                explanation: "level shift detected".to_string(),
            }],
            context: vec![ContextEvent::new(99, "mode", "cruise")],
            channels_involved: vec![0],
            resolution: None,
            unresolved: vec!["adaptation rolled back at tick 104".to_string()],
        }
    }

    #[test]
    fn investigation_report_includes_evidence_and_next_checks() {
        let incidents = vec![incident_with_unresolved(0)];
        let names = ["motor_current_A"];
        let report = investigation_report(&incidents, &names, 1000);

        assert!(report.contains("Incident #0"));
        assert!(report.contains("motor_current_A: LevelShift fired at tick 100"));
        assert!(report.contains("observed/threshold = 2.00"));
        assert!(report.contains("mode=cruise"));
        assert!(report.contains("1 incidents, 1 channels involved, 1 items unresolved"));
        assert!(report.contains("Next checks:"));
        assert!(report.contains("incident #0: adaptation rolled back at tick 104"));
    }

    #[test]
    fn investigation_report_skips_next_checks_when_nothing_unresolved() {
        let mut inc = incident_with_unresolved(0);
        inc.unresolved.clear();
        let report = investigation_report(&[inc], &[], 100);
        assert!(!report.contains("Next checks"));
    }

    #[test]
    fn replay_report_lists_missed_and_new() {
        let old = vec![incident_with_unresolved(0)];
        let mut new_inc = incident_with_unresolved(0);
        new_inc.id = 1;
        let new = vec![new_inc];
        let diff = ReplayDiff {
            matched: vec![],
            missed: vec![0],
            new_alarms: vec![1],
            timing_deltas: vec![],
        };
        let report = replay_report(&diff, &old, &new);
        assert!(report.starts_with("Replay comparison: 0 matched, 1 missed, 1 new"));
        assert!(report.contains("Missed"));
        assert!(report.contains("incident #0"));
        assert!(report.contains("New"));
        assert!(report.contains("incident #1"));
    }
}
