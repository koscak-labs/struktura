//! Regression tests for the parity-isolation redesign (sections A/B/C/E):
//! a full-mode cross-channel blame rule, escalation of a persistent
//! isolated spike to a quarantine, and a hold that stops an unresolved
//! ambiguous alarm from re-firing every `res_span` ticks and resetting
//! every other leg with it.
//!
//! Stream generators below port the judge's awk generators
//! (parity_plan.md HELPER section) byte-for-byte: same LCG, same
//! generation order per row, and every OUTPUT value rounded to 5 decimals
//! to match `printf "%.5f"` — the CSV text is what the real `struktura
//! guard` pipeline feeds the monitor, not awk's full-precision internal
//! accumulators (`s`, `cc` themselves stay full precision across rows,
//! exactly as awk keeps them).
//!
//! Each test's harness: calibrate on rows 0..1000, push rows 1000..3000
//! through AutoPilot with every channel valid (except T11, which marks one
//! channel invalid on a few rows). `row = 1000 + tick`.

use struktura::autopilot::{AutoPilot, Event};
use struktura::monitor::{HybridMonitor, Leg};

struct Lcg(u64);

impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0 * 16807 % 2_147_483_647;
        self.0 as f64 / 2_147_483_647.0
    }
    fn g(&mut self) -> f64 {
        self.u() + self.u() + self.u() + self.u() - 2.0
    }
}

/// Round to 5 decimals, matching `printf "%.5f"` (what the CSV pipeline
/// actually feeds the monitor).
fn round5(v: f64) -> f64 {
    format!("{v:.5}").parse().unwrap()
}

/// Two coupled channels [a, b]: a = s + .05g, b = 3s + .15g. `fault` is
/// (channel index 0|1, offset) applied from row 2000 onward.
fn stream_two(fault: Option<(usize, f64)>) -> Vec<Vec<f64>> {
    let mut lcg = Lcg(31);
    let mut s = 0.0f64;
    let mut out = vec![Vec::with_capacity(3000), Vec::with_capacity(3000)];
    for t in 0..3000usize {
        s = 0.98 * s + lcg.g();
        let mut v = [s + 0.05 * lcg.g(), 3.0 * s + 0.15 * lcg.g()];
        if t >= 2000 {
            if let Some((ch, d)) = fault {
                v[ch] += d;
            }
        }
        out[0].push(round5(v[0]));
        out[1].push(round5(v[1]));
    }
    out
}

/// Three channels [a, b, c]: a = s+.05g, b=3s+.15g, c=c_coef*s+.1g. `fault`
/// is (channel index 0..3, offset) applied from row 2000 onward.
fn stream_triplex(c_coef: f64, fault: Option<(usize, f64)>) -> Vec<Vec<f64>> {
    let mut lcg = Lcg(31);
    let mut s = 0.0f64;
    let mut out = vec![Vec::with_capacity(3000), Vec::with_capacity(3000), Vec::with_capacity(3000)];
    for t in 0..3000usize {
        s = 0.98 * s + lcg.g();
        let mut v = [s + 0.05 * lcg.g(), 3.0 * s + 0.15 * lcg.g(), c_coef * s + 0.1 * lcg.g()];
        if t >= 2000 {
            if let Some((ch, d)) = fault {
                v[ch] += d;
            }
        }
        for (o, vv) in out.iter_mut().zip(v.iter()) {
            o.push(round5(*vv));
        }
    }
    out
}

/// Four channels [a, b, c, d]: a=s+.05g, b=3s+.15g, c=2s+.1g, d=-s+.05g.
/// `stuck`: d freezes at its t=1499 value from t=1500 on. `drift`: c gets
/// +.002*(t-2000) from t=2000 on.
fn stream_four(stuck: bool, drift: bool) -> Vec<Vec<f64>> {
    let mut lcg = Lcg(31);
    let mut s = 0.0f64;
    let mut dz = 0.0f64;
    let mut out = vec![Vec::with_capacity(3000), Vec::with_capacity(3000), Vec::with_capacity(3000), Vec::with_capacity(3000)];
    for t in 0..3000usize {
        s = 0.98 * s + lcg.g();
        let a = s + 0.05 * lcg.g();
        let b = 3.0 * s + 0.15 * lcg.g();
        let mut c = 2.0 * s + 0.1 * lcg.g();
        let d_fresh = -s + 0.05 * lcg.g();
        let frozen = stuck && t >= 1500;
        let d = if frozen { dz } else { d_fresh };
        if !frozen {
            dz = d_fresh;
        }
        if drift && t >= 2000 {
            c += 0.002 * (t as f64 - 2000.0);
        }
        out[0].push(round5(a));
        out[1].push(round5(b));
        out[2].push(round5(c));
        out[3].push(round5(d));
    }
    out
}

/// Pair generator: a=s+.05g, b=3s+.15g (coupled), c independent AR(0.8).
/// b += db from row 2000; c += dc from row tc (no c fault if tc <= 0).
fn stream_pair(db: f64, dc: f64, tc: i64) -> Vec<Vec<f64>> {
    let mut lcg = Lcg(31);
    let (mut s, mut cc) = (0.0f64, 0.0f64);
    let mut out = vec![Vec::with_capacity(3000), Vec::with_capacity(3000), Vec::with_capacity(3000)];
    for t in 0..3000usize {
        s = 0.98 * s + lcg.g();
        cc = 0.8 * cc + lcg.g();
        let a = s + 0.05 * lcg.g();
        let mut b = 3.0 * s + 0.15 * lcg.g();
        let mut c = cc;
        if t >= 2000 {
            b += db;
        }
        if tc > 0 && (t as i64) >= tc {
            c += dc;
        }
        out[0].push(round5(a));
        out[1].push(round5(b));
        out[2].push(round5(c));
    }
    out
}

fn calibrate(data: &[Vec<f64>]) -> HybridMonitor {
    let calib: Vec<Vec<f64>> = data.iter().map(|c| c[..1000].to_vec()).collect();
    HybridMonitor::calibrate(&calib).expect("calibration")
}

/// T1: two coupled channels, one fault. Neither sensor is quarantined; the
/// alarm keeps today's tick, and gets class cross_channel_ambiguous.
#[test]
fn two_channel_parity_disagreement_quarantines_nobody() {
    for &(fault_ch, d) in &[(0usize, 0.5f64), (1usize, 1.5f64)] {
        let data = stream_two(Some((fault_ch, d)));
        let mon = calibrate(&data);
        let mut ap = AutoPilot::new(mon);
        let valid = [true, true];
        let mut sample = [0.0f64; 2];
        let mut quarantines = 0usize;
        let mut parity_alarms: Vec<(u64, &'static str)> = Vec::new();
        for tick in 0..2000u64 {
            let row = 1000 + tick as usize;
            sample[0] = data[0][row];
            sample[1] = data[1][row];
            for ev in ap.push(&sample, &valid) {
                match ev {
                    Event::Quarantined { .. } => quarantines += 1,
                    Event::Alarm { report, class, .. } if report.leg == Leg::Parity => {
                        parity_alarms.push((1000 + tick, class));
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(quarantines, 0, "fault ch{fault_ch}: {quarantines} quarantines");
        assert!(
            (1..=2).contains(&parity_alarms.len()),
            "fault ch{fault_ch}: {} parity alarms: {:?}",
            parity_alarms.len(),
            parity_alarms
        );
        assert!(
            parity_alarms.iter().all(|&(_, c)| c == "cross_channel_ambiguous"),
            "fault ch{fault_ch}: not all ambiguous: {:?}",
            parity_alarms
        );
        assert_eq!(parity_alarms[0].0, 2001, "fault ch{fault_ch}: first parity alarm at row {}", parity_alarms[0].0);
    }
}

/// T2: three channels, one coupling sign, one faulted. The blame rule must
/// quarantine the ACTUALLY faulted channel regardless of column order.
#[test]
fn triplex_parity_quarantines_the_faulty_sensor() {
    for &c_coef in &[2.0f64, -2.0f64] {
        for &(fault_ch, d) in &[(0usize, 0.5f64), (1usize, 1.5f64), (2usize, 1.0f64)] {
            let data = stream_triplex(c_coef, Some((fault_ch, d)));
            let mon = calibrate(&data);
            let mut ap = AutoPilot::new(mon);
            let valid = [true; 3];
            let mut sample = [0.0f64; 3];
            let mut quarantined: Option<(u64, usize)> = None;
            let mut alarm_channels: Vec<usize> = Vec::new();
            for tick in 0..2000u64 {
                let row = 1000 + tick as usize;
                for ch in 0..3 {
                    sample[ch] = data[ch][row];
                }
                for ev in ap.push(&sample, &valid) {
                    match ev {
                        Event::Quarantined { tick: qt, channel } => {
                            quarantined.get_or_insert((qt, channel));
                        }
                        Event::Alarm { report, .. } if report.leg == Leg::Parity => {
                            alarm_channels.push(report.channel);
                        }
                        _ => {}
                    }
                }
            }
            let (qt, qch) = quarantined
                .unwrap_or_else(|| panic!("c={c_coef} fault_ch={fault_ch}: no quarantine at all"));
            assert_eq!(qch, fault_ch, "c={c_coef} fault_ch={fault_ch}: quarantined ch{qch} instead");
            let row = 1000 + qt;
            assert!(row <= 2010, "c={c_coef} fault_ch={fault_ch}: quarantined at row {row}");
            assert!(
                alarm_channels.iter().all(|&ch| ch == fault_ch),
                "c={c_coef} fault_ch={fault_ch}: some parity alarm named a different channel: {:?}",
                alarm_channels
            );
        }
        // Control: F = none gives no events at all.
        let data = stream_triplex(c_coef, None);
        let mon = calibrate(&data);
        let mut ap = AutoPilot::new(mon);
        let valid = [true; 3];
        let mut sample = [0.0f64; 3];
        for tick in 0..2000u64 {
            let row = 1000 + tick as usize;
            for ch in 0..3 {
                sample[ch] = data[ch][row];
            }
            assert!(ap.push(&sample, &valid).is_empty(), "c={c_coef}: clean control raised an event at row {row}");
        }
    }
}

/// T3: a coupled pair (a, b) plus an unrelated third channel (c). A fault
/// on b must not be isolated — c is not independent evidence for the pair.
#[test]
fn coupled_pair_with_unrelated_channel_is_not_isolated() {
    let data = stream_pair(1.5, 0.0, 0);
    let mon = calibrate(&data);
    let mut ap = AutoPilot::new(mon);
    let valid = [true; 3];
    let mut sample = [0.0f64; 3];
    let mut quarantines = 0usize;
    let mut first_parity: Option<(u64, &'static str)> = None;
    for tick in 0..2000u64 {
        let row = 1000 + tick as usize;
        for ch in 0..3 {
            sample[ch] = data[ch][row];
        }
        for ev in ap.push(&sample, &valid) {
            match ev {
                Event::Quarantined { .. } => quarantines += 1,
                Event::Alarm { report, class, .. } if report.leg == Leg::Parity => {
                    first_parity.get_or_insert((tick, class));
                }
                _ => {}
            }
        }
    }
    assert_eq!(quarantines, 0, "{quarantines} quarantines");
    let (t, class) = first_parity.expect("a parity alarm must fire");
    assert_eq!(class, "cross_channel_ambiguous", "must be ambiguous, not isolated");
    assert_eq!(1000 + t, 2001, "first parity alarm at row {}", 1000 + t);
}

/// T4: an ambiguous alarm on the coupled pair must not blind the monitor to
/// an unrelated, later, real fault on the third channel.
#[test]
fn ambiguous_parity_alarm_does_not_blind_other_legs() {
    let data = stream_pair(5.0, 1.5, 2500);
    let mon = calibrate(&data);
    let mut ap = AutoPilot::new(mon);
    let valid = [true; 3];
    let mut sample = [0.0f64; 3];
    let mut parity_alarms = 0usize;
    let mut channel2_alarm: Option<u64> = None;
    for tick in 0..2000u64 {
        let row = 1000 + tick as usize;
        for ch in 0..3 {
            sample[ch] = data[ch][row];
        }
        for ev in ap.push(&sample, &valid) {
            if let Event::Alarm { report, .. } = &ev {
                if report.leg == Leg::Parity {
                    parity_alarms += 1;
                }
                if report.channel == 2 && (2500..3000).contains(&(row as u64)) {
                    channel2_alarm.get_or_insert(row as u64);
                }
            }
        }
    }
    assert!(parity_alarms <= 2, "{parity_alarms} parity alarms (must not storm)");
    assert!(channel2_alarm.is_some(), "channel 2's own fault (from row 2500) was never reported");
}

/// T6 (mutation guard): a slow drift first looks ambiguous (coupled with
/// the a/b pair's own noise), then gets isolated as more evidence
/// accumulates while it is held — not left ambiguous forever.
#[test]
fn drift_first_seen_ambiguous_is_isolated_when_evidence_grows() {
    let data = stream_four(false, true);
    let mon = calibrate(&data);
    let mut ap = AutoPilot::new(mon);
    let valid = [true; 4];
    let mut sample = [0.0f64; 4];
    let mut quarantines: Vec<(u64, usize)> = Vec::new();
    for tick in 0..2000u64 {
        let row = 1000 + tick as usize;
        for ch in 0..4 {
            sample[ch] = data[ch][row];
        }
        for ev in ap.push(&sample, &valid) {
            if let Event::Quarantined { tick: qt, channel } = ev {
                quarantines.push((1000 + qt, channel));
            }
        }
    }
    assert_eq!(quarantines.len(), 1, "expected exactly one quarantine: {:?}", quarantines);
    let (row, ch) = quarantines[0];
    assert_eq!(ch, 2, "quarantined channel {ch}, expected c (2)");
    assert!((2100..2200).contains(&row), "quarantined at row {row}, expected 2100..2200");
}

/// T11: a channel forward-filled (invalid) for a few rows must never be
/// blamed for the resulting apparent inconsistency; the real fault (b)
/// must still be isolated and quarantined shortly after.
#[test]
fn stale_forward_filled_channel_is_never_blamed() {
    let data = stream_triplex(2.0, Some((1, 1.5)));
    let mon = calibrate(&data);
    let mut ap = AutoPilot::new(mon);
    let mut sample = [0.0f64; 3];
    let mut quarantines: Vec<(u64, usize)> = Vec::new();
    let mut parity_alarm_channels: Vec<(u64, usize)> = Vec::new();
    for tick in 0..2000u64 {
        let row = 1000 + tick as usize;
        let c_valid = !(2000..=2002).contains(&(row as u64));
        sample[0] = data[0][row];
        sample[1] = data[1][row];
        sample[2] = if c_valid { data[2][row] } else { 0.0 };
        let valid = [true, true, c_valid];
        for ev in ap.push(&sample, &valid) {
            match ev {
                Event::Quarantined { tick: qt, channel } => quarantines.push((1000 + qt, channel)),
                Event::Alarm { report, .. } if report.leg == Leg::Parity => {
                    parity_alarm_channels.push((row as u64, report.channel));
                }
                _ => {}
            }
        }
    }
    assert!(
        !quarantines.iter().any(|&(_, ch)| ch == 2),
        "c must never be quarantined: {:?}",
        quarantines
    );
    assert!(
        !parity_alarm_channels.iter().any(|&(row, ch)| ch == 2 && (2000..2010).contains(&row)),
        "c must never be blamed for the forward-fill it was declared invalid during: {:?}",
        parity_alarm_channels
    );
    let b_q = quarantines.iter().find(|&&(_, ch)| ch == 1);
    assert!(b_q.is_some(), "b must be quarantined by row 2030: {:?}", quarantines);
    assert!(b_q.unwrap().0 <= 2030, "b quarantined at row {}, expected <= 2030", b_q.unwrap().0);
}
