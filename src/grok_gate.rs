//! Grok gate: an online detector of a "compress + generalize" event, the growth gate.
//!
//! Feed it one measurement at a time: training accuracy (or any fit score in `[0, 1]`), accuracy on
//! a validation slice carved from the TRAINING data (never the test set), and a complexity measure
//! where lower means more compressed (squared weight norm, parameter count, description length,
//! ...). It fires once, the first time all three hold:
//!
//! 1. **fitted**: training accuracy `>= fit`;
//! 2. **generalized**: validation accuracy `>= val_ok` for `val_hold` consecutive measurements;
//! 3. **compressed**: complexity at most `1 - drop` of its running peak.
//!
//! Validated in [`crate::brain_growth`] (thresholds pre-registered there): it fires on grokking runs
//! and stays silent on runs that compress without generalizing (too little data) and on runs that
//! generalize-free memorize (no weight decay). Rule: grow capacity only after it fires.
//!
//! no_std, no heap, deterministic.

/// Thresholds of a [`GrokGate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateCfg {
    pub fit: f32,
    pub val_ok: f32,
    pub val_hold: u8,
    pub drop: f32,
}

impl GateCfg {
    /// The thresholds pre-registered in [`crate::brain_growth`].
    pub const DEFAULT: GateCfg = GateCfg { fit: 0.99, val_ok: 0.95, val_hold: 3, drop: 0.05 };
}

/// Online compress + generalize detector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrokGate {
    pub cfg: GateCfg,
    peak: f32,
    hold: u8,
    seen: u32,
    fired_at: Option<u32>,
}

impl GrokGate {
    pub const fn new(cfg: GateCfg) -> Self { GrokGate { cfg, peak: 0.0, hold: 0, seen: 0, fired_at: None } }

    /// Feed one measurement; `true` exactly once, at the measurement where the event fires.
    pub fn observe(&mut self, train: f32, val: f32, complexity: f32) -> bool {
        let i = self.seen;
        self.seen += 1;
        if complexity > self.peak { self.peak = complexity; }
        self.hold = if val >= self.cfg.val_ok { self.hold.saturating_add(1) } else { 0 };
        if self.fired_at.is_some() { return false; }
        let fire = train >= self.cfg.fit && self.hold >= self.cfg.val_hold && complexity <= (1.0 - self.cfg.drop) * self.peak;
        if fire { self.fired_at = Some(i); }
        fire
    }

    /// Index (0-based, in measurements) of the event, if it fired.
    pub fn fired_at(&self) -> Option<u32> { self.fired_at }
    /// The running peak of the complexity measure.
    pub fn peak(&self) -> f32 { self.peak }
}

/// Offline: index of the event in recorded series (same rule as feeding [`GrokGate::observe`]).
pub fn grok_event(cfg: GateCfg, train: &[f32], val: &[f32], complexity: &[f32]) -> Option<usize> {
    let mut g = GrokGate::new(cfg);
    let n = train.len().min(val.len()).min(complexity.len());
    (0..n).find(|&i| g.observe(train[i], val[i], complexity[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: GateCfg = GateCfg::DEFAULT;

    #[test]
    fn fires_on_compress_then_generalize() {
        let train = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let val = [0.0, 0.0, 0.2, 0.6, 0.96, 0.97, 0.98, 0.99];
        let cx = [10.0, 20.0, 20.0, 19.0, 18.5, 18.0, 18.0, 18.0];
        // val holds from index 4; third consecutive at 6; complexity 18.0 <= 0.95 * 20 = 19
        assert_eq!(grok_event(C, &train, &val, &cx), Some(6));
    }

    #[test]
    fn silent_without_compression() {
        let train = [1.0; 8];
        let val = [0.99; 8];
        let cx = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // growing complexity
        assert_eq!(grok_event(C, &train, &val, &cx), None);
    }

    #[test]
    fn silent_when_compressing_without_generalizing() {
        let train = [1.0; 8];
        let val = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5];
        let cx = [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0];
        assert_eq!(grok_event(C, &train, &val, &cx), None);
    }

    #[test]
    fn silent_before_fit_and_on_a_broken_streak() {
        let train = [0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let val = [0.99, 0.99, 0.99, 0.5, 0.99, 0.99];
        let cx = [20.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        assert_eq!(grok_event(C, &train, &val, &cx), None, "streak broken at 3, only 2 after");
    }

    #[test]
    fn fires_once() {
        let mut g = GrokGate::new(C);
        let mut n = 0;
        for i in 0..20 { if g.observe(1.0, 0.99, if i == 0 { 10.0 } else { 5.0 }) { n += 1; } }
        assert_eq!((n, g.fired_at()), (1, Some(2)));
    }
}
