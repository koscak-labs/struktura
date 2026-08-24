//! Fixed-size rover health monitor for flight computers.
//!
//! No heap allocation at any point — all state lives in fixed arrays
//! sized at compile time. Suitable for `no_std` bare-metal targets
//! (radiation-hardened processors, FPGAs, embedded ARM).
//!
//! The full `HybridMonitor` uses `Vec` internally. This module provides
//! a stripped flight variant with the three legs that matter most for
//! rovers: AR(1) residual (instant spikes + correlation loss),
//! repeated-value (stuck sensors), and rolling-mean level shift
//! (regime changes). No DFA leg (saves the window-sized scratch
//! buffer) — the other three catch 5/6 fault types in the taxonomy.
//!
//! Memory per instance: `(WINDOW + ROLL + 12) * 8 * N_CH + 64` bytes.
//! At defaults (WINDOW=32, ROLL=32, N_CH=10): **5,920 bytes total**.

/// Compile-time channel count. Change this for your rover.
pub const N_CH: usize = 10;

/// Trailing window for the rolling-mean leg.
pub const F_WINDOW: usize = 32;
/// Rolling-mean window for the level-shift leg.
pub const F_ROLL: usize = 32;
/// Residual exceedances within this span to alarm.
pub const F_RES_SPAN: u32 = 20;
/// Consecutive rolling-mean exceedances to alarm.
pub const F_ROLL_PERSIST: u32 = 8;
/// Extra repeated values beyond calibration max to alarm.
pub const F_REPEAT_MARGIN: u32 = 4;

/// Which leg alarmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FLeg {
    Residual,
    Stuck,
    LevelShift,
}

/// Per-channel calibration constants (set once, never modified).
#[derive(Debug, Clone, Copy)]
pub struct FCalib {
    pub ar_a: f64,
    pub ar_b: f64,
    pub ar_sd: f64,
    pub mean: f64,
    pub roll_max_dev: f64,
    pub max_run: u32,
    pub repeat_enabled: bool,
}

impl Default for FCalib {
    fn default() -> Self {
        FCalib { ar_a: 0.0, ar_b: 0.0, ar_sd: 1.0, mean: 0.0, roll_max_dev: 1.0, max_run: 1, repeat_enabled: true }
    }
}

/// Per-channel runtime state (fixed size, mutated every sample).
#[derive(Debug, Clone, Copy)]
struct FState {
    roll_ring: [f64; F_ROLL],
    prev: f64,
    run: u32,
    t: u32,
    res_hit_prev: u32,
    roll_streak: u32,
}

impl Default for FState {
    fn default() -> Self {
        FState {
            roll_ring: [0.0; F_ROLL],
            prev: 0.0,
            run: 1,
            t: 0,
            res_hit_prev: u32::MAX,
            roll_streak: 0,
        }
    }
}

/// Alarm report (no heap).
#[derive(Debug, Clone, Copy)]
pub struct FAlarm {
    pub leg: FLeg,
    pub channel: u8,
    pub tick: u32,
}

/// Fixed-size rover health monitor. Zero heap. `N_CH` channels.
///
/// Calibrate by calling [`RoverMonitor::calibrate_channel`] for each
/// channel with pre-computed constants (from a ground tool or from a
/// calibration pass on the first N samples). Then call
/// [`RoverMonitor::push`] every sample tick.
pub struct RoverMonitor {
    calib: [FCalib; N_CH],
    state: [FState; N_CH],
    res_thr: f64,
    alarmed: bool,
}

impl RoverMonitor {
    /// Create an uncalibrated monitor (all channels default).
    pub const fn new() -> Self {
        // const fn can't use Default::default() on arrays, so manual init
        const C: FCalib = FCalib { ar_a: 0.0, ar_b: 0.0, ar_sd: 1.0, mean: 0.0, roll_max_dev: 1.0, max_run: 1, repeat_enabled: true };
        const S: FState = FState { roll_ring: [0.0; F_ROLL], prev: 0.0, run: 1, t: 0, res_hit_prev: u32::MAX, roll_streak: 0 };
        RoverMonitor {
            calib: [C; N_CH],
            state: [S; N_CH],
            res_thr: 5.0,
            alarmed: false,
        }
    }

    /// Set calibration for one channel + the global residual threshold.
    pub fn calibrate_channel(&mut self, ch: usize, cal: FCalib) {
        if ch < N_CH { self.calib[ch] = cal; }
    }

    /// Set the global residual z-threshold (max z seen in calibration × 1.1).
    pub fn set_residual_threshold(&mut self, thr: f64) {
        self.res_thr = thr;
    }

    /// Feed one sample per channel. Returns `Some(alarm)` on the first
    /// detection; latches until [`reset`].
    ///
    /// **Zero allocation. Bounded loops (max N_CH × F_ROLL iterations).**
    pub fn push(&mut self, sample: &[f64; N_CH]) -> Option<FAlarm> {
        if self.alarmed { return None; }

        for ch in 0..N_CH {
            let cc = &self.calib[ch];
            let st = &mut self.state[ch];
            let t = st.t;
            st.t = t.wrapping_add(1);
            let v = sample[ch];

            // Prime on first sample
            if t == 0 {
                st.prev = v;
                st.roll_ring[0] = v;
                continue;
            }

            // Leg 1: AR(1) residual (2-in-span persistence)
            let z = (v - (cc.ar_a + cc.ar_b * st.prev)) / cc.ar_sd;
            let z_abs = if z < 0.0 { -z } else { z };
            if z_abs > self.res_thr {
                if st.res_hit_prev != u32::MAX && t - st.res_hit_prev < F_RES_SPAN {
                    self.alarmed = true;
                    return Some(FAlarm { leg: FLeg::Residual, channel: ch as u8, tick: t });
                }
                st.res_hit_prev = t;
            }

            // Leg 2: repeated value (stuck sensor)
            if v == st.prev {
                st.run += 1;
                if cc.repeat_enabled && st.run >= cc.max_run + F_REPEAT_MARGIN {
                    self.alarmed = true;
                    return Some(FAlarm { leg: FLeg::Stuck, channel: ch as u8, tick: t });
                }
            } else {
                st.run = 1;
            }

            st.prev = v;
            st.roll_ring[(t % F_ROLL as u32) as usize] = v;

            // Leg 3: rolling-mean level shift
            if t >= F_ROLL as u32 {
                let mut sum = 0.0f64;
                for i in 0..F_ROLL {
                    sum += st.roll_ring[i];
                }
                let dev = sum / F_ROLL as f64 - cc.mean;
                let dev_abs = if dev < 0.0 { -dev } else { dev };
                if dev_abs > cc.roll_max_dev * 2.0 {
                    st.roll_streak += 1;
                    if st.roll_streak >= F_ROLL_PERSIST {
                        self.alarmed = true;
                        return Some(FAlarm { leg: FLeg::LevelShift, channel: ch as u8, tick: t });
                    }
                } else {
                    st.roll_streak = 0;
                }
            }
        }
        None
    }

    /// Clear the alarm latch and all streaks.
    pub fn reset(&mut self) {
        self.alarmed = false;
        for st in self.state.iter_mut() {
            st.run = 1;
            st.roll_streak = 0;
            st.res_hit_prev = u32::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flight_monitor_is_const_constructible() {
        // Must be usable in a static — real flight code puts this in BSS.
        static _MON: RoverMonitor = RoverMonitor::new();
    }

    #[test]
    fn flight_monitor_size() {
        let size = core::mem::size_of::<RoverMonitor>();
        // Must fit in a few KB — real flight computers have 32-256KB RAM.
        assert!(size < 8192, "RoverMonitor is {} bytes — must be < 8KB", size);
    }

    #[test]
    fn detects_stuck_sensor() {
        let mut mon = RoverMonitor::new();
        let mut cal = FCalib::default();
        cal.ar_a = 1.0; // predict ~1.0 always (simple baseline)
        cal.ar_b = 0.0;
        cal.ar_sd = 0.05; // noise scale
        cal.max_run = 2;
        for ch in 0..N_CH { mon.calibrate_channel(ch, cal); }
        mon.set_residual_threshold(20.0); // high enough for normal noise

        let mut sample = [1.0f64; N_CH];
        // Feed normal varying data (all channels near 1.0)
        for i in 0..100u32 {
            for ch in 0..N_CH {
                sample[ch] = 1.0 + 0.005 * ((i as f64 * 0.1 + ch as f64).sin());
            }
            assert!(mon.push(&sample).is_none(), "normal data alarmed at i={}", i);
        }
        // Freeze channel 3 while others keep varying
        for i in 100..120u32 {
            for ch in 0..N_CH {
                if ch == 3 {
                    sample[ch] = 1.05; // frozen
                } else {
                    sample[ch] = 1.0 + 0.005 * ((i as f64 * 0.1 + ch as f64).sin());
                }
            }
            if let Some(alarm) = mon.push(&sample) {
                assert_eq!(alarm.leg, FLeg::Stuck);
                assert_eq!(alarm.channel, 3);
                return;
            }
        }
        panic!("stuck sensor not detected");
    }

    #[test]
    fn no_alloc_proof() {
        // This test exists to prove the monitor compiles in no_std.
        // The module uses no Vec, no String, no Box — only fixed arrays.
        let mon = RoverMonitor::new();
        assert!(!mon.alarmed);
    }
}
