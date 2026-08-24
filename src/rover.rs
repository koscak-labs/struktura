//! Rover health monitoring — structural analysis for planetary exploration.
//!
//! Models the real subsystems of a Mars/Moon rover and the failure modes
//! that kill missions: wheel motor degradation, thermal runaway, battery
//! cell failure, suspension asymmetry, and communication fade.
//!
//! The key insight for rovers: you can't phone home for help. Mars has
//! 8-22 minute light delay; Moon has ~1.3 seconds but limited bandwidth.
//! The rover must detect, isolate, and adapt to faults AUTONOMOUSLY —
//! which is exactly what the AutoPilot + HybridMonitor stack does.
//!
//! ```
//! use struktura::rover::{RoverSim, RoverFault};
//! let mut sim = RoverSim::new(42);
//! sim.inject(1500, RoverFault::WheelBearing { wheel: 2, severity: 0.8 });
//! let telemetry = sim.run(3000);
//! // Feed to guard: struktura guard rover_telemetry.csv --baseline 1000
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Rover telemetry channels (matches real rover subsystems).
pub const ROVER_CHANNELS: usize = 10;
pub const ROVER_CHANNEL_NAMES: [&str; ROVER_CHANNELS] = [
    "wheel_fl_current",  // front-left wheel motor current (A)
    "wheel_fr_current",  // front-right
    "wheel_rl_current",  // rear-left
    "wheel_rr_current",  // rear-right
    "suspension_tilt",   // rocker-bogie tilt angle (deg)
    "battery_voltage",   // main bus voltage (V)
    "battery_soc",       // state of charge (0-1)
    "thermal_cpu",       // CPU temperature (°C)
    "thermal_motor_avg", // average motor temperature (°C)
    "comm_signal",       // downlink signal strength (dBm)
];

/// Faults that kill rover missions.
#[derive(Debug, Clone)]
pub enum RoverFault {
    /// Wheel bearing wear — vibration and current draw increase gradually.
    /// This is the #1 mechanical failure on Mars rovers (Spirit's right
    /// front wheel failed on sol 779).
    WheelBearing { wheel: usize, severity: f64 },
    /// Wheel motor stall — sudden overcurrent, RPM drops to zero.
    WheelStall { wheel: usize },
    /// Battery cell degradation — one cell's internal resistance rises,
    /// voltage sags under load, SOC readings become unreliable.
    BatteryCell { severity: f64 },
    /// Thermal runaway — CPU or motor temperature climbs due to blocked
    /// vent or failed heater controller.
    ThermalRunaway { channel: usize, rate: f64 },
    /// Communication fade — signal strength drops gradually (antenna
    /// misalignment, dust on the dish, orbital geometry).
    CommFade { rate: f64 },
    /// Suspension asymmetry — one side rides higher (stuck actuator,
    /// terrain wedge). Changes the cross-channel relationship between
    /// wheel currents — the parity leg's specialty.
    SuspensionAsymmetry { side: usize, offset: f64 },
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.max(1) } }
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller
        let u1 = (self.next() + 0.5).max(1e-10);
        let u2 = self.next() + 0.5;
        mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// Rover telemetry simulator with injectable faults.
pub struct RoverSim {
    rng: Rng,
    faults: Vec<(usize, RoverFault)>, // (start_sample, fault)
}

impl RoverSim {
    pub fn new(seed: u64) -> Self {
        RoverSim { rng: Rng::new(seed), faults: Vec::new() }
    }

    /// Schedule a fault to begin at sample `at`.
    pub fn inject(&mut self, at: usize, fault: RoverFault) {
        self.faults.push((at, fault));
    }

    /// Generate `n` samples of 10-channel rover telemetry.
    pub fn run(&mut self, n: usize) -> Vec<Vec<f64>> {
        let mut channels = vec![Vec::with_capacity(n); ROVER_CHANNELS];

        // State variables
        let mut soc = 0.92f64;
        let mut cpu_temp = -20.0f64; // Mars surface: -60 to +20°C
        let mut motor_temps = [15.0f64; 4];
        let mut comm = -85.0f64; // nominal signal strength

        for t in 0..n {
            let terrain = 0.3 * crate::sin(t as f64 * 0.01) + 0.1 * crate::sin(t as f64 * 0.037);
            let drive_load = 1.0 + 0.3 * terrain.abs();

            // Wheel currents: coupled through terrain + suspension
            let mut wheel_currents = [0.0f64; 4];
            for w in 0..4 {
                let side_bias = if w < 2 { terrain * 0.2 } else { -terrain * 0.2 };
                wheel_currents[w] = 0.8 * drive_load + side_bias + self.rng.normal(0.0, 0.03);
            }

            // Suspension tilt: follows terrain
            let tilt = terrain * 5.0 + self.rng.normal(0.0, 0.1);

            // Battery: slow discharge, voltage coupled to SOC + load
            soc = (soc - 0.00003 * drive_load + self.rng.normal(0.0, 0.00001)).clamp(0.1, 1.0);
            let voltage = 24.0 + 4.0 * soc - 0.5 * drive_load + self.rng.normal(0.0, 0.02);

            // Thermal: CPU tracks computational load, motors track current
            let cpu_target = -15.0 + 10.0 * (0.5 + 0.5 * crate::sin(t as f64 * 0.005));
            cpu_temp += 0.02 * (cpu_target - cpu_temp) + self.rng.normal(0.0, 0.05);
            for w in 0..4 {
                let target = 10.0 + 15.0 * wheel_currents[w];
                motor_temps[w] += 0.015 * (target - motor_temps[w]) + self.rng.normal(0.0, 0.1);
            }
            let motor_avg = motor_temps.iter().sum::<f64>() / 4.0;

            // Comm signal: orbital variation
            let orbital = -85.0 + 3.0 * crate::sin(t as f64 * 0.002);
            comm = orbital + self.rng.normal(0.0, 0.3);

            // Apply faults
            for (start, fault) in &self.faults {
                if t < *start { continue; }
                let dt = (t - start) as f64;
                match fault {
                    RoverFault::WheelBearing { wheel, severity } => {
                        let w = *wheel % 4;
                        let progress = (dt / 500.0).min(1.0) * severity;
                        // Bearing wear: current draw increases + vibration noise
                        wheel_currents[w] += progress * 0.5;
                        wheel_currents[w] += progress * 0.3 * self.rng.next();
                        motor_temps[w] += progress * 3.0;
                    }
                    RoverFault::WheelStall { wheel } => {
                        let w = *wheel % 4;
                        if dt < 5.0 {
                            wheel_currents[w] *= 3.0; // overcurrent spike
                        } else {
                            wheel_currents[w] = 0.01; // stalled
                        }
                    }
                    RoverFault::BatteryCell { severity } => {
                        let progress = (dt / 800.0).min(1.0) * severity;
                        soc -= progress * 0.0002;
                        // Voltage sags more under load as internal resistance rises
                        let extra_sag = progress * 1.5 * drive_load;
                        channels[5].last_mut().map(|v| *v -= extra_sag);
                    }
                    RoverFault::ThermalRunaway { channel, rate } => {
                        if *channel == 7 {
                            cpu_temp += rate * dt * 0.01;
                        }
                    }
                    RoverFault::CommFade { rate } => {
                        comm -= rate * dt * 0.01;
                    }
                    RoverFault::SuspensionAsymmetry { side, offset } => {
                        let progress = (dt / 200.0).min(1.0);
                        if *side == 0 {
                            wheel_currents[0] += progress * offset;
                            wheel_currents[1] += progress * offset;
                        } else {
                            wheel_currents[2] += progress * offset;
                            wheel_currents[3] += progress * offset;
                        }
                    }
                }
            }

            channels[0].push(wheel_currents[0]);
            channels[1].push(wheel_currents[1]);
            channels[2].push(wheel_currents[2]);
            channels[3].push(wheel_currents[3]);
            channels[4].push(tilt);
            channels[5].push(voltage);
            channels[6].push(soc);
            channels[7].push(cpu_temp);
            channels[8].push(motor_avg);
            channels[9].push(comm);
        }
        channels
    }

    /// Write telemetry to CSV with headers.
    #[cfg(feature = "std")]
    pub fn write_csv(&mut self, n: usize, path: &str) -> std::io::Result<()> {
        let data = self.run(n);
        let mut out = String::new();
        out.push_str(&ROVER_CHANNEL_NAMES.join(","));
        out.push('\n');
        for t in 0..n {
            for ch in 0..ROVER_CHANNELS {
                if ch > 0 { out.push(','); }
                out.push_str(&format!("{:.4}", data[ch][t]));
            }
            out.push('\n');
        }
        std::fs::write(path, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rover_sim_produces_10_channels() {
        let mut sim = RoverSim::new(42);
        let data = sim.run(500);
        assert_eq!(data.len(), ROVER_CHANNELS);
        assert_eq!(data[0].len(), 500);
    }

    #[test]
    fn rover_sim_is_deterministic() {
        let mut a = RoverSim::new(77);
        let mut b = RoverSim::new(77);
        assert_eq!(a.run(200)[3], b.run(200)[3]);
    }

    #[test]
    fn wheel_bearing_increases_current() {
        let mut clean = RoverSim::new(42);
        let clean_data = clean.run(2000);
        let clean_mean: f64 = clean_data[0][1000..].iter().sum::<f64>() / 1000.0;

        let mut faulted = RoverSim::new(42);
        faulted.inject(500, RoverFault::WheelBearing { wheel: 0, severity: 1.0 });
        let fault_data = faulted.run(2000);
        let fault_mean: f64 = fault_data[0][1000..].iter().sum::<f64>() / 1000.0;

        assert!(fault_mean > clean_mean + 0.1,
            "bearing fault must increase current: clean={:.3} fault={:.3}",
            clean_mean, fault_mean);
    }

    #[test]
    fn wheel_stall_spikes_then_drops() {
        let mut sim = RoverSim::new(42);
        sim.inject(500, RoverFault::WheelStall { wheel: 1 });
        let data = sim.run(600);
        // Spike in the first few samples after fault
        assert!(data[1][502] > 2.0, "stall must spike current");
        // Then drops to near zero
        assert!(data[1][510] < 0.1, "stalled motor draws no current");
    }

    #[test]
    fn guard_catches_bearing_fault() {
        use crate::monitor::HybridMonitor;
        let mut sim = RoverSim::new(99);
        sim.inject(1500, RoverFault::WheelBearing { wheel: 2, severity: 0.9 });
        let data = sim.run(3000);
        let calib: Vec<Vec<f64>> = data.iter().map(|c| c[..1000].to_vec()).collect();
        let mut mon = HybridMonitor::calibrate(&calib).expect("calibrate");
        let mut alarm_at = None;
        let mut sample = [0.0f64; ROVER_CHANNELS];
        for t in 1000..3000 {
            for ch in 0..ROVER_CHANNELS { sample[ch] = data[ch][t]; }
            if mon.push(&sample).is_some() {
                alarm_at = Some(t);
                break;
            }
        }
        let t = alarm_at.expect("bearing fault must be detected");
        assert!(t < 2000, "detection at {} should be well before end", t);
    }
}
