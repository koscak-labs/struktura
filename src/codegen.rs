/// The box-size table + `dfa_compute`, built from `struktura::dfa_box_sizes`
/// so the boxes (and therefore alpha) always match Rust's `dfa()` on the same
/// `window`, not a list good for one fixed window. `dfa_compute` sizes its
/// `log_s`/`log_f` scratch from `DFA_NUM_BOXES`/`DFA_BOX_SLOTS`, which the
/// caller must `#define` before splicing this in (both call sites below do).
/// It takes `values` as non-`const` and **overwrites it with the cumulative
/// profile** rather than using a separate `double y[DFA_WINDOW_SIZE]` local
/// (an 8-byte-per-sample stack frame that reached about 33 KB at `--window
/// 4096`, measured with `gcc -fstack-usage`): every caller passes a
/// caller-owned scratch array that it rebuilds from its ring on every call
/// (`ordered`) and does not read again afterward, so overwriting it in place
/// is safe. Shared by [`generate_c_monitor`] (embedded in the self-contained
/// monitor) and [`generate_dfa_core_h`] (a standalone header for the cFS/F
/// Prime/ROS directory generators), so there is exactly one DFA
/// implementation in the generated C, not a second one that can drift from
/// the tested one.
fn dfa_compute_block(window: usize) -> String {
    let (sizes, count) = crate::dfa_box_sizes(window);
    let boxes = if count == 0 {
        "0".to_string()
    } else {
        let list: Vec<String> = sizes[..count].iter().map(|b| b.to_string()).collect();
        list.join(", ")
    };
    let box_slots = count.max(1);
    format!(
        r#"#define DFA_NUM_BOXES   {num_boxes}
#define DFA_BOX_SLOTS   {box_slots}

/* The box sizes struktura's Rust dfa() uses for DFA_WINDOW_SIZE samples. */
static const int DFA_BOXES[DFA_BOX_SLOTS] = {{{boxes}}};

typedef struct {{
    double alpha;
    double r_squared;
}} dfa_result_t;

static dfa_result_t dfa_compute(double *values, int n) {{
    dfa_result_t result = {{0.5, 0.0}};
    if (n < 64) return result;

    double mean = 0.0;
    int i, seg, b;
    for (i = 0; i < n; i++) mean += values[i];
    mean /= (double)n;

    /* Cumulative profile, computed in place: `values` is scratch the caller
       rebuilds on every call and does not read again afterward, so this
       overwrites it instead of using a second window-sized array. */
    double cum = 0.0, ss = 0.0;
    for (i = 0; i < n; i++) {{
        cum += values[i] - mean;
        values[i] = cum;
        ss += cum * cum;
    }}
    /* A box whose F is at most 1e-12 of the profile's RMS measures only
       rounding (a box inside a constant run) and gives no point of the fit,
       as struktura's FLAT_BOX_REL. */
    double floor2 = 1e-12 * 1e-12 * (ss / (double)n);

    /* DFA: measure fluctuation at each box size */
    double log_s[DFA_BOX_SLOTS], log_f[DFA_BOX_SLOTS];
    int pts = 0;

    for (b = 0; b < DFA_NUM_BOXES && DFA_BOXES[b] <= n / 4; b++) {{
        int s = DFA_BOXES[b];
        int num_segs = n / s;
        if (num_segs == 0) continue;
        double f2_sum = 0.0;
        for (seg = 0; seg < num_segs; seg++) {{
            int start = seg * s;
            double sx = 0, sy = 0, sxy = 0, sx2 = 0;
            for (i = 0; i < s; i++) {{
                double xi = (double)i;
                sx += xi;
                sy += values[start + i];
                sxy += xi * values[start + i];
                sx2 += xi * xi;
            }}
            double k = (double)s;
            double det = k * sx2 - sx * sx;
            if (fabs(det) < 1e-15) continue;
            double a0 = (sx2 * sy - sx * sxy) / det;
            double a1 = (k * sxy - sx * sy) / det;
            double resid = 0.0;
            for (i = 0; i < s; i++) {{
                double d = values[start + i] - (a0 + a1 * (double)i);
                resid += d * d;
            }}
            f2_sum += resid / k;
        }}
        double f2 = f2_sum / (double)num_segs;
        if (f2 > floor2) {{
            log_s[pts] = log((double)s);
            log_f[pts] = log(sqrt(f2));
            pts++;
        }}
    }}

    if (pts < 3) return result;

    /* Log-log linear regression */
    double k = (double)pts;
    double sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (i = 0; i < pts; i++) {{
        sx += log_s[i]; sy += log_f[i];
        sxy += log_s[i] * log_f[i]; sx2 += log_s[i] * log_s[i];
    }}
    double slope = (k * sxy - sx * sy) / (k * sx2 - sx * sx);
    double ic = (sy - slope * sx) / k;
    double ym = sy / k;
    double sst = 0, ssr = 0;
    for (i = 0; i < pts; i++) {{
        sst += (log_f[i] - ym) * (log_f[i] - ym);
        ssr += (log_f[i] - slope * log_s[i] - ic) * (log_f[i] - slope * log_s[i] - ic);
    }}
    result.alpha = slope;
    result.r_squared = 1.0 - ssr / (sst > 1e-15 ? sst : 1e-15);
    return result;
}}
"#,
        num_boxes = count,
        box_slots = box_slots,
        boxes = boxes
    )
}

/// Generate `dfa_core.h`: a standalone DFA-only header (box table +
/// `dfa_compute`, no ring buffer) for the cFS/F Prime/ROS directory
/// generators (`struktura generate --cfs|--fprime|--ros`). Shares the exact
/// `dfa_compute` body [`generate_c_monitor`] uses, built from
/// `struktura::dfa_box_sizes(window)`, so its alpha matches Rust's `dfa()`
/// for whatever `--window` was requested, not just one baked-in size.
pub fn generate_dfa_core_h(window: usize) -> String {
    format!(
        r#"/* dfa_core.h -- DFA scaling analysis (compute only)
 * Generated by struktura. Zero dependencies beyond <math.h>.
 * Peng et al., Physical Review E 49(2), 1994.
 * https://github.com/koscak-labs/struktura
 */

#ifndef DFA_CORE_H
#define DFA_CORE_H

#include <math.h>

#define DFA_WINDOW_SIZE {window}

{core}
#endif /* DFA_CORE_H */
"#,
        window = window,
        core = dfa_compute_block(window)
    )
}

pub fn generate_c_monitor(window_size: usize, threshold: f64) -> String {
    // The box sizes the Rust DFA uses for this window, so a threshold set from
    // a Rust baseline (`dfa` on healthy data) means the same thing on board.
    let core = dfa_compute_block(window_size);
    format!(r#"/* dfa_monitor.c -- Generated by struktura codegen
 * Self-contained DFA structural health monitor.
 * Compile: gcc -Wall -Werror -O2 -lm -o dfa_monitor dfa_monitor.c
 * Zero dependencies. Pre-80s discipline.
 * https://github.com/koscak-labs/struktura
 */

#include <math.h>
#include <string.h>

#define DFA_WINDOW_SIZE {window}
#define DFA_THRESHOLD   {threshold:.4}

{core}

typedef struct {{
    double buffer[DFA_WINDOW_SIZE];
    double ordered[DFA_WINDOW_SIZE]; /* the window in time order, for dfa_compute */
    int pos;
    int filled;
    double baseline_alpha;
    int baseline_set;
    int learning_count;
    dfa_result_t last; /* the alpha/r_squared the most recent push actually used */
}} dfa_monitor_t;

/* Initialize monitor */
static void dfa_monitor_init(dfa_monitor_t *m) {{
    memset(m, 0, sizeof(*m));
}}

/* Push a sample. Returns: 0=learning, 1=healthy, 2=watch, 3=warning, 4=critical */
static int dfa_monitor_push(dfa_monitor_t *m, double value) {{
    m->buffer[m->pos] = value;
    m->pos = (m->pos + 1) % DFA_WINDOW_SIZE;
    if (m->pos == 0) m->filled = 1;
    if (!m->filled) return 0;

    m->learning_count++;
    /* The ring holds the oldest sample at pos; DFA needs the window in time order. */
    int i;
    for (i = 0; i < DFA_WINDOW_SIZE; i++)
        m->ordered[i] = m->buffer[(m->pos + i) % DFA_WINDOW_SIZE];
    dfa_result_t r = dfa_compute(m->ordered, DFA_WINDOW_SIZE);
    m->last = r;

    /* Learning phase: first 10 windows establish baseline */
    if (!m->baseline_set && m->learning_count >= 10 && r.r_squared > 0.7) {{
        m->baseline_alpha = r.alpha;
        m->baseline_set = 1;
        return 0;
    }}
    if (!m->baseline_set) return 0;

    /* Health check */
    double shift = fabs(r.alpha - m->baseline_alpha);
    if (shift < DFA_THRESHOLD * 0.375) return 1; /* healthy */
    if (shift < DFA_THRESHOLD)         return 2; /* watch */
    if (shift < DFA_THRESHOLD * 1.875) return 3; /* warning */
    return 4; /* critical */
}}
"#, window = window_size, threshold = threshold, core = core)
}

pub fn generate_fprime_component(name: &str, window_size: usize) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str(&format!("// {}.fpp -- Generated F Prime DFA health monitor\n", name));
    s.push_str("// Generated by: struktura codegen --fprime\n\n");
    s.push_str("module Svc {\n");
    s.push_str(&format!("    passive component {} {{\n\n", name));
    s.push_str("        sync input port schedIn: Svc.Sched\n");
    s.push_str("        guarded input port tlmIn: Fw.Tlm\n\n");
    s.push_str("        event StructuralShift(\n");
    s.push_str("            channelId: FwChanIdType\n");
    s.push_str("            baseline_alpha: F64\n");
    s.push_str("            current_alpha: F64\n");
    s.push_str("            delta: F64\n");
    s.push_str("        ) severity warning high\n\n");
    s.push_str("        event BaselineEstablished(\n");
    s.push_str("            channelId: FwChanIdType\n");
    s.push_str("            alpha: F64\n");
    s.push_str("            r_squared: F64\n");
    s.push_str("        ) severity activity high\n\n");
    s.push_str("        telemetry DfaAlpha: F64\n");
    s.push_str("        telemetry DfaR2: F64\n\n");
    s.push_str("        time get port timeCaller\n");
    s.push_str("        event port logOut\n");
    s.push_str("        telemetry port tlmOut\n");
    s.push_str("    }\n}\n\n");
    s.push_str(&format!("// Link with libstruktura.a, window size: {}\n", window_size));
    s.push_str("// https://github.com/koscak-labs/struktura\n");
    s
}

/// Generate an F Prime rover health component with the flight monitor.
///
/// 10-channel rover subsystems (4 wheels, suspension, battery V+SOC,
/// thermal CPU+motors, comms). Three detection legs (residual, stuck,
/// level shift). Autonomous quarantine events. No heap allocation.
pub fn generate_fprime_rover() -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("// RoverHealth.fpp -- F Prime autonomous health monitor\n");
    s.push_str("// Generated by: struktura generate --fprime --rover\n");
    s.push_str("// 10 channels, 3 detection legs, no heap. ~6KB RAM.\n\n");
    s.push_str("module Rover {\n");
    s.push_str("    passive component RoverHealth {\n\n");

    // Ports
    s.push_str("        // Rate-group driven: call schedIn every sample tick\n");
    s.push_str("        sync input port schedIn: Svc.Sched\n\n");

    // 10 telemetry input ports (one per rover channel)
    let channels = [
        ("wheelFL", "Front-left wheel motor current (A)"),
        ("wheelFR", "Front-right wheel motor current (A)"),
        ("wheelRL", "Rear-left wheel motor current (A)"),
        ("wheelRR", "Rear-right wheel motor current (A)"),
        ("suspTilt", "Rocker-bogie tilt angle (deg)"),
        ("batVoltage", "Main bus voltage (V)"),
        ("batSOC", "Battery state of charge (0-1)"),
        ("thermCPU", "CPU temperature (C)"),
        ("thermMotor", "Average motor temperature (C)"),
        ("commSignal", "Downlink signal strength (dBm)"),
    ];
    for (name, doc) in &channels {
        s.push_str(&format!("        @ {}\n", doc));
        s.push_str(&format!("        guarded input port {}: Fw.Tlm\n", name));
    }
    s.push('\n');

    // Events (human-readable, matches explain_alarm output)
    s.push_str("        @ Gradual drift detected on a channel\n");
    s.push_str("        event Drift(channel: string size 20) severity warning high\n\n");
    s.push_str("        @ Sensor appears stuck (same value repeating)\n");
    s.push_str("        event SensorStuck(channel: string size 20) severity warning high\n\n");
    s.push_str("        @ Signal shifted to a new operating level\n");
    s.push_str("        event LevelShift(channel: string size 20) severity warning high\n\n");
    s.push_str("        @ Channel quarantined, using reconstructed values\n");
    s.push_str("        event Quarantined(channel: string size 20) severity warning high\n\n");
    s.push_str("        @ Environment changed, learning new baseline\n");
    s.push_str("        event Adapting() severity activity high\n\n");
    s.push_str("        @ New baseline accepted\n");
    s.push_str("        event Recalibrated() severity activity high\n\n");
    s.push_str("        @ Adaptation rejected, fault confirmed\n");
    s.push_str("        event FaultConfirmed(channel: string size 20) severity warning high\n\n");

    // Telemetry
    for (name, _) in &channels {
        s.push_str(&format!("        telemetry {}_health: U8  @ 0=ok 1=warning 2=quarantined\n", name));
    }
    s.push('\n');

    // Standard ports
    s.push_str("        time get port timeCaller\n");
    s.push_str("        event port logOut\n");
    s.push_str("        telemetry port tlmOut\n\n");

    s.push_str("    }\n}\n\n");
    s.push_str("// Implementation: link with rover_flight.rs compiled as staticlib.\n");
    s.push_str("// RoverMonitor::new() is const, lives in BSS, zero init cost.\n");
    s.push_str("// RAM: ~6KB for 10 channels. No heap. Bounded worst-case per tick.\n");
    s.push_str("// Calibration constants baked at build time or loaded from EEPROM.\n");
    s.push_str("// https://github.com/koscak-labs/struktura\n");
    s
}


pub fn generate_cfs_app(name: &str, window_size: usize) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str(&format!("/* {}_app.c -- Generated cFS DFA health monitor app\n", name.to_lowercase()));
    s.push_str(" * Generated by: struktura codegen --cfs\n");
    s.push_str(" * Link with libstruktura.a or embed dfa_monitor.c\n */\n\n");
    s.push_str("#include \"cfe.h\"\n");
    s.push_str("#include \"struktura.h\"\n\n");
    s.push_str(&format!("#define {}_WINDOW_SIZE {}\n\n", name.to_uppercase(), window_size));
    s.push_str("typedef struct {\n");
    s.push_str(&format!("    double buffer[{}_WINDOW_SIZE];\n", name.to_uppercase()));
    s.push_str(&format!(
        "    double ordered[{}_WINDOW_SIZE]; /* the window in time order, for struktura_dfa */\n",
        name.to_uppercase()
    ));
    s.push_str("    uint32 pos;\n");
    s.push_str("    uint32 filled;\n");
    s.push_str("    double baseline;\n");
    s.push_str("    uint8 baseline_set;\n");
    s.push_str(&format!("}} {}_Data_t;\n\n", name));
    s.push_str(&format!("static {}_Data_t {}_Data;\n\n", name, name));
    s.push_str(&format!("void {}_Init(void) {{\n", name));
    s.push_str(&format!("    memset(&{}_Data, 0, sizeof({}_Data));\n", name, name));
    s.push_str("    CFE_EVS_SendEvent(1, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str(&format!("        \"{} DFA health monitor initialized (window={})\");\n", name, window_size));
    s.push_str("}\n\n");
    s.push_str(&format!("void {}_ProcessSample(double value) {{\n", name));
    s.push_str(&format!("    {}_Data.buffer[{}_Data.pos] = value;\n", name, name));
    s.push_str(&format!("    {}_Data.pos = ({}_Data.pos + 1) % {}_WINDOW_SIZE;\n", name, name, name.to_uppercase()));
    s.push_str(&format!("    if ({}_Data.pos == 0) {}_Data.filled = 1;\n", name, name));
    s.push_str(&format!("    if (!{}_Data.filled) return;\n\n", name));
    // The ring holds the oldest sample at pos; DFA needs the window in time order.
    s.push_str("    uint32 i;\n");
    s.push_str(&format!("    for (i = 0; i < {}_WINDOW_SIZE; i++)\n", name.to_uppercase()));
    s.push_str(&format!(
        "        {n}_Data.ordered[i] = {n}_Data.buffer[({n}_Data.pos + i) % {w}_WINDOW_SIZE];\n",
        n = name,
        w = name.to_uppercase()
    ));
    s.push_str(&format!("    struktura_dfa_result_t r = struktura_dfa({}_Data.ordered, {}_WINDOW_SIZE);\n", name, name.to_uppercase()));
    s.push_str(&format!("    if (!{}_Data.baseline_set && r.r_squared > 0.7) {{\n", name));
    s.push_str(&format!("        {}_Data.baseline = r.alpha;\n", name));
    s.push_str(&format!("        {}_Data.baseline_set = 1;\n", name));
    s.push_str("        CFE_EVS_SendEvent(2, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str("            \"DFA baseline established: alpha=%.3f R2=%.4f\", r.alpha, r.r_squared);\n");
    s.push_str("        return;\n    }\n");
    s.push_str(&format!("    if (!{}_Data.baseline_set) return;\n\n", name));
    s.push_str(&format!("    uint8_t verdict = struktura_health_check(r.alpha, {}_Data.baseline);\n", name));
    s.push_str("    if (verdict >= STRUKTURA_WARNING) {\n");
    s.push_str("        CFE_EVS_SendEvent(3, CFE_EVS_EventType_ERROR,\n");
    s.push_str(&format!("            \"{} structural shift: alpha=%.3f baseline=%.3f verdict=%d\",\n", name));
    s.push_str(&format!("            r.alpha, {}_Data.baseline, verdict);\n", name));
    s.push_str("    }\n}\n\n");
    s.push_str("// See: https://github.com/koscak-labs/struktura\n");
    s
}

/// Generate a self-contained C hybrid monitor with a calibrated
/// configuration baked in: every threshold learned from real calibration
/// data, no magic numbers. C99, no dependencies beyond libm, static
/// memory only, bounded loops only (flight-software discipline).
///
/// Compile: `gcc -std=c99 -Wall -Werror -O2 -o hybrid hybrid_monitor.c -lm`.
/// Define `HYBRID_STANDALONE_TEST` for a built-in stuck-sensor self-test.
pub fn generate_hybrid_c(export: &crate::monitor::MonitorExport) -> String {
    let nch = export.channels.len();
    let mut s = String::new();
    s.push_str("/* hybrid_monitor.c -- generated by struktura, calibration baked in.\n");
    s.push_str(" * Five-leg hybrid telemetry health monitor: residual (AR1),\n");
    s.push_str(" * repeated-value, windowed DFA, rolling-mean level, residual CUSUM.\n");
    s.push_str(" * All thresholds calibrated (Gumbel return levels).\n");
    s.push_str(" * Static memory only. Bounded loops only. C99 + libm.\n");
    s.push_str(" * Compile: gcc -std=c99 -Wall -Werror -O2 -o hybrid hybrid_monitor.c -lm\n");
    s.push_str(" * On 32-bit x86 add -msse2 -mfpmath=sse: x87 extended precision can\n");
    s.push_str(" * move an alarm against the Rust monitor (one of 198 alarms, seen).\n");
    let trended: Vec<(usize, f64)> = export
        .channels
        .iter()
        .enumerate()
        .filter(|(_, c)| c.tr_slope != 0.0)
        .map(|(i, c)| (i, c.tr_slope))
        .collect();
    if !trended.is_empty() {
        s.push_str(" *\n * Certified calibration trends (extra drift beyond the calibrated\n");
        s.push_str(" * band, sustained, still alarms):\n");
        for (i, slope) in &trended {
            s.push_str(&format!(" *   channel {}: {:.6e} per sample\n", i, slope));
        }
    }
    s.push_str(" *\n * Clock assumption: hyb_init() must run on the sample immediately\n");
    s.push_str(" * after the calibration rows end. A trended channel's reference line\n");
    s.push_str(" * (tr_mu/tr_t0) is anchored to that boundary; starting hyb_init later\n");
    s.push_str(" * offsets it by (slope * gap).\n");
    s.push_str(" */\n#include <math.h>\n#include <string.h>\n\n");
    s.push_str(&format!("#define HYB_CHANNELS   {}\n", nch));
    // Taken from the Rust monitor so the C stays in step with its calibration.
    s.push_str(&format!("#define HYB_WINDOW     {}\n", crate::monitor::WINDOW));
    s.push_str(&format!("#define HYB_ROLL       {}\n", crate::monitor::ROLL));
    s.push_str(&format!("#define HYB_DFA_STRIDE {}\n", crate::monitor::DFA_STRIDE));
    s.push_str("#define HYB_PHASE      (HYB_WINDOW * HYB_ROLL * HYB_DFA_STRIDE)\n");
    // Rolling window centre offset, exactly as Rust's level leg uses it
    // (`dl = (t - tr_t0) - 47.5` for ROLL=96): a C expression, not a baked
    // decimal literal, so it stays exact for any ROLL.
    s.push_str("#define HYB_ROLL_HALF  ((double)(HYB_ROLL - 1) / 2.0)\n");
    s.push_str(&format!("#define HYB_RES_THR    {:.17e}\n", export.res_thr));
    s.push_str(&format!("#define HYB_DFA_THR    {:.17e}\n", export.dfa_thr));
    s.push_str(&format!("#define HYB_CUSUM_THR  {:.17e}\n", export.cusum_thr));
    s.push_str("#define HYB_CUSUM_K    1.0\n#define HYB_REPEAT_MARGIN 4\n");
    s.push_str("#define HYB_DFA_PERSIST   5\n#define HYB_ROLL_PERSIST  10\n\n");
    s.push_str("typedef struct {\n    double ar_a, ar_b, ar_sd;\n");
    s.push_str("    double alpha_mean, alpha_sd;\n    double mean, roll_thr;\n");
    s.push_str("    int max_run;\n    int repeat_enabled;\n");
    s.push_str("    /* Certified calibration trend (0.0/trendless unless noted above). */\n");
    s.push_str("    double tr_mu, tr_slope, tr_t0, tr_q1, tr_q2, ar_slope;\n} hyb_calib_t;\n\n");
    s.push_str("static const hyb_calib_t HYB_CALIB[HYB_CHANNELS] = {\n");
    for c in &export.channels {
        s.push_str(&format!(
            "    {{ {:.17e}, {:.17e}, {:.17e}, {:.17e}, {:.17e}, {:.17e}, {:.17e}, {}, {}, \
             {:.17e}, {:.17e}, {:.17e}, {:.17e}, {:.17e}, {:.17e} }},\n",
            c.ar_a, c.ar_b, c.ar_sd, c.alpha_mean, c.alpha_sd, c.mean, c.roll_thr,
            c.max_run, if c.repeat_enabled { 1 } else { 0 },
            c.tr_mu, c.tr_slope, c.tr_t0, c.tr_q1, c.tr_q2, c.ar_slope
        ));
    }
    s.push_str("};\n\n");
    s.push_str("typedef enum {\n    HYB_OK = 0,\n    HYB_ALARM_RESIDUAL,\n");
    s.push_str("    HYB_ALARM_REPEATED,\n    HYB_ALARM_DFA,\n");
    s.push_str("    HYB_ALARM_LEVEL,\n    HYB_ALARM_CUSUM\n} hyb_verdict_t;\n\n");
    s.push_str("typedef struct {\n    double ring[HYB_WINDOW];\n");
    s.push_str("    double roll_ring[HYB_ROLL];\n    double prev;\n");
    s.push_str("    double cusum_pos, cusum_neg;\n    unsigned long long t;\n");
    s.push_str("    unsigned long long res_hit_prev;\n    int run;\n");
    s.push_str("    int dfa_streak;\n    int roll_streak;\n");
    s.push_str("    unsigned long ph; /* t mod HYB_PHASE: ring positions without 64-bit division */\n");
    s.push_str("} hyb_channel_t;\n\n");
    s.push_str("typedef struct {\n    hyb_channel_t ch[HYB_CHANNELS];\n");
    s.push_str("    int alarmed;\n} hyb_monitor_t;\n\n");
    s.push_str("static void hyb_init(hyb_monitor_t *m) {\n");
    s.push_str("    int c;\n    memset(m, 0, sizeof(*m));\n");
    s.push_str("    for (c = 0; c < HYB_CHANNELS; c++) {\n");
    s.push_str("        m->ch[c].run = 1;\n");
    s.push_str("        m->ch[c].res_hit_prev = (unsigned long long)-1;\n    }\n}\n\n");
    // The box sizes the Rust DFA uses for a window of HYB_WINDOW samples, so the
    // C alpha is on the same scale as the calibrated alpha_mean and alpha_sd.
    let (sizes, count) = crate::dfa_box_sizes(crate::monitor::WINDOW);
    let boxes: Vec<String> = sizes[..count].iter().map(|b| b.to_string()).collect();
    // Same operations, in the same order, as the Rust box_rss.
    s.push_str("/* Residual sum of squares of the least-squares line through y[0..s),\n");
    s.push_str(" * summed from the residuals: the one-pass identity cancels to rounding\n");
    s.push_str(" * on a box inside a constant run. As struktura's box_rss. */\n");
    s.push_str("static double hyb_box_rss(const double *y, int s) {\n");
    s.push_str("    double k = (double)s, ibar = (k - 1.0) / 2.0, ybar = 0.0, sxy = 0.0, rss = 0.0, b;\n");
    s.push_str("    int i;\n");
    s.push_str("    for (i = 0; i < s; i++) ybar += y[i];\n    ybar /= k;\n");
    s.push_str("    for (i = 0; i < s; i++) sxy += ((double)i - ibar) * (y[i] - ybar);\n");
    s.push_str("    b = sxy / (k * (k * k - 1.0) / 12.0);\n");
    s.push_str("    for (i = 0; i < s; i++) {\n");
    s.push_str("        double r = (y[i] - ybar) - b * ((double)i - ibar);\n");
    s.push_str("        rss += r * r;\n    }\n    return rss;\n}\n\n");
    s.push_str("static double hyb_dfa_alpha(const double *v, int n) {\n");
    s.push_str(&format!(
        "    static const int BOXES[{count}] = {{{}}};\n",
        boxes.join(", ")
    ));
    s.push_str("    double mean = 0.0, cum = 0.0, ss = 0.0, floor2;\n    double y[HYB_WINDOW];\n");
    s.push_str(&format!("    double log_s[{count}], log_f[{count}];\n    int i, b, pts = 0;\n"));
    s.push_str("    for (i = 0; i < n; i++) mean += v[i];\n    mean /= (double)n;\n");
    s.push_str("    for (i = 0; i < n; i++) { cum += v[i] - mean; y[i] = cum; ss += cum * cum; }\n");
    s.push_str("    /* A box whose F is at most 1e-12 of the profile's RMS measures only\n");
    s.push_str("     * rounding and gives no point of the fit (FLAT_BOX_REL). */\n");
    s.push_str("    floor2 = 1e-12 * 1e-12 * (ss / (double)n);\n");
    s.push_str(&format!("    for (b = 0; b < {count}; b++) {{\n        int s = BOXES[b];\n"));
    s.push_str("        int num_segs = n / s;\n        double k = (double)s;\n");
    s.push_str("        double sx = k * (k - 1.0) / 2.0;\n");
    s.push_str("        double sx2 = k * (k - 1.0) * (2.0 * k - 1.0) / 6.0;\n");
    s.push_str("        double det = k * sx2 - sx * sx;\n");
    s.push_str("        double f2 = 0.0;\n        int seg;\n");
    s.push_str("        if (num_segs == 0 || s > n / 4) continue;\n");
    s.push_str("        for (seg = 0; seg < num_segs; seg++) {\n");
    s.push_str("            int st = seg * s;\n");
    s.push_str("            double sy = 0, sxy = 0, sy2 = 0, a0, a1, resid;\n");
    s.push_str("            for (i = 0; i < s; i++) {\n");
    s.push_str("                double yi = y[st + i];\n");
    s.push_str("                sy += yi; sxy += (double)i * yi; sy2 += yi * yi;\n");
    s.push_str("            }\n");
    s.push_str("            a0 = (sx2 * sy - sx * sxy) / det;\n");
    s.push_str("            a1 = (k * sxy - sx * sy) / det;\n");
    s.push_str("            resid = sy2 - a0 * sy - a1 * sxy;\n");
    s.push_str("            /* Within rounding of its own terms: recompute (REFINE_REL). */\n");
    s.push_str("            if (resid <= 1e-9 * (sy2 + fabs(a0 * sy) + fabs(a1 * sxy)))\n");
    s.push_str("                resid = hyb_box_rss(y + st, s);\n");
    s.push_str("            f2 += resid / k;\n        }\n");
    s.push_str("        f2 /= (double)num_segs;\n");
    s.push_str("        if (f2 > floor2) { log_s[pts] = log((double)s); log_f[pts] = log(sqrt(f2)); pts++; }\n");
    s.push_str("    }\n    if (pts < 3) return 0.5;\n    {\n");
    s.push_str("        double n_ = (double)pts, sxa = 0, sya = 0, sxya = 0, sx2a = 0;\n");
    s.push_str("        for (i = 0; i < pts; i++) {\n");
    s.push_str("            sxa += log_s[i]; sya += log_f[i];\n");
    s.push_str("            sxya += log_s[i] * log_f[i]; sx2a += log_s[i] * log_s[i];\n");
    s.push_str("        }\n");
    s.push_str("        return (n_ * sxya - sxa * sya) / (n_ * sx2a - sxa * sxa);\n");
    s.push_str("    }\n}\n\n");
    s.push_str("/* Feed one sample for one channel (channels may arrive at different\n");
    s.push_str(" * rates). Returns HYB_OK or the alarm; the monitor then latches and\n");
    s.push_str(" * ignores every channel until hyb_reset. For one sample of all channels\n");
    s.push_str(" * use hyb_push_sample, which keeps feeding the channels after an alarm. */\n");
    s.push_str("static hyb_verdict_t hyb_push(hyb_monitor_t *m, int c, double v) {\n");
    s.push_str("    hyb_channel_t *st;\n    const hyb_calib_t *cc;\n");
    s.push_str("    unsigned long long t;\n    unsigned long ph;\n    double zs;\n");
    s.push_str("    double d1, d0, p1, p0, q, bq, base, pred;\n");
    s.push_str("    if (m->alarmed || c < 0 || c >= HYB_CHANNELS) return HYB_OK;\n");
    s.push_str("    st = &m->ch[c];\n    cc = &HYB_CALIB[c];\n    t = st->t++;\n");
    s.push_str("    ph = st->ph;\n    st->ph = ph + 1 == HYB_PHASE ? 0 : ph + 1;\n");
    s.push_str("    if (t == 0) {\n        st->prev = v;\n        st->ring[0] = v;\n");
    s.push_str("        st->roll_ring[0] = v;\n        return HYB_OK;\n    }\n");
    s.push_str("    /* One-step AR(1) prediction against the (possibly trending)\n");
    s.push_str("     * baseline; matches Rust's ar_pred statement for statement so a\n");
    s.push_str("     * trendless channel (ar_slope == 0.0, tr_t0 == 0.0) reduces to\n");
    s.push_str("     * ar_a + ar_b*prev bit for bit. */\n");
    s.push_str("    d1 = (double)t - cc->tr_t0;\n    d0 = d1 - 1.0;\n");
    s.push_str("    p1 = cc->ar_slope * d1;\n    p0 = cc->ar_slope * d0;\n");
    s.push_str("    q = st->prev - p0;\n    bq = cc->ar_b * q;\n");
    s.push_str("    base = cc->ar_a + p1;\n    pred = base + bq;\n");
    s.push_str("    zs = (v - pred) / cc->ar_sd;\n");
    s.push_str("    st->cusum_pos += zs - HYB_CUSUM_K;\n");
    s.push_str("    if (!(st->cusum_pos > 0.0)) st->cusum_pos = 0.0; /* NaN -> 0, as Rust max(0.0) */\n");
    s.push_str("    st->cusum_neg += -zs - HYB_CUSUM_K;\n");
    s.push_str("    if (!(st->cusum_neg > 0.0)) st->cusum_neg = 0.0;\n");
    s.push_str("    if (v == st->prev) {\n        st->run++;\n");
    s.push_str("        if (cc->repeat_enabled && st->run >= cc->max_run + HYB_REPEAT_MARGIN) {\n");
    s.push_str("            m->alarmed = 1; return HYB_ALARM_REPEATED;\n        }\n");
    s.push_str("    } else {\n        st->run = 1;\n    }\n");
    s.push_str("    st->prev = v;\n");
    s.push_str("    st->ring[ph % HYB_WINDOW] = v;\n");
    s.push_str("    st->roll_ring[ph % HYB_ROLL] = v;\n");
    s.push_str("    if (fabs(zs) > HYB_RES_THR) {\n");
    s.push_str("        if (st->res_hit_prev != (unsigned long long)-1 && t - st->res_hit_prev < 20) {\n");
    s.push_str("            m->alarmed = 1; return HYB_ALARM_RESIDUAL;\n        }\n");
    s.push_str("        st->res_hit_prev = t;\n    }\n");
    s.push_str("    if (st->cusum_pos > HYB_CUSUM_THR || st->cusum_neg > HYB_CUSUM_THR) {\n");
    s.push_str("        m->alarmed = 1; return HYB_ALARM_CUSUM;\n    }\n");
    s.push_str("    if (t >= HYB_WINDOW && ph % HYB_DFA_STRIDE == 0) {\n");
    s.push_str("        double lin[HYB_WINDOW];\n        double a;\n        int i;\n");
    s.push_str("        unsigned long start = (ph + 1) % HYB_WINDOW;\n");
    s.push_str("        for (i = 0; i < HYB_WINDOW; i++)\n");
    s.push_str("            lin[i] = st->ring[(start + (unsigned long)i) % HYB_WINDOW];\n");
    s.push_str("        a = hyb_dfa_alpha(lin, HYB_WINDOW);\n");
    s.push_str("        if (fabs(a - cc->alpha_mean) / cc->alpha_sd > HYB_DFA_THR) {\n");
    s.push_str("            st->dfa_streak += HYB_DFA_STRIDE;\n");
    s.push_str("            if (st->dfa_streak >= HYB_DFA_PERSIST) {\n");
    s.push_str("                m->alarmed = 1; return HYB_ALARM_DFA;\n            }\n");
    s.push_str("        } else {\n            st->dfa_streak = 0;\n        }\n    }\n");
    s.push_str("    if (t >= HYB_ROLL) {\n        double sum = 0.0;\n        int i;\n");
    s.push_str("        double dl, p, e, h, w1, hh, w2, thr, dev;\n");
    s.push_str("        for (i = 0; i < HYB_ROLL; i++) sum += st->roll_ring[i];\n");
    s.push_str("        dl = ((double)t - cc->tr_t0) - HYB_ROLL_HALF;\n");
    s.push_str("        p = cc->tr_slope * dl;\n        e = cc->tr_mu + p;\n");
    s.push_str("        h = fabs(dl);\n        w1 = cc->tr_q1 * h;\n");
    s.push_str("        hh = h * h;\n        w2 = cc->tr_q2 * hh;\n");
    s.push_str("        thr = cc->roll_thr + sqrt(w1 + w2);\n");
    s.push_str("        dev = sum / (double)HYB_ROLL - e;\n");
    s.push_str("        if (fabs(dev) > thr) {\n");
    s.push_str("            st->roll_streak++;\n");
    s.push_str("            if (st->roll_streak >= HYB_ROLL_PERSIST) {\n");
    s.push_str("                m->alarmed = 1; return HYB_ALARM_LEVEL;\n            }\n");
    s.push_str("        } else {\n            st->roll_streak = 0;\n        }\n    }\n");
    s.push_str("    return HYB_OK;\n}\n\n");
    // Whole-sample entry point and reset, matching HybridMonitor::push and
    // HybridMonitor::reset. `static inline` so an integrator who does not
    // use them gets no -Wunused-function error under -Wall -Werror.
    s.push_str("/* Clear the alarm latch and the detector streaks, as the Rust\n");
    s.push_str(" * HybridMonitor::reset does. Rings, previous values and sample\n");
    s.push_str(" * counters are kept. */\n");
    s.push_str("static inline void hyb_reset(hyb_monitor_t *m) {\n");
    s.push_str("    int c;\n    m->alarmed = 0;\n");
    s.push_str("    for (c = 0; c < HYB_CHANNELS; c++) {\n");
    s.push_str("        m->ch[c].cusum_pos = 0.0;\n        m->ch[c].cusum_neg = 0.0;\n");
    s.push_str("        m->ch[c].dfa_streak = 0;\n        m->ch[c].roll_streak = 0;\n");
    s.push_str("        m->ch[c].res_hit_prev = (unsigned long long)-1;\n    }\n}\n\n");
    s.push_str("/* Feed one sample of every channel, x[0..HYB_CHANNELS-1]. Every channel\n");
    s.push_str(" * takes its value even after an earlier channel alarms, so no channel\n");
    s.push_str(" * falls behind. Returns the first alarm of the sample (its channel in\n");
    s.push_str(" * *alarm_ch when not NULL) and latches until hyb_reset, as the Rust\n");
    s.push_str(" * HybridMonitor::push does. */\n");
    s.push_str("static inline hyb_verdict_t hyb_push_sample(hyb_monitor_t *m, const double *x,\n");
    s.push_str("                                            int *alarm_ch) {\n");
    s.push_str("    hyb_verdict_t first = HYB_OK;\n    int c;\n");
    s.push_str("    if (m->alarmed) return HYB_OK;\n");
    s.push_str("    for (c = 0; c < HYB_CHANNELS; c++) {\n");
    s.push_str("        hyb_verdict_t v = hyb_push(m, c, x[c]);\n");
    s.push_str("        if (v != HYB_OK) {\n");
    s.push_str("            if (first == HYB_OK) {\n");
    s.push_str("                first = v;\n");
    s.push_str("                if (alarm_ch) *alarm_ch = c;\n            }\n");
    s.push_str("            m->alarmed = 0;\n        }\n    }\n");
    s.push_str("    if (first != HYB_OK) m->alarmed = 1;\n");
    s.push_str("    return first;\n}\n\n");
    // The self-test freezes a channel whose repeat leg is enabled (channels
    // that legitimately saturate have it auto-disabled).
    let test_ch = export
        .channels
        .iter()
        .position(|c| c.repeat_enabled)
        .unwrap_or(0);
    s.push_str("#ifdef HYBRID_STANDALONE_TEST\n#include <stdio.h>\n");
    s.push_str(&format!("#define HYB_TEST_CH {}\n", test_ch));
    s.push_str("int main(void) {\n    hyb_monitor_t m;\n    int t, c, ch = -1;\n");
    s.push_str("    double x[HYB_CHANNELS];\n");
    s.push_str("    hyb_verdict_t v = HYB_OK;\n    hyb_init(&m);\n");
    s.push_str("    /* Small deterministic wobble around each channel mean; stuck\n");
    s.push_str("     * fault on a repeat-enabled channel from t=400 (value frozen). */\n");
    s.push_str("    for (t = 0; t < 700 && v == HYB_OK; t++) {\n");
    s.push_str("        for (c = 0; c < HYB_CHANNELS; c++) {\n");
    s.push_str("            /* tr_mu + tr_slope*((double)t - tr_t0) plus wobble: reduces to\n");
    s.push_str("             * mean + wobble for a trendless channel (tr_slope == 0.0). */\n");
    s.push_str("            x[c] = HYB_CALIB[c].tr_mu\n");
    s.push_str("                + HYB_CALIB[c].tr_slope * ((double)t - HYB_CALIB[c].tr_t0)\n");
    s.push_str("                + 0.5 * HYB_CALIB[c].ar_sd * sin(0.7 * (double)t + (double)c);\n");
    s.push_str("            if (c == HYB_TEST_CH && t >= 400) x[c] = HYB_CALIB[HYB_TEST_CH].mean;\n");
    s.push_str("        }\n");
    s.push_str("        v = hyb_push_sample(&m, x, &ch);\n    }\n");
    s.push_str("    if (v == HYB_ALARM_REPEATED && ch == HYB_TEST_CH && t > 400) {\n");
    s.push_str("        printf(\"SELFTEST PASS: stuck detected at t=%d on channel %d (leg=repeated)\\n\", t - 1, ch);\n");
    s.push_str("        hyb_reset(&m);\n        return 0;\n    }\n");
    s.push_str("    printf(\"SELFTEST FAIL: verdict=%d t=%d channel=%d\\n\", (int)v, t - 1, ch);\n");
    s.push_str("    return 1;\n}\n#endif\n");
    s
}

/// One telemetry channel for the cFS/F Prime/ROS directory generators
/// (`struktura generate --cfs|--fprime|--ros --db channels.json`).
///
/// `#[non_exhaustive]`: build one with [`Channel::new`], not a struct
/// literal, so adding a field later is not a breaking change.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Channel {
    /// The channel's identifier: names its per-channel ring
    /// (`dfa_ch_<name>`/`m_ch_<name>`/`ch_<name>`) and input port
    /// (`<name>In` for F Prime, `/<name>` for ROS), and appears in
    /// baseline/shift log messages.
    pub name: String,
    /// The cFS message topic this channel arrives on; the generator adds
    /// its `_MID` suffix. F Prime and ROS use `name` for their own
    /// port/topic naming and do not read this field.
    pub topic: String,
    /// The field within that cFS message holding the channel's value
    /// (cast to `double` when read). Unused by F Prime/ROS.
    pub field: String,
    /// The C message type for `topic` (cFS only), e.g. `sample_msg_t`.
    pub msg_type: String,
}

impl Channel {
    /// Build a channel. See the field docs above for what each generator
    /// reads.
    pub fn new(
        name: impl Into<String>,
        topic: impl Into<String>,
        field: impl Into<String>,
        msg_type: impl Into<String>,
    ) -> Self {
        Channel {
            name: name.into(),
            topic: topic.into(),
            field: field.into(),
            msg_type: msg_type.into(),
        }
    }
}

/// Generate `dfa_monitor_cfs.c` for `struktura generate --cfs`: a full NASA
/// cFS app (one `dfa_channel_t` ring per telemetry channel, wired to
/// `DFA_MONITOR_ProcessPkt`). Includes the shared `dfa_core.h`
/// ([`generate_dfa_core_h`]) for `dfa_compute`, and reorders each channel's
/// ring into time order (`ordered`) before calling it, the way
/// [`generate_c_monitor`]'s `dfa_monitor_push` does.
pub fn generate_cfs_main(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("/* dfa_monitor_cfs.c -- Generated by struktura generate --cfs\n");
    s.push_str(" * DFA structural health monitor for NASA cFS.\n");
    s.push_str(" * https://github.com/koscak-labs/struktura\n */\n\n");
    s.push_str("#include \"dfa_monitor_cfs.h\"\n");
    s.push_str("#include \"dfa_monitor_cfs_events.h\"\n");
    s.push_str("#include \"dfa_monitor_cfs_msgids.h\"\n");
    s.push_str("#include \"dfa_core.h\"\n\n");
    s.push_str(&format!("#define DFA_THRESHOLD   {:.4}\n", threshold));
    s.push_str("#define DFA_LEARN_WINDOWS 10\n");
    s.push_str("#define DFA_R2_MIN 0.7\n\n");

    s.push_str("typedef struct {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str(&format!(
        "    double ordered[{}]; /* the window in time order, for dfa_compute */\n",
        window
    ));
    s.push_str("    uint32 pos;\n    uint32 filled;\n");
    s.push_str("    double baseline_alpha;\n    uint8 baseline_set;\n    uint32 window_count;\n");
    s.push_str("    dfa_result_t last; /* the alpha/r_squared the most recent push actually used */\n");
    s.push_str("} dfa_channel_t;\n\n");

    for ch in channels {
        s.push_str(&format!("static dfa_channel_t dfa_ch_{};\n", ch.name));
    }
    s.push_str("\nstatic CFE_SB_PipeId_t DFA_MONITOR_Pipe;\n");
    s.push_str("static CFE_SB_Buffer_t *SBBufPtr;\n\n");

    s.push_str("void DFA_MONITOR_AppMain(void) {\n");
    s.push_str("    CFE_Status_t status;\n    uint32 RunStatus = CFE_ES_RunStatus_APP_RUN;\n");
    s.push_str("    status = DFA_MONITOR_Init();\n");
    s.push_str("    if (status != CFE_SUCCESS) RunStatus = CFE_ES_RunStatus_APP_ERROR;\n");
    s.push_str("    while (CFE_ES_RunLoop(&RunStatus) == true) {\n");
    s.push_str("        status = CFE_SB_ReceiveBuffer(&SBBufPtr, DFA_MONITOR_Pipe, 500);\n");
    s.push_str("        if (status == CFE_SUCCESS) DFA_MONITOR_ProcessPkt();\n");
    s.push_str("    }\n    CFE_ES_ExitApp(RunStatus);\n}\n\n");

    s.push_str("CFE_Status_t DFA_MONITOR_Init(void) {\n");
    s.push_str("    CFE_Status_t status;\n");
    s.push_str("    CFE_EVS_Register(NULL, 0, CFE_EVS_EventFilter_BINARY);\n");
    s.push_str("    status = CFE_SB_CreatePipe(&DFA_MONITOR_Pipe, 32, \"DFA_MON_PIPE\");\n");
    s.push_str("    if (status != CFE_SUCCESS) return status;\n\n");
    for ch in channels {
        s.push_str(&format!(
            "    CFE_SB_Subscribe(CFE_SB_ValueToMsgId({}_MID), DFA_MONITOR_Pipe);\n",
            ch.topic.to_uppercase()
        ));
        s.push_str(&format!(
            "    memset(&dfa_ch_{}, 0, sizeof(dfa_ch_{}));\n",
            ch.name, ch.name
        ));
    }
    s.push_str("\n    CFE_EVS_SendEvent(DFA_MON_INIT_EID, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str(&format!(
        "        \"DFA Monitor: {} channels, window={}, threshold={:.3}\");\n",
        channels.len(),
        window,
        threshold
    ));
    s.push_str("    return CFE_SUCCESS;\n}\n\n");

    s.push_str("static void dfa_push(dfa_channel_t *ch, double value, const char *name) {\n");
    s.push_str(&format!(
        "    ch->buffer[ch->pos] = value;\n    ch->pos = (ch->pos + 1) % {};\n",
        window
    ));
    s.push_str("    if (ch->pos == 0) ch->filled = 1;\n    if (!ch->filled) return;\n");
    s.push_str("    ch->window_count++;\n");
    s.push_str("    /* The ring holds the oldest sample at pos; DFA needs the window in time order. */\n");
    s.push_str("    uint32 i;\n");
    s.push_str(&format!("    for (i = 0; i < {}; i++)\n", window));
    s.push_str(&format!(
        "        ch->ordered[i] = ch->buffer[(ch->pos + i) % {}];\n",
        window
    ));
    s.push_str(&format!("    dfa_result_t r = dfa_compute(ch->ordered, {});\n", window));
    s.push_str("    ch->last = r;\n");
    s.push_str("    if (!ch->baseline_set && ch->window_count >= DFA_LEARN_WINDOWS && r.r_squared > DFA_R2_MIN) {\n");
    s.push_str("        ch->baseline_alpha = r.alpha;\n        ch->baseline_set = 1;\n");
    s.push_str("        CFE_EVS_SendEvent(DFA_MON_BASELINE_EID, CFE_EVS_EventType_INFORMATION,\n");
    s.push_str("            \"DFA baseline %s: alpha=%.3f R2=%.4f\", name, r.alpha, r.r_squared);\n");
    s.push_str("        return;\n    }\n    if (!ch->baseline_set) return;\n");
    s.push_str("    if (r.r_squared < DFA_R2_MIN) return;\n");
    s.push_str("    double shift = r.alpha - ch->baseline_alpha;\n");
    s.push_str("    if (shift < 0) shift = -shift;\n");
    s.push_str("    if (shift >= DFA_THRESHOLD) {\n");
    s.push_str("        CFE_EVS_SendEvent(DFA_MON_SHIFT_EID, CFE_EVS_EventType_ERROR,\n");
    s.push_str("            \"DFA SHIFT %s: alpha=%.3f baseline=%.3f delta=%.3f\",\n");
    s.push_str("            name, r.alpha, ch->baseline_alpha, r.alpha - ch->baseline_alpha);\n");
    s.push_str("    }\n}\n\n");

    s.push_str("void DFA_MONITOR_ProcessPkt(void) {\n");
    s.push_str("    CFE_SB_MsgId_t MsgId = CFE_SB_INVALID_MSG_ID;\n");
    s.push_str("    CFE_MSG_GetMsgId(&SBBufPtr->Msg, &MsgId);\n\n");
    for (i, ch) in channels.iter().enumerate() {
        let kw = if i == 0 { "if" } else { "else if" };
        s.push_str(&format!(
            "    {} (CFE_SB_MsgId_Equal(MsgId, CFE_SB_ValueToMsgId({}_MID))) {{\n",
            kw,
            ch.topic.to_uppercase()
        ));
        s.push_str(&format!(
            "        {} *msg = ({}*)&SBBufPtr->Msg;\n",
            ch.msg_type, ch.msg_type
        ));
        s.push_str(&format!(
            "        dfa_push(&dfa_ch_{}, (double)msg->{}, \"{}\");\n",
            ch.name, ch.field, ch.name
        ));
        s.push_str("    }\n");
    }
    s.push_str("}\n");
    s
}

/// Generate `dfa_monitor_cfs.h` for `struktura generate --cfs`.
pub fn generate_cfs_header(channels: &[Channel]) -> String {
    let mut s = String::new();
    s.push_str("#ifndef DFA_MONITOR_CFS_H\n#define DFA_MONITOR_CFS_H\n\n");
    s.push_str("#include \"cfe.h\"\n#include <string.h>\n#include <math.h>\n\n");
    s.push_str("void DFA_MONITOR_AppMain(void);\n");
    s.push_str("CFE_Status_t DFA_MONITOR_Init(void);\n");
    s.push_str("void DFA_MONITOR_ProcessPkt(void);\n\n");
    let _ = channels;
    s.push_str("#endif\n");
    s
}

/// Generate `dfa_monitor_cfs_msgids.h` for `struktura generate --cfs`.
pub fn generate_cfs_msgids(channels: &[Channel]) -> String {
    let mut s = String::new();
    s.push_str("#ifndef DFA_MONITOR_CFS_MSGIDS_H\n#define DFA_MONITOR_CFS_MSGIDS_H\n\n");
    for (i, ch) in channels.iter().enumerate() {
        s.push_str(&format!(
            "#define {}_MID 0x{:04X}\n",
            ch.topic.to_uppercase(),
            0x1900 + i
        ));
    }
    s.push_str("\n#endif\n");
    s
}

/// Generate `DfaMonitor.fpp` for `struktura generate --fprime`.
pub fn generate_fprime_fpp(channels: &[Channel]) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("module Svc {\n");
    s.push_str("  @ DFA structural health monitor\n");
    s.push_str("  @ Generated by: struktura generate --fprime\n");
    s.push_str("  @ https://github.com/koscak-labs/struktura\n");
    s.push_str("  passive component DfaMonitor {\n\n");
    s.push_str("    sync input port schedIn: Svc.Sched\n\n");
    for ch in channels {
        s.push_str(&format!("    guarded input port {}In: Fw.Tlm\n", ch.name));
    }
    s.push_str("\n    event StructuralShift(\n");
    s.push_str("      channelName: string size 32\n");
    s.push_str("      baseline_alpha: F64\n      current_alpha: F64\n      delta: F64\n");
    s.push_str("    ) severity warning high\n\n");
    s.push_str("    event BaselineEstablished(\n");
    s.push_str("      channelName: string size 32\n      alpha: F64\n      r_squared: F64\n");
    s.push_str("    ) severity activity high\n\n");
    s.push_str("    telemetry DfaAlpha: F64\n    telemetry DfaR2: F64\n\n");
    s.push_str("    time get port timeCaller\n    event port logOut\n    telemetry port tlmOut\n");
    s.push_str("  }\n}\n");
    s
}

/// Generate `DfaMonitor.cpp` for `struktura generate --fprime`: reorders each
/// channel's ring into time order (`ch.ordered`) before calling the shared
/// `dfa_compute` from `dfa_core.h` ([`generate_dfa_core_h`]), the way
/// [`generate_c_monitor`]'s `dfa_monitor_push` does.
pub fn generate_fprime_cpp(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("// DfaMonitor.cpp -- Generated by struktura generate --fprime\n");
    s.push_str("// https://github.com/koscak-labs/struktura\n\n");
    s.push_str("#include \"DfaMonitor.hpp\"\n#include \"dfa_core.h\"\n\n");
    s.push_str("namespace Svc {\n\n");
    s.push_str("DfaMonitor::DfaMonitor(const char* name) : DfaMonitorComponentBase(name) {\n");
    for ch in channels {
        s.push_str(&format!("    memset(&m_ch_{}, 0, sizeof(m_ch_{}));\n", ch.name, ch.name));
    }
    s.push_str("}\n\n");
    for ch in channels {
        s.push_str(&format!(
            "void DfaMonitor::{}In_handler(NATIVE_INT_TYPE portNum, FwTlmBuffer& val) {{\n",
            ch.name
        ));
        s.push_str("    F64 v; val.deserialize(v);\n");
        s.push_str(&format!("    pushSample(m_ch_{}, v, \"{}\");\n", ch.name, ch.name));
        s.push_str("}\n\n");
    }
    s.push_str("void DfaMonitor::pushSample(DfaChannel& ch, F64 value, const char* name) {\n");
    s.push_str(&format!(
        "    ch.buffer[ch.pos] = value;\n    ch.pos = (ch.pos + 1) % {};\n",
        window
    ));
    s.push_str("    if (ch.pos == 0) ch.filled = true;\n    if (!ch.filled) return;\n");
    s.push_str("    ch.windowCount++;\n");
    s.push_str("    // The ring holds the oldest sample at pos; DFA needs the window in time order.\n");
    s.push_str(&format!("    for (U32 i = 0; i < {}; i++)\n", window));
    s.push_str(&format!(
        "        ch.ordered[i] = ch.buffer[(ch.pos + i) % {}];\n",
        window
    ));
    s.push_str(&format!("    dfa_result_t r = dfa_compute(ch.ordered, {});\n", window));
    s.push_str("    ch.last = r;\n");
    s.push_str("    if (!ch.baselineSet && ch.windowCount >= 10 && r.r_squared > 0.7) {\n");
    s.push_str("        ch.baselineAlpha = r.alpha; ch.baselineSet = true;\n");
    s.push_str("        this->log_ACTIVITY_HI_BaselineEstablished(name, r.alpha, r.r_squared);\n");
    s.push_str("        return;\n    }\n    if (!ch.baselineSet || r.r_squared < 0.7) return;\n");
    s.push_str("    F64 shift = (r.alpha > ch.baselineAlpha) ? r.alpha - ch.baselineAlpha : ch.baselineAlpha - r.alpha;\n");
    s.push_str(&format!("    if (shift >= {:.4}) {{\n", threshold));
    s.push_str("        this->log_WARNING_HI_StructuralShift(name, ch.baselineAlpha, r.alpha, r.alpha - ch.baselineAlpha);\n");
    s.push_str("    }\n    this->tlmWrite_DfaAlpha(r.alpha);\n    this->tlmWrite_DfaR2(r.r_squared);\n");
    s.push_str("}\n\n} // namespace Svc\n");
    s
}

/// Generate `DfaMonitor.hpp` for `struktura generate --fprime`.
pub fn generate_fprime_hpp(channels: &[Channel], window: usize) -> String {
    let mut s = String::with_capacity(1024);
    s.push_str("#ifndef DFA_MONITOR_HPP\n#define DFA_MONITOR_HPP\n\n");
    s.push_str("#include \"DfaMonitorComponentAc.hpp\"\n");
    s.push_str("#include \"dfa_core.h\" // dfa_result_t, for DfaChannel::last\n\n");
    s.push_str("namespace Svc {\n\n");
    s.push_str("struct DfaChannel {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str(&format!(
        "    double ordered[{}]; // the window in time order, for dfa_compute\n",
        window
    ));
    s.push_str("    U32 pos; bool filled; double baselineAlpha;\n");
    s.push_str("    bool baselineSet; U32 windowCount;\n");
    s.push_str("    dfa_result_t last; // the alpha/r_squared the most recent push actually used\n};\n\n");
    s.push_str("class DfaMonitor : public DfaMonitorComponentBase {\n");
    s.push_str("  public:\n    DfaMonitor(const char* name);\n");
    s.push_str("  private:\n");
    for ch in channels {
        s.push_str(&format!(
            "    void {}In_handler(NATIVE_INT_TYPE portNum, FwTlmBuffer& val);\n",
            ch.name
        ));
    }
    s.push_str("    void pushSample(DfaChannel& ch, F64 value, const char* name);\n");
    for ch in channels {
        s.push_str(&format!("    DfaChannel m_ch_{};\n", ch.name));
    }
    s.push_str("};\n\n} // namespace Svc\n\n#endif\n");
    s
}

/// Generate `dfa_monitor_node.cpp` for `struktura generate --ros`: reorders
/// each channel's ring into time order (`ch.ordered`) before calling the
/// shared `dfa_compute` from `dfa_core.h` ([`generate_dfa_core_h`]), the way
/// [`generate_c_monitor`]'s `dfa_monitor_push` does.
pub fn generate_ros_monitor(channels: &[Channel], window: usize, threshold: f64) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("/* dfa_monitor_node.cpp -- Generated by struktura generate --ros\n");
    s.push_str(" * DFA structural health monitor for ROS 2.\n");
    s.push_str(" * https://github.com/koscak-labs/struktura\n */\n\n");
    s.push_str("#include <functional>\n#include <memory>\n#include <cmath>\n#include <cstring>\n\n");
    s.push_str("#include \"rclcpp/rclcpp.hpp\"\n");
    s.push_str("#include \"std_msgs/msg/float64.hpp\"\n");
    s.push_str("#include \"std_msgs/msg/string.hpp\"\n\n");
    s.push_str("#include \"dfa_core.h\"\n\n");
    s.push_str(&format!("#define DFA_THRESHOLD   {:.4}\n", threshold));
    s.push_str("#define DFA_LEARN_WINDOWS 10\n#define DFA_R2_MIN 0.7\n\n");
    s.push_str("using std::placeholders::_1;\n\n");

    s.push_str("struct DfaChannel {\n");
    s.push_str(&format!("    double buffer[{}];\n", window));
    s.push_str(&format!(
        "    double ordered[{}]; // the window in time order, for dfa_compute\n",
        window
    ));
    s.push_str("    uint32_t pos = 0;\n    bool filled = false;\n");
    s.push_str("    double baseline_alpha = 0.0;\n    bool baseline_set = false;\n");
    s.push_str("    uint32_t window_count = 0;\n");
    s.push_str("    dfa_result_t last{}; // the alpha/r_squared the most recent push actually used\n};\n\n");

    s.push_str("class DfaMonitorNode : public rclcpp::Node {\n");
    s.push_str("public:\n");
    s.push_str("    DfaMonitorNode() : Node(\"dfa_monitor\") {\n");
    for ch in channels {
        s.push_str(&format!(
            "        {}_sub_ = this->create_subscription<std_msgs::msg::Float64>(\n",
            ch.name
        ));
        s.push_str(&format!("            \"/{}\", 10,\n", ch.name));
        s.push_str(&format!(
            "            std::bind(&DfaMonitorNode::{}_callback, this, _1));\n",
            ch.name
        ));
        s.push_str(&format!(
            "        std::memset(&ch_{}, 0, sizeof(ch_{}));\n\n",
            ch.name, ch.name
        ));
    }
    s.push_str("        shift_pub_ = this->create_publisher<std_msgs::msg::String>(\"dfa/shift\", 10);\n");
    s.push_str(&format!(
        "        RCLCPP_INFO(this->get_logger(), \"DFA Monitor: {} channels, window={}, threshold={:.3}\");\n",
        channels.len(),
        window,
        threshold
    ));
    s.push_str("    }\n\nprivate:\n");

    s.push_str("    void push_sample(DfaChannel &ch, double value, const char *name) {\n");
    s.push_str(&format!(
        "        ch.buffer[ch.pos] = value;\n        ch.pos = (ch.pos + 1) % {};\n",
        window
    ));
    s.push_str("        if (ch.pos == 0) ch.filled = true;\n        if (!ch.filled) return;\n");
    s.push_str("        ch.window_count++;\n");
    s.push_str("        // The ring holds the oldest sample at pos; DFA needs the window in time order.\n");
    s.push_str(&format!("        for (uint32_t i = 0; i < {}; i++)\n", window));
    s.push_str(&format!(
        "            ch.ordered[i] = ch.buffer[(ch.pos + i) % {}];\n",
        window
    ));
    s.push_str(&format!(
        "        dfa_result_t r = dfa_compute(ch.ordered, {});\n",
        window
    ));
    s.push_str("        ch.last = r;\n");
    s.push_str("        if (!ch.baseline_set && ch.window_count >= DFA_LEARN_WINDOWS && r.r_squared > DFA_R2_MIN) {\n");
    s.push_str("            ch.baseline_alpha = r.alpha; ch.baseline_set = true;\n");
    s.push_str("            RCLCPP_INFO(this->get_logger(), \"DFA baseline %s: alpha=%.3f R2=%.4f\", name, r.alpha, r.r_squared);\n");
    s.push_str("            return;\n        }\n        if (!ch.baseline_set || r.r_squared < DFA_R2_MIN) return;\n");
    s.push_str("        double shift = std::abs(r.alpha - ch.baseline_alpha);\n");
    s.push_str("        if (shift >= DFA_THRESHOLD) {\n");
    s.push_str("            auto msg = std_msgs::msg::String();\n");
    s.push_str("            char buf[128];\n");
    s.push_str("            std::snprintf(buf, sizeof(buf), \"SHIFT %s: alpha=%.3f baseline=%.3f delta=%.3f\",\n");
    s.push_str("                name, r.alpha, ch.baseline_alpha, r.alpha - ch.baseline_alpha);\n");
    s.push_str("            msg.data = buf;\n            shift_pub_->publish(msg);\n");
    s.push_str("            RCLCPP_WARN(this->get_logger(), \"%s\", buf);\n");
    s.push_str("        }\n    }\n\n");

    for ch in channels {
        s.push_str(&format!(
            "    void {}_callback(const std_msgs::msg::Float64::SharedPtr msg) {{\n",
            ch.name
        ));
        s.push_str(&format!("        push_sample(ch_{}, msg->data, \"{}\");\n", ch.name, ch.name));
        s.push_str("    }\n\n");
    }

    for ch in channels {
        s.push_str(&format!(
            "    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr {}_sub_;\n",
            ch.name
        ));
        s.push_str(&format!("    DfaChannel ch_{};\n", ch.name));
    }
    s.push_str("    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr shift_pub_;\n");
    s.push_str("};\n\n");

    s.push_str("int main(int argc, char *argv[]) {\n");
    s.push_str("    rclcpp::init(argc, argv);\n");
    s.push_str("    rclcpp::spin(std::make_shared<DfaMonitorNode>());\n");
    s.push_str("    rclcpp::shutdown();\n    return 0;\n}\n");
    s
}

/// Generate `CMakeLists.txt` for `struktura generate --ros`.
pub fn generate_ros_cmake() -> String {
    let mut s = String::new();
    s.push_str("cmake_minimum_required(VERSION 3.8)\nproject(dfa_monitor)\n\n");
    s.push_str("find_package(ament_cmake REQUIRED)\nfind_package(rclcpp REQUIRED)\n");
    s.push_str("find_package(std_msgs REQUIRED)\n\n");
    s.push_str("add_executable(dfa_monitor_node src/dfa_monitor_node.cpp)\n");
    s.push_str("ament_target_dependencies(dfa_monitor_node rclcpp std_msgs)\n\n");
    s.push_str("install(TARGETS dfa_monitor_node DESTINATION lib/${PROJECT_NAME})\n");
    s.push_str("ament_package()\n");
    s
}

/// Generate `package.xml` for `struktura generate --ros`.
pub fn generate_ros_package() -> String {
    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<package format=\"3\">\n");
    s.push_str("  <name>dfa_monitor</name>\n  <version>1.0.0</version>\n");
    s.push_str("  <description>DFA structural health monitor for ROS 2</description>\n");
    s.push_str("  <maintainer email=\"maintainer@example.com\">koscak-labs</maintainer>\n");
    s.push_str("  <license>MIT</license>\n\n");
    s.push_str("  <buildtool_depend>ament_cmake</buildtool_depend>\n");
    s.push_str("  <depend>rclcpp</depend>\n  <depend>std_msgs</depend>\n\n");
    s.push_str("  <export>\n    <build_type>ament_cmake</build_type>\n  </export>\n");
    s.push_str("</package>\n");
    s
}
