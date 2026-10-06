//! Black-box recorder for [`crate::brain::Brain`] decisions.
//!
//! Every decision is written as a 32-byte record into a fixed ring that keeps
//! the newest `M`: brain clock, a hash of the situation, the action, whether
//! it abstained, how many episodes it cited, the expected reward, the
//! uncertainty bonus, the evidence, and the first cited memory slots. The ring
//! serializes to a little-endian byte stream for downlink.
//!
//! The brain is deterministic, so the ground can rebuild it from the same
//! learn sequence, recompute each decision with [`replay_one`] and compare
//! record for record, bit exact. A mismatch means the flight brain saw
//! different inputs (or its state was corrupted) at that decision.
//!
//! No heap, no std, bounded loops.

use crate::brain::{Brain, Decision};

/// Cited slots kept per record (the nearest ones).
pub const CITED: usize = 4;

/// Magic bytes of a downlink stream.
pub const MAGIC: [u8; 4] = *b"SKBR";
/// Stream format version.
pub const VERSION: u16 = 1;
/// Bytes per serialized record.
pub const RECORD_BYTES: usize = 32;
/// Bytes of the stream header: magic, version, record size, record count, total recorded.
pub const HEADER_BYTES: usize = 16;

/// One decision, 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionRecord {
    /// Brain clock (episodes learned) when the decision was made.
    pub clock: u32,
    /// FNV-1a over the situation's f32 bit patterns.
    pub situation_hash: u32,
    pub expected: f32,
    pub bonus: f32,
    pub evidence: f32,
    /// First cited memory slots, nearest first (only the first `n_cited` are meaningful).
    pub cited: [u16; CITED],
    pub action: u8,
    pub abstained: u8,
    pub n_cited: u8,
    pub _pad: u8,
}

impl DecisionRecord {
    pub const EMPTY: Self = DecisionRecord { clock: 0, situation_hash: 0, expected: 0.0, bonus: 0.0, evidence: 0.0,
        cited: [0; CITED], action: 0, abstained: 0, n_cited: 0, _pad: 0 };

    /// The record for `decision`, made by `brain` in situation `x`.
    pub fn new<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>, x: &[f32; D], decision: &Decision) -> Self {
        let mut cited = [0u16; CITED];
        let n = (decision.n_cited as usize).min(CITED);
        cited[..n].copy_from_slice(&decision.cited[..n]);
        DecisionRecord { clock: brain.clock(), situation_hash: situation_hash(x), expected: decision.expected,
            bonus: decision.bonus, evidence: decision.evidence, cited, action: decision.action,
            abstained: decision.abstained as u8, n_cited: decision.n_cited, _pad: 0 }
    }

    /// Little-endian bytes.
    pub fn to_le_bytes(&self) -> [u8; RECORD_BYTES] {
        let mut b = [0u8; RECORD_BYTES];
        b[0..4].copy_from_slice(&self.clock.to_le_bytes());
        b[4..8].copy_from_slice(&self.situation_hash.to_le_bytes());
        b[8..12].copy_from_slice(&self.expected.to_bits().to_le_bytes());
        b[12..16].copy_from_slice(&self.bonus.to_bits().to_le_bytes());
        b[16..20].copy_from_slice(&self.evidence.to_bits().to_le_bytes());
        for (k, c) in self.cited.iter().enumerate() { b[20 + 2 * k..22 + 2 * k].copy_from_slice(&c.to_le_bytes()); }
        b[28] = self.action;
        b[29] = self.abstained;
        b[30] = self.n_cited;
        b[31] = self._pad;
        b
    }

    pub fn from_le_bytes(b: &[u8; RECORD_BYTES]) -> Self {
        let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        let mut cited = [0u16; CITED];
        for (k, c) in cited.iter_mut().enumerate() { *c = u16::from_le_bytes([b[20 + 2 * k], b[21 + 2 * k]]); }
        DecisionRecord { clock: u32_at(0), situation_hash: u32_at(4), expected: f32::from_bits(u32_at(8)),
            bonus: f32::from_bits(u32_at(12)), evidence: f32::from_bits(u32_at(16)), cited,
            action: b[28], abstained: b[29], n_cited: b[30], _pad: b[31] }
    }

    /// Bit-exact equality (f32 compared by bit pattern, so NaN == NaN and -0 != +0).
    pub fn same_bits(&self, other: &Self) -> bool { self.to_le_bytes() == other.to_le_bytes() }
}

/// FNV-1a (32-bit) over the little-endian bit patterns of the situation.
pub fn situation_hash<const D: usize>(x: &[f32; D]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for v in x {
        for byte in v.to_bits().to_le_bytes() { h ^= byte as u32; h = h.wrapping_mul(0x0100_0193); }
    }
    h
}

/// Fixed ring of the newest `M` decision records.
pub struct Recorder<const M: usize> {
    records: [DecisionRecord; M],
    head: usize,
    total: u32,
}

impl<const M: usize> Default for Recorder<M> {
    fn default() -> Self { Self::new() }
}

impl<const M: usize> Recorder<M> {
    pub const fn new() -> Self { Recorder { records: [DecisionRecord::EMPTY; M], head: 0, total: 0 } }

    /// Decisions recorded since start (including those overwritten), wrapping.
    pub fn total(&self) -> u32 { self.total }

    /// Records currently held (at most `M`).
    pub fn len(&self) -> usize { (self.total as usize).min(M) }
    pub fn is_empty(&self) -> bool { self.total == 0 }

    /// Record `decision`, made by `brain` in situation `x`. Overwrites the oldest when full.
    pub fn record<const N: usize, const D: usize, const A: usize>(&mut self, brain: &Brain<N, D, A>, x: &[f32; D], decision: &Decision) {
        self.push(DecisionRecord::new(brain, x, decision));
    }

    pub fn push(&mut self, r: DecisionRecord) {
        if M == 0 { self.total = self.total.wrapping_add(1); return; }
        self.records[self.head] = r;
        self.head = (self.head + 1) % M;
        self.total = self.total.wrapping_add(1);
    }

    /// The `i`-th held record in time order (0 = oldest held).
    pub fn get(&self, i: usize) -> Option<&DecisionRecord> {
        let n = self.len();
        if i >= n { return None; }
        let start = if (self.total as usize) < M { 0 } else { self.head };
        Some(&self.records[(start + i) % M])
    }

    /// Sequence number (0-based, among all decisions ever recorded) of the oldest held record.
    pub fn first_seq(&self) -> u32 { self.total.wrapping_sub(self.len() as u32) }

    /// Serialize the held records, oldest first, into `buf`: a 16-byte header
    /// (magic "SKBR", version u16, record size u16, record count u32, total recorded u32)
    /// then 32 bytes per record, all little-endian. Writes as many whole records as fit
    /// (the newest are kept when the buffer is short) and returns the bytes written,
    /// or 0 if not even the header fits.
    pub fn downlink(&self, buf: &mut [u8]) -> usize {
        if buf.len() < HEADER_BYTES { return 0; }
        let fit = ((buf.len() - HEADER_BYTES) / RECORD_BYTES).min(self.len());
        let skip = self.len() - fit;
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..6].copy_from_slice(&VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&(RECORD_BYTES as u16).to_le_bytes());
        buf[8..12].copy_from_slice(&(fit as u32).to_le_bytes());
        buf[12..16].copy_from_slice(&self.total.to_le_bytes());
        for k in 0..fit {
            let off = HEADER_BYTES + k * RECORD_BYTES;
            buf[off..off + RECORD_BYTES].copy_from_slice(&self.get(skip + k).unwrap().to_le_bytes());
        }
        HEADER_BYTES + fit * RECORD_BYTES
    }
}

/// Why a downlink stream could not be read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StreamError { Short, BadMagic, BadVersion, BadRecordSize, Truncated }

/// Header of a downlink stream: (records in the stream, total decisions recorded on board).
pub fn read_header(buf: &[u8]) -> Result<(u32, u32), StreamError> {
    if buf.len() < HEADER_BYTES { return Err(StreamError::Short); }
    if buf[0..4] != MAGIC { return Err(StreamError::BadMagic); }
    if u16::from_le_bytes([buf[4], buf[5]]) != VERSION { return Err(StreamError::BadVersion); }
    if u16::from_le_bytes([buf[6], buf[7]]) as usize != RECORD_BYTES { return Err(StreamError::BadRecordSize); }
    let n = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let total = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    if buf.len() < HEADER_BYTES + n as usize * RECORD_BYTES { return Err(StreamError::Truncated); }
    Ok((n, total))
}

/// The `i`-th record of a downlink stream (oldest first). Call [`read_header`] first.
pub fn read_record(buf: &[u8], i: usize) -> Option<DecisionRecord> {
    let off = HEADER_BYTES + i * RECORD_BYTES;
    let b: &[u8; RECORD_BYTES] = buf.get(off..off + RECORD_BYTES)?.try_into().ok()?;
    Some(DecisionRecord::from_le_bytes(b))
}

/// Ground side: recompute one decision on a brain rebuilt from the same learn
/// sequence and compare it bit for bit with the flight record.
pub fn replay_one<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>, x: &[f32; D],
    allowed: &[bool; A], safe_default: u8, flight: &DecisionRecord) -> bool {
    let d = brain.decide(x, allowed, safe_default);
    DecisionRecord::new(brain, x, &d).same_bits(flight)
}

/// Running tally of a ground replay.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Verification {
    pub checked: u32,
    pub matched: u32,
    /// Sequence number of the first record that did not reproduce.
    pub first_mismatch: Option<u32>,
}

impl Verification {
    pub fn add(&mut self, seq: u32, ok: bool) {
        self.checked += 1;
        if ok { self.matched += 1 } else if self.first_mismatch.is_none() { self.first_mismatch = Some(seq) }
    }
    pub fn all_match(&self) -> bool { self.checked > 0 && self.matched == self.checked }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg { fn f(&mut self) -> f32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    fn world(x: &[f32; 4], a: u8) -> f32 {
        let best = if x[2] > 0.8 { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 };
        if a == best { 1.0 } else if a == 3 { 0.2 } else { 0.0 }
    }

    const STEPS: usize = 2000;

    /// The flight run: decide, record, act (rotating while abstaining early), learn.
    fn flight<const M: usize>(rec: &mut Recorder<M>) {
        let mut br: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut r = Lcg(42);
        for t in 0..STEPS {
            let x = [r.f(), r.f(), r.f(), 1.0];
            let d = br.decide(&x, &[true; 4], 3);
            rec.record(&br, &x, &d);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            br.learn(&x, a, world(&x, a));
        }
    }

    /// The ground replay from a downlink stream; `flip` toggles one bit of one situation.
    fn ground(stream: &[u8], flip: Option<(usize, usize, u32)>) -> Verification {
        let (n, total) = read_header(stream).unwrap();
        let first = total - n;
        let mut br: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut r = Lcg(42);
        let mut v = Verification::default();
        for t in 0..STEPS {
            let mut x = [r.f(), r.f(), r.f(), 1.0];
            let honest = x;
            if let Some((step, feat, bit)) = flip { if step == t { x[feat] = f32::from_bits(x[feat].to_bits() ^ (1 << bit)); } }
            if t as u32 >= first {
                let flight_rec = read_record(stream, t - first as usize).unwrap();
                v.add(t as u32, replay_one(&br, &x, &[true; 4], 3, &flight_rec));
            }
            // The learn sequence is the flight's, which the ground knows (commands + telemetry).
            let d = br.decide(&honest, &[true; 4], 3);
            let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
            br.learn(&honest, a, world(&honest, a));
        }
        v
    }

    #[test]
    fn record_is_32_bytes() {
        assert_eq!(core::mem::size_of::<DecisionRecord>(), 32);
        assert_eq!(RECORD_BYTES, 32);
    }

    #[test]
    fn falsifier_ground_replay_reproduces_every_record_bit_exact() {
        let mut rec: Box<Recorder<2048>> = Box::new(Recorder::new());
        flight(&mut rec);
        assert_eq!(rec.total() as usize, STEPS);
        let mut buf = vec![0u8; HEADER_BYTES + 2048 * RECORD_BYTES];
        let used = rec.downlink(&mut buf);
        assert_eq!(used, HEADER_BYTES + STEPS * RECORD_BYTES);
        let v = ground(&buf[..used], None);
        println!("recorder falsifier: {}/{} records reproduced bit-exact, {} bytes downlinked ({} B/record)", v.matched, v.checked, used, RECORD_BYTES);
        assert_eq!(v.checked as usize, STEPS);
        assert!(v.all_match(), "{:?}", v);
    }

    #[test]
    fn falsifier_one_flipped_situation_bit_is_detected() {
        let mut rec: Box<Recorder<2048>> = Box::new(Recorder::new());
        flight(&mut rec);
        let mut buf = vec![0u8; HEADER_BYTES + 2048 * RECORD_BYTES];
        let used = rec.downlink(&mut buf);
        // Lowest mantissa bit of drift at decision 1234: the smallest possible change to one input.
        let v = ground(&buf[..used], Some((1234, 0, 0)));
        println!("flipped-bit check: {}/{} matched, first mismatch at decision {:?}", v.matched, v.checked, v.first_mismatch);
        assert_eq!(v.first_mismatch, Some(1234));
        assert_eq!(v.matched, v.checked - 1, "only the corrupted decision differs");
    }

    #[test]
    fn ring_keeps_the_newest_and_replays_them() {
        let mut rec: Recorder<64> = Recorder::new();
        flight(&mut rec);
        assert_eq!((rec.len(), rec.total() as usize, rec.first_seq() as usize), (64, STEPS, STEPS - 64));
        let mut buf = [0u8; HEADER_BYTES + 64 * RECORD_BYTES];
        let used = rec.downlink(&mut buf);
        let (n, total) = read_header(&buf[..used]).unwrap();
        assert_eq!((n as usize, total as usize), (64, STEPS));
        // Held records are the last 64 decisions, oldest first: the full-ring replay matches exactly them.
        let v = ground(&buf[..used], None);
        assert_eq!((v.checked, v.matched), (64, 64));
        // A buffer with room for 10 records keeps the newest 10.
        let mut small = [0u8; HEADER_BYTES + 10 * RECORD_BYTES + 7];
        let used = rec.downlink(&mut small);
        assert_eq!(used, HEADER_BYTES + 10 * RECORD_BYTES);
        assert_eq!(read_record(&small, 9), rec.get(63).copied());
    }

    #[test]
    fn stream_errors_are_reported() {
        let rec: Recorder<4> = Recorder::new();
        let mut buf = [0u8; HEADER_BYTES];
        assert_eq!(rec.downlink(&mut buf[..8]), 0);
        assert_eq!(rec.downlink(&mut buf), HEADER_BYTES);
        assert_eq!(read_header(&buf), Ok((0, 0)));
        let mut bad = buf; bad[0] = b'X';
        assert_eq!(read_header(&bad), Err(StreamError::BadMagic));
        let mut bad = buf; bad[4] = 9;
        assert_eq!(read_header(&bad), Err(StreamError::BadVersion));
        let mut bad = buf; bad[8] = 3;
        assert_eq!(read_header(&bad), Err(StreamError::Truncated));
        assert_eq!(read_header(&buf[..10]), Err(StreamError::Short));
    }

    #[test]
    fn bytes_round_trip() {
        let r = DecisionRecord { clock: 7, situation_hash: 0xDEAD_BEEF, expected: -0.25, bonus: 1.5, evidence: 3.0,
            cited: [1, 2, 65535, 4], action: 3, abstained: 1, n_cited: 8, _pad: 0 };
        assert!(DecisionRecord::from_le_bytes(&r.to_le_bytes()).same_bits(&r));
        assert_ne!(situation_hash(&[0.0f32, 1.0]), situation_hash(&[1.0f32, 0.0]));
    }

    #[test]
    fn fits_in_a_static() {
        static _R: Recorder<1024> = Recorder::new();
        assert_eq!(core::mem::size_of::<Recorder<1024>>(), 1024 * 32 + 16);
    }
}
