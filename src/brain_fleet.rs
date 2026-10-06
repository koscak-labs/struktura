//! brain_fleet: fleet memory packets — move experience between brains.
//!
//! A ground testbed, a digital twin or a sister rover can hand episodes from
//! its [`Brain`] to another brain as a compact packet, so the receiver starts
//! with that experience instead of learning everything from its own trials.
//!
//! Two ways to choose what goes into a packet under a byte budget:
//!
//! - [`export_representative`]: evenly spaced across the sender's memory.
//!   **Use this to seed another brain.** Measured on the rover world of
//!   `struktura brain demo` (5 onboard seeds), 64 representative episodes cut
//!   the onboard trials needed to reach 85% right actions versus a blank brain.
//! - [`export_most_surprising`]: the episodes the sender predicted worst,
//!   first. Use this for a downlink to the ground (what to look at first).
//!   **Do not use it to seed a brain**: surprise-ranked episodes are almost all
//!   failures (63 of 64 in the measurement), and a brain seeded with them
//!   learned *slower* than a blank one (701 vs 340 trials).
//!
//! No heap: the caller provides the byte buffer. Deterministic: the same brain
//! and budget always produce the same bytes.
//!
//! Packet layout, all little-endian:
//!
//! ```text
//! offset  size  field
//! 0       4     magic "SKEP" (struktura episodes)
//! 4       1     version (1)
//! 5       1     D  (situation features)
//! 6       1     A  (actions)
//! 7       1     reserved (0)
//! 8       2     count (episodes)
//! 10      2     reserved (0)
//! 12      ...   count x episode: D x f32 situation, u8 action, f32 reward, f32 surprise
//! end-4   4     CRC-32 (IEEE) of every byte before it
//! ```
//!
//! An episode costs `4 D + 9` bytes (25 bytes at D = 4); a packet adds 16.
//!
//! **Integrity, not authenticity.** The CRC-32 detects corruption on the link
//! (bit flips, truncation). It is not a signature: anyone can forge a valid
//! packet. Authenticating the sender needs a cryptographic MAC or signature on
//! top, which this module does not provide.
//!
//! Import validates everything (magic, version, dimensions, exact length,
//! checksum, action range, finite numbers) before it touches the brain, so a
//! bad packet changes nothing.

use crate::brain::{Brain, Episode};

pub const MAGIC: [u8; 4] = *b"SKEP";
pub const VERSION: u8 = 1;
/// Header (12) + checksum (4).
pub const OVERHEAD: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Shorter than a header + checksum.
    TooShort,
    BadMagic,
    BadVersion(u8),
    /// The packet's (D, A) differ from the receiving brain's.
    DimMismatch { d: u8, a: u8 },
    /// Length does not match the episode count (truncated or padded).
    LengthMismatch,
    BadChecksum,
    /// An episode's action is >= A.
    BadAction,
    /// A situation, reward or surprise is NaN or infinite.
    NonFinite,
    /// The output buffer (or budget) cannot hold even an empty packet.
    BufferTooSmall,
}

/// Bytes per episode for `D` features.
pub const fn episode_bytes(d: usize) -> usize { 4 * d + 9 }

/// Total packet size for `count` episodes of `D` features.
pub const fn packet_bytes(d: usize, count: usize) -> usize { OVERHEAD + count * episode_bytes(d) }

/// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320), bitwise: no table, no heap.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 { c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 }; }
    }
    !c
}

fn put_f32(buf: &mut [u8], at: usize, x: f32) { buf[at..at + 4].copy_from_slice(&x.to_bits().to_le_bytes()); }
fn get_f32(buf: &[u8], at: usize) -> f32 { f32::from_bits(u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])) }

fn surprise_key(s: f32) -> f32 { if s.is_nan() { 0.0 } else { s } }

/// How many episodes fit, and the header written; Err if not even an empty packet fits.
fn begin<const N: usize, const D: usize, const A: usize>(brain: &Brain<N, D, A>, buf: &mut [u8], budget: usize) -> Result<usize, Error> {
    let room = budget.min(buf.len());
    if room < OVERHEAD || D > u8::MAX as usize || A > u8::MAX as usize { return Err(Error::BufferTooSmall); }
    let n = ((room - OVERHEAD) / episode_bytes(D)).min(brain.len()).min(u16::MAX as usize);
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4] = VERSION;
    buf[5] = D as u8;
    buf[6] = A as u8;
    buf[7] = 0;
    buf[8..10].copy_from_slice(&(n as u16).to_le_bytes());
    buf[10..12].copy_from_slice(&[0, 0]);
    Ok(n)
}

fn put_episode<const D: usize>(buf: &mut [u8], at: usize, e: &Episode<D>) -> usize {
    for i in 0..D { put_f32(buf, at + 4 * i, e.key[i]); }
    let p = at + 4 * D;
    buf[p] = e.action;
    put_f32(buf, p + 1, e.reward);
    put_f32(buf, p + 5, e.surprise);
    p + 9
}

fn finish(buf: &mut [u8], at: usize) -> usize {
    let crc = crc32(&buf[..at]);
    buf[at..at + 4].copy_from_slice(&crc.to_le_bytes());
    at + 4
}

/// Write up to `budget` bytes (and at most `buf.len()`) of the brain's memory,
/// evenly spaced across its memory slots: a representative sample to seed
/// another brain. Returns the packet length. An empty brain yields a valid
/// empty packet.
pub fn export_representative<const N: usize, const D: usize, const A: usize>(
    brain: &Brain<N, D, A>, buf: &mut [u8], budget: usize,
) -> Result<usize, Error> {
    let n = begin(brain, buf, budget)?;
    // Used slots are 0..len (memory fills in order and replaces in place).
    let len = brain.len();
    let mut at = 12;
    for i in 0..n {
        let slot = i * len / n;
        let e = brain.episode(slot).unwrap();
        at = put_episode(buf, at, e);
    }
    Ok(finish(buf, at))
}

/// Write up to `budget` bytes of the brain's memory, most surprising episodes
/// first (ties: lower memory slot first): a downlink ordering for ground
/// analysis. Not a good seed for another brain (see the module docs).
pub fn export_most_surprising<const N: usize, const D: usize, const A: usize>(
    brain: &Brain<N, D, A>, buf: &mut [u8], budget: usize,
) -> Result<usize, Error> {
    let n = begin(brain, buf, budget)?;
    // Selection without allocation: repeatedly take the largest (surprise, -slot)
    // strictly below the previous pick. O(n * N), bounded.
    let mut prev: Option<(f32, usize)> = None;
    let mut at = 12;
    for _ in 0..n {
        let mut best: Option<(f32, usize)> = None;
        for slot in 0..N {
            let Some(e) = brain.episode(slot) else { continue };
            let s = surprise_key(e.surprise);
            let below_prev = match prev { None => true, Some((ps, pslot)) => s < ps || (s == ps && slot > pslot) };
            if !below_prev { continue; }
            let better = match best { None => true, Some((bs, bslot)) => s > bs || (s == bs && slot < bslot) };
            if better { best = Some((s, slot)); }
        }
        let Some((s, slot)) = best else { break };
        at = put_episode(buf, at, brain.episode(slot).unwrap());
        prev = Some((s, slot));
    }
    Ok(finish(buf, at))
}

/// Header fields of a packet (after full validation of shape and checksum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header { pub d: u8, pub a: u8, pub count: u16 }

/// Validate magic, version, length and checksum (not the receiver's dimensions).
pub fn check(buf: &[u8]) -> Result<Header, Error> {
    if buf.len() < OVERHEAD { return Err(Error::TooShort); }
    if buf[0..4] != MAGIC { return Err(Error::BadMagic); }
    if buf[4] != VERSION { return Err(Error::BadVersion(buf[4])); }
    let h = Header { d: buf[5], a: buf[6], count: u16::from_le_bytes([buf[8], buf[9]]) };
    if buf.len() != packet_bytes(h.d as usize, h.count as usize) { return Err(Error::LengthMismatch); }
    let body = buf.len() - 4;
    let crc = u32::from_le_bytes([buf[body], buf[body + 1], buf[body + 2], buf[body + 3]]);
    if crc32(&buf[..body]) != crc { return Err(Error::BadChecksum); }
    Ok(h)
}

/// Episode `i` of a checked packet: (situation, action, reward, surprise).
pub fn read_episode<const D: usize>(buf: &[u8], i: usize) -> ([f32; D], u8, f32, f32) {
    let at = 12 + i * episode_bytes(D);
    let mut key = [0.0f32; D];
    for (j, k) in key.iter_mut().enumerate() { *k = get_f32(buf, at + 4 * j); }
    let p = at + 4 * D;
    (key, buf[p], get_f32(buf, p + 1), get_f32(buf, p + 5))
}

/// Validate a packet completely, then feed its episodes (in packet order)
/// through `learn`. Returns how many episodes were learned. On any error the
/// brain is untouched.
pub fn import<const N: usize, const D: usize, const A: usize>(brain: &mut Brain<N, D, A>, buf: &[u8]) -> Result<usize, Error> {
    let h = check(buf)?;
    if h.d as usize != D || h.a as usize != A { return Err(Error::DimMismatch { d: h.d, a: h.a }); }
    let n = h.count as usize;
    for i in 0..n {
        let (key, action, reward, surprise) = read_episode::<D>(buf, i);
        if action as usize >= A { return Err(Error::BadAction); }
        if !reward.is_finite() || !surprise.is_finite() || key.iter().any(|k| !k.is_finite()) { return Err(Error::NonFinite); }
    }
    for i in 0..n {
        let (key, action, reward, _) = read_episode::<D>(buf, i);
        brain.learn(&key, action, reward);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rover world A from `struktura brain demo`: [drift, spike, temperature, 1].
    fn truth(x: &[f32; 4]) -> u8 { if x[2] > 0.8 { 3 } else if x[1] > 0.6 { 2 } else if x[0] > 0.6 { 1 } else { 0 } }
    fn reward(x: &[f32; 4], a: u8) -> f32 { if a == truth(x) { 1.0 } else if a == 3 { 0.2 } else { 0.0 } }

    struct Xs(u64);
    impl Xs {
        fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 }
        fn sit(&mut self) -> [f32; 4] { [self.f(), self.f(), self.f(), 1.0] }
    }

    /// One onboard step with the demo's policy (rotate while abstaining early).
    fn step(br: &mut Brain<256, 4, 4>, x: &[f32; 4], t: usize) -> bool {
        let d = br.decide(x, &[true; 4], 3);
        let a = if d.abstained && t < 400 { (t % 4) as u8 } else { d.action };
        br.learn(x, a, reward(x, a));
        a == truth(x)
    }

    fn trained_ground() -> Brain<256, 4, 4> {
        let mut g: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        let mut r = Xs(0x1234_5678_9ABC_DEF1);
        for t in 0..4000 { let x = r.sit(); step(&mut g, &x, t); }
        g
    }

    /// Trials until the rolling right-action rate over the last `win` trials reaches `target`.
    fn trials_to(br: &mut Brain<256, 4, 4>, seed: u64, target: f32, win: usize, max: usize) -> Option<usize> {
        let mut r = Xs(seed);
        let mut hist = [false; 4096];
        let mut ok = 0usize;
        for t in 0..max {
            let x = r.sit();
            let good = step(br, &x, t);
            let slot = t % win;
            if t >= win && hist[slot] { ok -= 1; }
            hist[slot] = good;
            if good { ok += 1; }
            if t + 1 >= win && ok as f32 / win as f32 >= target { return Some(t + 1); }
        }
        None
    }

    fn seeded_from(buf: &[u8]) -> Brain<256, 4, 4> {
        let mut b: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        import(&mut b, buf).unwrap();
        b
    }

    #[test]
    fn crc_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn round_trip_is_bit_exact_for_both_exporters() {
        let g = trained_ground();
        for most_surprising in [true, false] {
            let mut buf = [0u8; 4096];
            let len = if most_surprising { export_most_surprising(&g, &mut buf, packet_bytes(4, 64)) }
                else { export_representative(&g, &mut buf, packet_bytes(4, 64)) }.unwrap();
            assert_eq!(len, packet_bytes(4, 64));
            assert_eq!(episode_bytes(4), 25);
            let h = check(&buf[..len]).unwrap();
            assert_eq!((h.d, h.a, h.count), (4, 4, 64));
            let mut last = f32::INFINITY;
            for i in 0..64 {
                let (key, action, rew, sur) = read_episode::<4>(&buf, i);
                if most_surprising { assert!(sur <= last, "most surprising first"); last = sur; }
                let found = (0..256).filter_map(|s| g.episode(s)).any(|e| e.action == action
                    && e.reward.to_bits() == rew.to_bits() && e.surprise.to_bits() == sur.to_bits()
                    && e.key.iter().zip(key.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
                assert!(found, "episode {} round-trips bit-exactly", i);
            }
            // Deterministic: same brain, same budget, same bytes.
            let mut buf2 = [0u8; 4096];
            let len2 = if most_surprising { export_most_surprising(&g, &mut buf2, packet_bytes(4, 64)) }
                else { export_representative(&g, &mut buf2, packet_bytes(4, 64)) }.unwrap();
            assert_eq!(buf[..len], buf2[..len2]);
        }
        // Budget smaller than one episode still gives a valid empty packet; smaller than the overhead is refused.
        let mut tiny = [0u8; 20];
        let n = export_representative(&g, &mut tiny, 40).unwrap();
        assert_eq!(n, OVERHEAD);
        assert_eq!(check(&tiny[..n]).unwrap().count, 0);
        assert_eq!(export_most_surprising(&g, &mut tiny, 10), Err(Error::BufferTooSmall));
        // An empty brain exports an empty packet that imports as 0 episodes.
        let empty: Brain<16, 4, 4> = Brain::new(0.3, 0.5);
        let mut eb = [0u8; 64];
        let el = export_representative(&empty, &mut eb, 64).unwrap();
        let mut rx: Brain<16, 4, 4> = Brain::new(0.3, 0.5);
        assert_eq!(import(&mut rx, &eb[..el]), Ok(0));
    }

    #[test]
    fn bad_packets_are_rejected_and_change_nothing() {
        let g = trained_ground();
        let mut buf = [0u8; 4096];
        let len = export_representative(&g, &mut buf, packet_bytes(4, 16)).unwrap();
        let good = &buf[..len];

        let mut fresh: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
        // corrupted byte: in an episode's reward, then in the header's count
        for at in [12 + 17, 9] {
            let mut bad = [0u8; 4096];
            bad[..len].copy_from_slice(good);
            bad[at] ^= 0x40;
            assert!(import(&mut fresh, &bad[..len]).is_err(), "flip at {}", at);
        }
        let mut bad = [0u8; 4096];
        bad[..len].copy_from_slice(good);
        bad[12 + 17] ^= 0x01;
        assert_eq!(import(&mut fresh, &bad[..len]), Err(Error::BadChecksum));
        bad[..len].copy_from_slice(good);
        bad[0] = b'X';
        assert_eq!(import(&mut fresh, &bad[..len]), Err(Error::BadMagic));
        bad[..len].copy_from_slice(good);
        bad[4] = 2;
        assert_eq!(import(&mut fresh, &bad[..len]), Err(Error::BadVersion(2)));
        // truncated
        assert_eq!(import(&mut fresh, &good[..len - 1]), Err(Error::LengthMismatch));
        assert_eq!(import(&mut fresh, &good[..10]), Err(Error::TooShort));
        // wrong dimensions
        let mut d3: Brain<64, 3, 4> = Brain::new(0.3, 0.5);
        assert_eq!(import(&mut d3, good), Err(Error::DimMismatch { d: 4, a: 4 }));
        let mut a2: Brain<64, 4, 2> = Brain::new(0.3, 0.5);
        assert_eq!(import(&mut a2, good), Err(Error::DimMismatch { d: 4, a: 4 }));
        // a valid checksum over a bad action or a NaN is still refused
        let mut forged = [0u8; 4096];
        forged[..len].copy_from_slice(good);
        forged[12 + 16] = 9;
        let crc = crc32(&forged[..len - 4]);
        forged[len - 4..len].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(import(&mut fresh, &forged[..len]), Err(Error::BadAction));
        forged[..len].copy_from_slice(good);
        forged[12..16].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        let crc = crc32(&forged[..len - 4]);
        forged[len - 4..len].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(import(&mut fresh, &forged[..len]), Err(Error::NonFinite));
        assert!(fresh.is_empty() && fresh.clock() == 0, "rejected packets leave the brain untouched");
        // the good packet imports
        assert_eq!(import(&mut fresh, good), Ok(16));
        assert_eq!(fresh.len(), 16);
    }

    /// FALSIFIER: a brain seeded with 64 episodes from a trained ground brain must
    /// reach 85% right actions (rolling 200) onboard in fewer trials than a blank
    /// brain, on every one of 5 onboard situation streams.
    #[test]
    fn falsifier_seeded_brain_learns_onboard_in_fewer_trials() {
        let g = trained_ground();
        let (target, win, max) = (0.85f32, 200usize, 4000usize);
        let mut rep = [0u8; 4096];
        let rl = export_representative(&g, &mut rep, packet_bytes(4, 64)).unwrap();
        let mut sur = [0u8; 4096];
        let sl = export_most_surprising(&g, &mut sur, packet_bytes(4, 64)).unwrap();
        let bad = (0..64).filter(|&i| read_episode::<4>(&sur[..sl], i).2 < 1.0).count();
        let seeds = [0x0DDB_A11C_AFE5_EED5u64, 0x5EED_0000_0000_0001, 0x5EED_0000_0000_0002, 0x5EED_0000_0000_0003, 0x5EED_0000_0000_0004];
        let (mut sb, mut sr, mut ss, mut wins) = (0usize, 0usize, 0usize, 0usize);
        for &seed in &seeds {
            let mut blank: Brain<256, 4, 4> = Brain::new(0.3, 0.5);
            let b = trials_to(&mut blank, seed, target, win, max).expect("blank reaches the target");
            let r = trials_to(&mut seeded_from(&rep[..rl]), seed, target, win, max).expect("seeded (representative) reaches the target");
            let s = trials_to(&mut seeded_from(&sur[..sl]), seed, target, win, max).unwrap_or(max);
            std::println!("fleet falsifier seed {:016x}: trials to {:.0}% (rolling {}): blank {}, seeded representative {}, seeded most-surprising {}",
                seed, target * 100.0, win, b, r, s);
            sb += b; sr += r; ss += s;
            if r < b { wins += 1; }
        }
        std::println!("fleet falsifier mean over {} seeds: blank {:.0}, seeded representative {:.0} ({} bytes), seeded most-surprising {:.0} ({} of 64 episodes were failures)",
            seeds.len(), sb as f32 / 5.0, sr as f32 / 5.0, rl, ss as f32 / 5.0, bad);
        assert_eq!(wins, seeds.len(), "representative seeding must win on every seed");
    }
}
