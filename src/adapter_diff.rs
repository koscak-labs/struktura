//! LoRA adapter diff: what a PEFT adapter actually changes, and how two checkpoints differ.
//!
//! Reads `.safetensors` (std-only parser: 8-byte LE header length, JSON header, raw data; dtypes
//! F32 / F16 / BF16 / F64, converted to f32 on load, all math in f64), pairs `<module>.lora_A` [r x in]
//! with `<module>.lora_B` [out x r] and reports, per module, the update `dW = scale * B A`:
//! Frobenius norm, spectral norm, effective rank (exp of the entropy of `sigma_i^2 / sum sigma^2`),
//! stable rank (`|dW|_F^2 / sigma_max^2`) and the top-k singular values.
//!
//! `dW` is never materialized (out x in). Everything happens in r-space:
//!
//! - `sigma_i(dW)^2 = scale^2 * eig_i(G H)` with `G = B^T B`, `H = A A^T` (both r x r). `G H` is not
//!   symmetric, so we take the similar symmetric form `S G S` with `S = H^{1/2}` (Jacobi on `H`,
//!   negative round-off eigenvalues clamped to 0, so no Cholesky ridge is needed) and run Jacobi on it.
//! - `|dW|_F^2 = scale^2 * tr(G H)`.
//! - Two checkpoints: `<Xa, Xb> = sa sb tr((Ba^T Bb)(Ab Aa^T))`, so
//!   `|dWb - dWa|_F^2 = |dWb|^2 + |dWa|^2 - 2 <Xa, Xb>` and `cos = <Xa, Xb> / (|dWa| |dWb|)`.
//!
//! Scale: `lora_alpha / r` (or `/ sqrt(r)` with `use_rslora`), from `adapter_config.json` next to the
//! file, overridable by the caller; unknown alpha means scale 1. Verified against a brute-force path
//! that materializes `dW` (tests below). CPU only, no inference.

use std::collections::BTreeMap;

// ---------------------------------------------------------------- minimal JSON (header + config)

/// A JSON value: just enough for safetensors headers and `adapter_config.json`.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self { Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, j)| j), _ => None }
    }
    pub fn num(&self) -> Option<f64> { if let Json::Num(x) = self { Some(*x) } else { None } }
    pub fn str(&self) -> Option<&str> { if let Json::Str(s) = self { Some(s) } else { None } }
}

/// Parse one JSON document. `None` on any syntax error or trailing garbage.
pub fn parse_json(s: &str) -> Option<Json> {
    let b = s.as_bytes();
    let mut i = 0;
    let v = value(b, &mut i)?;
    ws(b, &mut i);
    if i == b.len() { Some(v) } else { None }
}

fn ws(b: &[u8], i: &mut usize) { while *i < b.len() && b[*i].is_ascii_whitespace() { *i += 1 } }

fn value(b: &[u8], i: &mut usize) -> Option<Json> {
    ws(b, i);
    match *b.get(*i)? {
        b'{' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b'}') { *i += 1; return Some(Json::Obj(v)) }
            loop {
                ws(b, i);
                let k = string(b, i)?;
                ws(b, i);
                if b.get(*i) != Some(&b':') { return None }
                *i += 1;
                v.push((k, value(b, i)?));
                ws(b, i);
                match b.get(*i)? { b',' => *i += 1, b'}' => { *i += 1; return Some(Json::Obj(v)) } _ => return None }
            }
        }
        b'[' => {
            *i += 1;
            let mut v = Vec::new();
            ws(b, i);
            if b.get(*i) == Some(&b']') { *i += 1; return Some(Json::Arr(v)) }
            loop {
                v.push(value(b, i)?);
                ws(b, i);
                match b.get(*i)? { b',' => *i += 1, b']' => { *i += 1; return Some(Json::Arr(v)) } _ => return None }
            }
        }
        b'"' => string(b, i).map(Json::Str),
        b't' if b[*i..].starts_with(b"true") => { *i += 4; Some(Json::Bool(true)) }
        b'f' if b[*i..].starts_with(b"false") => { *i += 5; Some(Json::Bool(false)) }
        b'n' if b[*i..].starts_with(b"null") => { *i += 4; Some(Json::Null) }
        _ => {
            let st = *i;
            while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { *i += 1 }
            std::str::from_utf8(&b[st..*i]).ok()?.parse::<f64>().ok().map(Json::Num)
        }
    }
}

fn string(b: &[u8], i: &mut usize) -> Option<String> {
    if b.get(*i) != Some(&b'"') { return None }
    *i += 1;
    let mut out: Vec<u8> = Vec::new();
    loop {
        let c = *b.get(*i)?;
        *i += 1;
        match c {
            b'"' => return String::from_utf8(out).ok(),
            b'\\' => {
                let e = *b.get(*i)?;
                *i += 1;
                match e {
                    b'"' => out.push(b'"'), b'\\' => out.push(b'\\'), b'/' => out.push(b'/'),
                    b'b' => out.push(8), b'f' => out.push(12), b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'), b't' => out.push(b'\t'),
                    b'u' => {
                        let h = std::str::from_utf8(b.get(*i..*i + 4)?).ok()?;
                        let cp = u32::from_str_radix(h, 16).ok()?;
                        *i += 4;
                        // surrogate pairs are not expected in tensor names; map lone ones to U+FFFD
                        let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return None,
                }
            }
            _ => out.push(c),
        }
    }
}

// ---------------------------------------------------------------- safetensors

/// Element types we can read. Anything else is listed in [`SafeTensors::skipped`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype { F32, F16, BF16, F64 }

impl Dtype {
    pub fn parse(s: &str) -> Option<Dtype> {
        match s { "F32" => Some(Dtype::F32), "F16" => Some(Dtype::F16), "BF16" => Some(Dtype::BF16), "F64" => Some(Dtype::F64), _ => None }
    }
    pub fn name(self) -> &'static str {
        match self { Dtype::F32 => "F32", Dtype::F16 => "F16", Dtype::BF16 => "BF16", Dtype::F64 => "F64" }
    }
    pub fn size(self) -> usize { match self { Dtype::F16 | Dtype::BF16 => 2, Dtype::F32 => 4, Dtype::F64 => 8 } }
}

/// IEEE half -> f32 (subnormals, inf, NaN included).
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match (e, m) {
        (0, 0) => s,
        (0, _) => { let v = m as f32 * (1.0 / 16_777_216.0); return if s != 0 { -v } else { v } } // m * 2^-24
        (31, 0) => s | 0x7f80_0000,
        (31, _) => s | 0x7fc0_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half, round to nearest even; overflow to inf, tiny to subnormal / zero.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 { return s | 0x7c00 | if m != 0 { 0x200 } else { 0 } }
    let e16 = e - 112;
    if e16 >= 31 { return s | 0x7c00 }
    if e16 <= 0 {
        if e16 < -10 { return s }
        let mant = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) { r += 1 }
        return s | r as u16;
    }
    let mut r = ((e16 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) { r += 1 } // carry into exponent is correct rounding
    s | r as u16
}

pub fn bf16_to_f32(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }

/// f32 -> bf16, round to nearest even.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() { return ((b >> 16) | 0x40) as u16 }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// One loaded tensor, values widened to f32, row-major.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A parsed `.safetensors` file.
#[derive(Clone, Debug, Default)]
pub struct SafeTensors {
    pub tensors: Vec<Tensor>,
    /// `__metadata__` string pairs.
    pub metadata: Vec<(String, String)>,
    /// `(name, dtype)` of tensors with a dtype we do not read.
    pub skipped: Vec<(String, String)>,
}

impl SafeTensors {
    pub fn get(&self, name: &str) -> Option<&Tensor> { self.tensors.iter().find(|t| t.name == name) }
}

/// Parse a whole `.safetensors` file held in memory.
pub fn parse_safetensors(bytes: &[u8]) -> Result<SafeTensors, String> {
    if bytes.len() < 8 { return Err("file shorter than the 8-byte header length".into()) }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if n > bytes.len() - 8 { return Err(format!("header length {} exceeds file size {}", n, bytes.len())) }
    let head = std::str::from_utf8(&bytes[8..8 + n]).map_err(|_| "header is not UTF-8".to_string())?;
    let json = parse_json(head.trim_end_matches(' ')).ok_or("header is not valid JSON")?;
    let Json::Obj(entries) = json else { return Err("header is not a JSON object".into()) };
    let data = &bytes[8 + n..];
    let mut st = SafeTensors::default();
    for (name, v) in entries {
        if name == "__metadata__" {
            if let Json::Obj(m) = v { for (k, x) in m { st.metadata.push((k, x.str().unwrap_or("").to_string())) } }
            continue;
        }
        let dts = v.get("dtype").and_then(Json::str).ok_or_else(|| format!("{}: no dtype", name))?;
        let shape: Vec<usize> = match v.get("shape") {
            Some(Json::Arr(a)) => a.iter().map(|x| x.num().map(|f| f as usize)).collect::<Option<_>>().ok_or_else(|| format!("{}: bad shape", name))?,
            _ => return Err(format!("{}: no shape", name)),
        };
        let off: Vec<usize> = match v.get("data_offsets") {
            Some(Json::Arr(a)) if a.len() == 2 => a.iter().map(|x| x.num().map(|f| f as usize)).collect::<Option<_>>().ok_or_else(|| format!("{}: bad data_offsets", name))?,
            _ => return Err(format!("{}: no data_offsets", name)),
        };
        let Some(dt) = Dtype::parse(dts) else { st.skipped.push((name, dts.to_string())); continue };
        let (lo, hi) = (off[0], off[1]);
        let bytes_needed = shape.iter().try_fold(dt.size(), |a, &d| a.checked_mul(d)).ok_or_else(|| format!("{}: shape {:?} overflows", name, shape))?;
        if lo > hi || hi > data.len() { return Err(format!("{}: data_offsets [{}, {}] outside data ({} bytes)", name, lo, hi, data.len())) }
        if hi - lo != bytes_needed {
            return Err(format!("{}: {} bytes for {} x {}", name, hi - lo, bytes_needed / dt.size(), dt.name()))
        }
        let raw = &data[lo..hi];
        let vals: Vec<f32> = match dt {
            Dtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            Dtype::F64 => raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            Dtype::F16 => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Dtype::BF16 => raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        };
        st.tensors.push(Tensor { name, dtype: dt, shape, data: vals });
    }
    Ok(st)
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// Serialize tensors to `.safetensors` bytes (used by the round-trip test; header padded to 8).
pub fn write_safetensors(tensors: &[(&str, Dtype, &[usize], &[f32])], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    let mut head = String::from("{");
    if !metadata.is_empty() {
        head.push_str("\"__metadata__\":{");
        for (i, (k, v)) in metadata.iter().enumerate() {
            if i > 0 { head.push(',') }
            head.push_str(&format!("\"{}\":\"{}\"", esc(k), esc(v)));
        }
        head.push('}');
    }
    for (i, (name, dt, shape, vals)) in tensors.iter().enumerate() {
        let lo = data.len();
        for &x in vals.iter() {
            match dt {
                Dtype::F32 => data.extend_from_slice(&x.to_le_bytes()),
                Dtype::F64 => data.extend_from_slice(&(x as f64).to_le_bytes()),
                Dtype::F16 => data.extend_from_slice(&f32_to_f16(x).to_le_bytes()),
                Dtype::BF16 => data.extend_from_slice(&f32_to_bf16(x).to_le_bytes()),
            }
        }
        if i > 0 || !metadata.is_empty() { head.push(',') }
        let sh: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        head.push_str(&format!("\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", esc(name), dt.name(), sh.join(","), lo, data.len()));
    }
    head.push('}');
    while head.len() % 8 != 0 { head.push(' ') }
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(&data);
    out
}

// ---------------------------------------------------------------- small dense linear algebra (f64)

/// Row-major dense matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat { pub rows: usize, pub cols: usize, pub data: Vec<f64> }

impl Mat {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Mat { assert_eq!(data.len(), rows * cols); Mat { rows, cols, data } }
    pub fn zeros(rows: usize, cols: usize) -> Mat { Mat { rows, cols, data: vec![0.0; rows * cols] } }
    #[inline] pub fn at(&self, i: usize, j: usize) -> f64 { self.data[i * self.cols + j] }
    pub fn mul(&self, o: &Mat) -> Mat {
        assert_eq!(self.cols, o.rows);
        let mut c = Mat::zeros(self.rows, o.cols);
        for i in 0..self.rows { for k in 0..self.cols {
            let a = self.at(i, k);
            if a == 0.0 { continue }
            for j in 0..o.cols { c.data[i * o.cols + j] += a * o.at(k, j) }
        } }
        c
    }
    pub fn t(&self) -> Mat {
        let mut c = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { c.data[j * self.rows + i] = self.at(i, j) } }
        c
    }
}

/// `X^T Y` over shared rows: X [n x p], Y [n x q] -> [p x q]. Streams rows, never transposes.
pub fn gram_cols(x: &Mat, y: &Mat) -> Mat {
    assert_eq!(x.rows, y.rows);
    let mut c = Mat::zeros(x.cols, y.cols);
    for o in 0..x.rows {
        let xr = &x.data[o * x.cols..(o + 1) * x.cols];
        let yr = &y.data[o * y.cols..(o + 1) * y.cols];
        for (i, &a) in xr.iter().enumerate() { if a != 0.0 { for (j, &b) in yr.iter().enumerate() { c.data[i * y.cols + j] += a * b } } }
    }
    c
}

/// `X Y^T` over shared columns: X [p x n], Y [q x n] -> [p x q].
pub fn gram_rows(x: &Mat, y: &Mat) -> Mat {
    assert_eq!(x.cols, y.cols);
    let mut c = Mat::zeros(x.rows, y.rows);
    for i in 0..x.rows { for j in 0..y.rows {
        let (xr, yr) = (&x.data[i * x.cols..(i + 1) * x.cols], &y.data[j * y.cols..(j + 1) * y.cols]);
        c.data[i * y.rows + j] = xr.iter().zip(yr).map(|(a, b)| a * b).sum();
    } }
    c
}

/// `tr(P Q)` for P [p x q], Q [q x p].
pub fn trace_mul(p: &Mat, q: &Mat) -> f64 {
    assert!(p.cols == q.rows && p.rows == q.cols);
    let mut s = 0.0;
    for i in 0..p.rows { for j in 0..p.cols { s += p.at(i, j) * q.at(j, i) } }
    s
}

/// Cyclic Jacobi on a symmetric matrix. Returns (eigenvalues, eigenvectors as columns of V),
/// unsorted. Converges to machine precision for the small (r x r) matrices used here.
pub fn jacobi_eig(m: &Mat) -> (Vec<f64>, Mat) {
    let n = m.rows;
    assert_eq!(n, m.cols);
    let mut a = m.clone();
    let mut v = Mat::zeros(n, n);
    for i in 0..n { v.data[i * n + i] = 1.0 }
    let scale: f64 = a.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    if scale == 0.0 { return (vec![0.0; n], v) }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a.at(i, j).powi(2)).sum();
        if off.sqrt() <= 1e-15 * scale { break }
        for p in 0..n { for q in p + 1..n {
            let apq = a.at(p, q);
            if apq.abs() <= 1e-300 { continue }
            let theta = (a.at(q, q) - a.at(p, p)) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..n { // rotate columns p, q
                let (akp, akq) = (a.at(k, p), a.at(k, q));
                a.data[k * n + p] = c * akp - s * akq;
                a.data[k * n + q] = s * akp + c * akq;
            }
            for k in 0..n { // rotate rows p, q
                let (apk, aqk) = (a.at(p, k), a.at(q, k));
                a.data[p * n + k] = c * apk - s * aqk;
                a.data[q * n + k] = s * apk + c * aqk;
            }
            for k in 0..n {
                let (vkp, vkq) = (v.at(k, p), v.at(k, q));
                v.data[k * n + p] = c * vkp - s * vkq;
                v.data[k * n + q] = s * vkp + c * vkq;
            }
        } }
    }
    ((0..n).map(|i| a.at(i, i)).collect(), v)
}

/// Symmetric PSD square root via Jacobi; negative round-off eigenvalues clamp to 0.
pub fn psd_sqrt(h: &Mat) -> Mat {
    let n = h.rows;
    let (d, v) = jacobi_eig(h);
    let mut s = Mat::zeros(n, n);
    for i in 0..n { for j in 0..n {
        s.data[i * n + j] = (0..n).map(|k| v.at(i, k) * d[k].max(0.0).sqrt() * v.at(j, k)).sum();
    } }
    s
}

// ---------------------------------------------------------------- LoRA pairing and config

/// One LoRA-adapted module: `dW = scale * B A`, A [r x in], B [out x r].
#[derive(Clone, Debug)]
pub struct LoraModule {
    /// Path with `base_model.model.` and the `.lora_A.weight` suffix stripped.
    pub name: String,
    /// Module type: the last path component (`q_proj`, `down_proj`, ...).
    pub kind: String,
    pub a: Mat,
    pub b: Mat,
}

impl LoraModule {
    pub fn rank(&self) -> usize { self.a.rows }
}

fn split_lora(name: &str) -> Option<(String, char)> {
    // `<path>.lora_A.weight`, `<path>.lora_A.<adapter>.weight`, `<path>.lora_embedding_A[.<adapter>]`
    for pat in [".lora_embedding_", ".lora_"] {
        if let Some(p) = name.rfind(pat) {
            let rest = &name[p + pat.len()..];
            let ab = rest.chars().next()?;
            if ab != 'A' && ab != 'B' { continue }
            let tail = &rest[1..];
            if !(tail.is_empty() || tail.starts_with('.')) { continue }
            let path = name[..p].trim_start_matches("base_model.model.").to_string();
            return Some((path, ab));
        }
    }
    None
}

/// Pair `lora_A` / `lora_B` tensors per module. Returns modules sorted by name, plus the names of
/// tensors that are not part of a valid pair (with the reason).
pub fn pair_lora(st: &SafeTensors) -> (Vec<LoraModule>, Vec<String>) {
    let mut a_of: BTreeMap<String, &Tensor> = BTreeMap::new();
    let mut b_of: BTreeMap<String, &Tensor> = BTreeMap::new();
    let mut other = Vec::new();
    for t in &st.tensors {
        match split_lora(&t.name) {
            Some((p, 'A')) => { a_of.insert(p, t); }
            Some((p, _)) => { b_of.insert(p, t); }
            None => other.push(format!("{} (not a lora_A/lora_B tensor)", t.name)),
        }
    }
    let mut mods = Vec::new();
    for (p, ta) in &a_of {
        let Some(tb) = b_of.get(p) else { other.push(format!("{} (no lora_B)", ta.name)); continue };
        if ta.shape.len() != 2 || tb.shape.len() != 2 { other.push(format!("{} (not 2-D)", p)); continue }
        let to64 = |t: &Tensor| t.data.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        // Linear: A [r x in], B [out x r]. Embedding: A [r x vocab], B [dim x r]; dW = (B A)^T has the
        // same singular values, so the same r-space math applies with no transpose.
        let (a, b) = (Mat::new(ta.shape[0], ta.shape[1], to64(ta)), Mat::new(tb.shape[0], tb.shape[1], to64(tb)));
        if b.cols != a.rows { other.push(format!("{} (rank mismatch: A {:?}, B {:?})", p, ta.shape, tb.shape)); continue }
        let kind = p.rsplit('.').next().unwrap_or(p).to_string();
        mods.push(LoraModule { name: p.clone(), kind, a, b });
    }
    for (p, tb) in &b_of { if !a_of.contains_key(p) { other.push(format!("{} (no lora_A)", tb.name)) } }
    (mods, other)
}

/// LoRA scaling inputs. `scale(r)` = alpha / r (alpha / sqrt(r) with rsLoRA); 1 when alpha is unknown.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoraCfg { pub alpha: Option<f64>, pub r: Option<f64>, pub rslora: bool }

impl LoraCfg {
    /// Read `lora_alpha`, `r`, `use_rslora` from an `adapter_config.json` text.
    pub fn from_json(text: &str) -> Option<LoraCfg> {
        let j = parse_json(text)?;
        Some(LoraCfg {
            alpha: j.get("lora_alpha").and_then(Json::num),
            r: j.get("r").and_then(Json::num),
            rslora: matches!(j.get("use_rslora"), Some(Json::Bool(true))),
        })
    }
    /// Scale for a module of tensor rank `rank` (used when the config gives no `r`).
    pub fn scale(&self, rank: usize) -> f64 {
        match self.alpha {
            None => 1.0,
            Some(a) => {
                let r = self.r.unwrap_or(rank as f64).max(1e-300);
                if self.rslora { a / r.sqrt() } else { a / r }
            }
        }
    }
}

// ---------------------------------------------------------------- per-module statistics

/// What one module's `dW` looks like.
#[derive(Clone, Debug)]
pub struct ModuleStats {
    pub name: String,
    pub kind: String,
    pub rank: usize,
    pub out_dim: usize,
    pub in_dim: usize,
    pub scale: f64,
    /// `|dW|_F`.
    pub fro: f64,
    /// `sigma_max`.
    pub spectral: f64,
    /// exp(entropy of sigma_i^2 / sum sigma^2); 0 for dW = 0.
    pub eff_rank: f64,
    /// `|dW|_F^2 / sigma_max^2`; 0 for dW = 0.
    pub stable_rank: f64,
    /// All r singular values, descending.
    pub sv: Vec<f64>,
}

/// Singular values (descending) of `scale * B A`, computed in r-space.
pub fn singular_values(m: &LoraModule, scale: f64) -> Vec<f64> {
    let g = gram_cols(&m.b, &m.b); // B^T B
    let h = gram_rows(&m.a, &m.a); // A A^T
    let s = psd_sqrt(&h);
    let (mut ev, _) = jacobi_eig(&s.mul(&g).mul(&s));
    ev.iter_mut().for_each(|x| *x = scale.abs() * x.max(0.0).sqrt());
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    ev
}

/// `|dW|_F` and the r-space spectral summary of one module.
pub fn module_stats(m: &LoraModule, scale: f64) -> ModuleStats {
    let g = gram_cols(&m.b, &m.b);
    let h = gram_rows(&m.a, &m.a);
    let fro = (scale * scale * trace_mul(&g, &h).max(0.0)).sqrt();
    let sv = singular_values(m, scale);
    let spectral = sv.first().copied().unwrap_or(0.0);
    let ss: f64 = sv.iter().map(|s| s * s).sum();
    let (eff_rank, stable_rank) = if ss > 0.0 && spectral > 0.0 {
        let ent: f64 = sv.iter().map(|s| s * s / ss).filter(|&p| p > 0.0).map(|p| -p * p.ln()).sum();
        (ent.exp(), fro * fro / (spectral * spectral))
    } else { (0.0, 0.0) };
    ModuleStats { name: m.name.clone(), kind: m.kind.clone(), rank: m.rank(), out_dim: m.b.rows, in_dim: m.a.cols, scale, fro, spectral, eff_rank, stable_rank, sv }
}

/// How two checkpoints of one module differ.
#[derive(Clone, Debug)]
pub struct ModuleDiff {
    pub name: String,
    pub kind: String,
    pub fro_a: f64,
    pub fro_b: f64,
    /// `<dWa, dWb>` (Frobenius inner product).
    pub inner: f64,
    /// `|dWb - dWa|_F`.
    pub diff: f64,
    /// `<dWa, dWb> / (|dWa| |dWb|)`; `None` when either is 0.
    pub cos: Option<f64>,
}

/// `|dWb - dWa|_F` and the cosine, from r x r traces only. Errors on mismatched in/out dims.
pub fn module_diff(a: &LoraModule, sa: f64, b: &LoraModule, sb: f64) -> Result<ModuleDiff, String> {
    if a.a.cols != b.a.cols || a.b.rows != b.b.rows {
        return Err(format!("{}: shape {}x{} vs {}x{}", a.name, a.b.rows, a.a.cols, b.b.rows, b.a.cols));
    }
    let na = sa * sa * trace_mul(&gram_cols(&a.b, &a.b), &gram_rows(&a.a, &a.a));
    let nb = sb * sb * trace_mul(&gram_cols(&b.b, &b.b), &gram_rows(&b.a, &b.a));
    let inner = sa * sb * trace_mul(&gram_cols(&a.b, &b.b), &gram_rows(&b.a, &a.a)); // tr((Ba^T Bb)(Ab Aa^T))
    let (fa, fb) = (na.max(0.0).sqrt(), nb.max(0.0).sqrt());
    let diff = (na + nb - 2.0 * inner).max(0.0).sqrt();
    let cos = if fa > 0.0 && fb > 0.0 { Some((inner / (fa * fb)).clamp(-1.0, 1.0)) } else { None };
    Ok(ModuleDiff { name: a.name.clone(), kind: a.kind.clone(), fro_a: fa, fro_b: fb, inner, diff, cos })
}

// ---------------------------------------------------------------- loading and reports

/// A loaded adapter: modules, per-module scale, and where the scale came from.
#[derive(Clone, Debug)]
pub struct Adapter {
    pub path: String,
    pub modules: Vec<LoraModule>,
    pub unpaired: Vec<String>,
    pub skipped: Vec<(String, String)>,
    pub dtypes: Vec<&'static str>,
    pub cfg: LoraCfg,
    pub scale_source: String,
    pub n_tensors: usize,
}

impl Adapter {
    pub fn scale(&self, m: &LoraModule) -> f64 { self.cfg.scale(m.rank()) }
}

/// Load `path`; scale from `adapter_config.json` beside it, with `alpha` / `rank` overriding it.
pub fn load_adapter(path: &str, alpha: Option<f64>, rank: Option<f64>) -> Result<Adapter, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
    let st = parse_safetensors(&bytes).map_err(|e| format!("{}: {}", path, e))?;
    let cfg_path = std::path::Path::new(path).with_file_name("adapter_config.json");
    let file_cfg = std::fs::read_to_string(&cfg_path).ok().and_then(|t| LoraCfg::from_json(&t));
    let mut cfg = file_cfg.unwrap_or_default();
    let mut src = match file_cfg { Some(_) => format!("{}", cfg_path.display()), None => "no adapter_config.json".to_string() };
    if alpha.is_some() || rank.is_some() {
        if alpha.is_some() { cfg.alpha = alpha }
        if rank.is_some() { cfg.r = rank }
        src = format!("{} + CLI override", src);
    }
    if cfg.alpha.is_none() { src = format!("{}; alpha unknown -> scale 1", src) }
    let (modules, unpaired) = pair_lora(&st);
    let mut dtypes: Vec<&'static str> = st.tensors.iter().map(|t| t.dtype.name()).collect();
    dtypes.sort();
    dtypes.dedup();
    Ok(Adapter { path: path.to_string(), n_tensors: st.tensors.len() + st.skipped.len(), modules, unpaired, skipped: st.skipped, dtypes, cfg, scale_source: src })
}

/// Per-kind aggregate: count, `sqrt(sum |dW|_F^2)`, mean effective / stable rank, max spectral.
#[derive(Clone, Debug, Default)]
pub struct KindSummary { pub kind: String, pub n: usize, pub fro: f64, pub mean_eff_rank: f64, pub mean_stable_rank: f64, pub max_spectral: f64 }

pub fn group_stats(stats: &[ModuleStats]) -> Vec<KindSummary> {
    let mut g: BTreeMap<&str, KindSummary> = BTreeMap::new();
    for s in stats {
        let k = g.entry(&s.kind).or_insert_with(|| KindSummary { kind: s.kind.clone(), ..Default::default() });
        k.n += 1;
        k.fro += s.fro * s.fro;
        k.mean_eff_rank += s.eff_rank;
        k.mean_stable_rank += s.stable_rank;
        k.max_spectral = k.max_spectral.max(s.spectral);
    }
    let mut v: Vec<KindSummary> = g.into_values().map(|mut k| { k.fro = k.fro.sqrt(); k.mean_eff_rank /= k.n as f64; k.mean_stable_rank /= k.n as f64; k }).collect();
    v.sort_by(|a, b| b.fro.partial_cmp(&a.fro).unwrap());
    v
}

fn jn(x: f64) -> String { if x.is_finite() { format!("{:e}", x) } else { "null".into() } }
fn jo(x: Option<f64>) -> String { x.map_or("null".into(), jn) }

/// Single-adapter report (text or one JSON object).
pub fn report_single(ad: &Adapter, top: usize, json: bool) -> String {
    let mut stats: Vec<ModuleStats> = ad.modules.iter().map(|m| module_stats(m, ad.scale(m))).collect();
    stats.sort_by(|a, b| b.fro.partial_cmp(&a.fro).unwrap().then_with(|| a.name.cmp(&b.name)));
    let total = stats.iter().map(|s| s.fro * s.fro).sum::<f64>().sqrt();
    let zero = stats.iter().filter(|s| s.fro == 0.0).count();
    let nz: Vec<&ModuleStats> = stats.iter().filter(|s| s.fro > 0.0).collect();
    let mean_er = if nz.is_empty() { 0.0 } else { nz.iter().map(|s| s.eff_rank).sum::<f64>() / nz.len() as f64 };
    let groups = group_stats(&stats);
    let mut o = String::new();
    if json {
        o.push_str(&format!("{{\"mode\":\"single\",\"file\":\"{}\",\"tensors\":{},\"modules\":{},\"zero_modules\":{},\"dtypes\":[{}],\"alpha\":{},\"r\":{},\"rslora\":{},\"scale_source\":\"{}\",\"total_fro\":{},\"mean_eff_rank_nonzero\":{},",
            esc(&ad.path), ad.n_tensors, stats.len(), zero, ad.dtypes.iter().map(|d| format!("\"{}\"", d)).collect::<Vec<_>>().join(","),
            jo(ad.cfg.alpha), jo(ad.cfg.r), ad.cfg.rslora, esc(&ad.scale_source), jn(total), jn(mean_er)));
        o.push_str("\"by_kind\":[");
        o.push_str(&groups.iter().map(|k| format!("{{\"kind\":\"{}\",\"n\":{},\"fro\":{},\"mean_eff_rank\":{},\"mean_stable_rank\":{},\"max_spectral\":{}}}", esc(&k.kind), k.n, jn(k.fro), jn(k.mean_eff_rank), jn(k.mean_stable_rank), jn(k.max_spectral))).collect::<Vec<_>>().join(","));
        o.push_str("],\"modules_ranked\":[");
        o.push_str(&stats.iter().map(|s| format!("{{\"name\":\"{}\",\"kind\":\"{}\",\"r\":{},\"out\":{},\"in\":{},\"scale\":{},\"fro\":{},\"spectral\":{},\"eff_rank\":{},\"stable_rank\":{},\"top_sv\":[{}]}}",
            esc(&s.name), esc(&s.kind), s.rank, s.out_dim, s.in_dim, jn(s.scale), jn(s.fro), jn(s.spectral), jn(s.eff_rank), jn(s.stable_rank),
            s.sv.iter().take(top).map(|&x| jn(x)).collect::<Vec<_>>().join(","))).collect::<Vec<_>>().join(","));
        o.push_str(&format!("],\"unpaired\":[{}]}}\n", ad.unpaired.iter().map(|u| format!("\"{}\"", esc(u))).collect::<Vec<_>>().join(",")));
        return o;
    }
    o.push_str(&format!("adapter: {}\n", ad.path));
    o.push_str(&format!("  tensors {}  modules {}  dtypes {}  alpha {}  r {}{}  ({})\n", ad.n_tensors, stats.len(), ad.dtypes.join(","),
        ad.cfg.alpha.map_or("-".into(), |a| format!("{}", a)), ad.cfg.r.map_or("-".into(), |r| format!("{}", r)), if ad.cfg.rslora { "  rsLoRA" } else { "" }, ad.scale_source));
    o.push_str(&format!("  total |dW|_F = {:.6e}   zero modules {}/{}   mean eff-rank (nonzero) {:.3}\n", total, zero, stats.len(), mean_er));
    if !ad.unpaired.is_empty() { o.push_str(&format!("  unpaired/other tensors: {}\n", ad.unpaired.len())) }
    if !ad.skipped.is_empty() { o.push_str(&format!("  skipped (unsupported dtype): {}\n", ad.skipped.len())) }
    o.push_str("\nby module type (ranked by sqrt(sum |dW|_F^2)):\n");
    o.push_str(&format!("  {:<14} {:>4} {:>13} {:>13} {:>9} {:>9}\n", "kind", "n", "|dW|_F", "max sigma", "mean erk", "mean srk"));
    for k in &groups { o.push_str(&format!("  {:<14} {:>4} {:>13.6e} {:>13.6e} {:>9.3} {:>9.3}\n", k.kind, k.n, k.fro, k.max_spectral, k.mean_eff_rank, k.mean_stable_rank)) }
    o.push_str(&format!("\nmodules ranked by |dW|_F (top-{} singular values):\n", top));
    o.push_str(&format!("  {:>4} {:<52} {:>4} {:>11} {:>12} {:>12} {:>7} {:>7}  sigma\n", "#", "module", "r", "out x in", "|dW|_F", "sigma_max", "erank", "srank"));
    for (i, s) in stats.iter().enumerate() {
        let sv: Vec<String> = s.sv.iter().take(top).map(|x| format!("{:.3e}", x)).collect();
        o.push_str(&format!("  {:>4} {:<52} {:>4} {:>11} {:>12.5e} {:>12.5e} {:>7.3} {:>7.3}  {}\n", i + 1, s.name, s.rank, format!("{}x{}", s.out_dim, s.in_dim), s.fro, s.spectral, s.eff_rank, s.stable_rank, sv.join(" ")));
    }
    o
}

/// Checkpoint diff report a -> b (text or one JSON object).
pub fn report_diff(a: &Adapter, b: &Adapter, json: bool) -> Result<String, String> {
    let bm: BTreeMap<&str, &LoraModule> = b.modules.iter().map(|m| (m.name.as_str(), m)).collect();
    let am: BTreeMap<&str, &LoraModule> = a.modules.iter().map(|m| (m.name.as_str(), m)).collect();
    let mut diffs = Vec::new();
    for m in &a.modules { if let Some(mb) = bm.get(m.name.as_str()) { diffs.push(module_diff(m, a.scale(m), mb, b.scale(mb))?) } }
    let only_a: Vec<&str> = am.keys().filter(|k| !bm.contains_key(*k)).copied().collect();
    let only_b: Vec<&str> = bm.keys().filter(|k| !am.contains_key(*k)).copied().collect();
    diffs.sort_by(|x, y| y.diff.partial_cmp(&x.diff).unwrap().then_with(|| x.name.cmp(&y.name)));
    let sq = |f: &dyn Fn(&ModuleDiff) -> f64| diffs.iter().map(|d| f(d).powi(2)).sum::<f64>().sqrt();
    let (ta, tb, td) = (sq(&|d| d.fro_a), sq(&|d| d.fro_b), sq(&|d| d.diff));
    let tin: f64 = diffs.iter().map(|d| d.inner).sum();
    let gcos = if ta > 0.0 && tb > 0.0 { Some((tin / (ta * tb)).clamp(-1.0, 1.0)) } else { None };
    // by kind
    let mut g: BTreeMap<&str, (usize, f64, f64, f64, f64)> = BTreeMap::new();
    for d in &diffs { let e = g.entry(&d.kind).or_default(); e.0 += 1; e.1 += d.fro_a.powi(2); e.2 += d.fro_b.powi(2); e.3 += d.diff.powi(2); e.4 += d.inner }
    let mut kinds: Vec<(&str, usize, f64, f64, f64, Option<f64>)> = g.into_iter().map(|(k, (n, a2, b2, d2, inn))| {
        let (fa, fb) = (a2.sqrt(), b2.sqrt());
        (k, n, fa, fb, d2.sqrt(), if fa > 0.0 && fb > 0.0 { Some((inn / (fa * fb)).clamp(-1.0, 1.0)) } else { None })
    }).collect();
    kinds.sort_by(|x, y| y.4.partial_cmp(&x.4).unwrap());
    let rel = |d: f64, a: f64| if a > 0.0 { Some(d / a) } else { None };
    let mut o = String::new();
    if json {
        o.push_str(&format!("{{\"mode\":\"diff\",\"a\":\"{}\",\"b\":\"{}\",\"paired\":{},\"only_a\":{},\"only_b\":{},\"total_fro_a\":{},\"total_fro_b\":{},\"total_diff\":{},\"total_rel_to_a\":{},\"global_cos\":{},\"by_kind\":[",
            esc(&a.path), esc(&b.path), diffs.len(), only_a.len(), only_b.len(), jn(ta), jn(tb), jn(td), jo(rel(td, ta)), jo(gcos)));
        o.push_str(&kinds.iter().map(|k| format!("{{\"kind\":\"{}\",\"n\":{},\"fro_a\":{},\"fro_b\":{},\"diff\":{},\"cos\":{}}}", esc(k.0), k.1, jn(k.2), jn(k.3), jn(k.4), jo(k.5))).collect::<Vec<_>>().join(","));
        o.push_str("],\"modules_ranked\":[");
        o.push_str(&diffs.iter().map(|d| format!("{{\"name\":\"{}\",\"kind\":\"{}\",\"fro_a\":{},\"fro_b\":{},\"diff\":{},\"rel_to_a\":{},\"cos\":{}}}", esc(&d.name), esc(&d.kind), jn(d.fro_a), jn(d.fro_b), jn(d.diff), jo(rel(d.diff, d.fro_a)), jo(d.cos))).collect::<Vec<_>>().join(","));
        o.push_str("]}\n");
        return Ok(o);
    }
    let f = |x: Option<f64>, p: usize| x.map_or("-".to_string(), |v| format!("{:.*}", p, v));
    o.push_str(&format!("diff a -> b\n  a: {}  ({})\n  b: {}  ({})\n", a.path, a.scale_source, b.path, b.scale_source));
    o.push_str(&format!("  paired modules {}  only in a {}  only in b {}\n", diffs.len(), only_a.len(), only_b.len()));
    o.push_str(&format!("  total |dWa|_F {:.6e}  |dWb|_F {:.6e}  |dWb - dWa|_F {:.6e}  rel-to-a {}  global cos {}\n", ta, tb, td, f(rel(td, ta), 4), f(gcos, 6)));
    o.push_str("\nby module type (ranked by |dWb - dWa|_F):\n");
    o.push_str(&format!("  {:<14} {:>4} {:>13} {:>13} {:>13} {:>9}\n", "kind", "n", "|dWa|_F", "|dWb|_F", "|diff|_F", "cos"));
    for k in &kinds { o.push_str(&format!("  {:<14} {:>4} {:>13.6e} {:>13.6e} {:>13.6e} {:>9}\n", k.0, k.1, k.2, k.3, k.4, f(k.5, 5))) }
    o.push_str("\nmodules ranked by |dWb - dWa|_F:\n");
    o.push_str(&format!("  {:>4} {:<52} {:>12} {:>12} {:>12} {:>9} {:>9}\n", "#", "module", "|dWa|_F", "|dWb|_F", "|diff|_F", "rel-to-a", "cos"));
    for (i, d) in diffs.iter().enumerate() {
        o.push_str(&format!("  {:>4} {:<52} {:>12.5e} {:>12.5e} {:>12.5e} {:>9} {:>9}\n", i + 1, d.name, d.fro_a, d.fro_b, d.diff, f(rel(d.diff, d.fro_a), 4), f(d.cos, 5)));
    }
    for n in &only_a { o.push_str(&format!("  only in a: {}\n", n)) }
    for n in &only_b { o.push_str(&format!("  only in b: {}\n", n)) }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    // xorshift64*, deterministic
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
            ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn mat(&mut self, r: usize, c: usize) -> Mat { Mat::new(r, c, (0..r * c).map(|_| self.f()).collect()) }
    }

    fn module(a: Mat, b: Mat) -> LoraModule { LoraModule { name: "layers.0.self_attn.q_proj".into(), kind: "q_proj".into(), a, b } }

    /// Brute force: materialize dW, singular values from Jacobi on dW^T dW (in x in).
    fn brute_sv(m: &LoraModule, scale: f64) -> (Vec<f64>, f64, Mat) {
        let mut w = m.b.mul(&m.a);
        w.data.iter_mut().for_each(|x| *x *= scale);
        let (mut ev, _) = jacobi_eig(&w.t().mul(&w));
        ev.iter_mut().for_each(|x| *x = x.max(0.0).sqrt());
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let fro = w.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        (ev, fro, w)
    }

    fn close(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-12) }

    #[test]
    fn safetensors_round_trip_f32_f16_bf16() {
        let v32 = [1.0f32, -2.5, 3.25e-3, 1e10, -0.0, 7.0];
        let v16 = [1.0f32, -2.5, 0.5, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8]; // exact in f16 (max, min normal, min subnormal)
        let vbf = [1.0f32, -2.5, 3.0e38, 1.0e-30, 0.15625, -7.0];
        let bytes = write_safetensors(
            &[("w.f32", Dtype::F32, &[2, 3], &v32), ("w.f16", Dtype::F16, &[3, 2], &v16), ("w.bf16", Dtype::BF16, &[6], &vbf)],
            &[("format", "pt"), ("note", "q\"uote")],
        );
        let st = parse_safetensors(&bytes).unwrap();
        assert_eq!(st.tensors.len(), 3);
        assert_eq!(st.metadata, vec![("format".to_string(), "pt".to_string()), ("note".to_string(), "q\"uote".to_string())]);
        let t = st.get("w.f32").unwrap();
        assert_eq!((t.dtype, t.shape.clone()), (Dtype::F32, vec![2, 3]));
        assert_eq!(t.data, v32.to_vec());
        let t = st.get("w.f16").unwrap();
        assert_eq!((t.dtype, t.shape.clone()), (Dtype::F16, vec![3, 2]));
        assert_eq!(t.data, v16.to_vec());
        let t = st.get("w.bf16").unwrap();
        assert_eq!(t.dtype, Dtype::BF16);
        for (x, y) in t.data.iter().zip(vbf.iter()) { assert!(close(*x as f64, *y as f64, 1.0 / 256.0), "{} vs {}", x, y) }
        assert_eq!(t.data[0], 1.0);
        assert_eq!(t.data[5], -7.0);
        // f16 rounding: 1 + 2^-11 is a tie, rounds to even (1.0); 1 + 3*2^-11 rounds up
        assert_eq!(f16_to_f32(f32_to_f16(1.0 + 1.0 / 2048.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(1.0 + 3.0 / 2048.0)), 1.0 + 2.0 / 1024.0);
        // corrupt: offsets past the end
        let mut bad = bytes.clone();
        bad.truncate(bad.len() - 2);
        assert!(parse_safetensors(&bad).is_err());
    }

    #[test]
    fn pairs_peft_names() {
        let a = [0.0f32; 4 * 7];
        let b = [0.0f32; 5 * 4];
        let bytes = write_safetensors(&[
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", Dtype::F32, &[4, 7], &a),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", Dtype::F32, &[5, 4], &b),
            ("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight", Dtype::F32, &[4, 7], &a),
        ], &[]);
        let (mods, other) = pair_lora(&parse_safetensors(&bytes).unwrap());
        assert_eq!(mods.len(), 1);
        assert_eq!(mods[0].name, "model.layers.0.self_attn.q_proj");
        assert_eq!(mods[0].kind, "q_proj");
        assert_eq!(other.len(), 1);
        let c = LoraCfg::from_json("{\"lora_alpha\": 16, \"r\": 8, \"use_rslora\": false, \"target_modules\": [\"q_proj\"]}").unwrap();
        assert_eq!(c.scale(8), 2.0);
        assert_eq!(LoraCfg::default().scale(8), 1.0);
    }

    #[test]
    fn r_space_matches_brute_force() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for trial in 0..20 {
            let m = module(rng.mat(4, 7), rng.mat(5, 4));
            let scale = 0.5 + trial as f64 * 0.25;
            let s = module_stats(&m, scale);
            let (bsv, bfro, _) = brute_sv(&m, scale);
            assert!(close(s.fro, bfro, 1e-4), "fro {} vs {}", s.fro, bfro);
            for i in 0..4 { assert!(close(s.sv[i], bsv[i], 1e-4), "trial {} sv{}: {} vs {}", trial, i, s.sv[i], bsv[i]) }
            for &x in &bsv[4..] { assert!(x <= 1e-6 * bsv[0]) } // rank <= r
            assert!(close(s.stable_rank, bfro * bfro / (bsv[0] * bsv[0]), 1e-4));
        }
        // rank-deficient A (two equal rows): psd_sqrt must not need a ridge
        let mut a = rng.mat(4, 7);
        for j in 0..7 { a.data[7 + j] = a.data[j] }
        let m = module(a, rng.mat(5, 4));
        let (bsv, bfro, _) = brute_sv(&m, 1.0);
        let s = module_stats(&m, 1.0);
        assert!(close(s.fro, bfro, 1e-4));
        for i in 0..3 { assert!(close(s.sv[i], bsv[i], 1e-4), "deficient sv{}: {} vs {}", i, s.sv[i], bsv[i]) }
        assert!(s.sv[3] <= 1e-6 * s.sv[0]);
    }

    #[test]
    fn trace_diff_matches_brute_force() {
        let mut rng = Rng(42);
        for _ in 0..20 {
            let ma = module(rng.mat(4, 7), rng.mat(5, 4));
            let mb = module(rng.mat(4, 7), rng.mat(5, 4));
            let (sa, sb) = (2.0, 1.5);
            let d = module_diff(&ma, sa, &mb, sb).unwrap();
            let (_, fa, wa) = brute_sv(&ma, sa);
            let (_, fb, wb) = brute_sv(&mb, sb);
            let bd = wa.data.iter().zip(&wb.data).map(|(x, y)| (y - x).powi(2)).sum::<f64>().sqrt();
            let bin: f64 = wa.data.iter().zip(&wb.data).map(|(x, y)| x * y).sum();
            assert!(close(d.diff, bd, 1e-4), "diff {} vs {}", d.diff, bd);
            assert!(close(d.fro_a, fa, 1e-4) && close(d.fro_b, fb, 1e-4));
            assert!((d.cos.unwrap() - bin / (fa * fb)).abs() < 1e-6);
        }
        // different ranks a vs b (ra = 4, rb = 2)
        let ma = module(rng.mat(4, 7), rng.mat(5, 4));
        let mb = module(rng.mat(2, 7), rng.mat(5, 2));
        let d = module_diff(&ma, 1.0, &mb, 1.0).unwrap();
        let (_, _, wa) = brute_sv(&ma, 1.0);
        let (_, _, wb) = brute_sv(&mb, 1.0);
        let bd = wa.data.iter().zip(&wb.data).map(|(x, y)| (y - x).powi(2)).sum::<f64>().sqrt();
        assert!(close(d.diff, bd, 1e-4));
        // identical -> 0 diff, cos 1; zero B -> cos None
        let d = module_diff(&ma, 1.0, &ma, 1.0).unwrap();
        assert!(d.diff <= 1e-6 * d.fro_a && (d.cos.unwrap() - 1.0).abs() < 1e-9);
        let z = module(rng.mat(4, 7), Mat::zeros(5, 4));
        let d = module_diff(&z, 1.0, &ma, 1.0).unwrap();
        assert!(d.cos.is_none() && close(d.diff, d.fro_b, 1e-9));
    }

    #[test]
    fn rank_one_has_unit_ranks() {
        let mut rng = Rng(7);
        // B = u e1^T: only the first column nonzero -> dW = u a1^T, rank 1
        let mut b = Mat::zeros(5, 4);
        for o in 0..5 { b.data[o * 4] = rng.f() }
        let m = module(rng.mat(4, 7), b);
        let s = module_stats(&m, 2.0);
        assert!((s.eff_rank - 1.0).abs() < 1e-6, "erank {}", s.eff_rank);
        assert!((s.stable_rank - 1.0).abs() < 1e-6, "srank {}", s.stable_rank);
        assert!(close(s.spectral, s.fro, 1e-6));
        // zero dW reports zeros, not NaN
        let z = module_stats(&module(rng.mat(4, 7), Mat::zeros(5, 4)), 2.0);
        assert_eq!((z.fro, z.spectral, z.eff_rank, z.stable_rank), (0.0, 0.0, 0.0, 0.0));
        // equal singular values -> effective rank r
        let mut a = Mat::zeros(4, 7);
        let mut b = Mat::zeros(5, 4);
        for i in 0..4 { a.data[i * 7 + i] = 1.0; b.data[i * 4 + i] = 1.0 }
        let s = module_stats(&module(a, b), 1.0);
        assert!((s.eff_rank - 4.0).abs() < 1e-9 && (s.stable_rank - 4.0).abs() < 1e-9);
    }
}
