//! Genome: a brain's settings written as DNA and evolved by selection.
//!
//! A genome is a string of ACGT bases (2 bits each, packed), at most
//! [`MAX_CODONS`] codons long. It is read in codons (triplets, 64 values) by a
//! fixed reading frame: gene `g` is the pair of codons `2g, 2g+1`, a 12-bit
//! value (0..4096) mapped onto that setting's range. Genes past the end of a
//! short genome take the default value (like an unexpressed gene). Codon
//! duplication and deletion shift the reading frame, so they are real
//! frameshift mutations, not cosmetic ones.
//!
//! The phenotype is plain numbers ([`Phenotype`]); nothing else is wired here.
//! Evolution ([`Population`]) keeps `P` genomes in fixed arrays, ranks them by a
//! caller-supplied fitness (in the tests: prequential right-action rate of a
//! brain run on a training stream), keeps the best genome unchanged (elitism),
//! and fills the rest by tournament selection, crossover and mutation. Fitness
//! is cached per genome, so with a deterministic fitness the elite's fitness
//! never decreases. Generalization is judged by the caller on a held-out world.
//!
//! no_std, no heap, bounded loops, deterministic for a seed.

/// Maximum genome length in codons.
pub const MAX_CODONS: usize = 24;
/// Bytes holding MAX_CODONS * 3 bases at 2 bits each.
pub const GENOME_BYTES: usize = (MAX_CODONS * 3 + 3) / 4;
/// Number of genes (settings) read from the genome.
pub const GENES: usize = 9;

/// The settings a genome expresses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phenotype {
    /// Brain exploration weight.
    pub explore: f32,
    /// Brain minimum evidence before acting.
    pub min_evidence: f32,
    /// Brain memory half-life (decisions).
    pub half_life: u32,
    /// Grower: decisions between growth checks.
    pub grow_every: u32,
    /// Grower: minimum held-out gain to adopt a sense.
    pub grow_min_gain: f32,
    /// Grower: usable growth slots.
    pub grow_limit: u8,
    /// SS-LoRA sparsity.
    pub sslora_p: f32,
    /// Sleep: forget a memory when the model predicts it within this error.
    pub sleep_threshold: f32,
    /// Prune: drop a sense whose held-out contribution is below this.
    pub prune_threshold: f32,
}

impl Phenotype {
    /// The hand-set defaults used elsewhere in the crate.
    pub const DEFAULT: Phenotype = Phenotype {
        explore: 0.3, min_evidence: 0.5, half_life: 64, grow_every: 250, grow_min_gain: 0.05,
        grow_limit: 4, sslora_p: 0.5, sleep_threshold: 0.05, prune_threshold: 0.02,
    };
}

// (min, max) per gene; half_life is quadratic in the code so short half-lives get finer steps.
const RANGE: [(f32, f32); GENES] = [
    (0.0, 1.0),     // explore
    (0.0, 2.0),     // min_evidence
    (8.0, 512.0),   // half_life (quadratic)
    (50.0, 500.0),  // grow_every
    (0.005, 0.2),   // grow_min_gain
    (0.0, 4.0),     // grow_limit
    (0.0, 0.9),     // sslora_p
    (0.0, 0.5),     // sleep_threshold
    (0.0, 0.5),     // prune_threshold
];

const LEVELS: f32 = 4095.0;

fn level_of(gene: usize, v: f32) -> u16 {
    let (lo, hi) = RANGE[gene];
    let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.0 };
    let t = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
    let t = if gene == 2 { sqrtf(t) } else { t };
    (t * LEVELS + 0.5) as u16
}

fn value_of(gene: usize, level: u16) -> f32 {
    let (lo, hi) = RANGE[gene];
    let mut t = level as f32 / LEVELS;
    if gene == 2 { t *= t; }
    lo + t * (hi - lo)
}

fn sqrtf(x: f32) -> f32 { crate::sqrt(x as f64) as f32 }

/// A DNA genome: up to MAX_CODONS codons of ACGT bases, packed 2 bits per base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Genome {
    bases: [u8; GENOME_BYTES],
    /// Length in codons.
    pub len: u8,
}

const ACGT: [u8; 4] = *b"ACGT";

impl Genome {
    pub const EMPTY: Genome = Genome { bases: [0; GENOME_BYTES], len: 0 };

    pub fn base(&self, i: usize) -> u8 { (self.bases[i / 4] >> ((i % 4) * 2)) & 3 }

    fn set_base(&mut self, i: usize, b: u8) {
        let sh = (i % 4) * 2;
        self.bases[i / 4] = (self.bases[i / 4] & !(3 << sh)) | ((b & 3) << sh);
    }

    /// Codon `c` as a value 0..64.
    pub fn codon(&self, c: usize) -> u8 { (self.base(3 * c) << 4) | (self.base(3 * c + 1) << 2) | self.base(3 * c + 2) }

    fn set_codon(&mut self, c: usize, v: u8) {
        self.set_base(3 * c, (v >> 4) & 3); self.set_base(3 * c + 1, (v >> 2) & 3); self.set_base(3 * c + 2, v & 3);
    }

    /// The genome that expresses `p` (each gene quantized to 12 bits).
    pub fn encode(p: &Phenotype) -> Genome {
        let vals = [p.explore, p.min_evidence, p.half_life as f32, p.grow_every as f32, p.grow_min_gain,
            p.grow_limit as f32, p.sslora_p, p.sleep_threshold, p.prune_threshold];
        let mut g = Genome::EMPTY;
        g.len = (2 * GENES) as u8;
        for (gene, v) in vals.iter().enumerate() {
            let l = level_of(gene, *v);
            g.set_codon(2 * gene, (l >> 6) as u8);
            g.set_codon(2 * gene + 1, (l & 63) as u8);
        }
        g
    }

    /// Express the genome. Genes beyond the genome's length take the default.
    pub fn express(&self) -> Phenotype {
        let d = Phenotype::DEFAULT;
        let defaults = [d.explore, d.min_evidence, d.half_life as f32, d.grow_every as f32, d.grow_min_gain,
            d.grow_limit as f32, d.sslora_p, d.sleep_threshold, d.prune_threshold];
        let mut v = defaults;
        for gene in 0..GENES {
            if 2 * gene + 1 < self.len as usize {
                v[gene] = value_of(gene, ((self.codon(2 * gene) as u16) << 6) | self.codon(2 * gene + 1) as u16);
            }
        }
        Phenotype {
            explore: v[0], min_evidence: v[1], half_life: (v[2] + 0.5) as u32, grow_every: (v[3] + 0.5) as u32,
            grow_min_gain: v[4], grow_limit: (v[5] + 0.5) as u8, sslora_p: v[6], sleep_threshold: v[7], prune_threshold: v[8],
        }
    }

    /// Write the genome as ACGT letters into `out`; returns the number of bases written.
    pub fn letters(&self, out: &mut [u8]) -> usize {
        let n = (3 * self.len as usize).min(out.len());
        for i in 0..n { out[i] = ACGT[self.base(i) as usize]; }
        n
    }
}

/// Deterministic xorshift64.
#[derive(Clone, Copy, Debug)]
pub struct Rng(u64);
impl Rng {
    pub const fn new(seed: u64) -> Self { Rng(seed | 1) }
    pub fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    pub fn below(&mut self, n: usize) -> usize { if n == 0 { 0 } else { (self.next() % n as u64) as usize } }
    pub fn chance(&mut self, p: f32) -> bool { ((self.next() >> 40) as f32 / (1u64 << 24) as f32) < p }
}

/// Mutation rates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    /// Per-base substitution probability.
    pub point: f32,
    /// Per-offspring probability of duplicating one codon (shifts the frame right).
    pub duplicate: f32,
    /// Per-offspring probability of deleting one codon (shifts the frame left).
    pub delete: f32,
    /// Probability that an offspring comes from crossover rather than copying one parent.
    pub crossover: f32,
}

impl Rates {
    pub const DEFAULT: Rates = Rates { point: 0.03, duplicate: 0.05, delete: 0.05, crossover: 0.6 };
}

/// Point mutation: each base substituted with probability `rate`.
pub fn point_mutate(g: &mut Genome, rate: f32, rng: &mut Rng) {
    for i in 0..3 * g.len as usize {
        if rng.chance(rate) { let b = g.base(i); g.set_base(i, (b + 1 + rng.below(3) as u8) & 3); }
    }
}

/// Duplicate codon `at` (the frame after it shifts right; the last codon falls off at MAX_CODONS).
pub fn duplicate_codon(g: &mut Genome, at: usize) {
    let len = g.len as usize;
    if at >= len { return; }
    let new_len = (len + 1).min(MAX_CODONS);
    let mut c = new_len - 1;
    while c > at { let v = g.codon(c - 1); g.set_codon(c, v); c -= 1; }
    g.len = new_len as u8;
}

/// Delete codon `at` (the frame after it shifts left).
pub fn delete_codon(g: &mut Genome, at: usize) {
    let len = g.len as usize;
    if at >= len || len == 0 { return; }
    for c in at..len - 1 { let v = g.codon(c + 1); g.set_codon(c, v); }
    g.set_codon(len - 1, 0);
    g.len = (len - 1) as u8;
}

/// One-point crossover at a codon boundary: child = a[..cut] ++ b[cut..].
pub fn crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
    let la = a.len as usize; let lb = b.len as usize;
    let cut = rng.below(la.min(lb) + 1);
    let mut child = Genome::EMPTY;
    child.len = lb.max(cut) as u8;
    for c in 0..child.len as usize { child.set_codon(c, if c < cut { a.codon(c) } else { b.codon(c) }); }
    child
}

/// A population of `P` genomes with cached fitness.
pub struct Population<const P: usize> {
    pub genomes: [Genome; P],
    fitness: [f32; P],
    known: [bool; P],
    pub generation: u32,
    pub rates: Rates,
    /// Tournament size.
    pub tournament: usize,
    rng: Rng,
}

impl<const P: usize> Population<P> {
    /// Seed the population: genome 0 encodes `founder`, the rest are mutants of it.
    pub fn new(founder: &Phenotype, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let base = Genome::encode(founder);
        let mut genomes = [base; P];
        for g in genomes.iter_mut().skip(1) { point_mutate(g, 0.25, &mut rng); }
        Population { genomes, fitness: [0.0; P], known: [false; P], generation: 0, rates: Rates::DEFAULT, tournament: 3, rng }
    }

    fn evaluate(&mut self, fitness: &mut impl FnMut(&Phenotype) -> f32) {
        for i in 0..P {
            if !self.known[i] { self.fitness[i] = fitness(&self.genomes[i].express()); self.known[i] = true; }
        }
    }

    /// Index of the fittest genome (lowest index on ties).
    pub fn best(&self) -> usize {
        let mut b = 0;
        for i in 1..P { if self.fitness[i] > self.fitness[b] { b = i; } }
        b
    }

    pub fn fitness(&self, i: usize) -> Option<f32> { if i < P && self.known[i] { Some(self.fitness[i]) } else { None } }

    fn pick(&mut self) -> usize {
        let mut w = self.rng.below(P);
        for _ in 1..self.tournament.max(1) { let c = self.rng.below(P); if self.fitness[c] > self.fitness[w] { w = c; } }
        w
    }

    /// One generation: evaluate unknown genomes, keep the elite at index 0, breed the rest.
    /// Returns the elite's fitness after the step.
    pub fn step(&mut self, fitness: &mut impl FnMut(&Phenotype) -> f32) -> f32 {
        self.evaluate(fitness);
        let e = self.best();
        let (elite, ef) = (self.genomes[e], self.fitness[e]);
        let mut next = [Genome::EMPTY; P];
        next[0] = elite;
        for slot in next.iter_mut().skip(1) {
            let a = self.pick();
            let mut child = if self.rng.chance(self.rates.crossover) {
                let b = self.pick();
                crossover(&self.genomes[a], &self.genomes[b], &mut self.rng)
            } else { self.genomes[a] };
            point_mutate(&mut child, self.rates.point, &mut self.rng);
            if self.rng.chance(self.rates.duplicate) { let at = self.rng.below(child.len as usize); duplicate_codon(&mut child, at); }
            if self.rng.chance(self.rates.delete) { let at = self.rng.below(child.len as usize); delete_codon(&mut child, at); }
            *slot = child;
        }
        self.genomes = next;
        self.fitness = [0.0; P];
        self.known = [false; P];
        self.fitness[0] = ef; self.known[0] = true;
        self.generation += 1;
        self.evaluate(fitness);
        self.fitness[self.best()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::Brain;
    use crate::brain_grow::Grower;

    struct Lcg(u64);
    impl Lcg { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }

    /// Rover fault response with interactions (world B of `struktura brain demo`), its thresholds
    /// shifted by `shift` for the held-out variant.
    fn truth(x: &[f32; 4], shift: f32) -> u8 {
        let (t6, t4, t5) = (0.6 + shift, 0.4 + shift, 0.5 + shift);
        if x[2] > t6 && x[0] > t6 { 3 } else if x[1] > t6 && x[2] < t4 { 2 } else if (x[0] > t5) != (x[1] > t5) { 1 } else { 0 }
    }

    /// Prequential right-action rate (each decision scored before its outcome is learned).
    fn run(p: &Phenotype, seed: u64, steps: usize, shift: f32) -> f32 {
        let mut r = Lcg(seed);
        let mut br: Brain<128, 8, 4> = Brain::new(p.explore, p.min_evidence);
        br.half_life = p.half_life.max(1);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xD1A);
        gr.constant[3] = true;
        gr.every = p.grow_every.max(1);
        gr.min_gain = p.grow_min_gain;
        gr.set_limit(p.grow_limit as usize);
        let mut ok = 0usize;
        for t in 0..steps {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 200 { (t % 4) as u8 } else { d.action };
            let right = a == truth(&base, shift);
            if right { ok += 1; }
            br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            gr.after_learn(&mut br);
        }
        ok as f32 / steps as f32
    }

    #[test]
    fn genome_roundtrip_and_reading_frame() {
        let g = Genome::encode(&Phenotype::DEFAULT);
        let p = g.express();
        let d = Phenotype::DEFAULT;
        assert!((p.explore - d.explore).abs() < 1e-3 && (p.min_evidence - d.min_evidence).abs() < 1e-3);
        assert!((p.half_life as i64 - d.half_life as i64).abs() <= 1 && (p.grow_every as i64 - d.grow_every as i64).abs() <= 1);
        assert_eq!(p.grow_limit, d.grow_limit);
        assert!((p.sslora_p - d.sslora_p).abs() < 1e-3 && (p.grow_min_gain - d.grow_min_gain).abs() < 1e-3);
        let mut letters = [0u8; MAX_CODONS * 3];
        let n = g.letters(&mut letters);
        assert_eq!(n, 2 * GENES * 3);
        assert!(letters[..n].iter().all(|b| b"ACGT".contains(b)));
        // A deletion shifts the frame: genes after it change, the genome gets shorter.
        let mut h = g; delete_codon(&mut h, 0);
        assert_eq!(h.len as usize, 2 * GENES - 1);
        assert_ne!(h.express(), p);
        assert_eq!(h.express().prune_threshold, Phenotype::DEFAULT.prune_threshold, "last gene unexpressed -> default");
        // Duplication shifts it back the other way and never exceeds MAX_CODONS.
        let mut k = g; for _ in 0..40 { duplicate_codon(&mut k, 3); }
        assert_eq!(k.len as usize, MAX_CODONS);
        // Empty genome expresses the defaults.
        assert_eq!(Genome::EMPTY.express(), Phenotype::DEFAULT);
    }

    #[test]
    fn deterministic_evolution() {
        let evolve = || {
            let mut pop: Population<6> = Population::new(&Phenotype::DEFAULT, 77);
            let mut f = |p: &Phenotype| run(p, 5, 400, 0.0);
            for _ in 0..3 { pop.step(&mut f); }
            (pop.genomes[pop.best()], pop.fitness(pop.best()))
        };
        assert_eq!(evolve(), evolve());
    }

    /// Pre-registered falsifier (set before the run): a population of 8 genomes seeded from the
    /// defaults evolves for 12 generations on a TRAINING stream (rover world B, 1500 decisions).
    /// Required: (1) the elite's fitness never decreases across generations; (2) averaged over
    /// fresh seeds on a HELD-OUT world variant (thresholds shifted by +0.05, different situation
    /// streams, 2500 decisions), the evolved best phenotype beats the default settings.
    /// Checked for 3 evolution seeds.
    #[test]
    #[ignore = "FAILED as pre-registered: evolution on ONE training stream overfits it (train +4..+9 pts) and loses on the held-out world in 0/3 runs (0.761 / 0.755 / 0.762 vs default 0.763); see multi_stream_selection"]
    fn falsifier_evolved_genome_beats_defaults_on_a_held_out_world() {
        let held_seeds = [201u64, 203, 207, 209, 211, 223];
        let held = |p: &Phenotype| held_seeds.iter().map(|&s| run(p, s, 2500, 0.05)).sum::<f32>() / held_seeds.len() as f32;
        let default_held = held(&Phenotype::DEFAULT);
        let mut wins = 0;
        for evo_seed in [11u64, 23, 37] {
            let train_seed = evo_seed ^ 0x5EED;
            let mut f = |p: &Phenotype| run(p, train_seed, 1500, 0.0);
            let mut pop: Population<8> = Population::new(&Phenotype::DEFAULT, evo_seed);
            let mut last = f32::NEG_INFINITY;
            let mut curve = [0.0f32; 12];
            for g in 0..12 {
                let e = pop.step(&mut f);
                assert!(e >= last, "elitism: elite fitness fell from {} to {} at generation {}", last, e, g);
                last = e; curve[g] = e;
            }
            let best = pop.genomes[pop.best()];
            let p = best.express();
            let train_default = run(&Phenotype::DEFAULT, train_seed, 1500, 0.0);
            let ph = held(&p);
            let mut letters = [0u8; MAX_CODONS * 3];
            let n = best.letters(&mut letters);
            std::println!("genome evo seed {}: train default {:.3} -> elite {:.3} (curve {:.3} .. {:.3}); held-out default {:.3} vs evolved {:.3}",
                evo_seed, train_default, last, curve[0], curve[11], default_held, ph);
            std::println!("  evolved DNA ({} codons): {}", best.len, core::str::from_utf8(&letters[..n]).unwrap());
            std::println!("  evolved phenotype: {:?}", p);
            if ph > default_held { wins += 1; }
        }
        std::println!("held-out wins vs defaults: {}/3", wins);
        assert!(wins >= 2, "evolved genomes beat the defaults on the held-out world in only {}/3 runs", wins);
    }
}

#[cfg(test)]
mod follow_up {
    use super::*;

    /// Follow-up AFTER the failed falsifier (a new protocol, reported in full, not a pass of the
    /// original): select on fitness averaged over 3 independent training streams, so selection
    /// cannot reward settings that only fit one stream's noise. Same held-out world and seeds,
    /// same 8 genomes x 12 generations; streams are 1000 decisions to keep cost comparable.
    #[test]
    #[ignore = "also FAILS (recorded, 210 s): 3-stream selection, held-out 0.756 / 0.754 / 0.762 vs default 0.763, 0/3 wins"]
    fn multi_stream_selection() {
        let held_seeds = [201u64, 203, 207, 209, 211, 223];
        let held = |p: &Phenotype| held_seeds.iter().map(|&s| super::tests_api::run(p, s, 2500, 0.05)).sum::<f32>() / held_seeds.len() as f32;
        let default_held = held(&Phenotype::DEFAULT);
        let mut wins = 0;
        for evo_seed in [11u64, 23, 37] {
            let streams = [evo_seed ^ 0x5EED, evo_seed ^ 0xA11CE, evo_seed ^ 0xB0B];
            let mut f = |p: &Phenotype| streams.iter().map(|&s| super::tests_api::run(p, s, 1000, 0.0)).sum::<f32>() / 3.0;
            let mut pop: Population<8> = Population::new(&Phenotype::DEFAULT, evo_seed);
            let mut last = f32::NEG_INFINITY;
            for _ in 0..12 { let e = pop.step(&mut f); assert!(e >= last); last = e; }
            let p = pop.genomes[pop.best()].express();
            let ph = held(&p);
            std::println!("multi-stream evo seed {}: train elite {:.3}; held-out default {:.3} vs evolved {:.3}; {:?}", evo_seed, last, default_held, ph, p);
            if ph > default_held { wins += 1; }
        }
        std::println!("multi-stream held-out wins vs defaults: {}/3", wins);
    }
}

/// The rover-world run shared by the tests (prequential right-action rate).
#[cfg(test)]
mod tests_api {
    use super::Phenotype;
    use crate::brain::Brain;
    use crate::brain_grow::Grower;
    struct Lcg(u64);
    impl Lcg { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32) / (1u64 << 24) as f32 } }
    fn truth(x: &[f32; 4], shift: f32) -> u8 {
        let (t6, t4, t5) = (0.6 + shift, 0.4 + shift, 0.5 + shift);
        if x[2] > t6 && x[0] > t6 { 3 } else if x[1] > t6 && x[2] < t4 { 2 } else if (x[0] > t5) != (x[1] > t5) { 1 } else { 0 }
    }
    pub fn run(p: &Phenotype, seed: u64, steps: usize, shift: f32) -> f32 {
        let mut r = Lcg(seed);
        let mut br: Brain<128, 8, 4> = Brain::new(p.explore, p.min_evidence);
        br.half_life = p.half_life.max(1);
        let mut gr: Grower<4, 4> = Grower::new(seed ^ 0xD1A);
        gr.constant[3] = true;
        gr.every = p.grow_every.max(1);
        gr.min_gain = p.grow_min_gain;
        gr.set_limit(p.grow_limit as usize);
        let mut ok = 0usize;
        for t in 0..steps {
            let base = [r.f(), r.f(), r.f(), 1.0];
            let x: [f32; 8] = gr.situation(&base);
            let d = br.decide(&x, &[true; 4], 3);
            let a = if d.abstained && t < 200 { (t % 4) as u8 } else { d.action };
            let right = a == truth(&base, shift);
            if right { ok += 1; }
            br.learn(&x, a, if right { 1.0 } else if a == 3 { 0.2 } else { 0.0 });
            gr.after_learn(&mut br);
        }
        ok as f32 / steps as f32
    }
}
