//! COLONY v0.2: partially observed grid exploration research, not robot control.
//!
//! Planning has access to grid dimensions and the robot's *observed* occupancy
//! map only. Ground truth is used by the sensor and evaluation oracle.
//! No navigation or safety guarantees are claimed for physical robots.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    Unknown,
    Free,
    Blocked,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl Point {
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

/// Public planning geometry. Unlike `World`, this has no ground-truth cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    pub width: usize,
    pub height: usize,
}

impl Topology {
    pub fn inside(&self, p: Point) -> bool { p.x < self.width && p.y < self.height }
    pub fn idx(&self, p: Point) -> usize { p.y * self.width + p.x }
    pub fn point(&self, idx: usize) -> Point { Point::new(idx % self.width, idx / self.width) }
    pub fn neighbors(&self, p: Point) -> Vec<Point> {
        let mut result = Vec::with_capacity(4);
        if p.x > 0 { result.push(Point::new(p.x - 1, p.y)); }
        if p.x + 1 < self.width { result.push(Point::new(p.x + 1, p.y)); }
        if p.y > 0 { result.push(Point::new(p.x, p.y - 1)); }
        if p.y + 1 < self.height { result.push(Point::new(p.x, p.y + 1)); }
        result
    }
}

#[derive(Clone, Debug)]
pub struct World {
    pub width: usize,
    pub height: usize,
    /// Ground truth. Never read by any exploration target selector.
    pub cells: Vec<Tile>,
    pub start: Point,
}

impl World {
    pub fn topology(&self) -> Topology { Topology { width: self.width, height: self.height } }
    pub fn new(width: usize, height: usize, cells: Vec<Tile>, start: Point) -> Result<Self, String> {
        if width < 3 || height < 3 || cells.len() != width * height {
            return Err("invalid dimensions or cell count".into());
        }
        if cells.iter().any(|&tile| tile == Tile::Unknown) {
            return Err("ground truth cannot contain unknown cells".into());
        }
        let world = Self { width, height, cells, start };
        if !world.inside(start) || world.tile(start) != Tile::Free {
            return Err("start must be free".into());
        }
        Ok(world)
    }

    pub fn generated(seed: u64, width: usize, height: usize) -> Self {
        assert!(width >= 3 && height >= 3);
        let mut rng = seed ^ 0xa076_1d64_78bd_642f;
        let mut cells = vec![Tile::Free; width * height];
        for y in 0..height {
            for x in 0..width {
                // Independent deterministic obstacles; borders always blocked.
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let fraction = (rng >> 11) as f64 / ((1u64 << 53) as f64);
                let border = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
                cells[y * width + x] = if border || fraction < 0.17 {
                    Tile::Blocked
                } else {
                    Tile::Free
                };
            }
        }
        cells[width + 1] = Tile::Free;
        Self::new(width, height, cells, Point::new(1, 1)).expect("valid generated grid")
    }

    pub fn inside(&self, p: Point) -> bool {
        p.x < self.width && p.y < self.height
    }

    pub fn idx(&self, p: Point) -> usize {
        p.y * self.width + p.x
    }

    pub fn point(&self, idx: usize) -> Point {
        Point::new(idx % self.width, idx / self.width)
    }

    pub fn tile(&self, p: Point) -> Tile {
        if self.inside(p) {
            self.cells[self.idx(p)]
        } else {
            Tile::Blocked
        }
    }

    pub fn neighbors(&self, p: Point) -> Vec<Point> {
        let mut result = Vec::with_capacity(4);
        if p.x > 0 { result.push(Point::new(p.x - 1, p.y)); }
        if p.x + 1 < self.width { result.push(Point::new(p.x + 1, p.y)); }
        if p.y > 0 { result.push(Point::new(p.x, p.y - 1)); }
        if p.y + 1 < self.height { result.push(Point::new(p.x, p.y + 1)); }
        result
    }
}

/// Bresenham ray. A blocking cell can itself be observed, but hides later cells.
fn sensor_ray(world: &World, known: &mut [Tile], a: Point, b: Point) {
    let (mut x, mut y) = (a.x as i64, a.y as i64);
    let (x1, y1) = (b.x as i64, b.y as i64);
    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let sy = if y < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        let p = Point::new(x as usize, y as usize);
        let id = world.idx(p);
        known[id] = world.cells[id];
        if world.cells[id] == Tile::Blocked || (x == x1 && y == y1) { break; }
        let e2 = 2 * error;
        if e2 >= dy { error += dy; x += sx; }
        if e2 <= dx { error += dx; y += sy; }
    }
}

/// Idealized range-limited line-of-sight observation, no SLAM or sensor noise.
pub fn observe(world: &World, known: &mut [Tile], position: Point, radius: usize) {
    assert!(radius > 0 && known.len() == world.cells.len());
    assert!(world.inside(position) && world.tile(position) == Tile::Free);
    for y in position.y.saturating_sub(radius)..=(position.y + radius).min(world.height - 1) {
        for x in position.x.saturating_sub(radius)..=(position.x + radius).min(world.width - 1) {
            let dx = position.x.abs_diff(x);
            let dy = position.y.abs_diff(y);
            if dx * dx + dy * dy <= radius * radius {
                sensor_ray(world, known, position, Point::new(x, y));
            }
        }
    }
}

/// Evaluation-only truth reachability. Never use this for navigation decisions.
pub fn truth_reachable(world: &World) -> Vec<usize> {
    let mut seen = vec![false; world.cells.len()];
    let start = world.idx(world.start);
    let mut queue = VecDeque::from([start]);
    seen[start] = true;
    let mut result = Vec::new();
    while let Some(at) = queue.pop_front() {
        result.push(at);
        for next in world.neighbors(world.point(at)) {
            let id = world.idx(next);
            if world.cells[id] == Tile::Free && !seen[id] {
                seen[id] = true;
                queue.push_back(id);
            }
        }
    }
    result
}

pub fn distances(topology: &Topology, known: &[Tile], source: Point) -> Vec<usize> {
    let mut out = vec![usize::MAX; known.len()];
    if !topology.inside(source) || known[topology.idx(source)] != Tile::Free {
        return out;
    }
    let root = topology.idx(source);
    out[root] = 0;
    let mut queue = VecDeque::from([root]);
    while let Some(at) = queue.pop_front() {
        for p in topology.neighbors(topology.point(at)) {
            let id = topology.idx(p);
            if known[id] == Tile::Free && out[id] == usize::MAX {
                out[id] = out[at] + 1;
                queue.push_back(id);
            }
        }
    }
    out
}

/// Shortest path through observed FREE cells. Unknown cells are never traversed.
pub fn shortest_path(topology: &Topology, known: &[Tile], source: Point, goal: Point) -> Option<Vec<Point>> {
    if !topology.inside(source) || !topology.inside(goal) { return None; }
    let start = topology.idx(source);
    let end = topology.idx(goal);
    if known[start] != Tile::Free || known[end] != Tile::Free { return None; }
    let mut prev = vec![usize::MAX; known.len()];
    prev[start] = start;
    let mut queue = VecDeque::from([start]);
    while let Some(at) = queue.pop_front() {
        if at == end {
            let mut route = vec![topology.point(at)];
            let mut cur = at;
            while cur != start {
                cur = prev[cur];
                route.push(topology.point(cur));
            }
            route.reverse();
            return Some(route);
        }
        for next in topology.neighbors(topology.point(at)) {
            let id = topology.idx(next);
            if known[id] == Tile::Free && prev[id] == usize::MAX {
                prev[id] = at;
                queue.push_back(id);
            }
        }
    }
    None
}

pub fn frontiers(topology: &Topology, known: &[Tile]) -> Vec<usize> {
    (0..known.len()).filter(|&id| {
        known[id] == Tile::Free && topology.neighbors(topology.point(id)).iter()
            .any(|p| known[topology.idx(*p)] == Tile::Unknown)
    }).collect()
}

fn information_gain(topology: &Topology, known: &[Tile], at: usize) -> usize {
    topology.neighbors(topology.point(at)).iter()
        .filter(|p| known[topology.idx(**p)] == Tile::Unknown).count()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Frontier,
    Growth,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self { Self::Frontier => "frontier", Self::Growth => "growth" }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub target: Option<Point>,
    pub growth_nodes: usize,
    pub frontier_count: usize,
}

/// Influence-limited attraction-tree growth over currently observed FREE cells.
pub fn choose_target(topology: &Topology, known: &[Tile], current: Point, policy: Policy) -> Decision {
    let dists = distances(topology, known, current);
    let mut targets = frontiers(topology, known);
    targets.retain(|&id| dists[id] != usize::MAX && id != topology.idx(current));
    if targets.is_empty() {
        return Decision { target: None, growth_nodes: 0, frontier_count: 0 };
    }
    targets.sort_by_key(|&id| (dists[id], usize::MAX - information_gain(topology, known, id), id));
    let nearest = targets[0];
    if policy == Policy::Frontier {
        return Decision { target: Some(topology.point(nearest)), growth_nodes: 0, frontier_count: targets.len() };
    }

    let influence_radius = 6isize;
    let radius_squared = influence_radius * influence_radius;
    let mut nodes = vec![topology.idx(current)];
    let mut occupied = vec![false; known.len()];
    occupied[nodes[0]] = true;
    for _ in 0..60 {
        let mut direction = vec![(0.0f64, 0.0f64); known.len()];
        let mut influenced = vec![false; known.len()];
        for &target in &targets {
            let p = topology.point(target);
            let &tip = nodes.iter().min_by_key(|&&node| {
                let n = topology.point(node);
                n.x.abs_diff(p.x).pow(2) + n.y.abs_diff(p.y).pow(2)
            }).expect("nonempty tree");
            let n = topology.point(tip);
            let dx = p.x as isize - n.x as isize;
            let dy = p.y as isize - n.y as isize;
            let dist2 = dx * dx + dy * dy;
            if dist2 == 0 || dist2 > radius_squared { continue; }
            let magnitude = (dist2 as f64).sqrt();
            let weight = 1.0 + 0.35 * information_gain(topology, known, target) as f64;
            direction[tip].0 += weight * dx as f64 / magnitude;
            direction[tip].1 += weight * dy as f64 / magnitude;
            influenced[tip] = true;
        }
        let mut next = Vec::new();
        for &tip in &nodes {
            if !influenced[tip] { continue; }
            let (dx, dy) = direction[tip];
            let origin = topology.point(tip);
            let mut selected = None;
            let mut best = 1e-9f64;
            for p in topology.neighbors(origin) {
                let candidate = topology.idx(p);
                if known[candidate] != Tile::Free || occupied[candidate] { continue; }
                let alignment = (p.x as isize - origin.x as isize) as f64 * dx
                    + (p.y as isize - origin.y as isize) as f64 * dy;
                if alignment > best {
                    best = alignment;
                    selected = Some(candidate);
                }
            }
            if let Some(candidate) = selected {
                occupied[candidate] = true;
                next.push(candidate);
            }
        }
        if next.is_empty() { break; }
        nodes.extend(next);
        if nodes.len() >= known.len() { break; }
    }

    let mut best = nearest;
    let mut best_score = f64::NEG_INFINITY;
    for &target in &targets {
        if !occupied[target] { continue; }
        let score = (1.0 + information_gain(topology, known, target) as f64)
            / (2.0 + dists[target] as f64);
        if score > best_score || (score == best_score && dists[target] < dists[best]) {
            best_score = score;
            best = target;
        }
    }
    Decision { target: Some(topology.point(best)), growth_nodes: nodes.len(), frontier_count: targets.len() }
}

#[derive(Clone, Copy, Debug)]
pub struct Telemetry {
    pub localization_confidence: f64,
    pub sensor_healthy: bool,
    pub anomaly: bool,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self { localization_confidence: 1.0, sensor_healthy: true, anomaly: false }
    }
}

impl Telemetry {
    pub fn may_move(self) -> bool {
        self.sensor_healthy && !self.anomaly && self.localization_confidence.is_finite()
            && (0.70..=1.0).contains(&self.localization_confidence)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub radius: usize,
    pub max_steps: usize,
}

impl Default for Config {
    fn default() -> Self { Self { radius: 3, max_steps: 3_000 } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeResult {
    pub policy: Policy,
    pub reachable_cells: usize,
    pub observed_reachable: usize,
    pub visited_reachable: usize,
    pub coverage: f64,
    pub steps: usize,
    pub decisions: usize,
    pub growth_nodes: usize,
    pub collisions: usize,
    pub stopped_for_safety: bool,
    pub completed: bool,
    pub exhausted_budget: bool,
}

pub fn run_episode(world: &World, policy: Policy, config: Config) -> Result<EpisodeResult, String> {
    run_episode_with_telemetry(world, policy, config, |_, _| Telemetry::default())
}

pub fn run_episode_with_telemetry<F>(
    world: &World, policy: Policy, config: Config, mut gate: F,
) -> Result<EpisodeResult, String>
where F: FnMut(usize, Point) -> Telemetry {
    if config.radius == 0 || config.max_steps == 0 { return Err("invalid config".into()); }
    let truth = truth_reachable(world);
    let topology = world.topology();
    let mut known = vec![Tile::Unknown; world.cells.len()];
    let mut pos = world.start;
    let mut visited = vec![false; world.cells.len()];
    visited[world.idx(pos)] = true;
    let mut steps = 0usize;
    let mut decisions = 0usize;
    let mut growth_nodes = 0usize;
    let mut collisions = 0usize;
    let mut stopped = false;
    observe(world, &mut known, pos, config.radius);

    while steps < config.max_steps {
        let choice = choose_target(&topology, &known, pos, policy);
        decisions += 1;
        growth_nodes += choice.growth_nodes;
        let target = match choice.target { Some(p) => p, None => break };
        let path = shortest_path(&topology, &known, pos, target)
            .ok_or_else(|| "planner selected unreachable target".to_string())?;
        if path.len() < 2 { return Err("planner selected current position".into()); }
        for next in path.into_iter().skip(1) {
            if steps >= config.max_steps { break; }
            if !gate(steps, pos).may_move() {
                stopped = true;
                break;
            }
            if known[world.idx(next)] != Tile::Free {
                return Err("planner attempted unknown or blocked terrain".into());
            }
            if world.tile(next) != Tile::Free {
                collisions += 1;
                stopped = true;
                break;
            }
            pos = next;
            visited[world.idx(pos)] = true;
            steps += 1;
            observe(world, &mut known, pos, config.radius);
        }
        if stopped { break; }
    }
    let observed_reachable = truth.iter().filter(|&&id| known[id] == Tile::Free).count();
    let visited_reachable = truth.iter().filter(|&&id| visited[id]).count();
    let coverage = observed_reachable as f64 / truth.len() as f64;
    Ok(EpisodeResult {
        policy, reachable_cells: truth.len(), observed_reachable, visited_reachable,
        coverage, steps, decisions, growth_nodes, collisions,
        stopped_for_safety: stopped, completed: observed_reachable == truth.len(),
        exhausted_budget: steps >= config.max_steps && observed_reachable < truth.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_world() -> World {
        let mut cells = vec![Tile::Blocked; 7 * 5];
        for y in 1..4 { for x in 1..6 { cells[y * 7 + x] = Tile::Free; } }
        World::new(7, 5, cells, Point::new(1, 1)).unwrap()
    }

    #[test]
    fn shortest_path_never_traverses_unknown() {
        let world = open_world();
        let mut known = vec![Tile::Unknown; world.cells.len()];
        observe(&world, &mut known, world.start, 2);
        assert!(shortest_path(&world.topology(), &known, world.start, Point::new(5, 1)).is_none());
        assert!(shortest_path(&world.topology(), &known, world.start, Point::new(2, 1)).is_some());
    }

    #[test]
    fn wall_occludes_cells_behind_it() {
        let mut world = open_world();
        let obstacle = world.idx(Point::new(3, 1));
        world.cells[obstacle] = Tile::Blocked;
        let mut known = vec![Tile::Unknown; world.cells.len()];
        observe(&world, &mut known, world.start, 4);
        assert_eq!(known[world.idx(Point::new(3, 1))], Tile::Blocked);
        assert_eq!(known[world.idx(Point::new(4, 1))], Tile::Unknown);
    }

    #[test]
    fn both_policies_complete_simple_world() {
        let world = open_world();
        for policy in [Policy::Frontier, Policy::Growth] {
            let r = run_episode(&world, policy, Config { radius: 2, max_steps: 500 }).unwrap();
            assert!(r.completed, "{r:?}");
            assert_eq!(r.collisions, 0);
            assert_eq!(r.coverage, 1.0);
        }
    }

    #[test]
    fn anomalous_telemetry_halts_before_first_motion() {
        let world = open_world();
        let r = run_episode_with_telemetry(&world, Policy::Growth, Config::default(), |_, _| {
            Telemetry { anomaly: true, ..Telemetry::default() }
        }).unwrap();
        assert_eq!(r.steps, 0);
        assert!(r.stopped_for_safety);
    }

    #[test]
    fn fixed_seeds_are_reproducible() {
        let world = World::generated(7, 25, 19);
        let a = run_episode(&world, Policy::Growth, Config::default()).unwrap();
        let b = run_episode(&world, Policy::Growth, Config::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn target_selection_does_not_read_hidden_truth() {
        let world = open_world();
        let mut known = vec![Tile::Unknown; world.cells.len()];
        observe(&world, &mut known, world.start, 2);
        let a = choose_target(&world.topology(), &known, world.start, Policy::Growth);
        let mut different = world.clone();
        different.cells[world.idx(Point::new(5, 3))] = Tile::Blocked;
        let b = choose_target(&different.topology(), &known, world.start, Policy::Growth);
        assert_eq!(a, b);
    }
}
