"""Ground-truth grid, line-of-sight observation and connectivity oracle.

The WORLD must never be consulted by exploration target selection or routing.
It is used only by the sensor, simulated motion, and post-run evaluator.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from random import Random

UNKNOWN = -1
FREE = 0
BLOCKED = 1
Point = tuple[int, int]


@dataclass(frozen=True)
class Topology:
    """Only grid geometry. Deliberately cannot inspect ground-truth occupancy."""
    width: int
    height: int

    def idx(self, p: Point) -> int:
        return p[1] * self.width + p[0]

    def inside(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def neighbors(self, p: Point):
        x, y = p
        for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if self.inside(q):
                yield q


@dataclass(frozen=True)
class World:
    width: int
    height: int
    cells: tuple[int, ...]
    start: Point = (1, 1)

    def __post_init__(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("grid must be >= 3x3")
        if len(self.cells) != self.width * self.height:
            raise ValueError("cell count mismatch")
        if any(c not in (FREE, BLOCKED) for c in self.cells):
            raise ValueError("ground truth cannot contain unknown cells")
        if self.tile(self.start) != FREE:
            raise ValueError("start must be free")

    def topology(self) -> Topology:
        return Topology(self.width, self.height)

    def idx(self, p: Point) -> int:
        return p[1] * self.width + p[0]

    def inside(self, p: Point) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def tile(self, p: Point) -> int:
        if not self.inside(p):
            return BLOCKED
        return self.cells[self.idx(p)]

    def neighbors(self, p: Point):
        x, y = p
        for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if self.inside(q):
                yield q

    @classmethod
    def generated(cls, seed: int, width: int = 25, height: int = 19,
                  obstacle_fraction: float = 0.17) -> "World":
        if not 0.0 <= obstacle_fraction < 1.0:
            raise ValueError("invalid obstacle fraction")
        rng = Random(seed)
        cells = []
        for y in range(height):
            for x in range(width):
                # A few wall clusters make line-of-sight occlusion meaningful.
                boundary = x in (0, width - 1) or y in (0, height - 1)
                cells.append(BLOCKED if boundary or rng.random() < obstacle_fraction else FREE)
        cells[width + 1] = FREE
        return cls(width, height, tuple(cells))

    @classmethod
    def from_rows(cls, rows: tuple[str, ...] | list[str]) -> "World":
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("invalid rows")
        start: Point | None = None
        cells = []
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c not in ("#", ".", "S"):
                    raise ValueError("use '#', '.', 'S' in scenario rows")
                if c == "S":
                    if start is not None:
                        raise ValueError("multiple start cells")
                    start = (x, y)
                cells.append(BLOCKED if c == "#" else FREE)
        if start is None:
            raise ValueError("missing start cell")
        return cls(len(rows[0]), len(rows), tuple(cells), start)


def line_cells(a: Point, b: Point):
    """Bresenham cells from a to b inclusive (discrete ray sensor model)."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield (x0, y0)
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def observe(world: World, known: list[int], position: Point,
            sensor_radius: int) -> int:
    """Reveal only in-range cells not occluded by a closer blocked cell.

    A blocked target itself can be seen; nothing beyond it on the same ray.
    This idealized model does not include noisy range readings or SLAM.
    """
    if sensor_radius < 1:
        raise ValueError("sensor_radius must be >= 1")
    if len(known) != len(world.cells):
        raise ValueError("incorrect knowledge size")
    if world.tile(position) != FREE:
        raise ValueError("sensor pose not in free space")
    px, py = position
    newly_seen = 0
    r2 = sensor_radius * sensor_radius
    for y in range(max(0, py - sensor_radius), min(world.height, py + sensor_radius + 1)):
        for x in range(max(0, px - sensor_radius), min(world.width, px + sensor_radius + 1)):
            if (px - x)**2 + (py - y)**2 > r2:
                continue
            target = (x, y)
            for cell in line_cells(position, target):
                index = world.idx(cell)
                if known[index] == UNKNOWN:
                    known[index] = world.tile(cell)
                    newly_seen += 1
                if world.tile(cell) == BLOCKED:
                    break
    return newly_seen


def reachable(world: World) -> set[Point]:
    """Evaluation-only oracle; no navigation policy may use this function."""
    seen = {world.start}
    queue = deque([world.start])
    while queue:
        for neighbor in world.neighbors(queue.popleft()):
            if neighbor not in seen and world.tile(neighbor) == FREE:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def challenge_world(seed: int, family: str, width: int = 19,
                    height: int = 15) -> World:
    """Procedural environment families, seeded and independent of planners.

    Families intentionally stress different topologies. All boundary cells are
    blocked; start (1,1) is connected to an interior corridor by construction.
    """
    if width < 9 or height < 9:
        raise ValueError("challenge worlds need at least 9x9")
    rng = Random(seed)
    cells = [[BLOCKED] * width for _ in range(height)]
    if family == "rooms":
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                cells[y][x] = FREE
        # Dividers with variable doorways.
        for x in range(5, width - 1, 6):
            door = rng.randrange(1, height - 1)
            for y in range(1, height - 1):
                if y != door:
                    cells[y][x] = BLOCKED
        for y in range(5, height - 1, 6):
            door = rng.randrange(1, width - 1)
            for x in range(1, width - 1):
                if x != door:
                    cells[y][x] = BLOCKED
    elif family == "corridors":
        # Perfect odd-cell lattice maze generated by randomized depth-first carve.
        stack = [(1, 1)]
        cells[1][1] = FREE
        while stack:
            x, y = stack[-1]
            directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            rng.shuffle(directions)
            advance = None
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < width - 1 and 1 <= ny < height - 1 and cells[ny][nx] == BLOCKED:
                    advance = (nx, ny, x + dx // 2, y + dy // 2)
                    break
            if advance is None:
                stack.pop()
            else:
                nx, ny, mx, my = advance
                cells[my][mx] = cells[ny][nx] = FREE
                stack.append((nx, ny))
    elif family == "scatter":
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                cells[y][x] = BLOCKED if rng.random() < 0.21 else FREE
        # Clear a guaranteed central spine from the start to avoid degenerate runs.
        for x in range(1, width - 1):
            cells[1][x] = FREE
        for y in range(1, height - 1):
            cells[y][1] = FREE
    else:
        raise ValueError(f"unknown world family {family}")
    cells[1][1] = FREE
    return World(width, height, tuple(c for row in cells for c in row))
