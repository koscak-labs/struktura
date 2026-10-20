"""Paired-seed, partial-observability COLONY benchmark (stdlib only)."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from .simulation import run_episode
from .world import World


def benchmark(*, seeds: int = 12, width: int = 25, height: int = 19,
              radius: int = 3, max_steps: int = 3000) -> dict:
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    episodes = []
    for seed in range(seeds):
        world = World.generated(seed, width, height)
        for policy in ("frontier", "growth"):
            started = time.perf_counter()
            record = run_episode(world, policy, seed=seed,
                                 sensor_radius=radius, max_steps=max_steps).as_dict()
            record["runtime_seconds"] = round(time.perf_counter() - started, 6)
            episodes.append(record)
    rows = {p: [r for r in episodes if r["policy"] == p] for p in ("frontier", "growth")}
    summary = {}
    for policy, records in rows.items():
        summary[policy] = {
            "mean_coverage": statistics.mean(r["coverage"] for r in records),
            "mean_steps": statistics.mean(r["steps"] for r in records),
            "completed": sum(r["completed"] for r in records),
            "collisions": sum(r["collisions"] for r in records),
            "mean_decisions": statistics.mean(r["decisions"] for r in records),
            "mean_runtime_seconds": statistics.mean(r["runtime_seconds"] for r in records),
        }
    pairs = [(rows["growth"][i]["steps"] - rows["frontier"][i]["steps"])
             for i in range(seeds) if rows["growth"][i]["completed"]
             and rows["frontier"][i]["completed"]]
    return {
        "version": "COLONY-0.2",
        "model": "static hidden occupancy grid, idealized line-of-sight range sensor",
        "policies": ["frontier", "growth"],
        "seeds": seeds, "width": width, "height": height,
        "radius": radius, "max_steps": max_steps,
        "summary": summary,
        "paired_complete_cases": len(pairs),
        "paired_growth_minus_frontier_steps_mean": statistics.mean(pairs) if pairs else None,
        "paired_growth_minus_frontier_steps_stddev": statistics.stdev(pairs) if len(pairs) > 1 else None,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--height", type=int, default=19)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = benchmark(seeds=args.seeds, width=args.width, height=args.height,
                       radius=args.radius, max_steps=args.max_steps)
    if args.json:
        args.json.parent.mkdir(exist_ok=True, parents=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    if args.csv:
        args.csv.parent.mkdir(exist_ok=True, parents=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=result["episodes"][0].keys(),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(result["episodes"])
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
