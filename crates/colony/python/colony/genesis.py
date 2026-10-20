"""GENESIS: deterministic, budgeted, MAP-Elites-inspired growth policy search.

Search is training-only. Evaluation on disjoint seeds and world families happens
once, after selecting the champion. This is simulation research, not evidence of
physical robot safety or autonomous general intelligence.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from random import Random
from statistics import mean
from .simulation import run_episode
from .world import challenge_world, reachable

FAMILIES = ("rooms", "scatter", "corridors")


@dataclass(frozen=True)
class Genome:
    radius: int = 6
    budget: int = 60
    gain_weight: float = 0.35
    distance_penalty: float = 2.0

    def parameters(self) -> dict:
        return {"influence_radius": self.radius,
                "growth_budget": self.budget,
                "gain_weight": self.gain_weight,
                "distance_penalty": self.distance_penalty}

    def niche(self) -> tuple[int, int]:
        # Behavior descriptor: local/global attractor horizon vs shallow/deep growth.
        return (min(3, self.radius // 3), min(3, self.budget // 25))


def mutate(genome: Genome, rng: Random) -> Genome:
    choice = rng.randrange(4)
    fields = asdict(genome)
    key = ("radius", "budget", "gain_weight", "distance_penalty")[choice]
    step = (-1 if rng.random() < .5 else 1)
    if key == "radius":
        fields[key] = min(12, max(2, fields[key] + step * rng.choice((1, 2))))
    elif key == "budget":
        fields[key] = min(100, max(10, fields[key] + step * 10))
    elif key == "gain_weight":
        fields[key] = round(min(2.0, max(0.0, fields[key] + step * .15)), 3)
    else:
        fields[key] = round(min(8.0, max(.5, fields[key] + step * .5)), 3)
    return Genome(**fields)


def evaluate(genome: Genome, worlds: list[tuple[str, int]],
             max_steps: int = 500) -> dict:
    """Evaluation never exposes ground truth to policy; oracle only scores outcomes."""
    observations = []
    for family, seed in worlds:
        world = challenge_world(seed, family)
        record = run_episode(world, "growth", seed=seed, sensor_radius=3,
                             max_steps=max_steps, growth_parameters=genome.parameters())
        # Guard against selective easy-map scoring: failure is very expensive.
        # Episode steps are bounded; failed coverage adds a penalty for missing area.
        penalty = (1.0 - record.coverage) * max_steps + (0 if record.completed else max_steps)
        observations.append({"family": family, "seed": seed, "steps": record.steps,
                             "coverage": record.coverage, "completed": record.completed,
                             "collisions": record.collisions,
                             "cost": round(record.steps + penalty, 5),
                             "reachable": len(reachable(world))})
    return {"cost": round(mean(r["cost"] for r in observations), 5),
            "mean_steps": mean(r["steps"] for r in observations),
            "mean_coverage": mean(r["coverage"] for r in observations),
            "completed": sum(r["completed"] for r in observations),
            "collisions": sum(r["collisions"] for r in observations),
            "episodes": observations}


def archive_search(*, budget: int = 24, seed: int = 17,
                   train_seeds: int = 3, max_steps: int = 500) -> dict:
    if budget < 1 or train_seeds < 1:
        raise ValueError("budget and train_seeds must be positive")
    rng = Random(seed)
    worlds = [(family, 1000 + i) for family in FAMILIES[:2]
              for i in range(train_seeds)]
    archive: dict[tuple[int, int], tuple[Genome, dict]] = {}
    evaluated: dict[Genome, dict] = {}
    history = []
    initial = Genome()
    for iteration in range(budget):
        if iteration == 0:
            genome = initial
        elif iteration % 5 == 0:
            genome = Genome(rng.randint(2, 12), rng.randrange(10, 101, 10),
                            round(rng.uniform(0, 2), 3), round(rng.uniform(.5, 8), 3))
        else:
            parent = rng.choice(list(archive.values()))[0]
            genome = mutate(parent, rng)
        # Cached duplicate genomes do not receive extra stochastic evaluations.
        if genome not in evaluated:
            evaluated[genome] = evaluate(genome, worlds, max_steps)
        outcome = evaluated[genome]
        niche = genome.niche()
        previous = archive.get(niche)
        replaced = previous is None or outcome["cost"] < previous[1]["cost"]
        if replaced:
            archive[niche] = (genome, outcome)
        history.append({"iteration": iteration, "genome": asdict(genome),
                        "niche": niche, "cost": outcome["cost"],
                        "new_elite": replaced})
    champion, champ_result = min(archive.values(), key=lambda pair: pair[1]["cost"])
    return {"algorithm": "mutation + 2D quality-diversity archive (MAP-Elites-inspired)",
            "seed": seed, "budget": budget, "train_seeds": train_seeds,
            "train_families": list(FAMILIES[:2]),
            "train_world_ids": worlds, "unique_evaluations": len(evaluated),
            "archive": [{"niche": niche, "genome": asdict(g),
                         "train_cost": score["cost"]}
                        for niche, (g, score) in sorted(archive.items())],
            "history": history, "champion": asdict(champion),
            "champion_train": champ_result}


def discovery_report(*, budget: int = 24, seed: int = 17,
                     train_seeds: int = 3, test_seeds: int = 5,
                     max_steps: int = 500) -> dict:
    if test_seeds < 1:
        raise ValueError("test_seeds must be positive")
    training = archive_search(budget=budget, seed=seed,
                              train_seeds=train_seeds, max_steps=max_steps)
    # Test split differs in seed AND topology family (strict OOD corridor challenge).
    heldout = [("corridors", 50000 + i) for i in range(test_seeds)]
    control = Genome()
    champion = Genome(**training["champion"])
    controls = {"evolved": evaluate(champion, heldout, max_steps),
                "untuned_growth": evaluate(control, heldout, max_steps)}
    frontier_rows = []
    for family, world_seed in heldout:
        world = challenge_world(world_seed, family)
        r = run_episode(world, "frontier", seed=world_seed,
                        sensor_radius=3, max_steps=max_steps)
        cost = r.steps + (1.0-r.coverage)*max_steps + (0 if r.completed else max_steps)
        frontier_rows.append({"family": family, "seed": world_seed,
                              "cost": round(cost, 5), "steps": r.steps,
                              "completed": r.completed, "coverage": r.coverage})
    controls["frontier"] = {"cost": round(mean(r["cost"] for r in frontier_rows), 5),
                            "mean_steps": mean(r["steps"] for r in frontier_rows),
                            "completed": sum(r["completed"] for r in frontier_rows),
                            "episodes": frontier_rows}
    diffs = [controls["evolved"]["episodes"][i]["cost"] - r["cost"]
             for i, r in enumerate(frontier_rows)]
    return {"experiment": "COLONY GENESIS 0.3", "protocol": {
                "train": "rooms+scatter seeds 1000.., optimization only",
                "test": "corridors seeds 50000.., single post-selection evaluation",
                "metric": "steps + (1-coverage)*step_cap + incomplete*step_cap; lower is better",
                "caveat": "toy grid and ideal sensing; held-out families not real-world robustness"},
            "training": training, "heldout_worlds": heldout,
            "heldout": controls,
            "paired_evolved_minus_frontier_cost": diffs,
            "paired_delta_mean": round(mean(diffs), 5),
            "beats_frontier_on_this_sample": (mean(diffs) < 0 and
                                             controls["evolved"]["collisions"] == 0 and
                                             controls["evolved"]["completed"] == test_seeds),
            "evolution_gain_over_untuned": round(controls["untuned_growth"]["cost"] -
                                                   controls["evolved"]["cost"], 5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--test-seeds", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = discovery_report(budget=args.budget, seed=args.seed,
                              train_seeds=args.train_seeds,
                              test_seeds=args.test_seeds, max_steps=args.max_steps)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"champion": report["training"]["champion"],
                      "niches": len(report["training"]["archive"]),
                      "unique_evaluations": report["training"]["unique_evaluations"],
                      "heldout_costs": {k: v["cost"] for k,v in report["heldout"].items()},
                      "paired_delta_mean": report["paired_delta_mean"],
                      "beats_frontier_on_this_sample": report["beats_frontier_on_this_sample"],
                      "evolution_gain_over_untuned": report["evolution_gain_over_untuned"]}, indent=2))


if __name__ == "__main__":
    main()
