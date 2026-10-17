"""Unittest suite covering observation, navigation, reproducibility and safety."""
import unittest
from colony import (World, FREE, BLOCKED, UNKNOWN, observe, reachable,
                    shortest_path, frontiers, choose_target, run_episode, Telemetry)
from colony.benchmark import benchmark


class WorldTests(unittest.TestCase):
    def test_occlusion_hides_cells_behind_wall(self):
        world = World.from_rows(["########", "#S.#...#", "#......#", "########"])
        known = [UNKNOWN] * len(world.cells)
        observe(world, known, (1, 1), 6)
        self.assertEqual(known[world.idx((3, 1))], BLOCKED)
        self.assertEqual(known[world.idx((4, 1))], UNKNOWN)
        self.assertEqual(known[world.idx((2, 1))], FREE)

    def test_disconnected_area_not_counted_as_reachable(self):
        world = World.from_rows(["#######", "#S.#..#", "#..#..#", "#######"])
        self.assertEqual(len(reachable(world)), 4)

    def test_invalid_truth_or_start(self):
        with self.assertRaises(ValueError):
            World.from_rows(["###", "#.#", "###"])
        with self.assertRaises(ValueError):
            World.generated(1, 8, 8, obstacle_fraction=1.0)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.world = World.from_rows(["########", "#S.....#", "#..#...#", "#......#", "########"])
        self.known = [UNKNOWN] * len(self.world.cells)
        observe(self.world, self.known, self.world.start, 2)

    def test_bfs_never_routes_over_unknown(self):
        self.assertIsNone(shortest_path(self.world.topology(), self.known, (1, 1), (5, 1)))
        self.assertEqual(shortest_path(self.world.topology(), self.known, (1, 1), (2, 1)),
                         [(1, 1), (2, 1)])

    def test_both_policies_select_reachable_free_frontier(self):
        for policy in ("frontier", "growth"):
            with self.subTest(policy=policy):
                decision = choose_target(self.world.topology(), self.known, self.world.start, policy)
                self.assertIn(decision.target, frontiers(self.world.topology(), self.known))
                self.assertEqual(self.known[self.world.idx(decision.target)], FREE)
                self.assertIsNotNone(shortest_path(self.world.topology(), self.known,
                                                   self.world.start, decision.target))

    def test_growth_nodes_bounded_and_no_truth_lookup(self):
        decision = choose_target(self.world.topology(), self.known, self.world.start, "growth")
        self.assertLessEqual(decision.growth_nodes, len(self.world.cells))
        # Hold knowledge constant but change hidden truth: target must be unchanged.
        inverted_truth = tuple(FREE if t == BLOCKED else BLOCKED
                               for t in self.world.cells)
        cells = list(inverted_truth)
        cells[self.world.idx(self.world.start)] = FREE
        other = World(self.world.width, self.world.height, tuple(cells), self.world.start)
        self.assertEqual(decision, choose_target(other.topology(), self.known, other.start, "growth"))

    def test_no_unknown_path(self):
        self.assertIsNone(shortest_path(self.world.topology(), self.known, (1, 1), (6, 3)))


class EpisodeTests(unittest.TestCase):
    def test_both_cover_open_world(self):
        world = World.from_rows(["#######", "#S....#", "#.....#", "#.....#", "#######"])
        for policy in ("frontier", "growth"):
            with self.subTest(policy=policy):
                result = run_episode(world, policy, sensor_radius=2, max_steps=500)
                self.assertTrue(result.completed, result)
                self.assertEqual(result.collisions, 0)
                self.assertEqual(result.coverage, 1.0)

    def test_safety_gate_halts_without_moving(self):
        world = World.generated(3)
        result = run_episode(world, "growth", telemetry=lambda step, p: Telemetry(anomaly=True))
        self.assertEqual(result.steps, 0)
        self.assertTrue(result.stopped_for_safety)
        self.assertEqual(result.collisions, 0)

    def test_low_confidence_prevents_move(self):
        world = World.generated(3)
        result = run_episode(world, "frontier",
                             telemetry=lambda step, p: Telemetry(localization_confidence=0.45))
        self.assertEqual(result.steps, 0)
        self.assertTrue(result.stopped_for_safety)

    def test_fixed_seed_reproducible(self):
        world = World.generated(7)
        self.assertEqual(run_episode(world, "growth", seed=7),
                         run_episode(world, "growth", seed=7))

    def test_benchmark_pairs_seeds(self):
        data = benchmark(seeds=3, width=13, height=11, max_steps=300)
        self.assertEqual(len(data["episodes"]), 6)
        self.assertEqual({row["seed"] for row in data["episodes"]}, {0, 1, 2})
        self.assertTrue(all(row["collisions"] == 0 for row in data["episodes"]))


if __name__ == "__main__":
    unittest.main()

# Keep visualization smoke validation independent of external graphics packages.
class VisualizationTests(unittest.TestCase):
    def test_svg_contains_policy_labels_and_trajectories(self):
        from colony.visualize import draw_svg
        svg = draw_svg(World.generated(1, 10, 9), seed=1, radius=2, max_steps=300)
        self.assertIn('Nearest frontier BFS', svg)
        self.assertIn('SCA-inspired growth', svg)
        self.assertIn('<polyline', svg)
