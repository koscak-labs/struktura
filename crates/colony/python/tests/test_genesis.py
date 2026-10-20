"""GENESIS: reproducibility, split isolation and finite search budget."""
import unittest
from random import Random
from colony.genesis import Genome, mutate, archive_search, discovery_report, evaluate
from colony.world import challenge_world, reachable


class GenesisTests(unittest.TestCase):
    def test_family_coverage_and_connectivity(self):
        for family in ("rooms", "scatter", "corridors"):
            world = challenge_world(123, family)
            self.assertGreater(len(reachable(world)), 10)
            self.assertEqual(world.tile(world.start), 0)

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            challenge_world(0, "alien")

    def test_mutation_reproducible_and_bounded(self):
        a = Random(9)
        b = Random(9)
        x = Genome()
        for _ in range(30):
            ga, gb = mutate(x, a), mutate(x, b)
            self.assertEqual(ga, gb)
            self.assertTrue(2 <= ga.radius <= 12)
            self.assertTrue(10 <= ga.budget <= 100)
            self.assertTrue(0 <= ga.gain_weight <= 2)
            self.assertTrue(.5 <= ga.distance_penalty <= 8)
            x = ga

    def test_deterministic_replay(self):
        a = archive_search(budget=5, train_seeds=1)
        b = archive_search(budget=5, train_seeds=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a["history"]), 5)
        self.assertLessEqual(a["unique_evaluations"], 5)

    def test_disjoint_heldout_and_control(self):
        r = discovery_report(budget=3, train_seeds=1, test_seeds=2)
        self.assertTrue(set(map(tuple, r["training"]["train_world_ids"]))
                        .isdisjoint(set(map(tuple, r["heldout_worlds"]))))
        self.assertEqual({f for f,_ in r["heldout_worlds"]}, {"corridors"})
        self.assertEqual(len(r["paired_evolved_minus_frontier_cost"]), 2)
        self.assertIn("untuned_growth", r["heldout"])

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            archive_search(budget=0)

    def test_coverage_metric_does_not_reward_failure(self):
        r = evaluate(Genome(), [("rooms", 4)], max_steps=1)
        self.assertGreater(r["cost"], 1)
        self.assertEqual(r["completed"], 0)


if __name__ == "__main__":
    unittest.main()
