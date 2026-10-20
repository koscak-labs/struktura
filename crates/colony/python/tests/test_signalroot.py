import unittest
from colony.signalroot import *

class SignalRootTests(unittest.TestCase):
    def test_all_have_identical_startup(self):
        for m in MODES:
            self.assertEqual(run(5,"compact",m,5).steps,5)
    def test_budget_and_unique_locations(self):
        for m in MODES:
            self.assertEqual(run(2,"decoy",m,20).steps,20)
    def test_reproducible(self):
        for m in MODES:
            self.assertEqual(run(8,"dropout",m,30),run(8,"dropout",m,30))
    def test_no_truth_in_proposals(self):
        seen={p:0.1 for p in startup()}
        import random
        for m in MODES:
            self.assertTrue(walkable(proposal(m,seen,random.Random(5),(12,12))))
    def test_missing_is_not_zero(self):
        observed={p:None for p in startup()}
        for m in ("signal_sca","signal_greedy"):
            self.assertTrue(walkable(proposal(m,observed,Random(1),(12,12))))
    def test_invalid(self):
        with self.assertRaises(ValueError): benchmark(0,10)
        with self.assertRaises(ValueError): run(1,"compact","invalid")
    def test_oracle_is_separate(self):
        w=World(4,"compact")
        observed={p:w.sample(p,i) for i,p in enumerate(startup())}
        clone=dict(observed)
        proposal("signal_sca",observed,Random(1),(12,12))
        self.assertEqual(observed,clone)
    def test_all_families(self):
        b=benchmark(2,10)
        self.assertEqual(len(b["episodes"]),40)
        self.assertEqual(len(b["results"]),4)
