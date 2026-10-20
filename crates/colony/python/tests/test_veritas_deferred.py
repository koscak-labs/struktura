import unittest
from colony.veritas_deferred import *
from colony.veritas import FAMILIES

class DeferredTests(unittest.TestCase):
    def make(self,value=.8):return {(x,y):(value,.9) for y in range(N) for x in range(N)}
    def test_requires_confirmation(self):
        first=self.make(.8)
        second=self.make(.1)
        r=deferred_episode(2,'correlation','signal_sca',first,second,.7)
        self.assertFalse(r.claimed)
    def test_repeated_measurement_counts_against_budget(self):
        first=self.make(.8);r=deferred_episode(2,'correlation','signal_sca',first,first,.7)
        self.assertTrue(r.claimed)
        self.assertLessEqual(r.acquired,24)
        self.assertEqual(r.acquired,r.distinct_places+r.repetitions)
    def test_non_probes_cannot_be_claimed(self):
        first=self.make(.1)
        first[(0,0)]=(.95,.9)
        r=deferred_episode(3,'correlation','signal_sca',first,first,.7)
        self.assertFalse(r.claimed)
    def test_reproducible(self):
        first=self.make(.9)
        a=deferred_episode(15,'dropout','trustroot',first,first,.7)
        self.assertEqual(a,deferred_episode(15,'dropout','trustroot',first,first,.7))
    def test_shuffled_cannot_be_true(self):
        first=self.make(.9)
        a=deferred_episode(15,'shuffled','coverage',first,first,.7)
        self.assertFalse(a.true_claim)
    def test_bad_budgets(self):
        with self.assertRaises(ValueError):deferred_episode(1,'correlation','coverage',self.make(),self.make(),.7,budget=8)
    def test_travel_bound(self):
        first=self.make(.1)
        for m in METHODS:
            r=deferred_episode(8,'correlation',m,first,first,.7,travel_cap=20)
            self.assertLessEqual(r.traveled,20)
