import unittest
import math
from colony.veritas import *

class VeritasTests(unittest.TestCase):
    def test_independent_replicate_different(self):
        for family in FAMILIES:
            self.assertNotEqual(independent_trace(42,family,(6,6),0),
                                independent_trace(42,family,(6,6),1))
    def test_shuffling_destroys_order_preserves_values(self):
        # Independent_trace() always computes its unshuffled correlated counterpart.
        a=independent_trace(2,'correlation',(9,9),0)
        b=independent_trace(2,'shuffled',(9,9),0)
        self.assertEqual(sorted(a),sorted(b))
        self.assertNotEqual(a,b)
    def test_missing_never_passes_confirmation(self):
        field={(x,y):None for y in range(N) for x in range(N)}
        r=episode(2,'dropout','trustroot',field,field,threshold=.8,budget=15)
        self.assertEqual(r.confirm_reads,0)
        self.assertFalse(r.claim)
    def test_no_claim_from_single_window(self):
        field={(x,y):(.99,.9) for y in range(N) for x in range(N)}
        unavailable={(x,y):None for y in range(N) for x in range(N)}
        r=episode(3,'correlation','signal_sca',field,unavailable,threshold=.8,budget=13)
        self.assertFalse(r.claim)
    def test_claim_counter_is_budgeted(self):
        field={(x,y):(.99,.9) for y in range(N) for x in range(N)}
        r=episode(3,'correlation','signal_sca',field,field,threshold=.8,budget=13)
        self.assertTrue(r.claim)
        self.assertEqual(r.readings,2)
        self.assertEqual(r.confirm_reads,1)
        self.assertEqual(r.primary_locations,1)
    def test_shuffled_claim_is_false_even_near_hidden_target(self):
        field={(x,y):(.99,.9) for y in range(N) for x in range(N)}
        r=episode(3,'shuffled','signal_sca',field,field,threshold=.8,budget=13)
        self.assertTrue(r.false_claim)
        self.assertFalse(r.true_claim)
    def test_maximum_travel(self):
        field={(x,y):(.6,.9) for y in range(N) for x in range(N)}
        for m in METHODS:
            r=episode(3,'correlation',m,field,field,threshold=.9,budget=20,travel_cap=20)
            self.assertLessEqual(r.travel,20)
    def test_reproducible(self):
        field={(x,y):(.6,.9) for y in range(N) for x in range(N)}
        a=episode(3,'correlation','trustroot',field,field,.9,budget=20)
        b=episode(3,'correlation','trustroot',field,field,.9,budget=20)
        self.assertEqual(a,b)
    def test_calibration_disjoint_validation(self):
        with self.assertRaises(ValueError):run(seeds=1,seed_start=1,command=None)
    def test_bad_field(self):
        with self.assertRaises(ValueError):independent_trace(4,'bad',(1,1),0)
        with self.assertRaises(ValueError):independent_trace(4,'correlation',(1,1),3)
    def test_bad_replica_data(self):
        field={(x,y):(.6,.9) for y in range(N) for x in range(N)}
        confirm=dict(field);confirm.pop((0,0))
        with self.assertRaises(ValueError):episode(2,'correlation','coverage',field,confirm,.8)
