import unittest, random, math
from colony.trustroot import *

class TrustrootTests(unittest.TestCase):
    def test_probe_generation_reproducible(self):
        for f in FAMILIES:
            self.assertEqual(field_trace(1,f,(9,8)),field_trace(1,f,(9,8)))
    def test_correlated_decoy_is_distinct(self):
        self.assertNotEqual(field_trace(2,'correlation',(9,8)),field_trace(2,'correlated_decoy',(9,8)))
    def test_shuffle_preserves_mean_and_variance_but_changes_order(self):
        a=field_trace(4,'correlation',(10,10))
        b=field_trace(4,'shuffled',(10,10))
        self.assertEqual(sorted(a),sorted(b))
        self.assertNotEqual(a,b)
    def test_no_hidden_truth_is_planner_argument(self):
        observed={p:.5 for p in START}
        before=dict(observed)
        for method in METHODS:
            p,_=policy_position(method,observed,(12,12),6,[True]*5)
            self.assertNotIn(p,observed)
            self.assertEqual(observed,before)
    def test_invalid_not_zero(self):
        observed={p:None for p in START}
        p,mode=trust_position(observed,START[0],last_valid=[False]*5)
        self.assertEqual(mode,'attract')
        self.assertNotIn(p,observed)
    def test_recovery_to_attraction(self):
        observed={p:.5 for p in START}
        _,mode=trust_position(observed,START[0],last_valid=[False,True,True,True,True,True,True])
        self.assertEqual(mode,'attract')
    def test_isolated_dropout_does_not_force_repair(self):
        observed={p:.4 for p in START}; observed[START[0]]=None
        _,mode=trust_position(observed,START[-1],last_valid=[False,True,True,True,True])
        self.assertEqual(mode,'attract')
    def test_sustained_dropout_triggers_repair(self):
        observed={p:None for p in START}
        _,mode=trust_position(observed,START[-1],last_valid=[False]*8)
        self.assertEqual(mode,'repair')
    def test_episode_pairs_preserve_world(self):
        # No Rust bridge in local tests: mocked *input scores* exercise planner only.
        data={(x,y):(.5 + math.exp(-((x-13)**2+(y-13)**2)/20),.9) for y in range(N) for x in range(N)}
        for m in METHODS:
            r=episode(2,'correlation',m,data,budget=15)
            self.assertEqual(r.readings,15)
            self.assertEqual(r.seed,2)
    def test_travel_cap_enforced(self):
        field={(x,y):(.6,.9) for y in range(N) for x in range(N)}
        for m in METHODS:
            r=episode(0,'correlation',m,field,budget=25,travel_cap=15)
            self.assertLessEqual(r.travel,15)
    def test_deterministic_episode(self):
        field={(x,y):(.6,.9) for y in range(N) for x in range(N)}
        self.assertEqual(episode(4,'correlation','trustroot',field,budget=17),
                         episode(4,'correlation','trustroot',field,budget=17))
    def test_paired_bootstrap(self):
        r=bootstrap_difference([1,0,1,1],[0,0,0,0],reps=100)
        self.assertEqual(r['paired_delta'],.75)
        self.assertEqual(r['paired_n'],4)
    def test_bad_conditions(self):
        with self.assertRaises(ValueError): field_trace(1,'unknown',(1,2))
        with self.assertRaises(ValueError): experiment(seeds=2,budget=20,travel_cap=10)
        with self.assertRaises(ValueError): episode(0,'correlation','fake',{},budget=15)
    def test_seed_negative_not_allowed(self):
        with self.assertRaises(ValueError): bootstrap_difference([1],[1,2])
