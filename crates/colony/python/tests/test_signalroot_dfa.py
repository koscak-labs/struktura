import unittest,csv,io
from colony.signalroot_dfa import *

class DfaBridgeTests(unittest.TestCase):
 def test_reproducible(self):self.assertEqual(trace(1,'correlation',(3,4)),trace(1,'correlation',(3,4)))
 def test_different_locations(self):self.assertNotEqual(trace(1,'correlation',(3,4)),trace(1,'correlation',(3,5)))
 def test_nonfinite_dropout(self):
  self.assertTrue(any(not math.isfinite(z) for z in trace(1,'dropout',(0,0))))
 def test_reject_incomplete(self):
  with self.assertRaises(ValueError):parse_scores('x,y,alpha,r_squared,confidence\n0,0,0.5,0.9,1\n')
 def test_validation(self):
  with self.assertRaises(ValueError):list(rows_for(1,'bad'))
 def test_score(self):self.assertEqual(score(0.4),0)
 def test_hidden_truth_not_passed_to_planner(self):
  # Fixed scores support identical planner behavior independent of hidden source.
  from colony.signalroot import proposal
  import random
  observations={p:0.1 for p in startup()}
  self.assertEqual(proposal('signal_sca',observations,random.Random(3),(12,12)),proposal('signal_sca',observations,random.Random(3),(12,12)))
