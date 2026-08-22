-- | DFA structural health monitoring expressed as a Copilot specification.
-- This shows how DFA integrates INTO the Copilot ecosystem as a new primitive,
-- not as a parallel tool.
--
-- The DFA computation runs in C (extern). The monitoring logic — baseline
-- comparison, shift detection, exact-or-abstain gating — is expressed in
-- Copilot's temporal stream DSL. Ogma generates both the C code and the
-- Copilot glue.
--
-- Example: monitor a sensor channel for structural degradation.
--
-- Usage with ogma:
--   ogma cfs --input-file copilot_dfa_spec.hs --variable-db channels.json
--
-- Reference: https://github.com/koscak-labs/struktura

module CopilotDFA where

import Language.Copilot
import Copilot.Compile.C99

-- External C functions (provided by dfa_core.h / libstruktura)
--
-- These are the bridge: Copilot calls the DFA computation as an external
-- C function, then reasons about the result using temporal logic.
dfaAlpha :: Stream Double -> Stream Double
dfaAlpha channel = extern "dfa_compute_alpha" [channel]

dfaR2 :: Stream Double -> Stream Double
dfaR2 channel = extern "dfa_compute_r2" [channel]

-- | Baseline alpha learned during healthy operation.
-- In practice this would be set by the learning phase.
-- For this spec, it's a parameter.
baselineAlpha :: Stream Double
baselineAlpha = constant 0.875  -- e.g., from Voyager 1 healthy period

-- | Shift threshold — how much alpha can deviate before alerting.
shiftThreshold :: Stream Double
shiftThreshold = constant 0.08

-- | R² quality gate — below this, the measurement is unreliable.
r2Minimum :: Stream Double
r2Minimum = constant 0.7

-- | The core monitor: fires when structural health degrades.
--
-- This is the key insight: the DETECTION LOGIC is in Copilot (temporal,
-- formally verifiable), while the COMPUTATION is in C (efficient, proven).
--
-- Copilot's guarantees (constant memory, constant time, deterministic)
-- apply to the monitoring logic. DFA's guarantees (convergence, bounded
-- error, exact-or-abstain) apply to the computation.
structuralShiftDetected :: Stream Double -> Stream Bool
structuralShiftDetected channel =
  let alpha = dfaAlpha channel
      r2    = dfaR2 channel
      shift = abs (alpha - baselineAlpha)
      reliable = r2 > r2Minimum           -- exact-or-abstain gate
      shifted  = shift > shiftThreshold   -- structural change detected
  in reliable && shifted

-- | The Copilot specification.
spec :: Spec
spec = do
  -- Sensor channel (populated by ogma from db.json subscription)
  let sensorAccel = extern "accel_x" :: Stream Double

  -- Trigger: fire when structural shift detected on reliable data
  trigger "handler_structuralShift"
    (structuralShiftDetected sensorAccel)
    [ arg (dfaAlpha sensorAccel)    -- current alpha
    , arg (dfaR2 sensorAccel)       -- current R²
    , arg baselineAlpha             -- baseline for comparison
    ]

-- | Compile to C99 (ogma handles the framework glue)
main :: IO ()
main = reify spec >>= compile "copilot_dfa"

-- WHAT THIS MEANS FOR OGMA:
--
-- 1. DFA becomes a Copilot EXTERN — the computation happens in C,
--    the monitoring logic is in Copilot's verified DSL.
--
-- 2. Ogma generates the glue: subscribe to cFS/ROS/fprime topics,
--    feed samples to the DFA extern, check the Copilot triggers.
--
-- 3. The generated C code gets BOTH:
--    - Copilot's bisimulation guarantee (ICFP 2023) on the monitor logic
--    - DFA's convergence guarantee (Peng 1994) on the computation
--
-- 4. No changes to Copilot or ogma core needed — just an extern + dfa_core.h
--
-- This is the integration path: not a new ogma backend, but a new
-- Copilot PRIMITIVE that ogma already knows how to generate around.
