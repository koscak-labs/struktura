-- | Unit tests for the cFS, F Prime, and ROS application generation backends.
--
-- These three major backends currently have ZERO unit tests in ogma-core.
-- The existing tests only cover the standalone backend and C struct parsing.
--
-- These tests follow the exact pattern from ogma-core/tests/Main.hs:
--   1. Build a CommandOptions with the example files
--   2. Call the command function
--   3. Assert isSuccess on the result
--
-- To integrate into ogma-core's test suite, add these test cases to the
-- `tests` list in ogma-core/tests/Main.hs and add the necessary imports:
--
--   import qualified Command.CFSApp
--   import qualified Command.FPrimeApp
--   import qualified Command.ROSApp
--
-- The test data referenced lives in ogma-cli/examples/.
--
-- This file was written as a contribution proposal for nasa/ogma.
-- See: https://github.com/koscak-labs/struktura

module Main where

import Data.Monoid                    ( mempty )
import Test.Framework                 ( Test, defaultMainWithOpts )
import Test.Framework.Providers.HUnit ( testCase )
import Test.HUnit                     ( assertBool )
import System.Directory               ( getTemporaryDirectory )

-- Internal imports
import           Command.Result    ( isSuccess )
import qualified Command.CFSApp
import qualified Command.FPrimeApp
import qualified Command.ROSApp

-- | Run all backend unit tests.
main :: IO ()
main = defaultMainWithOpts tests mempty

-- | All backend tests.
tests :: [Test.Framework.Test]
tests =
  [
    -- cFS backend tests
    testCase "cfs-cmd-hello-ogma-ok"
      (testCFSApp
         "ogma-cli/examples/cfs-001-hello-ogma/expressions.json"
         "ogma-cli/examples/cfs-001-hello-ogma/json-format.cfg"
         (Just "ogma-cli/examples/cfs-001-hello-ogma/db.json")
         (Just "ogma-cli/examples/cfs-001-hello-ogma/extra-vars.json")
         True
      )
    -- Should pass: valid expressions, variable DB, and extra vars

  , testCase "cfs-cmd-no-input-ok"
      (testCFSAppMinimal True)
    -- Should pass: no input file, just variable DB

  , testCase "cfs-cmd-bad-input-file"
      (testCFSApp
         "nonexistent-file.json"
         "ogma-cli/examples/cfs-001-hello-ogma/json-format.cfg"
         Nothing
         Nothing
         False
      )
    -- Should fail: input file does not exist

    -- F Prime backend tests
  , testCase "fprime-cmd-fcs-ok"
      (testFPrimeApp
         "ogma-cli/examples/fcs-1.json"
         (Just "ogma-cli/examples/cfs-001-hello-ogma/db.json")
         True
      )
    -- Should pass: valid FCS input, variable DB available

  , testCase "fprime-cmd-bad-input-file"
      (testFPrimeApp
         "nonexistent-file.json"
         Nothing
         False
      )
    -- Should fail: input file does not exist

    -- ROS backend tests
  , testCase "ros-cmd-copilot-example-ok"
      (testROSApp
         "ogma-cli/examples/ros-copilot/document.json"
         "ogma-cli/examples/ros-copilot/json-format.cfg"
         (Just "ogma-cli/examples/ros-copilot/vars-db.json")
         True
      )
    -- Should pass: valid ROS Copilot example
  ]

-- | Test CFS application generation with full options.
--
-- This test mirrors testStandaloneFCS from ogma-core/tests/Main.hs but
-- targets the cFS backend instead.
--
-- The commandFormat field is either a built-in format name (e.g. "fcs")
-- resolved from ogma-core's data directory, or a file path to a JSON
-- format configuration (e.g. "ogma-cli/examples/.../json-format.cfg").
-- See Data.Spec.Parser.readInputFile for resolution logic.
testCFSApp :: FilePath      -- ^ Input specification file
           -> FilePath      -- ^ Input format (name or file path)
           -> Maybe FilePath -- ^ Variable database (db.json)
           -> Maybe FilePath -- ^ Extra template variables
           -> Bool          -- ^ Expected success?
           -> IO ()
testCFSApp inputFile inputFormat varDB extraVars success = do
    targetDir <- getTemporaryDirectory
    let opts = Command.CFSApp.CommandOptions
                 { Command.CFSApp.commandConditionExpr = Nothing
                 , Command.CFSApp.commandInputFiles    = [ inputFile ]
                 , Command.CFSApp.commandTargetDir     = targetDir
                 , Command.CFSApp.commandTemplateDir   = Nothing
                 , Command.CFSApp.commandVariables     = Nothing
                 , Command.CFSApp.commandVariableDB    = varDB
                 , Command.CFSApp.commandHandlers      = Nothing
                 , Command.CFSApp.commandFormat        = inputFormat
                 , Command.CFSApp.commandPropFormat    = "literal"
                 , Command.CFSApp.commandPropVia       = Nothing
                 , Command.CFSApp.commandDiagramMode   = "calculate"
                 , Command.CFSApp.commandExtraVars     = extraVars
                 }
    result <- Command.CFSApp.command opts

    let testPass = success == isSuccess result

    assertBool errorMsg testPass
  where
    errorMsg = "The result of CFS app generation from "
               ++ inputFile ++ " was unexpected."

-- | Test CFS app with minimal options (no input files).
testCFSAppMinimal :: Bool -> IO ()
testCFSAppMinimal success = do
    targetDir <- getTemporaryDirectory
    let opts = Command.CFSApp.CommandOptions
                 { Command.CFSApp.commandConditionExpr = Nothing
                 , Command.CFSApp.commandInputFiles    = []
                 , Command.CFSApp.commandTargetDir     = targetDir
                 , Command.CFSApp.commandTemplateDir   = Nothing
                 , Command.CFSApp.commandVariables     = Nothing
                 , Command.CFSApp.commandVariableDB    = Nothing
                 , Command.CFSApp.commandHandlers      = Nothing
                 , Command.CFSApp.commandFormat        = "fcs"
                 , Command.CFSApp.commandPropFormat    = "smv"
                 , Command.CFSApp.commandPropVia       = Nothing
                 , Command.CFSApp.commandDiagramMode   = "calculate"
                 , Command.CFSApp.commandExtraVars     = Nothing
                 }
    result <- Command.CFSApp.command opts

    let testPass = success == isSuccess result

    assertBool "Minimal CFS app generation result was unexpected." testPass

-- | Test F Prime component generation.
--
-- Note: The F Prime CommandOptions does not have a commandDiagramMode field.
testFPrimeApp :: FilePath       -- ^ Input specification file
              -> Maybe FilePath -- ^ Variable database
              -> Bool           -- ^ Expected success?
              -> IO ()
testFPrimeApp inputFile varDB success = do
    targetDir <- getTemporaryDirectory
    let opts = Command.FPrimeApp.CommandOptions
                 { Command.FPrimeApp.commandConditionExpr = Nothing
                 , Command.FPrimeApp.commandInputFiles    = [ inputFile ]
                 , Command.FPrimeApp.commandTargetDir     = targetDir
                 , Command.FPrimeApp.commandTemplateDir   = Nothing
                 , Command.FPrimeApp.commandVariables     = Nothing
                 , Command.FPrimeApp.commandVariableDB    = varDB
                 , Command.FPrimeApp.commandHandlers      = Nothing
                 , Command.FPrimeApp.commandFormat        = "fcs"
                 , Command.FPrimeApp.commandPropFormat    = "smv"
                 , Command.FPrimeApp.commandPropVia       = Nothing
                 , Command.FPrimeApp.commandExtraVars     = Nothing
                 }
    result <- Command.FPrimeApp.command opts

    let testPass = success == isSuccess result

    assertBool errorMsg testPass
  where
    errorMsg = "The result of F Prime component generation from "
               ++ inputFile ++ " was unexpected."

-- | Test ROS application generation.
--
-- Note: ROSApp CommandOptions has additional fields for testing apps/vars.
testROSApp :: FilePath       -- ^ Input specification file
           -> FilePath       -- ^ Input format configuration
           -> Maybe FilePath -- ^ Variable database
           -> Bool           -- ^ Expected success?
           -> IO ()
testROSApp inputFile inputFormat varDB success = do
    targetDir <- getTemporaryDirectory
    let opts = Command.ROSApp.CommandOptions
                 { Command.ROSApp.commandConditionExpr = Nothing
                 , Command.ROSApp.commandInputFiles    = [ inputFile ]
                 , Command.ROSApp.commandTargetDir     = targetDir
                 , Command.ROSApp.commandTemplateDir   = Nothing
                 , Command.ROSApp.commandVariables     = Nothing
                 , Command.ROSApp.commandVariableDB    = varDB
                 , Command.ROSApp.commandHandlers      = Nothing
                 , Command.ROSApp.commandFormat        = inputFormat
                 , Command.ROSApp.commandPropFormat    = "smv"
                 , Command.ROSApp.commandPropVia       = Nothing
                 , Command.ROSApp.commandExtraVars     = Nothing
                 , Command.ROSApp.commandTestingApps   = []
                 , Command.ROSApp.commandTestingVars   = []
                 }
    result <- Command.ROSApp.command opts

    let testPass = success == isSuccess result

    assertBool errorMsg testPass
  where
    errorMsg = "The result of ROS app generation from "
               ++ inputFile ++ " was unexpected."
