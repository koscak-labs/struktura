# Proposal: Distinct Error Codes for ogma-core

## Current state

25 error code constants across 10 modules. **24 of 25 are `= 1`.** Only `ecDiagramTemplateError = 2`.
Scripts and CI cannot distinguish "file not found" from "parse error" from "template expansion failed."

## All error codes (current)

| Code | Name | Module | Condition |
|------|------|--------|-----------|
| 1 | `ecWrongArguments` | Common.hs | Invalid CLI arguments |
| 1 | `ecCannotOpenDBFile` | Common.hs | Variable DB file missing/unreadable |
| 1 | `ecCannotOpenVarFile` | Common.hs | Variable file missing/unreadable |
| 1 | `ecCannotOpenHandlersFile` | Common.hs | Handlers file missing/unreadable |
| 1 | `ecCannotOpenTemplateVarsFile` | Common.hs | Template vars file missing/unreadable |
| 1 | `ecCannotReadObjectTemplateVarsFile` | Common.hs | Template vars JSON parse failure |
| 1 | `ecCannotCopyTemplate` | Common.hs | Template expansion/copy failed |
| 1 | `ecCannotOpenInputFile` | Spec/Parser.hs | Input spec file missing/unreadable |
| 1 | `ecIncorrectFormatFile` | Spec/Parser.hs | Format config file parse failure |
| 1 | `ecCannotReadConditionExpr` | Spec/Parser.hs | Condition expression parse failure |
| 1 | `ecCannotReadDiagram` | Diagram/Parser.hs | Diagram file parse failure |
| 1 | `ecCannotMergeVariableDB` | VariableDB.hs | Variable DB merge conflict |
| 1 | `ecMissingSpec` | Standalone.hs | No input specification provided |
| 1 | `ecMultipleInputTypes` | Standalone.hs | Mixed spec + diagram inputs |
| 1 | `ecIncorrectSpec` | Standalone.hs | Spec formalization failed |
| 1 | `ecMultipleInputTypes` | CFSApp.hs | Mixed spec + diagram inputs |
| 1 | `ecWrongDiagramMode` | CFSApp.hs | Invalid diagram execution mode |
| 1 | `ecMultipleInputTypes` | FPrimeApp.hs | Mixed spec + diagram inputs |
| 1 | `ecMultipleInputTypes` | ROSApp.hs | Mixed spec + diagram inputs |
| 1 | `ecDiagramError` | Diagram.hs | Diagram processing failed |
| **2** | `ecDiagramTemplateError` | Diagram.hs | Diagram template expansion failed |
| 1 | `ecCStructError` | CStructs2Copilot.hs | C struct parse failure |
| 1 | `ecCStructError` | CStructs2MsgHandlers.hs | C struct parse failure |
| 1 | `ecSearchError` | Search.hs | Search operation failed |
| 1 | `ecOverviewError` | Overview.hs | Overview generation failed |
| 1 | `ecCannotAnalyzeError` | Report.hs | Report analysis failed |

## Proposed numbering

Group by error category. Leaves room for growth within each range.

| Code | Name | Category |
|------|------|----------|
| **10** | `ecWrongArguments` | Arguments |
| **11** | `ecMissingSpec` | Arguments |
| **12** | `ecMultipleInputTypes` | Arguments |
| **13** | `ecWrongDiagramMode` | Arguments |
| **20** | `ecCannotOpenInputFile` | File I/O |
| **21** | `ecCannotOpenDBFile` | File I/O |
| **22** | `ecCannotOpenVarFile` | File I/O |
| **23** | `ecCannotOpenHandlersFile` | File I/O |
| **24** | `ecCannotOpenTemplateVarsFile` | File I/O |
| **30** | `ecIncorrectFormatFile` | Parse |
| **31** | `ecCannotReadConditionExpr` | Parse |
| **32** | `ecIncorrectSpec` | Parse |
| **33** | `ecCannotReadDiagram` | Parse |
| **34** | `ecCStructError` | Parse |
| **35** | `ecCannotReadObjectTemplateVarsFile` | Parse |
| **36** | `ecCannotMergeVariableDB` | Parse / merge |
| **40** | `ecCannotCopyTemplate` | Generation |
| **41** | `ecDiagramError` | Generation |
| **42** | `ecDiagramTemplateError` | Generation |
| **50** | `ecSearchError` | Analysis |
| **51** | `ecOverviewError` | Analysis |
| **52** | `ecCannotAnalyzeError` | Analysis |

### Notes

- `ecMultipleInputTypes` is defined identically in 4 modules (Standalone, CFSApp, FPrimeApp, ROSApp). Should be consolidated into Common.hs.
- `ecCStructError` is defined identically in 2 modules (CStructs2Copilot, CStructs2MsgHandlers). Same consolidation opportunity.
- Ranges 10-19 arguments, 20-29 file I/O, 30-39 parse, 40-49 generation, 50-59 analysis — each has room for 10 future codes.
- `ecDiagramTemplateError = 2` is the only existing non-1 code; proposed `= 42` keeps it distinct but in the generation range.
