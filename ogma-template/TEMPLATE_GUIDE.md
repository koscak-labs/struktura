# How to Build a Custom Ogma Template

This guide documents how to create a custom monitoring template for
[nasa/ogma](https://github.com/nasa/ogma) that generates cFS, F Prime,
or ROS applications without writing Haskell. It uses the DFA structural
health monitor as a worked example.

Re: [nasa/ogma discussion #315](https://github.com/nasa/ogma/discussions/315)

---

## 1. Understand the Template System

Ogma uses [Mustache](https://mustache.github.io/) templates. When you run
`ogma cfs --template-dir my_template/ ...`, ogma:

1. Parses your `--variable-db` (db.json) to discover input variables
2. Parses your `--input-file` to extract expressions/properties
3. Serializes everything into a JSON context
4. Expands Mustache tags in every file under `--template-dir`
5. Writes the expanded files to `--target-dir`

Both file **contents** and file **paths** are Mustache templates.

## 2. Available Mustache Variables by Backend

### cFS Backend (`ogma cfs`)

From `ogma-core/src/Command/CFSApp.hs`:

| Section | Fields | Description |
|---------|--------|-------------|
| `{{#variables}}` | `varDeclName`, `varDeclType` | One block per input variable |
| `{{#msgIds}}` | (value is the MID string) | Message IDs to subscribe to |
| `{{#msgCases}}` | `msgInfoId`, `msgInfoDesc`, `msgInfoExtra` | Switch cases for message dispatch |
| `{{#msgHandlers}}` | `msgDataDesc`, `msgDataFromType`, `msgDataFromField`, `msgDataVarName`, `msgDataVarType`, `msgDataActive` | Handler functions per message type |
| `{{#triggers}}` | `triggerName`, `triggerType`, `triggerMsgType` | Copilot violation handlers |
| `{{#copilot}}` | `copilot.specName` | Copilot specification name |
| `{{#impl_extra_header}}` | (string value) | Extra `#include` lines (from `--template-vars`) |
| `{{#included_libraries}}` | (path strings) | Additional include directories |

### F Prime Backend (`ogma fprime`)

From `ogma-core/src/Command/FPrimeApp.hs`:

| Section | Fields | Description |
|---------|--------|-------------|
| `{{#variables}}` | `varDeclName`, `varDeclType`, `varDeclFPrimeType` | One block per variable |
| `{{#monitors}}` | `monitorName`, `monitorUC`, `monitorType`, `monitorPortType` | Monitor declarations |
| `{{#copilot}}` | `copilot.specName` | Copilot specification name |

### ROS Backend (`ogma ros`)

From `ogma-core/src/Command/ROSApp.hs`:

| Section | Fields | Description |
|---------|--------|-------------|
| `{{#variables}}` | `varDeclName`, `varDeclType`, `varDeclId`, `varDeclMsgType`, `varDeclMsgField`, `varDeclRandom` | One block per variable |
| `{{#monitors}}` | `monitorName`, `monitorType`, `monitorMsgType` | Monitor declarations |
| `{{#testingVariables}}` | (same as variables, filtered) | For test node generation |
| `{{#copilot}}` | `copilot.specName` | Copilot specification name |

All backends also expand any key/value pairs from `--template-vars` (extra-vars.json).

## 3. The db.json Format

The variable database tells ogma where each input variable comes from:

```json
{ "inputs":
    [ { "name": "accel_x"
      , "type": "double"
      , "active": true
      , "connections":
          [ { "scope": "cfs"
            , "topic": "SENSOR_ACCEL_MID"
            , "field": "x"
            }
          ]
      }
    ]
, "topics":
    [ { "scope": "cfs"
      , "topic": "SENSOR_ACCEL_MID"
      , "type":  "sensor_accel_msg_t"
      }
    ]
, "types":
    [ { "fromScope": "cfs"
      , "fromType":  "sensor_accel_msg_t"
      , "fromField": "x"
      , "toScope":   "C"
      , "toType":    "double"
      }
    ]
}
```

The three sections must be consistent: input connections reference topics,
topics reference types, types map framework-specific structs to C types.

## 4. Replacing Copilot with Custom Logic

The default cFS template calls `copilot_step()` whenever an active
variable arrives. To use custom monitoring logic instead:

1. Remove references to `copilot.specName`, `copilot.h`, `Properties.hs`
2. Replace `copilot_step()` with your own function call
3. Replace `{{#triggers}}` violation handlers with your own event reporting
4. Keep `{{#variables}}`, `{{#msgIds}}`, `{{#msgCases}}`, `{{#msgHandlers}}`
   for the subscription/dispatch infrastructure

### Example: DFA instead of Copilot

In the default template, the handler calls:
```c
copilot_step();  // evaluate all Copilot monitors
```

In our DFA template, the handler calls:
```c
dfa_push(&dfa_ch_{{msgDataVarName}}, (double){{msgDataVarName}}, "{{msgDataVarName}}");
```

The DFA function maintains a per-channel sliding window, learns a baseline
alpha, and emits events when the scaling exponent shifts.

## 5. The extra-vars.json File

Custom template variables that ogma expands but doesn't interpret:

```json
{ "impl_extra_header": "#include \"sensor_defs.h\""
, "CFS_CMD_MID":       "0x18A0"
, "DFA_WINDOW_SIZE":   "256"
, "DFA_THRESHOLD":     "0.08"
}
```

These appear as `{{{CFS_CMD_MID}}}` (triple-brace = no HTML escaping)
in template files.

## 6. Testing the Generated Output

Without a full cFS/fprime installation, you can verify:

1. The generated `.c` files parse correctly (syntax check)
2. `dfa_core.h` compiles standalone: `gcc -Wall -Werror -O2 -lm -c dfa_core.h`
3. MID definitions are consistent across headers and source
4. All variables from db.json appear in the generated code

With a cFS installation:
```sh
mv generated_app/ cfs/apps/dfa_monitor
# Add to sample_defs/targets.cmake
make native_std.prep && make native_std.install
```

## 7. Alternative: struktura generate

For DFA monitoring specifically, Struktura can generate the same output
without ogma installed:

```sh
cargo install struktura
struktura generate --cfs --db channels.json -o dfa_monitor/
struktura generate --fprime --db channels.json -o dfa_component/
```

This reads ogma's db.json format and produces complete, ready-to-compile
applications. No Haskell required.

---

*Generated from the [Struktura](https://github.com/koscak-labs/struktura) DFA template project.*
