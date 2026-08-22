module Ref {

{{#variables}}
    port {{varDeclFPrimeType}}Value(value: {{varDeclFPrimeType}})

{{/variables}}
  @ DFA structural health monitor component
  @ Detects behavioral degradation via scaling analysis.
  @ Algorithm: Peng et al., Physical Review E 49(2), 1994.
  @ Library: https://github.com/koscak-labs/struktura
  queued component DfaMonitor {

    # ----------------------------------------------------------------------
    # Telemetry input ports (one per monitored variable)
    # ----------------------------------------------------------------------

{{#variables}}
    async input port {{varDeclName}}In : {{varDeclFPrimeType}}Value

{{/variables}}
    # ----------------------------------------------------------------------
    # Special ports
    # ----------------------------------------------------------------------

    @ Command receive
    command recv port cmdIn

    @ Command registration
    command reg port cmdRegOut

    @ Command response
    command resp port cmdResponseOut

    @ Event
    event port eventOut

    @ Parameter get
    param get port prmGetOut

    @ Parameter set
    param set port prmSetOut

    @ Telemetry
    telemetry port tlmOut

    @ Text event
    text event port textEventOut

    @ Time get
    time get port timeGetOut

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ Structural shift detected on a monitored channel
    event STRUCTURAL_SHIFT( \
      channel: string size 32, \
      baseline_alpha: F64, \
      current_alpha: F64, \
      delta: F64 \
    ) \
      severity warning high \
      id 0 \
      format "DFA structural shift on {}: baseline={.3f} current={.3f} delta={+.3f}"

    @ Baseline established after learning phase
    event BASELINE_ESTABLISHED( \
      channel: string size 32, \
      alpha: F64, \
      r_squared: F64 \
    ) \
      severity activity high \
      id 1 \
      format "DFA baseline for {}: alpha={.3f} R2={.4f}"

    @ Signal has insufficient structure for monitoring
    event INSUFFICIENT_STRUCTURE( \
      channel: string size 32, \
      r_squared: F64 \
    ) \
      severity activity low \
      id 2 \
      format "DFA {}: R2={.4f} below threshold, monitoring suspended"

    # ----------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------

    @ Run DFA analysis on all channels and check for structural shifts
    sync command CHECK_MONITORS()

    # ----------------------------------------------------------------------
    # Telemetry
    # ----------------------------------------------------------------------

    @ Current DFA scaling exponent (last computed)
    telemetry DfaAlpha: F64

    @ Current DFA fit quality (last computed)
    telemetry DfaR2: F64

    @ Number of channels with established baselines
    telemetry BaselineCount: U32

    @ Number of channels currently in structural shift
    telemetry ShiftCount: U32

  }

}
