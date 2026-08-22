// ======================================================================
// \title  DfaMonitor.cpp
// \author Ogma + Struktura
// \brief  cpp file for DfaMonitor component implementation class
//
// DFA structural health monitor for F Prime.
// Detects behavioral degradation via DFA scaling analysis.
// Zero external dependencies. Fixed buffers. No heap allocation.
//
// Algorithm: Peng et al., Physical Review E 49(2), 1994.
// Library: https://github.com/koscak-labs/struktura
// ======================================================================

#include <Ref/DfaMonitor/DfaMonitor.hpp>
#include "Fw/Types/BasicTypes.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include "dfa_core.h"

#ifdef __cplusplus
}
#endif

#include <cstring>
#include <cmath>

namespace Ref {

  // ----------------------------------------------------------------------
  // Construction, initialization, and destruction
  // ----------------------------------------------------------------------

  DfaMonitor ::
    DfaMonitor(
        const char *const compName
    ) : DfaMonitorComponentBase(compName),
        m_baselineCount(0),
        m_shiftCount(0)
  {
{{#variables}}
    memset(&m_ch_{{varDeclName}}, 0, sizeof(DfaChannel));
{{/variables}}
  }

  void DfaMonitor ::
    init(
        const NATIVE_INT_TYPE queueDepth,
        const NATIVE_INT_TYPE instance
    )
  {
    DfaMonitorComponentBase::init(queueDepth, instance);
  }

  DfaMonitor ::
    ~DfaMonitor()
  {

  }

  // ----------------------------------------------------------------------
  // DFA channel push
  // ----------------------------------------------------------------------

  void DfaMonitor ::
    pushSample(DfaChannel &ch, double value, const char *name)
  {
    ch.buffer[ch.pos] = value;
    ch.pos = (ch.pos + 1) % DFA_WINDOW_SIZE;
    if (ch.pos == 0) {
      ch.filled = 1;
    }
    if (!ch.filled) {
      return;
    }

    ch.window_count++;
    dfa_result_t r = dfa_compute(ch.buffer, DFA_WINDOW_SIZE);

    /* Report latest alpha/R2 as telemetry */
    this->tlmWrite_DfaAlpha(r.alpha);
    this->tlmWrite_DfaR2(r.r_squared);

    /* Learning phase: establish baseline */
    if (!ch.baseline_set &&
        ch.window_count >= DFA_LEARN_WINDOWS &&
        r.r_squared > DFA_R2_MIN) {
      ch.baseline_alpha = r.alpha;
      ch.baseline_set = 1;
      m_baselineCount++;
      this->tlmWrite_BaselineCount(m_baselineCount);

      Fw::String chName(name);
      this->log_ACTIVITY_HI_BASELINE_ESTABLISHED(
          chName, r.alpha, r.r_squared);
      return;
    }
    if (!ch.baseline_set) {
      return;
    }

    /* Exact-or-abstain: skip if fit is poor */
    if (r.r_squared < DFA_R2_MIN) {
      Fw::String chName(name);
      this->log_ACTIVITY_LO_INSUFFICIENT_STRUCTURE(chName, r.r_squared);
      return;
    }

    /* Check for structural shift */
    double shift = fabs(r.alpha - ch.baseline_alpha);
    if (shift >= DFA_THRESHOLD) {
      m_shiftCount++;
      this->tlmWrite_ShiftCount(m_shiftCount);

      Fw::String chName(name);
      this->log_WARNING_HI_STRUCTURAL_SHIFT(
          chName,
          ch.baseline_alpha,
          r.alpha,
          r.alpha - ch.baseline_alpha);
    }
  }

  // ----------------------------------------------------------------------
  // Handler implementations for user-defined typed input ports
  // ----------------------------------------------------------------------

{{#variables}}
  void DfaMonitor ::
    {{varDeclName}}In_handler(
        const NATIVE_INT_TYPE portNum,
        {{varDeclType}} value
    )
  {
    pushSample(m_ch_{{varDeclName}}, static_cast<double>(value),
               "{{varDeclName}}");
  }

{{/variables}}
  // ----------------------------------------------------------------------
  // Command handler implementations
  // ----------------------------------------------------------------------

  void DfaMonitor ::
    CHECK_MONITORS_cmdHandler(
        const FwOpcodeType opCode,
        const U32 cmdSeq
    )
  {
    this->tlmWrite_BaselineCount(m_baselineCount);
    this->tlmWrite_ShiftCount(m_shiftCount);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
  }

} // end namespace Ref
