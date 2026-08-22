// ======================================================================
// \title  DfaMonitor.hpp
// \author Ogma + Struktura
// \brief  hpp file for DfaMonitor component implementation class
//
// DFA structural health monitor for F Prime.
// Algorithm: Peng et al., Physical Review E 49(2), 1994.
// Library: https://github.com/koscak-labs/struktura
// ======================================================================

#ifndef DfaMonitor_HPP
#define DfaMonitor_HPP

#include "Ref/DfaMonitor/DfaMonitorComponentAc.hpp"

#ifndef DFA_WINDOW_SIZE
#define DFA_WINDOW_SIZE 256
#endif

#ifndef DFA_THRESHOLD
#define DFA_THRESHOLD 0.08
#endif

#ifndef DFA_LEARN_WINDOWS
#define DFA_LEARN_WINDOWS 10
#endif

#ifndef DFA_R2_MIN
#define DFA_R2_MIN 0.7
#endif

namespace Ref {

  struct DfaChannel {
      double buffer[DFA_WINDOW_SIZE];
      U32 pos;
      U32 filled;
      double baseline_alpha;
      U8 baseline_set;
      U32 window_count;
  };

  class DfaMonitor :
    public DfaMonitorComponentBase
  {

    public:

      // ----------------------------------------------------------------------
      // Construction, initialization, and destruction
      // ----------------------------------------------------------------------

      DfaMonitor(
          const char *const compName
      );

      void init(
          const NATIVE_INT_TYPE queueDepth,
          const NATIVE_INT_TYPE instance = 0
      );

      ~DfaMonitor();

    PRIVATE:

      // ----------------------------------------------------------------------
      // Handler implementations for user-defined typed input ports
      // ----------------------------------------------------------------------

{{#variables}}
      void {{varDeclName}}In_handler(
            const NATIVE_INT_TYPE portNum,
            {{varDeclType}} value
        );

{{/variables}}
    PRIVATE:

      // ----------------------------------------------------------------------
      // Command handler implementations
      // ----------------------------------------------------------------------

      void CHECK_MONITORS_cmdHandler(
          const FwOpcodeType opCode,
          const U32 cmdSeq
      );

      // ----------------------------------------------------------------------
      // DFA state
      // ----------------------------------------------------------------------

{{#variables}}
      DfaChannel m_ch_{{varDeclName}};
{{/variables}}
      U32 m_baselineCount;
      U32 m_shiftCount;

      void pushSample(DfaChannel &ch, double value, const char *name);

    };

} // end namespace Ref

#endif
