# Register the standard build
set(SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/dfa_core.h"
  "${CMAKE_CURRENT_LIST_DIR}/DfaMonitor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/DfaMonitor.fpp"
)
register_fprime_module()
