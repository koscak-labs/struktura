#ifndef _dfa_monitor_app_h_
#define _dfa_monitor_app_h_

#include "cfe.h"
#include "cfe_error.h"
#include "cfe_evs.h"
#include "cfe_sb.h"
#include "cfe_es.h"

#include <string.h>
#include <math.h>

#define DFA_MONITOR_PIPE_DEPTH 32

void DFA_MONITOR_AppMain(void);
CFE_Status_t DFA_MONITOR_AppInit(void);
void DFA_MONITOR_ProcessCommandPacket(void);
{{#msgCases}}
void DFA_MONITOR_Process{{msgInfoDesc}}(void);
{{/msgCases}}

#endif /* _dfa_monitor_app_h_ */
