#ifndef _dfa_monitor_cfs_msg_h_
#define _dfa_monitor_cfs_msg_h_

typedef struct {
    CFE_MSG_TelemetryHeader_t TelemetryHeader;
    uint32 dfa_monitor_command_count;
    uint32 dfa_monitor_command_error_count;
} dfa_monitor_hk_tlm_t;

#define DFA_MONITOR_HK_TLM_LNGTH sizeof(dfa_monitor_hk_tlm_t)

#endif /* _dfa_monitor_cfs_msg_h_ */
