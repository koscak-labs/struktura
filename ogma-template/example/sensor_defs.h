/* sensor_defs.h -- USER-PROVIDED header for the Ogma-generated DFA monitor.
 * The template pulls this in via extra-vars `impl_extra_header` (TEMPLATE_GUIDE.md section 133).
 * It must define the message IDs and payload types of the telemetry the monitor subscribes to.
 * This is the minimal definition used for the 2026-09-17 cFS bundle build proof.
 */
#ifndef SENSOR_DEFS_H
#define SENSOR_DEFS_H

#include "cfe.h"

#define SENSOR_ACCEL_MID 0x08B0
#define SENSOR_GYRO_MID  0x08B1
#define BATT_TLM_MID     0x08B2

typedef struct {
    CFE_MSG_TelemetryHeader_t TlmHeader;
    float x;
    float y;
    float z;
} sensor_accel_msg_t;

typedef struct {
    CFE_MSG_TelemetryHeader_t TlmHeader;
    float x;
    float y;
    float z;
} sensor_gyro_msg_t;

typedef struct {
    CFE_MSG_TelemetryHeader_t TlmHeader;
    float voltage;
    float current;
} batt_tlm_msg_t;

#endif /* SENSOR_DEFS_H */
