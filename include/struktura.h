/* struktura.h -- C interface for Struktura DFA anomaly detection
 *
 * Link with: -lstruktura (static) or struktura.dll (dynamic)
 * Zero dependencies. Pre-80s discipline: fixed-size types, no allocator.
 *
 * https://github.com/koscak-labs/struktura
 */

#ifndef STRUKTURA_H
#define STRUKTURA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quality levels (matches LawQuality enum) */
#define STRUKTURA_EXACT        0
#define STRUKTURA_STRONG       1
#define STRUKTURA_GOOD         2
#define STRUKTURA_APPROX       3
#define STRUKTURA_ABSTAIN      4
#define STRUKTURA_INSUFFICIENT 5

/* Health verdicts */
#define STRUKTURA_HEALTHY  0
#define STRUKTURA_WATCH    1
#define STRUKTURA_WARNING  2
#define STRUKTURA_CRITICAL 3

typedef struct {
    double alpha;
    double r_squared;
} struktura_dfa_result_t;

typedef struct {
    double dfa_alpha;
    double dfa_r2;
    double acr_alpha;
    double acr_r2;
    double hurst;
    double mean;
    double std_dev;
    double kurtosis;
    uint32_t n;
    uint8_t quality;
} struktura_law_t;

/* Compute DFA scaling exponent. Needs >= 64 samples. */
struktura_dfa_result_t struktura_dfa(const double *data, uint32_t len);

/* Full structural analysis. Needs >= 20 samples. */
struktura_law_t struktura_analyze(const double *data, uint32_t len);

/* Compare current alpha against baseline. Returns verdict (0-3). */
uint8_t struktura_health_check(double current_alpha, double baseline_alpha);

#ifdef __cplusplus
}
#endif

#endif /* STRUKTURA_H */
