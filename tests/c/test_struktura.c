/* test_struktura.c -- verify C FFI works */
#include <stdio.h>
#include <math.h>
#include "../../include/struktura.h"

int main(void) {
    /* Generate a simple signal: sin wave */
    double data[256];
    int i;
    for (i = 0; i < 256; i++) {
        data[i] = sin((double)i * 0.1);
    }

    /* Test DFA */
    struktura_dfa_result_t dfa = struktura_dfa(data, 256);
    printf("DFA: alpha=%.3f R2=%.4f\n", dfa.alpha, dfa.r_squared);

    /* Test analyze */
    struktura_law_t law = struktura_analyze(data, 256);
    printf("Analyze: alpha=%.3f hurst=%.3f quality=%d n=%u\n",
           law.dfa_alpha, law.hurst, law.quality, law.n);

    /* Test health check */
    uint8_t verdict = struktura_health_check(0.2, 0.6);
    printf("Health: verdict=%d (expected CRITICAL=3)\n", verdict);

    /* Verify */
    int pass = 0;
    if (dfa.r_squared >= 0.0) { printf("[PASS] DFA R2 valid\n"); pass++; }
    if (law.n == 256) { printf("[PASS] N correct\n"); pass++; }
    if (verdict == STRUKTURA_CRITICAL) { printf("[PASS] Verdict CRITICAL\n"); pass++; }

    printf("\n%d/3 passed\n", pass);
    return (pass == 3) ? 0 : 1;
}
