/* test_dfa_core.c -- Verify dfa_core.h compiles and produces correct results.
 * Compile: gcc -Wall -Werror -O2 -lm -o test_dfa_core test_dfa_core.c
 * Run: ./test_dfa_core
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../cfs/dfa_monitor/fsw/src/dfa_core.h"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(const char *name, int condition) {
    if (condition) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("FAIL: %s\n", name);
    }
}

static void test_white_noise(void) {
    double data[256];
    unsigned int seed = 42;
    int i;
    for (i = 0; i < 256; i++) {
        seed = seed * 1103515245 + 12345;
        data[i] = (double)((seed >> 16) & 0x7FFF) / 32768.0;
    }
    dfa_result_t r = dfa_compute(data, 256);
    check("white_noise alpha near 0.5", fabs(r.alpha - 0.5) < 0.25);
    check("white_noise R2 > 0", r.r_squared > 0.0);
}

static void test_correlated_signal(void) {
    double data[256];
    int i;
    data[0] = 0.0;
    unsigned int seed = 7;
    for (i = 1; i < 256; i++) {
        seed = seed * 1103515245 + 12345;
        double noise = ((double)((seed >> 16) & 0x7FFF) / 32768.0 - 0.5) * 0.1;
        data[i] = data[i-1] * 0.95 + noise;
    }
    dfa_result_t r = dfa_compute(data, 256);
    check("correlated alpha > 0.5", r.alpha > 0.5);
    check("correlated R2 reasonable", r.r_squared > 0.3);
}

static void test_too_short(void) {
    double data[32];
    int i;
    for (i = 0; i < 32; i++) data[i] = (double)i;
    dfa_result_t r = dfa_compute(data, 32);
    check("short signal alpha = 0.5", fabs(r.alpha - 0.5) < 0.001);
    check("short signal R2 = 0.0", fabs(r.r_squared) < 0.001);
}

static void test_constant_signal(void) {
    double data[128];
    int i;
    for (i = 0; i < 128; i++) data[i] = 42.0;
    dfa_result_t r = dfa_compute(data, 128);
    check("constant alpha = 0.5 (no structure)", fabs(r.alpha - 0.5) < 0.001);
}

static void test_deterministic(void) {
    double data[256];
    int i;
    unsigned int seed = 99;
    for (i = 0; i < 256; i++) {
        seed = seed * 1103515245 + 12345;
        data[i] = (double)((seed >> 16) & 0x7FFF) / 32768.0;
    }
    dfa_result_t r1 = dfa_compute(data, 256);
    dfa_result_t r2 = dfa_compute(data, 256);
    check("deterministic: same alpha", r1.alpha == r2.alpha);
    check("deterministic: same R2", r1.r_squared == r2.r_squared);
}

int main(void) {
    printf("dfa_core.h test suite\n");
    printf("====================\n\n");

    test_white_noise();
    test_correlated_signal();
    test_too_short();
    test_constant_signal();
    test_deterministic();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
