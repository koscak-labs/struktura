/* bench/flight/harness.c
 * Equivalence + cost harness for the struktura-generated C99 hybrid
 * monitor (hybrid_monitor.c), run either natively (x86 gcc, full libc) or
 * bare-metal freestanding on an emulated Cortex-M3 (arm-none-eabi-gcc +
 * qemu-system-arm, -machine mps2-an385, ARM semihosting for output, no
 * newlib/libc). Builds a fresh hyb_monitor_t, replays the fixed
 * STREAM[][] from stream_data.h (Rust-generated, includes an injected
 * stuck-sensor fault), records the first-alarm tick + leg, per-tick DWT
 * cycle counts on ARM (wall-clock ns on native), and runs twice to check
 * determinism. No malloc anywhere; insertion sort avoids pulling in
 * qsort/libc on the bare-metal build.
 */
#if !defined(__arm__)
#define _POSIX_C_SOURCE 199309L
#endif
#include "hybrid_monitor.c"
#include "stream_data.h"

#if defined(__arm__)
/* ---- bare-metal ARM: no libc. ARM semihosting output + DWT cycles. ---- */
#define DWT_CTRL   (*(volatile unsigned long *)0xE0001000)
#define DWT_CYCCNT (*(volatile unsigned long *)0xE0001004)
#define DEMCR      (*(volatile unsigned long *)0xE000EDFC)

static void cyc_init(void) {
    DEMCR |= (1UL << 24);
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1UL;
}
static unsigned long cyc_now(void) { return DWT_CYCCNT; }

static void sh_write0(const char *s) {
    register unsigned long r0 __asm__("r0") = 0x04; /* SYS_WRITE0 */
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt 0xab" : : "r"(r0), "r"(r1) : "memory");
}
static void sh_exit(int code) {
    static unsigned long block[2];
    register unsigned long r0 __asm__("r0") = 0x18; /* SYS_EXIT */
    block[0] = 0x20026UL; /* ADP_Stopped_ApplicationExit */
    block[1] = (unsigned long)code;
    {
        register unsigned long *r1 __asm__("r1") = block;
        __asm__ volatile("bkpt 0xab" : : "r"(r0), "r"(r1) : "memory");
    }
    while (1) {}
}
static void put_ulong(unsigned long v) {
    char tmp[24];
    int i = 0, j;
    if (v == 0) { sh_write0("0"); return; }
    while (v) { tmp[i++] = (char)('0' + (v % 10)); v /= 10; }
    {
        char out[24];
        for (j = 0; i > 0; j++) out[j] = tmp[--i];
        out[j] = 0;
        sh_write0(out);
    }
}
static void put_long(long v) {
    if (v < 0) { sh_write0("-"); put_ulong((unsigned long)(-v)); }
    else put_ulong((unsigned long)v);
}
#define OUT_STR(s)   sh_write0(s)
#define OUT_ULONG(v) put_ulong(v)
#define OUT_LONG(v)  put_long((long)(v))
#define PLATFORM_EXIT(code) sh_exit(code)

#else
/* ---- native (x86) build: full libc, wall-clock ns via clock_gettime. --- */
#include <stdio.h>
#include <time.h>
static void cyc_init(void) {}
static unsigned long cyc_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)((unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec);
}
#define OUT_STR(s)   fputs(s, stdout)
#define OUT_ULONG(v) printf("%lu", (unsigned long)(v))
#define OUT_LONG(v)  printf("%ld", (long)(v))
#define PLATFORM_EXIT(code) return (code)
#endif

static unsigned long cycles[STREAM_N];
static unsigned long cycles_sorted[STREAM_N];

/* Insertion sort: STREAM_N is a few thousand, this runs once, off the
 * measured per-tick path, and needs no libc. */
static void sort_ul(unsigned long *a, int n) {
    int i, j;
    for (i = 1; i < n; i++) {
        unsigned long key = a[i];
        j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}

/* One full replay of STREAM[][] through a fresh monitor. Returns 1 if an
 * alarm fired, 0 if the stream ran out first. */
static int run_once(int *out_t, int *out_leg) {
    hyb_monitor_t m;
    hyb_verdict_t v = HYB_OK;
    int t, c;
    hyb_init(&m);
    for (t = 0; t < STREAM_N; t++) {
        unsigned long c0 = cyc_now();
#if defined(DEBUG_TRACE)
        OUT_STR("t="); OUT_LONG(t); OUT_STR("\n");
#endif
        for (c = 0; c < STREAM_CH; c++) {
#if defined(DEBUG_TRACE)
            OUT_STR(" c="); OUT_LONG(c);
#endif
            v = hyb_push(&m, c, STREAM[t][c]);
#if defined(DEBUG_TRACE)
            OUT_STR(" ok\n");
#endif
            if (v != HYB_OK) break;
        }
        cycles[t] = cyc_now() - c0;
        if (v != HYB_OK) {
            *out_t = t;
            *out_leg = (int)v;
            return 1;
        }
    }
    *out_t = -1;
    *out_leg = 0;
    return 0;
}

int main(void) {
    int t1, l1, t2, l2, fired1, fired2, i, ok;

    cyc_init();

    OUT_STR("MARK-A\n");
    fired1 = run_once(&t1, &l1);
    OUT_STR("MARK-B\n");
    for (i = 0; i < STREAM_N; i++) cycles_sorted[i] = cycles[i];
    OUT_STR("MARK-C\n");
    sort_ul(cycles_sorted, STREAM_N);
    OUT_STR("MARK-D\n");

    OUT_STR("HYB_CHANNELS="); OUT_LONG(HYB_CHANNELS);
    OUT_STR(" STREAM_N="); OUT_LONG(STREAM_N); OUT_STR("\n");
    OUT_STR("RUN1 fired="); OUT_LONG(fired1);
    OUT_STR(" t="); OUT_LONG(t1);
    OUT_STR(" leg="); OUT_LONG(l1); OUT_STR("\n");
    OUT_STR("CYCLES_PER_TICK min="); OUT_ULONG(cycles_sorted[0]);
    OUT_STR(" median="); OUT_ULONG(cycles_sorted[STREAM_N / 2]);
    OUT_STR(" max="); OUT_ULONG(cycles_sorted[STREAM_N - 1]); OUT_STR("\n");

    fired2 = run_once(&t2, &l2);
    OUT_STR("RUN2 fired="); OUT_LONG(fired2);
    OUT_STR(" t="); OUT_LONG(t2);
    OUT_STR(" leg="); OUT_LONG(l2); OUT_STR("\n");

    ok = (fired1 == fired2 && t1 == t2 && l1 == l2) ? 1 : 0;
    OUT_STR("DETERMINISTIC="); OUT_LONG(ok); OUT_STR("\n");

    PLATFORM_EXIT(ok ? 0 : 1);
}
