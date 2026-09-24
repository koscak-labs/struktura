/* Minimal caller that references hyb_init/hyb_push so the linker can't
 * dead-strip them, used ONLY to measure the generated monitor's own
 * flash/RAM footprint in isolation from the equivalence/cost harness
 * (which also embeds a 144000-byte test stream that is not part of the
 * deployed monitor). */
#include "hybrid_monitor.c"

int main(void) {
    static hyb_monitor_t m;
    hyb_init(&m);
    return (int)hyb_push(&m, 0, 1.0);
}
