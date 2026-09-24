/* Minimal Cortex-M3 startup for QEMU mps2-an385: vector table + reset
 * handler (copy .data, zero .bss, call main). No vendor SDK, no CMSIS.
 */
extern unsigned long _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

void Reset_Handler(void);
static void Default_Handler(void) { while (1) {} }

/* Bare semihosting write so a fault is visible instead of a silent hang
 * (SYS_WRITE0 = 0x04). Kept free of any float/double use. */
static void fault_print(const char *s) {
    register unsigned long r0 __asm__("r0") = 0x04;
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt 0xab" : : "r"(r0), "r"(r1) : "memory");
}
static void fault_print_hex(unsigned long v) {
    char buf[11];
    int i;
    buf[0] = '0'; buf[1] = 'x';
    for (i = 0; i < 8; i++) {
        unsigned long nib = (v >> ((7 - i) * 4)) & 0xF;
        buf[2 + i] = (char)(nib < 10 ? '0' + nib : 'a' + (nib - 10));
    }
    buf[10] = 0;
    fault_print(buf);
}

#define CFSR (*(volatile unsigned long *)0xE000ED28)
#define HFSR (*(volatile unsigned long *)0xE000ED2C)
#define MMFAR (*(volatile unsigned long *)0xE000ED34)
#define BFAR (*(volatile unsigned long *)0xE000ED38)

void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) {
    fault_print("!! HardFault_Handler CFSR=");
    fault_print_hex(CFSR);
    fault_print(" HFSR=");
    fault_print_hex(HFSR);
    fault_print(" MMFAR=");
    fault_print_hex(MMFAR);
    fault_print(" BFAR=");
    fault_print_hex(BFAR);
    fault_print("\n");
    while (1) {}
}
void MemManage_Handler(void) {
    fault_print("!! MemManage_Handler CFSR=");
    fault_print_hex(CFSR);
    fault_print(" MMFAR=");
    fault_print_hex(MMFAR);
    fault_print("\n");
    while (1) {}
}
void BusFault_Handler(void) {
    fault_print("!! BusFault_Handler CFSR=");
    fault_print_hex(CFSR);
    fault_print(" BFAR=");
    fault_print_hex(BFAR);
    fault_print("\n");
    while (1) {}
}
void UsageFault_Handler(void) {
    fault_print("!! UsageFault_Handler CFSR=");
    fault_print_hex(CFSR);
    fault_print("\n");
    while (1) {}
}
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

typedef void (*isr_t)(void);

__attribute__((section(".isr_vector"), used))
const isr_t isr_vector[16] = {
    (isr_t)&_estack,
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};

/* Freestanding build (-nostdlib): hybrid_monitor.c needs memset(), and
 * newlib's libm error path (w_sqrt/w_log) needs __errno(); neither pulls
 * in the rest of libc. Both stubs are trivial and bounded. */
void *memset(void *dst, int val, unsigned int n) {
    unsigned char *d = (unsigned char *)dst;
    while (n--) *d++ = (unsigned char)val;
    return dst;
}

static int __errno_storage;
int *__errno(void) { return &__errno_storage; }

#define SHCSR (*(volatile unsigned long *)0xE000ED24)

void Reset_Handler(void) {
    unsigned long *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    dst = &_sbss;
    while (dst < &_ebss) *dst++ = 0;
    /* Enable MemManage/BusFault/UsageFault as distinct handlers instead of
     * always escalating to HardFault, so a fault is diagnosable. */
    SHCSR |= (1UL << 16) | (1UL << 17) | (1UL << 18);
    main();
    while (1) {}
}
