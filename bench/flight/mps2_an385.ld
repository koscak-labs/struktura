/* Linker script for QEMU -M mps2-an385 (Cortex-M3): ZBT SRAM at 0x0
 * used as "flash" (an385 has no real flash model; code+rodata live in the
 * 4MB SRAM at 0x00000000), data SRAM at 0x20000000 for .data/.bss/stack.
 */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(Reset_Handler)

SECTIONS
{
    .isr_vector : {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
    } > FLASH

    .text : {
        . = ALIGN(4);
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    _sidata = .;

    .data : AT (_sidata) {
        . = ALIGN(4);
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    ._stack_dummy (COPY) : { . = . + 0x2000; } > RAM

    _estack = ORIGIN(RAM) + LENGTH(RAM);
}
