#!/usr/bin/env bash
# bench/flight/run.sh -- reproduce the embedded-monitor equivalence + cost
# measurement described in bench/flight/README.md. Run from inside WSL/Linux
# (needs: rustc/cargo, gcc, arm-none-eabi-gcc, qemu-system-arm).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. generate hybrid_monitor.c + stream fixtures via the Rust monitor =="
if command -v cargo >/dev/null 2>&1; then
    (cd genrs && cargo build --release)
    GENRS_BIN=genrs/target/release/genrs
    [ -x "$GENRS_BIN" ] || GENRS_BIN=genrs/target/release/genrs.exe
    "$GENRS_BIN"
else
    echo "cargo not found: the generated monitor and fixtures are not committed;" \
         "install Rust (rustup) in this shell and re-run" >&2
    exit 1
fi

echo
echo "== 2. native x86 build (equivalence + determinism sanity check) =="
gcc -std=c99 -Wall -Werror -O2 -o harness_native harness.c -lm
./harness_native

echo
echo "== 3. size_probe: generated monitor's OWN static footprint (Cortex-M3) =="
# -fno-early-inlining: see the note in step 5 -- required at -O2 on ARM here,
# harmless at every other level.
arm-none-eabi-gcc -std=c99 -Wall -Werror -O2 -fno-early-inlining -mcpu=cortex-m3 -mthumb \
    -mfloat-abi=soft -ffreestanding -fno-builtin -nostdlib \
    -T mps2_an385.ld -Wl,--gc-sections \
    -o size_probe.elf startup.c size_probe.c -lm -lgcc
arm-none-eabi-size size_probe.elf

echo
echo "== 4. malloc/free check on the generated C (must be empty) =="
grep -n 'malloc\|calloc\|realloc\|free(' hybrid_monitor.c && exit 1 || echo "OK: no malloc/free references"

echo
echo "== 5. bare-metal ARM build + QEMU run (mps2-an385, Cortex-M3) =="
echo "   NOTE: -O2 (struktura's own suggested compile line) needs"
echo "   -fno-early-inlining here: without it, a stack slot in run_once()"
echo "   (harness.c) is overwritten once hyb_push()+hyb_dfa_alpha() are"
echo "   inlined into it, and the run BusFaults (CFSR=0x8200). The generated"
echo "   C shows no undefined behaviour under ASan/UBSan; whether this is a"
echo "   GCC 13.2.1 code-generation bug is not established. See"
echo "   bench/flight/README.md for the evidence."
arm-none-eabi-gcc -std=c99 -Wall -Wno-error -O2 -fno-early-inlining -mcpu=cortex-m3 -mthumb \
    -mfloat-abi=soft -ffreestanding -fno-builtin -nostdlib \
    -T mps2_an385.ld -Wl,--gc-sections \
    -o harness_arm_o2.elf startup.c harness.c -lm -lgcc
timeout 60 qemu-system-arm -M mps2-an385 -cpu cortex-m3 -semihosting -nographic \
    -kernel harness_arm_o2.elf | tee qemu_out.log

echo
echo "== done: compare alarms_rust.csv (Rust monitor) against RUN1/RUN2 t=/leg= above =="
cat alarms_rust.csv
