//! Tests for the shared test helpers in `tests/common/mod.rs`.
//!
//! These live in their own binary on purpose: `common` is compiled into every
//! test that uses it, so a `#[cfg(test)] mod tests` inside it would add its
//! test count to every one of those binaries — and `docs/claims.tsv` asserts
//! exact "test result: ok. N passed" counts for the generated-C comparisons.

mod common;

use common::{c_compiler, link_flags, needs_libm};

#[test]
fn msvc_target_does_not_need_libm() {
    assert!(!needs_libm("x86_64-pc-windows-msvc"));
    assert!(!needs_libm("aarch64-pc-windows-msvc"));
    assert!(!needs_libm("clang version 22.1.8 (target: x86_64-pc-windows-msvc)"));
}

#[test]
fn unix_and_mingw_targets_need_libm() {
    assert!(needs_libm("x86_64-pc-linux-gnu"));
    assert!(needs_libm("x86_64-apple-darwin"));
    assert!(needs_libm("x86_64-w64-mingw32"));
    assert!(needs_libm("gcc (GCC) 13.2.0"));
}

/// The flags must be right for the compiler actually in use, or every
/// generated-C comparison silently fails to link.
#[test]
fn link_flags_match_the_live_compiler() {
    let Some(cc) = c_compiler() else {
        return;
    };
    let flags = link_flags(&cc);
    let mut description = String::new();
    for args in [&["-dumpmachine"][..], &["--version"][..]] {
        if let Ok(o) = std::process::Command::new(&cc).args(args).output() {
            description.push_str(&String::from_utf8_lossy(&o.stdout));
        }
    }
    if needs_libm(&description) {
        assert_eq!(flags, vec!["-lm"], "non-MSVC compiler must get -lm");
    } else {
        assert!(flags.is_empty(), "MSVC target must not get -lm, got {flags:?}");
    }
}

/// Two `scratch` calls must not land in the same directory: the tests inside
/// one binary run on separate threads, and sharing a directory lets one
/// overwrite the other's `prog.exe` mid-run.
#[test]
fn scratch_directories_are_unique() {
    let a = common::scratch("uniq");
    let b = common::scratch("uniq");
    assert_ne!(a, b);
    assert!(a.is_dir() && b.is_dir());
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}
