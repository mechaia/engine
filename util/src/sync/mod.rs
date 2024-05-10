pub mod half;
pub mod spsc;

use core::sync::atomic;

/// Fastest, smallest atomic type for the platform.
///
/// Some architectures such as RISC-V don't have native atomic instructions for u8/u16
/// and generate a whole lot more code for it than u32/u64.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type FastAtomic = atomic::AtomicU8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
type FastAtomic = atomic::AtomicU32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type FastNonAtomic = u8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
type FastNonAtomic = u32;
