//! Shared input, animation, and gesture logic for all platforms.
//!
//! Uses `f64` timestamps (seconds) instead of `std::time::Instant` so the same
//! code works on both native (Instant → f64) and WASM (performance.now()/1000).
//!
//! # Modules
//! - [`AnimationController`]: smooth zoom, inertia panning, double-click, tile fades
//! - [`TouchGestureState`]: multi-touch gestures with grace period for finger transitions
//! - [`crossfade`]: tile crossfade overlay computation (shared between native and web)

mod animation;
mod crossfade;
mod gestures;

pub use animation::*;
pub use crossfade::*;
pub use gestures::*;
