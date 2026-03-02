//! Run the Phase 1 verification chain.
//!
//! This is the Karpathy "gradient checking" for x-planets.
//! No GPU needed. Pure CPU. Tests every assumption.
//!
//! Run: cargo run --example verify
//!
//! Expected output:
//!   ══════════════════════════════════════════════════
//!     Verification Chain  (9 steps)
//!   ══════════════════════════════════════════════════
//!     [1/9] GeoCoord normalize clamps correctly ... ✓ OK
//!     [2/9] Mercator roundtrip (6 world cities) ... ✓ OK
//!     ...
//!     [9/9] compute_load_requests: excludes cached tiles ... ✓ OK
//!   ══════════════════════════════════════════════════
//!     All 9 steps passed ✓
//!   ══════════════════════════════════════════════════
//!
//! If any step fails, it stops immediately and tells you what broke.
//! Fix that step before moving on. Don't skip. Don't ignore.

fn main() {
    let chain = x_planets_core::verify_chain::phase1_chain();
    let result = chain.run();

    match result {
        x_planets_core::verify_chain::ChainResult::AllPassed { steps } => {
            println!("Phase 1 foundations solid. {} checks passed.", steps);
            println!();
            println!("You can now proceed to the GPU steps:");
            println!("  cargo run --example step00_triangle");
            println!("  cargo run --example step01_colored_quad");
            println!("  cargo run --example step02_textured_quad");
        }
        x_planets_core::verify_chain::ChainResult::Failed { step, name } => {
            eprintln!();
            eprintln!("BLOCKED at step {}: '{}'", step + 1, name);
            eprintln!("Fix this before doing anything else.");
            std::process::exit(1);
        }
    }
}
