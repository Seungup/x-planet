//! Step-by-step verification chain.
//!
//! Karpathy's gradient checking, applied to a rendering engine:
//! run each pipeline stage independently, verify its output,
//! and stop at the first failure.
//!
//! Usage:
//! ```rust,no_run
//! use x_planets_core::verify_chain::{VerifyChain, StepResult};
//! let mut chain = VerifyChain::new();
//! chain.add("math basics", || StepResult::pass());
//! chain.add("tile coords", || StepResult::pass());
//! chain.run(); // stops at first failure
//! ```

use std::collections::HashMap;

/// Result of a single verification step.
pub struct StepResult {
    pub passed: bool,
    pub details: String,
    pub metrics: HashMap<String, f64>,
}

impl StepResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            details: String::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn pass_with(details: impl Into<String>) -> Self {
        Self {
            passed: true,
            details: details.into(),
            metrics: HashMap::new(),
        }
    }

    pub fn fail(details: impl Into<String>) -> Self {
        Self {
            passed: false,
            details: details.into(),
            metrics: HashMap::new(),
        }
    }

    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }
}

/// Result of running the full chain.
#[derive(Debug)]
pub enum ChainResult {
    AllPassed { steps: usize },
    Failed { step: usize, name: String },
}

impl ChainResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, ChainResult::AllPassed { .. })
    }
}

/// A chain of verification steps that run sequentially.
/// Stops at the first failure — no point checking further
/// if the foundations are broken.
pub struct VerifyChain {
    steps: Vec<(String, Box<dyn Fn() -> StepResult>)>,
}

impl VerifyChain {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a verification step.
    pub fn add(&mut self, name: impl Into<String>, f: impl Fn() -> StepResult + 'static) {
        self.steps.push((name.into(), Box::new(f)));
    }

    /// Run all steps. Prints results. Stops at first failure.
    pub fn run(&self) -> ChainResult {
        let total = self.steps.len();
        println!();
        println!("══════════════════════════════════════════════════");
        println!("  Verification Chain  ({} steps)", total);
        println!("══════════════════════════════════════════════════");

        for (i, (name, verify_fn)) in self.steps.iter().enumerate() {
            print!("  [{}/{}] {} ... ", i + 1, total, name);
            let result = verify_fn();

            if result.passed {
                println!("✓ OK");
                if !result.details.is_empty() {
                    println!("         {}", result.details);
                }
                for (k, v) in &result.metrics {
                    println!("         {} = {:.6e}", k, v);
                }
            } else {
                println!("✗ FAIL");
                println!("         {}", result.details);
                for (k, v) in &result.metrics {
                    println!("         {} = {:.6e}", k, v);
                }
                println!("══════════════════════════════════════════════════");
                println!("  STOPPED at step {} — fix this before continuing", i + 1);
                println!("══════════════════════════════════════════════════");
                return ChainResult::Failed {
                    step: i,
                    name: name.clone(),
                };
            }
        }

        println!("══════════════════════════════════════════════════");
        println!("  All {} steps passed ✓", total);
        println!("══════════════════════════════════════════════════");
        println!();
        ChainResult::AllPassed { steps: total }
    }
}

impl Default for VerifyChain {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-built verification chain for Phase 1
// ═══════════════════════════════════════════════════════════════════

use x_planets_math::{geo_to_mercator, mercator_to_geo, GeoCoord, TileCoord};
use x_planets_projection::{Mercator, ProjectionPlugin};

/// Build the standard Phase 1 verification chain.
///
/// This checks every assumption from math primitives up to
/// tile geometry, in order of dependency.
pub fn phase1_chain() -> VerifyChain {
    let mut chain = VerifyChain::new();

    // ── Level 0: Math primitives ──────────────────────────────

    chain.add("GeoCoord normalize clamps correctly", || {
        let c = GeoCoord::new(100.0, 200.0).normalize();
        if (c.lat - 90.0).abs() > 1e-10 {
            return StepResult::fail(format!("lat {} should be 90", c.lat));
        }
        if (c.lon - (-160.0)).abs() > 1e-10 {
            return StepResult::fail(format!("lon {} should be -160", c.lon));
        }
        StepResult::pass()
    });

    chain.add("Mercator roundtrip (6 world cities)", || {
        let cities = [
            ("London", 51.5074, -0.1278),
            ("Tokyo", 35.6762, 139.6503),
            ("NYC", 40.7128, -74.0060),
            ("Sydney", -33.8688, 151.2093),
            ("São Paulo", -23.5505, -46.6333),
            ("Seoul", 37.5665, 126.9780),
        ];

        let mut max_err = 0.0f64;
        let mut worst = "";

        for (name, lat, lon) in &cities {
            let coord = GeoCoord::new(*lat, *lon);
            let merc = geo_to_mercator(&coord);
            let back = mercator_to_geo(merc);
            let err = ((coord.lat - back.lat).powi(2) + (coord.lon - back.lon).powi(2)).sqrt();
            if err > max_err {
                max_err = err;
                worst = name;
            }
        }

        StepResult {
            passed: max_err < 1e-10,
            details: format!("worst city: {} (err={:.2e})", worst, max_err),
            metrics: [("max_roundtrip_error".into(), max_err)].into(),
        }
    });

    // ── Level 1: TileCoord system ─────────────────────────────

    chain.add("TileCoord::from_geo containment (19 zoom levels)", || {
        let test_coords = [
            GeoCoord::new(0.0, 0.0),
            GeoCoord::new(51.5, -0.1),
            GeoCoord::new(-33.9, 151.2),
            GeoCoord::new(37.6, 127.0),
            GeoCoord::new(85.0, 0.0),   // near pole
            GeoCoord::new(-85.0, 0.0),  // near south pole
        ];

        let mut failures = 0;
        let mut total = 0;

        for coord in &test_coords {
            for zoom in 0..=18u8 {
                total += 1;
                let tile = TileCoord::from_geo(coord, zoom);
                let bounds = tile.to_geo_bounds();
                if !bounds.contains(coord) {
                    failures += 1;
                }
            }
        }

        StepResult {
            passed: failures == 0,
            details: format!("{}/{} passed", total - failures, total),
            metrics: [("failures".into(), failures as f64)].into(),
        }
    });

    chain.add("TileCoord parent/child consistency", || {
        let mut failures = 0;

        for z in 1..=10u8 {
            let n = 1u32 << z;
            // Sample a few tiles at each zoom
            for x in (0..n).step_by((n as usize / 4).max(1)) {
                for y in (0..n).step_by((n as usize / 4).max(1)) {
                    let tile = TileCoord::new(z, x, y);
                    let parent = tile.parent().unwrap();
                    let children = parent.children();

                    if !children.contains(&tile) {
                        failures += 1;
                    }
                }
            }
        }

        StepResult {
            passed: failures == 0,
            details: format!("{} parent/child mismatches", failures),
            metrics: [("failures".into(), failures as f64)].into(),
        }
    });

    // ── Level 2: Projection accuracy ──────────────────────────

    chain.add("Mercator projection known values", || {
        let proj = Mercator;

        let checks = [
            ("origin→(0.5,0.5)", glam::DVec3::new(0.0, 0.0, 0.0), 0.5, 0.5),
            ("equator-left→(0,0.5)", glam::DVec3::new(0.0, -180.0, 0.0), 0.0, 0.5),
            ("equator-right→(1,0.5)", glam::DVec3::new(0.0, 180.0, 0.0), 1.0, 0.5),
        ];

        for (label, input, exp_x, exp_y) in &checks {
            let out = proj.project_cpu(*input);
            let err = ((out.x - exp_x).powi(2) + (out.y - exp_y).powi(2)).sqrt();
            if err > 1e-6 {
                return StepResult::fail(format!(
                    "{}: expected ({},{}) got ({:.6},{:.6})",
                    label, exp_x, exp_y, out.x, out.y
                ))
                .with_metric("error", err);
            }
        }

        StepResult::pass_with("all known values match")
    });

    chain.add("Mercator roundtrip 1000 points (< 1e-8 error)", || {
        let proj = Mercator;
        let mut max_err = 0.0f64;

        for i in 0..50 {
            let lat = -80.0 + 160.0 * (i as f64 / 49.0);
            for j in 0..20 {
                let lon = -179.0 + 358.0 * (j as f64 / 19.0);
                let p = glam::DVec3::new(lat, lon, 0.0);
                let projected = proj.project_cpu(p);
                let recovered = proj.unproject_cpu(projected);
                let err = (p - recovered).length();
                max_err = max_err.max(err);
            }
        }

        StepResult {
            passed: max_err < 1e-8,
            details: format!("max roundtrip error = {:.2e}", max_err),
            metrics: [("max_error".into(), max_err)].into(),
        }
    });

    // ── Level 3: Pipeline pure functions ──────────────────────

    chain.add("build_tile_mesh: zoom 0 RTE centered at origin", || {
        use crate::pipeline::build_tile_mesh;

        let tiles = vec![TileCoord::new(0, 0, 0)];
        let (verts, indices) = build_tile_mesh(&tiles);

        if verts.len() != 4 {
            return StepResult::fail(format!("expected 4 verts, got {}", verts.len()));
        }
        if indices.len() != 6 {
            return StepResult::fail(format!("expected 6 indices, got {}", indices.len()));
        }

        // RTE: vertices are centered at origin, half-size = 0.5 at zoom 0
        let min_x = verts.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
        let max_x = verts.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);
        let min_y = verts.iter().map(|v| v.position[1]).fold(f32::MAX, f32::min);
        let max_y = verts.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);

        if (min_x + 0.5).abs() > 1e-6 || (max_x - 0.5).abs() > 1e-6
            || (min_y + 0.5).abs() > 1e-6 || (max_y - 0.5).abs() > 1e-6
        {
            return StepResult::fail(format!(
                "expected (-0.5,-0.5)→(0.5,0.5) got ({},{})→({},{})",
                min_x, min_y, max_x, max_y
            ));
        }

        StepResult::pass()
    });

    chain.add("build_tile_mesh: zoom 1 tiles all same RTE size", || {
        use crate::pipeline::build_tile_mesh;

        let tiles = vec![
            TileCoord::new(1, 0, 0),
            TileCoord::new(1, 1, 0),
            TileCoord::new(1, 0, 1),
            TileCoord::new(1, 1, 1),
        ];
        let (verts, _) = build_tile_mesh(&tiles);

        // RTE: each tile's 4 vertices should be at ±0.25 (half-size at zoom 1)
        for (i, chunk) in verts.chunks(4).enumerate() {
            let min_x = chunk.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min);
            let max_x = chunk.iter().map(|v| v.position[0]).fold(f32::MIN, f32::max);

            if (min_x + 0.25).abs() > 1e-6 || (max_x - 0.25).abs() > 1e-6 {
                return StepResult::fail(format!(
                    "tile {}: x range: {}→{}, expected -0.25→0.25", i, min_x, max_x
                ));
            }
        }

        StepResult::pass_with(format!("{} vertices, 4 tiles", verts.len()))
    });

    chain.add("compute_load_requests: excludes cached tiles", || {
        use crate::pipeline::compute_load_requests;
        use std::collections::HashSet;

        let visible = vec![
            TileCoord::new(3, 0, 0),
            TileCoord::new(3, 1, 0),
            TileCoord::new(3, 2, 0),
        ];
        let mut cached = HashSet::new();
        cached.insert(TileCoord::new(3, 1, 0));

        let requests = compute_load_requests(&visible, &cached, &GeoCoord::new(0.0, 0.0));

        if requests.len() != 2 {
            return StepResult::fail(format!("expected 2 requests, got {}", requests.len()));
        }
        if requests.iter().any(|r| r.coord == TileCoord::new(3, 1, 0)) {
            return StepResult::fail("cached tile should not be in requests");
        }

        StepResult::pass()
    });

    chain
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result_constructors() {
        assert!(StepResult::pass().passed);
        assert!(!StepResult::fail("oops").passed);

        let r = StepResult::pass_with("good").with_metric("acc", 0.99);
        assert!(r.passed);
        assert_eq!(r.metrics["acc"], 0.99);
    }

    #[test]
    fn test_verify_chain_all_pass() {
        let mut chain = VerifyChain::new();
        chain.add("always pass", || StepResult::pass());
        chain.add("also pass", || StepResult::pass_with("ok"));

        let result = chain.run();
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_chain_stops_on_failure() {
        let mut chain = VerifyChain::new();
        chain.add("pass", || StepResult::pass());
        chain.add("fail", || StepResult::fail("broken"));
        chain.add("never reached", || {
            panic!("should not run after failure")
        });

        let result = chain.run();
        assert!(!result.is_ok());
        match result {
            ChainResult::Failed { step, name } => {
                assert_eq!(step, 1);
                assert_eq!(name, "fail");
            }
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn test_phase1_chain_passes() {
        let chain = phase1_chain();
        let result = chain.run();
        assert!(result.is_ok(), "Phase 1 verification chain should pass");
    }
}
