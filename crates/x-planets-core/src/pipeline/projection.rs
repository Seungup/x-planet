//! Stage 5: Projection transform (CPU reference).

use x_planets_projection::ProjectionPlugin;

/// Apply a projection to a list of world-space positions (CPU).
///
/// This is the "ground truth" against which GPU results are compared.
///
/// Pure function.
pub fn project_positions_cpu(
    plugin: &dyn ProjectionPlugin,
    positions: &[glam::DVec3],
) -> Vec<glam::DVec3> {
    positions.iter().map(|p| plugin.project_cpu(*p)).collect()
}

/// Verify projection roundtrip accuracy.
///
/// Pure function. Returns max error across all test points.
pub fn verify_projection_roundtrip(
    plugin: &dyn ProjectionPlugin,
    test_points: &[glam::DVec3],
) -> f64 {
    test_points
        .iter()
        .map(|p| {
            let projected = plugin.project_cpu(*p);
            let recovered = plugin.unproject_cpu(projected);
            (*p - recovered).length()
        })
        .fold(0.0f64, f64::max)
}
