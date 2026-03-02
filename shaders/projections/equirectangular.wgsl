// Equirectangular / Plate Carrée Projection (EPSG:4326)
//
// Direct mapping of geographic coordinates to a rectangular grid.
// Simple linear transformation without distortion compensation.

fn project_equirectangular(world_pos: vec3<f32>) -> vec3<f32> {
    let lat = world_pos.x;
    let lon = world_pos.y;
    let alt = world_pos.z;

    let x = (lon + 180.0) / 360.0;
    let y = (90.0 - lat) / 180.0;

    return vec3<f32>(x, y, alt);
}

fn unproject_equirectangular(projected: vec3<f32>) -> vec3<f32> {
    let lon = projected.x * 360.0 - 180.0;
    let lat = 90.0 - projected.y * 180.0;

    return vec3<f32>(lat, lon, projected.z);
}
