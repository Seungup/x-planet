// Web Mercator Projection (EPSG:3857)
//
// Transforms geographic coordinates (lat, lon, alt) to
// normalized Mercator coordinates (0..1, 0..1, alt).
//
// Usage: Include this file and call project_mercator() in your shader.

const PI: f32 = 3.14159265358979323846;

fn project_mercator(world_pos: vec3<f32>) -> vec3<f32> {
    let lat = world_pos.x;
    let lon = world_pos.y;
    let alt = world_pos.z;

    let x = (lon + 180.0) / 360.0;
    let lat_rad = radians(lat);
    let sin_lat = sin(lat_rad);
    let y = 0.5 - 0.5 * log((1.0 + sin_lat) / (1.0 - sin_lat)) / (2.0 * PI);

    return vec3<f32>(x, y, alt);
}

fn unproject_mercator(projected: vec3<f32>) -> vec3<f32> {
    let lon = projected.x * 360.0 - 180.0;
    let n = PI - 2.0 * PI * projected.y;
    let lat = degrees(atan(0.5 * (exp(n) - exp(-n))));

    return vec3<f32>(lat, lon, projected.z);
}
