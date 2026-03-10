//! Camera controller for map navigation (pan, zoom, pitch, bearing).

use x_planets_math::{geo_to_mercator, mercator_to_geo};

use super::globe_visible_half_angle;

/// Controls camera movement (pan, zoom, pitch, bearing).
pub struct CameraController {
    /// Pan sensitivity (pixels per degree).
    pub pan_speed: f64,
    /// Zoom speed (scroll units per level).
    pub zoom_speed: f64,
    /// Minimum zoom level.
    pub min_zoom: f64,
    /// Maximum zoom level.
    pub max_zoom: f64,
    /// Maximum pitch angle in degrees.
    pub max_pitch: f64,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            pan_speed: 2.0,
            zoom_speed: 1.0,
            min_zoom: 0.0,
            max_zoom: 22.0,
            max_pitch: 60.0,
        }
    }

    /// Pan the viewport by a screen-space pixel delta.
    ///
    /// dx > 0 = rightward drag, dy > 0 = upward drag (already negated by caller).
    /// Rotates the delta by the current bearing so panning always follows the screen.
    pub fn pan(&self, viewport: &mut super::Viewport, dx: f64, dy: f64) {
        let scale = 2.0_f64.powf(-viewport.zoom);
        let aspect = viewport.width as f64 / viewport.height.max(1) as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let dx_n = dx / viewport.width.max(1) as f64 * scale * aspect * self.pan_speed;
        let dy_n = dy / viewport.height.max(1) as f64 * scale * self.pan_speed;

        let merc_dx = -(cos_b * dx_n + sin_b * dy_n);
        let merc_dy = -(sin_b * dx_n - cos_b * dy_n);

        let mut center_merc = geo_to_mercator(&viewport.center);
        center_merc.x = (center_merc.x + merc_dx).rem_euclid(1.0);
        center_merc.y = (center_merc.y + merc_dy).clamp(0.0, 1.0);

        viewport.center = mercator_to_geo(center_merc);
    }

    /// Zoom the viewport by a delta (positive = zoom in).
    pub fn zoom(&self, viewport: &mut super::Viewport, delta: f64) {
        viewport.zoom = (viewport.zoom + delta * self.zoom_speed)
            .clamp(self.min_zoom, self.max_zoom);
    }

    /// Zoom with projection-aware minimum.
    pub fn zoom_for_mode(
        &self,
        viewport: &mut super::Viewport,
        delta: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        let effective_min = match mode {
            x_planets_math::ProjectionMode::Mercator => self.min_zoom.max(2.0),
            _ => self.min_zoom,
        };
        viewport.zoom = (viewport.zoom + delta * self.zoom_speed)
            .clamp(effective_min, self.max_zoom);
    }

    /// Set absolute pitch angle (clamped to 0–max_pitch degrees).
    pub fn set_pitch(&self, viewport: &mut super::Viewport, degrees: f64) {
        viewport.pitch = degrees.clamp(0.0, self.max_pitch);
    }

    /// Set absolute bearing (0-360 degrees, clockwise from north).
    pub fn set_bearing(&self, viewport: &mut super::Viewport, degrees: f64) {
        viewport.bearing = degrees.rem_euclid(360.0);
    }

    /// Zoom toward a specific screen point (zoom-to-pointer).
    pub fn zoom_at(
        &self,
        viewport: &mut super::Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
    ) {
        self.zoom_at_for_mode(
            viewport,
            delta,
            screen_x,
            screen_y,
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    /// Zoom toward a specific screen point, projection-aware.
    pub fn zoom_at_for_mode(
        &self,
        viewport: &mut super::Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        if mode == x_planets_math::ProjectionMode::Globe {
            return self.zoom_at_globe(viewport, delta, screen_x, screen_y);
        }

        let old_scale = 2.0_f64.powf(-viewport.zoom);

        self.zoom_for_mode(viewport, delta, mode);

        let new_scale = 2.0_f64.powf(-viewport.zoom);
        let scale_diff = old_scale - new_scale;
        if scale_diff == 0.0 {
            return;
        }

        let aspect = viewport.width as f64 / viewport.height.max(1) as f64;

        let dx_norm = (screen_x - viewport.width as f64 * 0.5) / viewport.width.max(1) as f64;
        let dy_norm = (screen_y - viewport.height as f64 * 0.5) / viewport.height.max(1) as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let merc_dx = (dx_norm * aspect * cos_b - dy_norm * sin_b) * scale_diff;
        let merc_dy = (dx_norm * aspect * sin_b + dy_norm * cos_b) * scale_diff;

        let mut center_merc = geo_to_mercator(&viewport.center);
        center_merc.x = (center_merc.x + merc_dx).rem_euclid(1.0);
        center_merc.y = (center_merc.y + merc_dy).clamp(0.0, 1.0);
        viewport.center = mercator_to_geo(center_merc);
    }

    /// Pan the viewport in globe mode using angular deltas.
    pub fn pan_globe(&self, viewport: &mut super::Viewport, dx: f64, dy: f64) {
        let unit_altitude = super::globe_unit_altitude(viewport.zoom, &viewport.body);
        let visible_half = globe_visible_half_angle(unit_altitude);
        let visible_deg = visible_half.to_degrees() * 2.0;

        let deg_per_px = visible_deg / viewport.height.max(1) as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let dx_deg = dx * deg_per_px;
        let dy_deg = dy * deg_per_px;

        let dlat = sin_b * dx_deg - cos_b * dy_deg;

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let dlon = -(cos_b * dx_deg + sin_b * dy_deg) / cos_lat;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    /// Zoom toward a screen point in globe mode.
    pub fn zoom_at_globe(
        &self,
        viewport: &mut super::Viewport,
        delta: f64,
        screen_x: f64,
        screen_y: f64,
    ) {
        let unit_altitude = super::globe_unit_altitude(viewport.zoom, &viewport.body);
        let half_angle_old = globe_visible_half_angle(unit_altitude).to_degrees();

        let dx_norm = (screen_x - viewport.width as f64 * 0.5) / viewport.width.max(1) as f64;
        let dy_norm = (screen_y - viewport.height as f64 * 0.5) / viewport.height.max(1) as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let cursor_lon_off = (dx_norm * cos_b - dy_norm * sin_b) * half_angle_old * 2.0
            / cos_lat;
        let cursor_lat_off = -(dx_norm * sin_b + dy_norm * cos_b) * half_angle_old * 2.0;

        self.zoom(viewport, delta);

        let unit_altitude_new = super::globe_unit_altitude(viewport.zoom, &viewport.body);
        let half_angle_new = globe_visible_half_angle(unit_altitude_new).to_degrees();

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let new_cursor_lon_off = (dx_norm * cos_b - dy_norm * sin_b) * half_angle_new * 2.0
            / cos_lat;
        let new_cursor_lat_off = -(dx_norm * sin_b + dy_norm * cos_b) * half_angle_new * 2.0;

        let dlat = cursor_lat_off - new_cursor_lat_off;
        let dlon = cursor_lon_off - new_cursor_lon_off;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    /// Pan in centered Mercator mode using angular deltas.
    pub fn pan_centered(&self, viewport: &mut super::Viewport, dx: f64, dy: f64) {
        let scale = 2.0_f64.powf(-viewport.zoom);
        let edge_y = (0.5 + scale * 0.5).min(0.9999);
        let edge_geo = mercator_to_geo(glam::DVec2::new(0.5, edge_y));
        let visible_deg = edge_geo.lat.abs() * 2.0;
        let deg_per_px = visible_deg / viewport.height.max(1) as f64;

        let bearing_rad = viewport.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let dx_deg = dx * deg_per_px * self.pan_speed;
        let dy_deg = dy * deg_per_px * self.pan_speed;

        let dlat = sin_b * dx_deg - cos_b * dy_deg;

        let cos_lat = viewport.center.lat.to_radians().cos().max(0.05);
        let dlon = -(cos_b * dx_deg + sin_b * dy_deg) / cos_lat;

        viewport.center.lat = (viewport.center.lat + dlat).clamp(-89.9, 89.9);
        viewport.center.lon = ((viewport.center.lon + dlon) + 180.0).rem_euclid(360.0) - 180.0;
    }

    pub fn pan_for_mode(
        &self,
        viewport: &mut super::Viewport,
        dx: f64,
        dy: f64,
        mode: x_planets_math::ProjectionMode,
    ) {
        match mode {
            x_planets_math::ProjectionMode::Globe => {
                self.pan_globe(viewport, dx, dy)
            }
            x_planets_math::ProjectionMode::Mercator => {
                self.pan_centered(viewport, dx, dy)
            }
        }
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}
