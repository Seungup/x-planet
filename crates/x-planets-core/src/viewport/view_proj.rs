//! View-projection matrix computation for the viewport.

use x_planets_math::{geo_to_mercator, ViewportUniforms};

impl super::Viewport {
    /// Compute the view-projection matrix in f64 for high-precision per-tile MVP.
    ///
    /// Same camera geometry as `to_uniforms()` but uses `DMat4` throughout
    /// to avoid f32 precision loss at high zoom levels.  Each tile computes
    /// `MVP = VP_f64 * translate(tile_center_f64)` in f64, then casts to f32.
    /// This eliminates the ~14px jitter at zoom 18+ caused by f32 VP.
    pub fn to_view_proj_f64(&self) -> glam::DMat4 {
        self.to_view_proj_f64_projected(x_planets_math::ProjectionMode::Mercator)
    }

    /// Like [`to_view_proj_f64`] but positions the camera using the given projection.
    pub fn to_view_proj_f64_projected(&self, mode: x_planets_math::ProjectionMode) -> glam::DMat4 {
        if mode == x_planets_math::ProjectionMode::Globe {
            return self.to_globe_view_proj_f64();
        }
        // Centered (oblique) Mercator: viewport center → (0.5, 0.5)
        let center = glam::DVec2::new(0.5, 0.5);
        let center_merc = center;
        let scale = 2.0_f64.powf(self.zoom);
        let aspect = self.width as f64 / self.height as f64;
        let cx = center_merc.x;
        let cy = center_merc.y;

        let half_h = 1.0 / scale;

        let fov_y: f64 = std::f64::consts::FRAC_PI_3; // 60°
        let cam_h = half_h / (fov_y * 0.5).tan();

        let pitch_rad = self.pitch.to_radians();
        let bearing_rad = self.bearing.to_radians();
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        let up = glam::DVec3::new(sin_b, -cos_b, 0.0);

        let eye = glam::DVec3::new(
            cx - sin_b * cam_h * pitch_rad.sin(),
            cy + cos_b * cam_h * pitch_rad.sin(),
            cam_h * pitch_rad.cos(),
        );
        let target = glam::DVec3::new(cx, cy, 0.0);

        let view = glam::DMat4::look_at_rh(eye, target, up);
        // Adaptive near/far: tighten the ratio to preserve depth-buffer
        // precision, but extend far plane for pitched views.
        let pitch_rad = self.pitch.to_radians();
        let far_mult = if pitch_rad > 0.01 {
            // At high pitch, distant tiles are much farther than cam_h.
            // tan(pitch) gives the horizontal reach; camera→ground distance
            // is sqrt(cam_h² + reach²).  Use a generous multiplier.
            4.0 + 8.0 * pitch_rad.sin()
        } else {
            4.0
        };
        let near = cam_h * 0.1;
        let far = cam_h * far_mult;
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, near, far);

        let flip_x = glam::DMat4::from_diagonal(glam::DVec4::new(-1.0, 1.0, 1.0, 1.0));
        flip_x * proj * view
    }

    /// Compute the globe view-projection matrix in f64.
    ///
    /// Orbital camera around a unit sphere.  The camera looks at the surface
    /// point corresponding to `self.center`, positioned at a distance
    /// determined by `self.zoom`.  Pitch and bearing rotate the camera.
    pub fn to_globe_view_proj_f64(&self) -> glam::DMat4 {
        let lat_rad = self.center.lat.to_radians();
        let lon_rad = self.center.lon.to_radians();

        // Surface point (look-at target) on unit sphere
        let surface_point = x_planets_math::geo_to_unit_sphere(lat_rad, lon_rad);

        // Camera altitude above sphere surface (unit-sphere radius = 1.0).
        // Accelerated descent at high zoom to match Mercator visible area.
        let unit_altitude = super::globe_unit_altitude(self.zoom);

        // Local ENU (East-North-Up) basis at surface point
        let up_surface = surface_point.normalize();
        let east = glam::DVec3::new(-lon_rad.sin(), lon_rad.cos(), 0.0).normalize();
        let north = up_surface.cross(east).normalize();

        // Apply bearing: rotate horizontal component
        let bearing_rad = self.bearing.to_radians();
        let cos_b = bearing_rad.cos();
        let sin_b = bearing_rad.sin();
        let forward_h = north * cos_b + east * sin_b;

        // Apply pitch: at pitch=0, camera directly above, looking down.
        let pitch_rad = self.pitch.to_radians();
        let backward = -forward_h * pitch_rad.sin() + up_surface * pitch_rad.cos();
        let camera_pos = surface_point + backward.normalize() * unit_altitude;

        let camera_up = if pitch_rad.abs() < 0.01 {
            forward_h.normalize()
        } else {
            let forward = (surface_point - camera_pos).normalize();
            let right = forward.cross(up_surface).normalize();
            right.cross(forward).normalize()
        };

        let view = glam::DMat4::look_at_rh(camera_pos, surface_point, camera_up);

        let aspect = self.width as f64 / self.height.max(1) as f64;
        let fov_y: f64 = std::f64::consts::FRAC_PI_3; // 60°
        // At high zoom the camera is extremely close to the surface; use
        // adaptive near/far to preserve depth-buffer precision.
        let horizon_dist = (2.0 * unit_altitude).sqrt();
        let near = (unit_altitude * 0.1).max(1e-7);
        let far = (unit_altitude + horizon_dist) * 2.0 + 0.1;
        let proj = glam::DMat4::perspective_rh(fov_y, aspect, near, far);

        proj * view
    }

    /// Compute GPU uniforms for this viewport.
    ///
    /// Uses perspective projection so pitch and bearing work correctly.
    /// When pitch=0 and bearing=0 this is equivalent to orthographic top-down.
    pub fn to_uniforms(&self) -> ViewportUniforms {
        let center_merc = geo_to_mercator(&self.center);
        let scale = 2.0_f64.powf(self.zoom) as f32;
        let aspect = self.width as f32 / self.height as f32;
        let cx = center_merc.x as f32;
        let cy = center_merc.y as f32;

        let half_h = 1.0 / scale;

        // FOV set so that at pitch=0 the visible height matches orthographic half_h.
        // tan(fov/2) = half_h / cam_h → cam_h = half_h / tan(fov/2)
        let fov_y = std::f32::consts::FRAC_PI_3; // 60°
        let cam_h = half_h / (fov_y * 0.5).tan();

        let pitch_rad = self.pitch.to_radians() as f32;
        let bearing_rad = self.bearing.to_radians() as f32;
        let sin_b = bearing_rad.sin();
        let cos_b = bearing_rad.cos();

        // Camera "up" = the direction bearing points (north rotated clockwise by bearing).
        // At bearing=0: up = (0, -1, 0) = north in Mercator (Y=0 = north).
        // At bearing=90: up = (1, 0, 0) = east.
        let up = glam::Vec3::new(sin_b, -cos_b, 0.0);

        // Camera displaced "backward" from the look direction by pitch.
        // Look direction (forward) = (sin_b, -cos_b, 0) horizontally.
        // Backward = (-sin_b, cos_b, 0).
        // At bearing=0, pitch>0: eye displaced south (+Y) above center. ✓
        let eye = glam::Vec3::new(
            cx - sin_b * cam_h * pitch_rad.sin(),
            cy + cos_b * cam_h * pitch_rad.sin(),
            cam_h * pitch_rad.cos(),
        );
        let target = glam::Vec3::new(cx, cy, 0.0);

        let view = glam::Mat4::look_at_rh(eye, target, up);
        let far_mult = if pitch_rad > 0.01 {
            4.0 + 8.0 * pitch_rad.sin()
        } else {
            4.0
        };
        let near = cam_h * 0.1;
        let far = cam_h * far_mult;
        let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);

        // look_at_rh with up=(sin_b,-cos_b,0) makes camera_right = world(-cos_b,-sin_b,0),
        // which flips X when bearing=0. Correct with a -X scale so world east → screen right.
        let flip_x = glam::Mat4::from_diagonal(glam::Vec4::new(-1.0, 1.0, 1.0, 1.0));
        let view_proj = (flip_x * proj * view).to_cols_array();

        // Small-circle clipping center on unit sphere.
        let clip_center = x_planets_math::geo_to_unit_sphere(
            self.center.lat.to_radians(),
            self.center.lon.to_radians(),
        );
        // Clip angle: 85° from center — covers nearly a full hemisphere
        // while avoiding the oblique Mercator singularity at 90°.
        let cos_clip_angle = 85.0_f64.to_radians().cos() as f32;

        ViewportUniforms {
            view_proj,
            resolution: [
                self.width as f32,
                self.height as f32,
                1.0 / self.width as f32,
                1.0 / self.height as f32,
            ],
            camera: [cx, cy, self.zoom as f32, self.pitch as f32],
            clip_sphere: [
                clip_center.x as f32,
                clip_center.y as f32,
                clip_center.z as f32,
                cos_clip_angle,
            ],
        }
    }
}
