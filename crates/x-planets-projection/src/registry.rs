//! Projection plugin registry for dynamic registration and lookup.

use crate::ProjectionPlugin;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry for projection plugins.
///
/// Allows registering and retrieving projections by name.
/// Built-in projections are registered automatically on creation.
pub struct ProjectionRegistry {
    plugins: HashMap<String, Arc<dyn ProjectionPlugin>>,
}

impl ProjectionRegistry {
    /// Create a new registry with built-in projections pre-registered.
    pub fn new() -> Self {
        let mut registry = Self {
            plugins: HashMap::new(),
        };

        // Register built-in projections
        registry.register(Arc::new(crate::builtins::Mercator));
        registry.register(Arc::new(crate::builtins::Globe));

        registry
    }

    /// Create an empty registry (no built-ins).
    pub fn empty() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a projection plugin.
    pub fn register(&mut self, plugin: Arc<dyn ProjectionPlugin>) {
        let name = plugin.name().to_string();
        log::info!("Registered projection: {}", name);
        self.plugins.insert(name, plugin);
    }

    /// Get a projection by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn ProjectionPlugin>> {
        self.plugins.get(name).cloned()
    }

    /// Get a projection by EPSG code.
    pub fn get_by_epsg(&self, code: &str) -> Option<Arc<dyn ProjectionPlugin>> {
        self.plugins
            .values()
            .find(|p| p.epsg_code() == Some(code))
            .cloned()
    }

    /// List all registered projection names.
    pub fn list(&self) -> Vec<&str> {
        self.plugins.keys().map(|s| s.as_str()).collect()
    }

    /// Look up the rendering mode for a projection by name.
    ///
    /// Returns the `ProjectionMode` declared by the plugin, or
    /// the default `Mercator` if the name is not registered.
    pub fn rendering_mode_for(&self, name: &str) -> x_planets_math::ProjectionMode {
        self.plugins
            .get(name)
            .map(|p| p.rendering_mode())
            .unwrap_or_default()
    }

    /// Number of registered projections.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for ProjectionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A custom projection loaded from a WGSL shader file.
///
/// Allows users to add projections at runtime by providing shader code.
pub struct CustomProjection {
    name: String,
    epsg: Option<String>,
    shader: String,
    function_name: String,
}

impl CustomProjection {
    pub fn new(name: impl Into<String>, shader: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            epsg: None,
            shader: shader.into(),
            function_name: "project".to_string(),
        }
    }

    pub fn with_epsg(mut self, code: impl Into<String>) -> Self {
        self.epsg = Some(code.into());
        self
    }

    pub fn with_function_name(mut self, name: impl Into<String>) -> Self {
        self.function_name = name.into();
        self
    }
}

impl ProjectionPlugin for CustomProjection {
    fn name(&self) -> &str {
        &self.name
    }

    fn epsg_code(&self) -> Option<&str> {
        self.epsg.as_deref()
    }

    fn shader_source(&self) -> &str {
        &self.shader
    }

    fn shader_function_name(&self) -> &str {
        &self.function_name
    }

    fn project_cpu(&self, world_pos: glam::DVec3) -> glam::DVec3 {
        // CPU fallback not available for custom projections by default.
        // Users should override if needed.
        log::warn!(
            "CPU projection fallback not implemented for custom projection '{}'",
            self.name
        );
        world_pos
    }

    fn unproject_cpu(&self, projected: glam::DVec3) -> glam::DVec3 {
        log::warn!(
            "CPU un-projection fallback not implemented for custom projection '{}'",
            self.name
        );
        projected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_builtins() {
        let registry = ProjectionRegistry::new();
        assert!(registry.get("Web Mercator").is_some());
        assert!(registry.get("Globe").is_some());
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_registry_by_epsg() {
        let registry = ProjectionRegistry::new();
        let proj = registry.get_by_epsg("EPSG:3857").unwrap();
        assert_eq!(proj.name(), "Web Mercator");
    }

    #[test]
    fn test_custom_projection() {
        let mut registry = ProjectionRegistry::new();
        let custom = CustomProjection::new("My Projection", "fn project(p: vec3<f32>) -> vec3<f32> { return p; }");
        registry.register(Arc::new(custom));
        assert!(registry.get("My Projection").is_some());
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_rendering_mode_for_builtins() {
        let registry = ProjectionRegistry::new();

        assert_eq!(
            registry.rendering_mode_for("Web Mercator"),
            x_planets_math::ProjectionMode::Mercator,
        );
        assert_eq!(
            registry.rendering_mode_for("Globe"),
            x_planets_math::ProjectionMode::Globe,
        );
    }

    #[test]
    fn test_rendering_mode_for_unknown_defaults_to_mercator() {
        let registry = ProjectionRegistry::new();
        assert_eq!(
            registry.rendering_mode_for("NonExistent"),
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    #[test]
    fn test_custom_projection_default_rendering_mode() {
        let mut registry = ProjectionRegistry::new();
        let custom = CustomProjection::new(
            "Stereographic",
            "fn project(p: vec3<f32>) -> vec3<f32> { return p; }",
        );
        registry.register(Arc::new(custom));

        // CustomProjection uses the default rendering_mode() = Mercator
        assert_eq!(
            registry.rendering_mode_for("Stereographic"),
            x_planets_math::ProjectionMode::Mercator,
        );
    }

    #[test]
    fn test_registry_empty() {
        let registry = ProjectionRegistry::empty();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get("Web Mercator").is_none());
    }

    #[test]
    fn test_registry_list() {
        let registry = ProjectionRegistry::new();
        let names = registry.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Web Mercator"));
        assert!(names.contains(&"Globe"));
    }

    #[test]
    fn test_registry_overwrite_same_name() {
        let mut registry = ProjectionRegistry::new();
        let custom = CustomProjection::new("Web Mercator", "// overridden shader");
        registry.register(Arc::new(custom));

        // Should still have 2 entries (overwritten, not duplicated)
        assert_eq!(registry.len(), 2);
        let proj = registry.get("Web Mercator").unwrap();
        assert_eq!(proj.shader_source(), "// overridden shader");
    }

    #[test]
    fn test_custom_projection_with_epsg() {
        let custom = CustomProjection::new("Test", "// shader")
            .with_epsg("EPSG:9999");
        assert_eq!(custom.epsg_code(), Some("EPSG:9999"));
    }

    #[test]
    fn test_custom_projection_with_function_name() {
        let custom = CustomProjection::new("Test", "// shader")
            .with_function_name("my_project");
        assert_eq!(custom.shader_function_name(), "my_project");
    }

    #[test]
    fn test_custom_projection_cpu_fallback_returns_input() {
        let custom = CustomProjection::new("Test", "// shader");
        let input = glam::DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(custom.project_cpu(input), input);
        assert_eq!(custom.unproject_cpu(input), input);
    }

    #[test]
    fn test_compose_shaders_base_only() {
        let result = crate::compose_shaders("fn base() {}", &[]);
        assert!(result.contains("fn base() {}"));
        assert!(result.contains("// === Base Projection ==="));
        assert!(!result.contains("Transform"));
    }

    #[test]
    fn test_compose_shaders_with_transforms() {
        let result = crate::compose_shaders(
            "fn base() {}",
            &["fn t0() {}", "fn t1() {}"],
        );
        assert!(result.contains("// === Base Projection ==="));
        assert!(result.contains("fn base() {}"));
        assert!(result.contains("// === Transform 0 ==="));
        assert!(result.contains("fn t0() {}"));
        assert!(result.contains("// === Transform 1 ==="));
        assert!(result.contains("fn t1() {}"));
    }

    #[test]
    fn test_mercator_epsg_code() {
        let merc = crate::builtins::Mercator;
        assert_eq!(merc.epsg_code(), Some("EPSG:3857"));
    }

    #[test]
    fn test_globe_epsg_code() {
        let globe = crate::builtins::Globe;
        assert_eq!(globe.epsg_code(), Some("EPSG:4326"));
    }

    #[test]
    fn test_mercator_latitude_range() {
        let merc = crate::builtins::Mercator;
        let (min, max) = merc.latitude_range();
        assert!((min - (-85.0511)).abs() < 0.001);
        assert!((max - 85.0511).abs() < 0.001);
    }
}
