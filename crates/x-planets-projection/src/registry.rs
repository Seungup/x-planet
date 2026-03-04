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
}
