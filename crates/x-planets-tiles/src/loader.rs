//! Async tile loading and priority queue.

use async_trait::async_trait;
use thiserror::Error;
use x_planets_math::TileCoord;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("Network error: {0}")]
    Network(String),
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Tile not found: {0}")]
    NotFound(TileCoord),
    #[error("HTTP {status}: {url}")]
    HttpError { status: u16, url: String },
}

/// Tile data source abstraction.
///
/// Implement this trait to provide tile data from different backends
/// (HTTP, local files, IndexedDB, etc.)
///
/// On native platforms the trait requires `Send + Sync` for multi-threaded use.
/// On wasm32 it is single-threaded so these bounds are relaxed.
#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
pub trait TileSource: Send + Sync {
    /// Fetch raw tile bytes for a given coordinate.
    async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, LoadError>;

    /// Build the URL/path for a tile coordinate.
    fn tile_url(&self, coord: &TileCoord) -> String;
}

/// Tile data source abstraction (wasm32 — single-threaded, no Send/Sync).
#[cfg(target_arch = "wasm32")]
#[async_trait(?Send)]
pub trait TileSource {
    /// Fetch raw tile bytes for a given coordinate.
    async fn fetch(&self, coord: TileCoord) -> Result<Vec<u8>, LoadError>;

    /// Build the URL/path for a tile coordinate.
    fn tile_url(&self, coord: &TileCoord) -> String;
}

/// URL template-based tile source for XYZ/TMS tile servers.
pub struct UrlTileSource {
    /// URL template with {z}, {x}, {y} placeholders.
    /// Example: "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    pub url_template: String,
    /// Whether to use TMS y-axis convention (flipped).
    pub tms: bool,
}

impl UrlTileSource {
    pub fn new(url_template: impl Into<String>) -> Self {
        Self {
            url_template: url_template.into(),
            tms: false,
        }
    }

    pub fn with_tms(mut self, tms: bool) -> Self {
        self.tms = tms;
        self
    }
}

impl UrlTileSource {
    fn build_url(&self, coord: &TileCoord) -> String {
        let y = if self.tms {
            (1 << coord.z) - 1 - coord.y
        } else {
            coord.y
        };

        self.url_template
            .replace("{z}", &coord.z.to_string())
            .replace("{x}", &coord.x.to_string())
            .replace("{y}", &y.to_string())
    }
}

/// Manages tile loading with a priority queue.
///
/// Tiles closer to the viewport center are loaded first.
pub struct TileLoader {
    queue: BinaryHeap<TileRequest>,
    max_concurrent: usize,
    active_count: usize,
}

/// A request to load a specific tile.
#[derive(Debug, Clone)]
pub struct TileRequest {
    pub coord: TileCoord,
    pub priority: f32, // Lower = higher priority (distance from center)
}

impl PartialEq for TileRequest {
    fn eq(&self, other: &Self) -> bool {
        self.coord == other.coord
    }
}

impl Eq for TileRequest {}

impl PartialOrd for TileRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TileRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: lower priority value = higher in the heap
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
    }
}

impl TileLoader {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            queue: BinaryHeap::new(),
            max_concurrent,
            active_count: 0,
        }
    }

    /// Enqueue a tile request.
    pub fn enqueue(&mut self, request: TileRequest) {
        self.queue.push(request);
    }

    /// Dequeue the highest priority request if capacity allows.
    pub fn dequeue(&mut self) -> Option<TileRequest> {
        if self.active_count >= self.max_concurrent {
            return None;
        }
        self.queue.pop().map(|req| {
            self.active_count += 1;
            req
        })
    }

    /// Mark a tile load as completed.
    pub fn complete(&mut self) {
        if self.active_count > 0 {
            self.active_count -= 1;
        }
    }

    /// Clear all pending requests.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Number of active (in-flight) requests.
    pub fn active_count(&self) -> usize {
        self.active_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_tile_source() {
        let source = UrlTileSource::new("https://tile.openstreetmap.org/{z}/{x}/{y}.png");
        let url = source.build_url(&TileCoord::new(2, 1, 1));
        assert_eq!(url, "https://tile.openstreetmap.org/2/1/1.png");
    }

    #[test]
    fn test_url_tile_source_tms() {
        let source =
            UrlTileSource::new("https://example.com/{z}/{x}/{y}.png").with_tms(true);
        let coord = TileCoord::new(2, 1, 1);
        let url = source.build_url(&coord);
        // TMS y = (1<<2) - 1 - 1 = 2
        assert_eq!(url, "https://example.com/2/1/2.png");
    }

    #[test]
    fn test_tile_loader_priority() {
        let mut loader = TileLoader::new(2);

        loader.enqueue(TileRequest {
            coord: TileCoord::new(1, 0, 0),
            priority: 10.0,
        });
        loader.enqueue(TileRequest {
            coord: TileCoord::new(1, 1, 0),
            priority: 1.0, // Higher priority (closer)
        });

        let first = loader.dequeue().unwrap();
        assert_eq!(first.coord, TileCoord::new(1, 1, 0)); // Closer tile first
    }

    #[test]
    fn test_tile_loader_max_concurrent() {
        let mut loader = TileLoader::new(1);
        loader.enqueue(TileRequest {
            coord: TileCoord::new(0, 0, 0),
            priority: 1.0,
        });
        loader.enqueue(TileRequest {
            coord: TileCoord::new(1, 0, 0),
            priority: 2.0,
        });

        assert!(loader.dequeue().is_some());
        assert!(loader.dequeue().is_none()); // Max concurrent reached

        loader.complete();
        assert!(loader.dequeue().is_some()); // Now can dequeue again
    }
}
