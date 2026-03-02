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
    #[cfg(test)]
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

    /// Returns the maximum number of concurrent tile loads.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
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
        self.queue.pop().inspect(|_req| {
            self.active_count += 1;
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

    #[test]
    fn test_tile_loader_clear_preserves_active_count() {
        let mut loader = TileLoader::new(4);

        // Enqueue 3 tiles and dequeue 2 (now 2 active, 1 in queue)
        loader.enqueue(TileRequest { coord: TileCoord::new(1, 0, 0), priority: 1.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(1, 1, 0), priority: 2.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(1, 0, 1), priority: 3.0 });
        assert!(loader.dequeue().is_some());
        assert!(loader.dequeue().is_some());
        assert_eq!(loader.active_count(), 2);
        assert_eq!(loader.pending_count(), 1);

        // Clear queue: drops pending but active stays
        loader.clear();
        assert_eq!(loader.pending_count(), 0);
        assert_eq!(loader.active_count(), 2);

        // Re-enqueue new tiles — these should be dequeued (2 active + 2 more = 4 max)
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 10, 10), priority: 0.5 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 11, 10), priority: 0.8 });
        assert!(loader.dequeue().is_some()); // active=3
        assert!(loader.dequeue().is_some()); // active=4
        assert!(loader.dequeue().is_none()); // at max
        assert_eq!(loader.active_count(), 4);

        // Complete old tasks → free slots
        loader.complete();
        loader.complete();
        assert_eq!(loader.active_count(), 2);
    }

    #[test]
    fn test_tile_loader_abort_and_requeue() {
        // Simulate: user zooms from z=3 to z=5, stale z=3 tiles should be dropped
        let mut loader = TileLoader::new(2);

        // Frame 1: enqueue z=3 tiles
        loader.enqueue(TileRequest { coord: TileCoord::new(3, 1, 1), priority: 1.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(3, 2, 1), priority: 2.0 });
        let req1 = loader.dequeue().unwrap(); // z=3 tile in-flight
        assert_eq!(req1.coord.z, 3);
        assert_eq!(loader.active_count(), 1);

        // Frame 2: user zooms to z=5, clear queue and re-enqueue
        loader.clear(); // drops the remaining z=3 tile in queue
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 10, 10), priority: 0.5 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 11, 10), priority: 0.8 });

        // Should dequeue z=5 tile (higher priority, current view)
        let req2 = loader.dequeue().unwrap();
        assert_eq!(req2.coord.z, 5);
        assert_eq!(loader.active_count(), 2); // z=3 in-flight + z=5 in-flight
        assert!(loader.dequeue().is_none()); // at max concurrent

        // z=3 task completes → free slot
        loader.complete();
        let req3 = loader.dequeue().unwrap();
        assert_eq!(req3.coord.z, 5); // now the second z=5 tile
    }

    #[test]
    fn test_clear_and_reenqueue_pattern() {
        // Reproduces the NativeApp frame loop: enqueue visible tiles, clear()
        // next frame drops un-dequeued tiles, then re-enqueue them fresh.
        // This verifies tiles are NOT lost after clear().
        let mut loader = TileLoader::new(2);

        // Frame 1: enqueue 5 tiles, only 2 dequeued (max_concurrent=2)
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 0, 0), priority: 1.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 1, 0), priority: 2.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 0, 1), priority: 3.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 1, 1), priority: 4.0 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 2, 0), priority: 5.0 });

        let d1 = loader.dequeue().unwrap(); // priority 1.0
        let d2 = loader.dequeue().unwrap(); // priority 2.0
        assert!(loader.dequeue().is_none()); // at max
        assert_eq!(d1.coord, TileCoord::new(5, 0, 0));
        assert_eq!(d2.coord, TileCoord::new(5, 1, 0));
        assert_eq!(loader.pending_count(), 3); // 3 still in queue
        assert_eq!(loader.active_count(), 2);

        // Frame 2: clear() drops the 3 un-dequeued tiles. One in-flight completes.
        loader.clear();
        assert_eq!(loader.pending_count(), 0);
        assert_eq!(loader.active_count(), 2); // in-flight preserved

        loader.complete(); // d1 done → active=1

        // Re-enqueue the 3 tiles that were dropped (with fresh priorities)
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 0, 1), priority: 1.5 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 1, 1), priority: 2.5 });
        loader.enqueue(TileRequest { coord: TileCoord::new(5, 2, 0), priority: 3.5 });

        // Now can dequeue 1 (active=1, max=2)
        let d3 = loader.dequeue().unwrap();
        assert_eq!(d3.coord, TileCoord::new(5, 0, 1)); // highest priority
        assert_eq!(loader.active_count(), 2);
        assert!(loader.dequeue().is_none()); // at max again

        // d2 completes → active=1
        loader.complete();
        let d4 = loader.dequeue().unwrap();
        assert_eq!(d4.coord, TileCoord::new(5, 1, 1));

        // Verify the last tile can still be dequeued
        loader.complete();
        let d5 = loader.dequeue().unwrap();
        assert_eq!(d5.coord, TileCoord::new(5, 2, 0));
    }
}
