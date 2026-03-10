//! LRU tile cache with configurable memory limits.

use std::collections::HashMap;
use x_planets_math::TileCoord;

/// Cache statistics for observability.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits since creation/reset.
    pub hits: u64,
    /// Number of cache misses since creation/reset.
    pub misses: u64,
    /// Current number of entries in the cache.
    pub entries: usize,
    /// Maximum number of entries allowed.
    pub max_entries: usize,
    /// Number of evictions since creation/reset.
    pub evictions: u64,
}

impl CacheStats {
    /// Hit rate as a percentage (0.0–100.0). Returns 0.0 if no lookups.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { (self.hits as f64 / total as f64) * 100.0 }
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "entries={}/{} hits={} misses={} hit_rate={:.1}% evictions={}",
            self.entries, self.max_entries, self.hits, self.misses,
            self.hit_rate(), self.evictions,
        )
    }
}

/// In-memory LRU cache for decoded tiles.
///
/// Tracks tiles by TileCoord with a maximum entry limit.
/// Uses a simple clock-based approximation of LRU.
pub struct TileCache<T> {
    entries: HashMap<TileCoord, CacheEntry<T>>,
    max_entries: usize,
    access_counter: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

struct CacheEntry<T> {
    value: T,
    last_access: u64,
}

impl<T> TileCache<T> {
    /// Create a new tile cache with the given max entry count.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_entries),
            max_entries,
            access_counter: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get a cached tile, updating its access time.
    pub fn get(&mut self, coord: &TileCoord) -> Option<&T> {
        self.access_counter += 1;
        let counter = self.access_counter;
        if let Some(entry) = self.entries.get_mut(coord) {
            entry.last_access = counter;
            self.hits += 1;
            Some(&entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Peek at a cached tile **without** updating its access time.
    /// Use during rendering when you need read-only access and
    /// don't want to perturb LRU ordering.
    pub fn peek(&self, coord: &TileCoord) -> Option<&T> {
        self.entries.get(coord).map(|entry| &entry.value)
    }

    /// Iterate over all (coord, value) pairs in the cache.
    pub fn iter(&self) -> impl Iterator<Item = (&TileCoord, &T)> {
        self.entries.iter().map(|(k, v)| (k, &v.value))
    }

    /// Iterate over all cached tile coordinates.
    pub fn keys(&self) -> impl Iterator<Item = &TileCoord> {
        self.entries.keys()
    }

    /// Insert a tile into the cache. Evicts the least recently used if full.
    pub fn insert(&mut self, coord: TileCoord, value: T) {
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&coord) {
            self.evict_one();
        }

        self.access_counter += 1;
        self.entries.insert(
            coord,
            CacheEntry {
                value,
                last_access: self.access_counter,
            },
        );
    }

    /// Check if a tile is in the cache.
    pub fn contains(&self, coord: &TileCoord) -> bool {
        self.entries.contains_key(coord)
    }

    /// Remove a specific tile from cache.
    pub fn remove(&mut self, coord: &TileCoord) -> Option<T> {
        self.entries.remove(coord).map(|e| e.value)
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the least recently used entry.
    fn evict_one(&mut self) {
        if let Some((&lru_key, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
        {
            self.entries.remove(&lru_key);
            self.evictions += 1;
        }
    }

    /// Get current cache statistics for observability.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len(),
            max_entries: self.max_entries,
            evictions: self.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = TileCache::new(10);
        let coord = TileCoord::new(1, 0, 0);
        cache.insert(coord, "tile_data");
        assert_eq!(cache.get(&coord), Some(&"tile_data"));
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = TileCache::new(2);
        let c1 = TileCoord::new(1, 0, 0);
        let c2 = TileCoord::new(1, 1, 0);
        let c3 = TileCoord::new(1, 0, 1);

        cache.insert(c1, "a");
        cache.insert(c2, "b");

        // Access c1 to make c2 the LRU
        cache.get(&c1);

        // Insert c3 should evict c2 (least recently used)
        cache.insert(c3, "c");

        assert!(cache.contains(&c1));
        assert!(!cache.contains(&c2));
        assert!(cache.contains(&c3));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = TileCache::new(10);
        cache.insert(TileCoord::new(0, 0, 0), 1);
        cache.insert(TileCoord::new(1, 0, 0), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_peek_no_lru_update() {
        let mut cache = TileCache::new(2);
        let c1 = TileCoord::new(1, 0, 0);
        let c2 = TileCoord::new(1, 1, 0);
        let c3 = TileCoord::new(1, 0, 1);

        cache.insert(c1, "a");
        cache.insert(c2, "b");

        // Peek at c1 — should NOT update LRU.
        assert_eq!(cache.peek(&c1), Some(&"a"));

        // Insert c3 — should evict c1 (oldest by insert order, peek didn't update).
        cache.insert(c3, "c");

        // c1 was evicted because peek doesn't bump LRU.
        assert!(!cache.contains(&c1));
        assert!(cache.contains(&c2));
        assert!(cache.contains(&c3));
    }

    #[test]
    fn test_cache_iter_and_keys() {
        let mut cache = TileCache::new(10);
        let c1 = TileCoord::new(1, 0, 0);
        let c2 = TileCoord::new(1, 1, 0);
        cache.insert(c1, "a");
        cache.insert(c2, "b");

        let keys: Vec<&TileCoord> = cache.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&&c1));
        assert!(keys.contains(&&c2));

        let items: Vec<(&TileCoord, &&str)> = cache.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = TileCache::new(10);
        let c1 = TileCoord::new(1, 0, 0);
        cache.insert(c1, "a");
        assert!(cache.contains(&c1));

        let removed = cache.remove(&c1);
        assert_eq!(removed, Some("a"));
        assert!(!cache.contains(&c1));
        assert_eq!(cache.len(), 0);

        // Remove non-existent returns None
        assert!(cache.remove(&c1).is_none());
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = TileCache::new(10);
        let c1 = TileCoord::new(1, 0, 0);
        let c2 = TileCoord::new(1, 1, 0);

        assert!(!cache.contains(&c1));
        cache.insert(c1, "a");
        assert!(cache.contains(&c1));
        assert!(!cache.contains(&c2));
    }

    #[test]
    fn test_cache_len_and_is_empty() {
        let mut cache = TileCache::new(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        cache.insert(TileCoord::new(0, 0, 0), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);

        cache.insert(TileCoord::new(1, 0, 0), 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_duplicate_insert_updates_value() {
        let mut cache = TileCache::new(10);
        let coord = TileCoord::new(1, 0, 0);
        cache.insert(coord, "old");
        cache.insert(coord, "new");

        assert_eq!(cache.len(), 1); // No duplicate
        assert_eq!(cache.get(&coord), Some(&"new")); // Updated value
    }

    #[test]
    fn test_cache_capacity_one() {
        let mut cache = TileCache::new(1);
        let c1 = TileCoord::new(1, 0, 0);
        let c2 = TileCoord::new(1, 1, 0);

        cache.insert(c1, "a");
        assert_eq!(cache.len(), 1);

        cache.insert(c2, "b");
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(&c1)); // Evicted
        assert!(cache.contains(&c2));
    }

    #[test]
    fn test_cache_get_nonexistent() {
        let mut cache: TileCache<&str> = TileCache::new(10);
        assert!(cache.get(&TileCoord::new(0, 0, 0)).is_none());
    }

    #[test]
    fn test_cache_clear_then_get() {
        let mut cache = TileCache::new(10);
        let coord = TileCoord::new(0, 0, 0);
        cache.insert(coord, 42);
        cache.clear();
        assert!(cache.get(&coord).is_none());
        assert!(cache.is_empty());
    }
}
