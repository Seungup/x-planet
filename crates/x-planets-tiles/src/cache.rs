//! LRU tile cache with configurable memory limits.

use std::collections::HashMap;
use x_planets_math::TileCoord;

/// In-memory LRU cache for decoded tiles.
///
/// Tracks tiles by TileCoord with a maximum entry limit.
/// Uses a simple clock-based approximation of LRU.
pub struct TileCache<T> {
    entries: HashMap<TileCoord, CacheEntry<T>>,
    max_entries: usize,
    access_counter: u64,
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
        }
    }

    /// Get a cached tile, updating its access time.
    pub fn get(&mut self, coord: &TileCoord) -> Option<&T> {
        self.access_counter += 1;
        let counter = self.access_counter;
        self.entries.get_mut(coord).map(|entry| {
            entry.last_access = counter;
            &entry.value
        })
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
}
