use std::{collections::HashMap, fmt::Display, fs, io, path::PathBuf};

use serde::{Deserialize, Serialize};

use super::hash_key;

/// The cache format this release reads and writes.
///
/// Version 2 hashes keys with SHA-256, which is stable across Rust releases,
/// and marks vectors computed from raw text. Files from earlier releases hold
/// vectors computed from JSON-quoted text, so they load as an empty cache.
pub const CACHE_FORMAT_VERSION: u32 = 2;

/// Why reading or writing a cache file failed.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// The file couldn't be read or written.
    #[error("cache file {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        #[source]
        source: io::Error,
    },
    /// The file isn't a cache.
    #[error("cache file {path} is not valid JSON: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// A map from a model-and-input hash to previously generated embeddings.
///
/// Cache keys include both the model name and the inputs, so the same input
/// can safely be embedded with several models. A cache can be cloned into a
/// [`JobQueue`](crate::jobs::job_queue::JobQueue) and persisted as JSON between
/// runs.
///
/// # Example
///
/// ```
/// use lvv::cache::cache_embeddings::Cache;
/// let mut cache = Cache::new();
/// cache.add_embedding(
///     "embed-model".into(),
///     vec!["first record".into()],
///     vec![vec![0.1, 0.2]],
/// );
/// assert_eq!(
///     cache.get_embedding("embed-model".into(), vec!["first record".into()]),
///     Some(&vec![vec![0.1, 0.2]]),
/// );
/// ```
pub struct Cache {
    /// The format version; see [`CACHE_FORMAT_VERSION`]. Files written before
    /// versions existed deserialize as `0`.
    #[serde(default)]
    pub version: u32,
    /// Stored embeddings indexed by the input hash.
    pub cache: HashMap<String, Vec<Vec<f32>>>,
}

impl Default for Cache {
    fn default() -> Self {
        Cache {
            version: CACHE_FORMAT_VERSION,
            cache: HashMap::new(),
        }
    }
}

impl Display for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cache with {} entries", self.cache.len())
    }
}

impl Cache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Cache::default()
    }

    fn key(model: &str, data: &[String]) -> String {
        let version = CACHE_FORMAT_VERSION.to_le_bytes();
        let count = (data.len() as u64).to_le_bytes();
        hash_key(
            [
                b"lvv-embedding-cache".as_slice(),
                version.as_slice(),
                model.as_bytes(),
                count.as_slice(),
            ]
            .into_iter()
            .chain(data.iter().map(|item| item.as_bytes())),
        )
    }

    /// Returns cached embeddings for `model` and `data`, if present.
    pub fn get_embedding(&self, model: String, data: Vec<String>) -> Option<&Vec<Vec<f32>>> {
        self.cache.get(&Self::key(&model, &data))
    }

    /// Stores embeddings unless the same model-and-data key already exists.
    pub fn add_embedding(&mut self, model: String, data: Vec<String>, embeddings: Vec<Vec<f32>>) {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.cache.entry(Self::key(&model, &data))
        {
            tracing::debug!(model, inputs = data.len(), "added embeddings to the cache");
            entry.insert(embeddings);
        }
    }

    /// Loads a cache from a JSON file.
    ///
    /// A file written in an older format loads as an empty cache, since its
    /// vectors don't match what this release embeds.
    ///
    /// # Errors
    ///
    /// Returns an error when the file can't be read or isn't a cache.
    pub fn from_json_file(file_name: &str) -> Result<Self, CacheError> {
        let path = PathBuf::from(file_name);
        let text = fs::read_to_string(&path).map_err(|source| CacheError::Io {
            path: path.clone(),
            source,
        })?;
        let cache: Cache =
            serde_json::from_str(&text).map_err(|source| CacheError::Json { path, source })?;
        if cache.version != CACHE_FORMAT_VERSION {
            tracing::warn!(
                file = file_name,
                found = cache.version,
                expected = CACHE_FORMAT_VERSION,
                "ignoring an embedding cache written in an older format"
            );
            return Ok(Cache::new());
        }
        Ok(cache)
    }

    /// Serializes the cache to a JSON file, replacing its contents.
    ///
    /// # Errors
    ///
    /// Returns an error when the file can't be written.
    pub fn to_json_file(&self, file_name: &str) -> Result<(), CacheError> {
        let path = PathBuf::from(file_name);
        let text = serde_json::to_string(self).map_err(|source| CacheError::Json {
            path: path.clone(),
            source,
        })?;
        fs::write(&path, text).map_err(|source| CacheError::Io { path, source })?;
        tracing::debug!(file = file_name, "embedding cache written");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, contents: &str) -> String {
        let path =
            std::env::temp_dir().join(format!("lvv-cache-test-{}-{name}", std::process::id()));
        fs::write(&path, contents).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn a_file_from_an_older_release_misses() {
        // Written by lvv 0.5: `DefaultHasher` keys and no version.
        let file = temp_file("old.json", r#"{"cache":{"1234567890":[[0.1,0.2]]}}"#);
        let cache = Cache::from_json_file(&file).unwrap();
        assert!(cache.cache.is_empty());
        assert_eq!(cache.version, CACHE_FORMAT_VERSION);
        assert_eq!(cache.get_embedding("m".into(), vec!["x".into()]), None);
        fs::remove_file(file).ok();
    }

    #[test]
    fn a_file_from_this_release_hits() {
        let mut cache = Cache::new();
        cache.add_embedding("m".into(), vec!["x".into()], vec![vec![1.0]]);
        let file = temp_file("new.json", "");
        cache.to_json_file(&file).unwrap();
        let loaded = Cache::from_json_file(&file).unwrap();
        assert_eq!(
            loaded.get_embedding("m".into(), vec!["x".into()]),
            Some(&vec![vec![1.0]])
        );
        fs::remove_file(file).ok();
    }

    #[test]
    fn keys_depend_on_model_and_inputs() {
        let mut cache = Cache::new();
        cache.add_embedding("m".into(), vec!["x".into()], vec![vec![1.0]]);
        assert_eq!(cache.get_embedding("other".into(), vec!["x".into()]), None);
        assert_eq!(cache.get_embedding("m".into(), vec!["y".into()]), None);
    }
}
