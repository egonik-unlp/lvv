// TODO: reemplazar anyhow con thiserror
use std::{
    collections::HashMap,
    fmt::Display,
    fs::{OpenOptions, read_to_string},
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
};

use anyhow::{Context, Ok};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
/// A map from a model-and-input hash to previously generated embeddings.
///
/// Cache keys include both the model name and serialized input records, so the
/// same input can safely be embedded with several models. A cache can be
/// cloned into a [`JobQueue`](crate::jobs::job_queue::JobQueue) and persisted
/// as JSON between runs.
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
    /// Stored embeddings indexed by the deterministic input hash.
    pub cache: HashMap<u64, Vec<Vec<f32>>>,
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
    /// Returns cached embeddings for `model` and `data`, if present.
    pub fn get_embedding(&self, model: String, data: Vec<String>) -> Option<&Vec<Vec<f32>>> {
        let mut hasher = DefaultHasher::new();
        let hash = {
            model.hash(&mut hasher);
            data.hash(&mut hasher);
            hasher.finish()
        };
        self.cache.get(&hash)
    }
    /// Stores embeddings unless the same model-and-data key already exists.
    pub fn add_embedding(&mut self, model: String, data: Vec<String>, embeddings: Vec<Vec<f32>>) {
        let mut hasher = DefaultHasher::new();
        let hash = {
            model.hash(&mut hasher);
            data.hash(&mut hasher);
            hasher.finish()
        };
        if !self.cache.keys().any(|key| key.eq(&hash)) {
            println!("Added embedding with model {} to cache", model);
            self.cache.insert(hash, embeddings);
        }
    }
    /// Loads a cache from a JSON file.
    pub fn from_json_file(file_name: &str) -> anyhow::Result<Self> {
        let file_text = read_to_string(file_name).context("Couldn't open file for cache")?;
        let data: Cache = serde_json::from_str(&file_text).context("Error deserializando")?;
        Ok(data)
    }
    /// Serializes the cache to a JSON file, replacing its contents.
    pub fn to_json_file(&self, file_name: &str) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_name)
            .context("Couldnt't open dump file")?;
        let string = serde_json::to_string(self).context("Error serializando")?;
        file.write_all(string.as_bytes())
            .context("Error escribiendo archivo")?;
        println!("Cache written to {file_name}");
        Ok(())
    }
}
