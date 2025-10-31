use std::{
    collections::HashMap,
    fs::{OpenOptions, read_to_string},
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
};

use anyhow::{Context, Ok};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Cache {
    pub cache: HashMap<u64, Vec<Vec<f32>>>,
}

impl Cache {
    pub fn new() -> Self {
        Cache::default()
    }
    pub fn get_embedding(&self, model: String, data: Vec<String>) -> Option<&Vec<Vec<f32>>> {
        let mut hasher = DefaultHasher::new();
        let hash = {
            model.hash(&mut hasher);
            data.hash(&mut hasher);
            hasher.finish()
        };
        self.cache.get(&hash)
    }
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
    pub fn from_json_file(file_name: &str) -> anyhow::Result<Self> {
        let file_text = read_to_string(file_name).context("Couldn't open file for cache")?;
        let data: Cache = serde_json::from_str(&file_text).context("Error deserializando")?;
        Ok(data)
    }
    pub fn to_json_file(&self, file_name: &str) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("database.json")
            .context("Couldnt't open dump file")?;
        let string = serde_json::to_string(self).context("Error serializando")?;
        file.write_all(string.as_bytes())
            .context("Error escribiendo archivo")?;
        Ok(())
    }
}
