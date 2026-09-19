use std::{
    collections::HashMap,
    io,
    path::{Path, PathBuf},
    pin::Pin,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::error::CompletionCacheError;
use crate::{
    backend::{BackendIdentity, OutputSchema},
    cache::hash_key,
};

/// The line format this release reads and writes.
const FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Line {
    v: u32,
    key: String,
    output: Value,
}

/// Completed outputs, one JSON line each, keyed by everything that affects
/// the output: backend, model, prompt, schema and input.
///
/// Lines are only ever appended, so an interrupted run leaves at most one
/// broken last line, which the next run skips.
pub(crate) struct CompletionCache {
    path: PathBuf,
    entries: HashMap<String, Value>,
    writer: Pin<Box<dyn AsyncWrite + Send>>,
}

impl CompletionCache {
    /// Loads `path` if it exists, and opens it for appending.
    pub(crate) async fn open(path: &Path) -> Result<Self, CompletionCacheError> {
        let error = |source: io::Error| CompletionCacheError {
            path: path.to_path_buf(),
            source,
        };
        let (entries, ends_mid_line) = match tokio::fs::read_to_string(path).await {
            Ok(text) => (
                parse_lines(&text, path),
                !text.is_empty() && !text.ends_with('\n'),
            ),
            Err(err) if err.kind() == io::ErrorKind::NotFound => (HashMap::new(), false),
            Err(err) => return Err(error(err)),
        };
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(error)?;
        if ends_mid_line {
            // Finish the broken line so the next entry starts on its own.
            file.write_all(b"\n").await.map_err(error)?;
        }
        Ok(CompletionCache {
            path: path.to_path_buf(),
            entries,
            writer: Box::pin(file),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_writer(writer: impl AsyncWrite + Send + 'static) -> Self {
        CompletionCache {
            path: PathBuf::from("<test>"),
            entries: HashMap::new(),
            writer: Box::pin(writer),
        }
    }

    pub(crate) fn key(
        identity: &BackendIdentity,
        prompt: &str,
        schema: Option<&OutputSchema>,
        input: &str,
    ) -> String {
        let schema = schema
            .map(|s| format!("{}\u{0}{}", s.name, s.schema))
            .unwrap_or_default();
        let version = FORMAT_VERSION.to_le_bytes();
        hash_key([
            b"lvv-completion-cache".as_slice(),
            version.as_slice(),
            identity.kind.as_bytes(),
            identity.base_url.as_bytes(),
            identity.model.as_bytes(),
            prompt.as_bytes(),
            schema.as_bytes(),
            input.as_bytes(),
        ])
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    /// Appends one entry and flushes it, so it survives a crash right after.
    pub(crate) async fn insert(
        &mut self,
        key: String,
        output: Value,
    ) -> Result<(), CompletionCacheError> {
        let line = Line {
            v: FORMAT_VERSION,
            key,
            output,
        };
        let result = async {
            let mut bytes = serde_json::to_vec(&line).map_err(io::Error::other)?;
            bytes.push(b'\n');
            self.writer.write_all(&bytes).await?;
            self.writer.flush().await
        }
        .await;
        result.map_err(|source| CompletionCacheError {
            path: self.path.clone(),
            source,
        })?;
        self.entries.insert(line.key, line.output);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

fn parse_lines(text: &str, path: &Path) -> HashMap<String, Value> {
    let mut entries = HashMap::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Line>(line) {
            Ok(entry) if entry.v == FORMAT_VERSION => {
                entries.insert(entry.key, entry.output);
            }
            Ok(entry) => tracing::warn!(
                file = %path.display(),
                line = number + 1,
                version = entry.v,
                "skipping a completion cache entry in an unknown format"
            ),
            Err(err) => tracing::warn!(
                file = %path.display(),
                line = number + 1,
                error = %err,
                "skipping an unreadable completion cache line"
            ),
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lvv-completion-cache-{}-{name}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[tokio::test]
    async fn entries_survive_reopening() {
        let path = temp_path("reopen");
        let mut cache = CompletionCache::open(&path).await.unwrap();
        cache.insert("a".into(), Value::from("one")).await.unwrap();
        cache.insert("b".into(), Value::from("two")).await.unwrap();
        drop(cache);
        let cache = CompletionCache::open(&path).await.unwrap();
        assert_eq!(cache.get("a"), Some(&Value::from("one")));
        assert_eq!(cache.get("b"), Some(&Value::from("two")));
        std::fs::remove_file(path).ok();
    }

    #[tokio::test]
    async fn a_truncated_last_line_is_skipped_and_not_glued_to_the_next() {
        let path = temp_path("truncated");
        std::fs::write(
            &path,
            "{\"v\":1,\"key\":\"a\",\"output\":\"one\"}\n{\"v\":1,\"key\":\"b\",\"outp",
        )
        .unwrap();
        let mut cache = CompletionCache::open(&path).await.unwrap();
        assert_eq!(cache.len(), 1);
        cache
            .insert("c".into(), Value::from("three"))
            .await
            .unwrap();
        drop(cache);
        let cache = CompletionCache::open(&path).await.unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("c"), Some(&Value::from("three")));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn keys_change_with_every_part() {
        let identity = BackendIdentity {
            kind: "ollama".into(),
            base_url: "http://x".into(),
            model: "m".into(),
        };
        let base = CompletionCache::key(&identity, "p", None, "i");
        assert_eq!(base, CompletionCache::key(&identity, "p", None, "i"));
        assert_ne!(base, CompletionCache::key(&identity, "p2", None, "i"));
        assert_ne!(base, CompletionCache::key(&identity, "p", None, "i2"));
        let other_model = BackendIdentity {
            model: "m2".into(),
            ..identity.clone()
        };
        assert_ne!(base, CompletionCache::key(&other_model, "p", None, "i"));
        let schema = OutputSchema {
            name: "T".into(),
            schema: serde_json::json!({}),
        };
        assert_ne!(
            base,
            CompletionCache::key(&identity, "p", Some(&schema), "i")
        );
    }
}
