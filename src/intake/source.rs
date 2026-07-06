//! Data-origin connectors that produce embeddable [`DataSet`]s.
//!
//! [`Source`] decouples *where the data comes from* from the rest of the
//! pipeline (embed → sink). A connector reads its origin and yields one or more
//! `DataSet<Value>` batches. `serde_json::Value` is the canonical row type
//! because SQL/HTTP/file origins don't know their schema at compile time; a
//! caller with a strongly-typed row can still build `DataSet<MyType>` by hand.

use crate::intake::dataset::DataSet;
use anyhow::Context;
use async_trait::async_trait;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// A data origin that can be read into embeddable [`DataSet`]s.
///
/// Implementors map their origin's records to JSON objects and return them
/// chunked into one `DataSet<Value>` per batch, so large origins don't have to
/// be materialised in memory all at once.
#[async_trait]
pub trait Source: Send + Sync {
    /// Read the whole origin, returning one [`DataSet`] per batch.
    async fn fetch(&self) -> anyhow::Result<Vec<DataSet<Value>>>;
}

/// Chunk rows into `DataSet`s: one per `batch_size` rows (0 = a single set).
fn chunk_into_datasets(
    rows: Vec<Value>,
    filename: &str,
    identifier: &str,
    batch_size: usize,
) -> Vec<DataSet<Value>> {
    if batch_size == 0 || rows.len() <= batch_size {
        return vec![DataSet::new(filename, identifier, rows)];
    }
    rows.chunks(batch_size)
        .enumerate()
        .map(|(i, chunk)| DataSet::new(filename, format!("{identifier}_{i}"), chunk.to_vec()))
        .collect()
}

/// Supported flat-file encodings for [`FileSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Comma-separated values with a header row. Every field is read as text.
    Csv,
    /// A single JSON array of values.
    Json,
    /// JSON Lines / NDJSON: one JSON value per line.
    Jsonl,
}

impl FileFormat {
    fn from_path(path: &Path) -> anyhow::Result<Self> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("csv") => Ok(FileFormat::Csv),
            Some("json") => Ok(FileFormat::Json),
            Some("jsonl") | Some("ndjson") => Ok(FileFormat::Jsonl),
            other => anyhow::bail!(
                "cannot infer file format from extension {other:?}; \
                 set it explicitly with FileSource::with_format"
            ),
        }
    }
}

/// Reads CSV, JSON (array) and JSONL flat files into `DataSet<Value>`.
#[derive(Debug, Clone)]
pub struct FileSource {
    path: PathBuf,
    identifier: String,
    format: FileFormat,
    batch_size: usize,
}

impl FileSource {
    /// Build a source over `path`, inferring the format from its extension.
    pub fn new(path: impl Into<PathBuf>, identifier: impl Into<String>) -> anyhow::Result<Self> {
        let path = path.into();
        let format = FileFormat::from_path(&path)?;
        Ok(Self {
            path,
            identifier: identifier.into(),
            format,
            batch_size: 0,
        })
    }

    /// Override the auto-detected format.
    pub fn with_format(mut self, format: FileFormat) -> Self {
        self.format = format;
        self
    }

    /// Emit one `DataSet` per `batch_size` rows (0 = a single `DataSet`).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn read_rows(&self) -> anyhow::Result<Vec<Value>> {
        match self.format {
            FileFormat::Csv => {
                let mut reader = csv::Reader::from_path(&self.path)
                    .with_context(|| format!("opening CSV {}", self.path.display()))?;
                let mut rows = Vec::new();
                for (i, rec) in reader
                    .deserialize::<std::collections::BTreeMap<String, String>>()
                    .enumerate()
                {
                    // Report the offending record instead of silently truncating.
                    let rec = rec.with_context(|| {
                        format!("parsing CSV record {} in {}", i + 1, self.path.display())
                    })?;
                    let map = rec
                        .into_iter()
                        .map(|(k, v)| (k, Value::String(v)))
                        .collect();
                    rows.push(Value::Object(map));
                }
                Ok(rows)
            }
            FileFormat::Json => {
                let text = std::fs::read_to_string(&self.path)
                    .with_context(|| format!("reading {}", self.path.display()))?;
                let value: Value = serde_json::from_str(&text)
                    .with_context(|| format!("parsing JSON {}", self.path.display()))?;
                match value {
                    Value::Array(items) => Ok(items),
                    other => Ok(vec![other]),
                }
            }
            FileFormat::Jsonl => {
                let text = std::fs::read_to_string(&self.path)
                    .with_context(|| format!("reading {}", self.path.display()))?;
                let mut rows = Vec::new();
                for (i, line) in text.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(line).with_context(|| {
                        format!("parsing JSONL line {} in {}", i + 1, self.path.display())
                    })?;
                    rows.push(value);
                }
                Ok(rows)
            }
        }
    }
}

#[async_trait]
impl Source for FileSource {
    async fn fetch(&self) -> anyhow::Result<Vec<DataSet<Value>>> {
        let rows = self.read_rows()?;
        let filename = self.path.to_string_lossy().into_owned();
        Ok(chunk_into_datasets(
            rows,
            &filename,
            &self.identifier,
            self.batch_size,
        ))
    }
}

#[cfg(feature = "postgres")]
pub use postgres_source::PostgresSource;

#[cfg(feature = "postgres")]
mod postgres_source {
    use super::*;
    use tokio_postgres::{NoTls, types::Type};

    /// Reads rows from PostgreSQL via a `SELECT` into `DataSet<Value>`.
    ///
    /// Each row becomes a JSON object keyed by column name; the whole object is
    /// what the pipeline embeds and stores as payload. Common column types are
    /// mapped to their JSON equivalents, anything else falls back to text.
    #[derive(Debug, Clone)]
    pub struct PostgresSource {
        conn_str: String,
        query: String,
        identifier: String,
        batch_size: usize,
    }

    impl PostgresSource {
        pub fn new(
            conn_str: impl Into<String>,
            query: impl Into<String>,
            identifier: impl Into<String>,
        ) -> Self {
            Self {
                conn_str: conn_str.into(),
                query: query.into(),
                identifier: identifier.into(),
                batch_size: 0,
            }
        }

        pub fn with_batch_size(mut self, batch_size: usize) -> Self {
            self.batch_size = batch_size;
            self
        }
    }

    #[async_trait]
    impl Source for PostgresSource {
        async fn fetch(&self) -> anyhow::Result<Vec<DataSet<Value>>> {
            let (client, connection) = tokio_postgres::connect(&self.conn_str, NoTls)
                .await
                .context("connecting to PostgreSQL source")?;
            // Drive the connection on a task; it resolves once `client` drops.
            let handle = tokio::spawn(async move {
                if let Err(e) = connection.await {
                    eprintln!("postgres source connection error: {e}");
                }
            });
            let rows = client
                .query(&self.query, &[])
                .await
                .context("running source query");
            drop(client);
            let _ = handle.await;
            let json_rows = rows?.iter().map(row_to_json).collect::<Vec<_>>();
            Ok(chunk_into_datasets(
                json_rows,
                &self.query,
                &self.identifier,
                self.batch_size,
            ))
        }
    }

    fn row_to_json(row: &tokio_postgres::Row) -> Value {
        let mut map = serde_json::Map::new();
        for (i, col) in row.columns().iter().enumerate() {
            map.insert(col.name().to_string(), column_to_json(row, i, col.type_()));
        }
        Value::Object(map)
    }

    fn column_to_json(row: &tokio_postgres::Row, i: usize, ty: &Type) -> Value {
        use serde_json::json;
        match *ty {
            Type::BOOL => row.try_get::<_, Option<bool>>(i).ok().flatten().map(|v| json!(v)),
            Type::INT2 => row.try_get::<_, Option<i16>>(i).ok().flatten().map(|v| json!(v)),
            Type::INT4 => row.try_get::<_, Option<i32>>(i).ok().flatten().map(|v| json!(v)),
            Type::INT8 => row.try_get::<_, Option<i64>>(i).ok().flatten().map(|v| json!(v)),
            Type::FLOAT4 => row.try_get::<_, Option<f32>>(i).ok().flatten().map(|v| json!(v)),
            Type::FLOAT8 => row.try_get::<_, Option<f64>>(i).ok().flatten().map(|v| json!(v)),
            Type::JSON | Type::JSONB => row.try_get::<_, Option<Value>>(i).ok().flatten(),
            // varchar/text/uuid/timestamp/... — read as text.
            _ => row
                .try_get::<_, Option<String>>(i)
                .ok()
                .flatten()
                .map(Value::String),
        }
        .unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("lvv_src_{}_{}", std::process::id(), name));
        p
    }

    #[tokio::test]
    async fn jsonl_maps_one_item_per_line() {
        let path = tmp("a.jsonl");
        std::fs::write(&path, "{\"id\":1,\"t\":\"x\"}\n\n{\"id\":2,\"t\":\"y\"}\n").unwrap();
        let sets = FileSource::new(&path, "things").unwrap().fetch().await.unwrap();
        assert_eq!(sets.len(), 1);
        let data = sets[0].data.as_ref().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], serde_json::json!(1));
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn csv_uses_header_as_fields() {
        let path = tmp("b.csv");
        std::fs::write(&path, "name,city\nada,london\ngrace,nyc\n").unwrap();
        let sets = FileSource::new(&path, "people").unwrap().fetch().await.unwrap();
        let data = sets[0].data.as_ref().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["name"], serde_json::json!("ada"));
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn json_array_splits_into_batches() {
        let path = tmp("c.json");
        std::fs::write(&path, "[{\"n\":1},{\"n\":2},{\"n\":3}]").unwrap();
        let sets = FileSource::new(&path, "nums")
            .unwrap()
            .with_batch_size(2)
            .fetch()
            .await
            .unwrap();
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].data.as_ref().unwrap().len(), 2);
        assert_eq!(sets[1].data.as_ref().unwrap().len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn malformed_jsonl_reports_offending_line() {
        let path = tmp("d.jsonl");
        std::fs::write(&path, "{\"ok\":1}\nNOT JSON\n").unwrap();
        let err = FileSource::new(&path, "x").unwrap().fetch().await.unwrap_err();
        assert!(format!("{err:#}").contains("line 2"), "error was: {err:#}");
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn unknown_extension_requires_explicit_format() {
        let path = tmp("e.dat");
        std::fs::write(&path, "{\"n\":1}").unwrap();
        assert!(FileSource::new(&path, "x").is_err());
        let sets = FileSource::new(&path, "x")
            .or_else(|_| {
                // build with an explicit format instead of inferring
                Ok::<_, anyhow::Error>(FileSource {
                    path: path.clone(),
                    identifier: "x".into(),
                    format: FileFormat::Json,
                    batch_size: 0,
                })
            })
            .unwrap()
            .fetch()
            .await
            .unwrap();
        assert_eq!(sets[0].data.as_ref().unwrap().len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
