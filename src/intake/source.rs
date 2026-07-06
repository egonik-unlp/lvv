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

#[cfg(feature = "sql")]
pub use sql_source::{SqlEngine, SqlSource};

#[cfg(feature = "sql")]
mod sql_source {
    use super::*;

    /// SQL engine a [`SqlSource`] talks to. PostgreSQL has its own dedicated
    /// [`PostgresSource`]; this covers the other engines.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SqlEngine {
        /// MySQL/MariaDB via a `mysql://…` connection URL.
        MySql,
        /// SQLite via a file path (or `:memory:`).
        Sqlite,
    }

    /// Reads rows from SQLite or MySQL via a query into `DataSet<Value>`, with
    /// the same row→JSON-object mapping across engines.
    #[derive(Debug, Clone)]
    pub struct SqlSource {
        engine: SqlEngine,
        /// SQLite file path, or MySQL connection URL.
        target: String,
        query: String,
        identifier: String,
        batch_size: usize,
    }

    impl SqlSource {
        /// A SQLite source over `path` (a file, or `:memory:`).
        pub fn sqlite(
            path: impl Into<String>,
            query: impl Into<String>,
            identifier: impl Into<String>,
        ) -> Self {
            Self {
                engine: SqlEngine::Sqlite,
                target: path.into(),
                query: query.into(),
                identifier: identifier.into(),
                batch_size: 0,
            }
        }

        /// A MySQL source over a `mysql://…` URL.
        pub fn mysql(
            url: impl Into<String>,
            query: impl Into<String>,
            identifier: impl Into<String>,
        ) -> Self {
            Self {
                engine: SqlEngine::MySql,
                target: url.into(),
                query: query.into(),
                identifier: identifier.into(),
                batch_size: 0,
            }
        }

        pub fn with_batch_size(mut self, batch_size: usize) -> Self {
            self.batch_size = batch_size;
            self
        }

        async fn read_rows(&self) -> anyhow::Result<Vec<Value>> {
            match self.engine {
                SqlEngine::Sqlite => {
                    // rusqlite is blocking; keep it off the async runtime.
                    let (target, query) = (self.target.clone(), self.query.clone());
                    tokio::task::spawn_blocking(move || read_sqlite(&target, &query))
                        .await
                        .context("sqlite worker panicked")?
                }
                SqlEngine::MySql => read_mysql(&self.target, &self.query).await,
            }
        }
    }

    #[async_trait]
    impl Source for SqlSource {
        async fn fetch(&self) -> anyhow::Result<Vec<DataSet<Value>>> {
            let rows = self.read_rows().await?;
            Ok(chunk_into_datasets(
                rows,
                &self.query,
                &self.identifier,
                self.batch_size,
            ))
        }
    }

    fn read_sqlite(path: &str, query: &str) -> anyhow::Result<Vec<Value>> {
        use rusqlite::types::ValueRef;
        let conn = rusqlite::Connection::open(path)
            .with_context(|| format!("opening SQLite {path}"))?;
        let mut stmt = conn.prepare(query).context("preparing SQLite query")?;
        let col_names: Vec<String> = stmt
            .column_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        let mut rows = stmt.query([]).context("running SQLite query")?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().context("reading SQLite row")? {
            let mut map = serde_json::Map::new();
            for (i, name) in col_names.iter().enumerate() {
                let value = match row.get_ref(i).context("reading SQLite column")? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(n) => serde_json::json!(n),
                    ValueRef::Real(f) => serde_json::json!(f),
                    ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
                    ValueRef::Blob(b) => serde_json::json!(b),
                };
                map.insert(name.clone(), value);
            }
            out.push(Value::Object(map));
        }
        Ok(out)
    }

    async fn read_mysql(url: &str, query: &str) -> anyhow::Result<Vec<Value>> {
        use mysql_async::{Value as MyVal, prelude::Queryable};
        let pool = mysql_async::Pool::new(url);
        let mut conn = pool.get_conn().await.context("connecting to MySQL")?;
        let rows: Vec<mysql_async::Row> = conn.query(query).await.context("running MySQL query")?;
        drop(conn);
        pool.disconnect().await.ok();

        let out = rows
            .iter()
            .map(|row| {
                let mut map = serde_json::Map::new();
                for (i, col) in row.columns_ref().iter().enumerate() {
                    let value = match row.as_ref(i) {
                        None | Some(MyVal::NULL) => Value::Null,
                        Some(MyVal::Int(n)) => serde_json::json!(n),
                        Some(MyVal::UInt(n)) => serde_json::json!(n),
                        Some(MyVal::Float(f)) => serde_json::json!(f),
                        Some(MyVal::Double(f)) => serde_json::json!(f),
                        Some(MyVal::Bytes(b)) => Value::String(String::from_utf8_lossy(b).into_owned()),
                        // Date/Time — render as their debug string.
                        Some(other) => Value::String(format!("{other:?}")),
                    };
                    map.insert(col.name_str().into_owned(), value);
                }
                Value::Object(map)
            })
            .collect();
        Ok(out)
    }
}

#[cfg(feature = "http")]
pub use http_source::{HttpSource, Pagination};

#[cfg(feature = "http")]
mod http_source {
    use super::*;

    /// How to page through an endpoint.
    #[derive(Debug, Clone)]
    pub enum Pagination {
        /// One request, no paging.
        None,
        /// Append `?<param>=<n>` (or `&…` if the URL already has a query),
        /// starting at `start`, until a page yields no items.
        PageParam { param: String, start: u64 },
    }

    /// Pulls data from an HTTP/JSON endpoint into `DataSet<Value>`.
    #[derive(Debug, Clone)]
    pub struct HttpSource {
        url: String,
        /// Optional JSON pointer (e.g. `/data/items`) to the array of items.
        /// When absent, the whole body is used (array → items, else one item).
        pointer: Option<String>,
        pagination: Pagination,
        identifier: String,
        batch_size: usize,
    }

    impl HttpSource {
        pub fn new(url: impl Into<String>, identifier: impl Into<String>) -> Self {
            Self {
                url: url.into(),
                pointer: None,
                pagination: Pagination::None,
                identifier: identifier.into(),
                batch_size: 0,
            }
        }
        pub fn with_pointer(mut self, pointer: impl Into<String>) -> Self {
            self.pointer = Some(pointer.into());
            self
        }
        pub fn with_pagination(mut self, pagination: Pagination) -> Self {
            self.pagination = pagination;
            self
        }
        pub fn with_batch_size(mut self, batch_size: usize) -> Self {
            self.batch_size = batch_size;
            self
        }
    }

    #[async_trait]
    impl Source for HttpSource {
        async fn fetch(&self) -> anyhow::Result<Vec<DataSet<Value>>> {
            let client = reqwest::Client::new();
            let mut all = Vec::new();
            match &self.pagination {
                Pagination::None => {
                    let body: Value = client
                        .get(&self.url)
                        .send()
                        .await
                        .context("HTTP request failed")?
                        .error_for_status()
                        .context("HTTP error status")?
                        .json()
                        .await
                        .context("decoding JSON response")?;
                    all.extend(extract_items(&body, self.pointer.as_deref())?);
                }
                Pagination::PageParam { param, start } => {
                    let mut page = *start;
                    loop {
                        let sep = if self.url.contains('?') { '&' } else { '?' };
                        let url = format!("{}{sep}{param}={page}", self.url);
                        let body: Value = client
                            .get(&url)
                            .send()
                            .await
                            .context("HTTP request failed")?
                            .error_for_status()
                            .context("HTTP error status")?
                            .json()
                            .await
                            .context("decoding JSON response")?;
                        let items = extract_items(&body, self.pointer.as_deref())?;
                        if items.is_empty() {
                            break;
                        }
                        all.extend(items);
                        page += 1;
                    }
                }
            }
            Ok(chunk_into_datasets(
                all,
                &self.url,
                &self.identifier,
                self.batch_size,
            ))
        }
    }

    /// Pure extraction of the items array from a response body (unit-tested).
    pub(super) fn extract_items(body: &Value, pointer: Option<&str>) -> anyhow::Result<Vec<Value>> {
        let target = match pointer {
            Some(p) => body
                .pointer(p)
                .with_context(|| format!("JSON pointer {p:?} not found in response"))?,
            None => body,
        };
        match target {
            Value::Array(items) => Ok(items.clone()),
            other => Ok(vec![other.clone()]),
        }
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

    #[cfg(feature = "sql")]
    #[tokio::test]
    async fn sqlite_maps_rows_to_json_objects() {
        let path = tmp("sql.db");
        std::fs::remove_file(&path).ok();
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE t (id INTEGER, name TEXT, score REAL);
                 INSERT INTO t VALUES (1, 'ada', 9.5), (2, 'grace', NULL);",
            )
            .unwrap();
        }
        let src = SqlSource::sqlite(
            path.to_string_lossy().into_owned(),
            "SELECT id, name, score FROM t ORDER BY id",
            "t",
        );
        let sets = src.fetch().await.unwrap();
        let data = sets[0].data.as_ref().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], serde_json::json!(1));
        assert_eq!(data[0]["name"], serde_json::json!("ada"));
        assert_eq!(data[0]["score"], serde_json::json!(9.5));
        assert_eq!(data[1]["score"], serde_json::json!(null));
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "http")]
    #[test]
    fn http_extract_items_with_and_without_pointer() {
        use super::http_source::extract_items;
        let body = serde_json::json!({"data": {"items": [{"a": 1}, {"a": 2}]}});
        let items = extract_items(&body, Some("/data/items")).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["a"], serde_json::json!(1));

        let array = serde_json::json!([{"x": 1}]);
        assert_eq!(extract_items(&array, None).unwrap().len(), 1);

        assert!(extract_items(&body, Some("/nope")).is_err());
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
