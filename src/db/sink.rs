//! Destinations for embedded data.
//!
//! A [`Sink`] receives a job's embeddings + JSON rows and persists them. The
//! pipeline can register several sinks and write to all of them in one run
//! (e.g. vectors to Qdrant and metadata to PostgreSQL). Sinks are held as
//! `Arc<dyn Sink>` so a [`crate::jobs::job_queue::JobQueue`] stays `Clone`.

use anyhow::Context;
use async_trait::async_trait;
use qdrant_client::Payload;
use serde_json::Value;

use crate::db::{QdrantDatabase, vector_database::DatabaseParams};

/// Everything a [`Sink`] needs to persist one job's output.
pub struct SinkContext<'a> {
    /// Logical collection/target name for this job.
    pub collection_name: &'a str,
    /// Embedding dimensionality.
    pub dims: u64,
    /// Honour the job's `extends` flag (don't overwrite a populated target).
    pub extends: bool,
    /// The embeddings, one vector per row, in order.
    pub embeddings: &'a [Vec<f32>],
    /// One JSON object per embedding, in the same order.
    pub rows: &'a [Value],
}

/// A destination that persists a job's embeddings/rows.
///
/// `Debug` is a supertrait so a `Vec<Arc<dyn Sink>>` (and thus `JobQueue`) can
/// derive `Debug`.
#[async_trait]
pub trait Sink: Send + Sync + std::fmt::Debug {
    /// Short name, used in progress output and partial-failure reports.
    fn name(&self) -> &str;
    /// Persist `ctx` to this destination.
    async fn write(&self, ctx: &SinkContext<'_>) -> anyhow::Result<()>;
}

/// Writes vectors to Qdrant.
///
/// Wraps the existing [`QdrantDatabase`] upload path: creates the collection if
/// missing, honours `extends` (skips a populated collection when `extends` is
/// false), and upserts points with random UUIDs + payloads.
#[derive(Debug, Clone)]
pub struct QdrantSink {
    params: DatabaseParams,
}

impl QdrantSink {
    pub fn new(params: DatabaseParams) -> Self {
        Self { params }
    }
}

#[async_trait]
impl Sink for QdrantSink {
    fn name(&self) -> &str {
        "qdrant"
    }

    async fn write(&self, ctx: &SinkContext<'_>) -> anyhow::Result<()> {
        let db = QdrantDatabase::new_with_database_params(self.params.clone())
            .connect()
            .context("connecting to Qdrant")?;
        if let QdrantDatabase::Connected(db) = db {
            if db
                .collection_exists_and_is_not_empty(ctx.collection_name, ctx.extends)
                .await
                .map_err(|e| anyhow::anyhow!("Qdrant collection check failed: {e}"))?
            {
                // Already populated and not extending: leave it untouched.
                return Ok(());
            }
            let payloads = ctx
                .rows
                .iter()
                .map(|v| Payload::try_from(v.clone()).context("building Qdrant payload"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            db.upload_embedddings(
                ctx.collection_name,
                ctx.dims,
                ctx.embeddings.to_vec(),
                payloads,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Qdrant upload failed: {e}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "postgres")]
pub use postgres_sink::PostgresSink;

#[cfg(feature = "postgres")]
mod postgres_sink {
    use super::*;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use tokio_postgres::NoTls;

    /// Persists each row as a JSONB record in a PostgreSQL table, idempotently.
    ///
    /// The target table is created if missing as
    /// `(id text primary key, collection text, payload jsonb)`. Re-running the
    /// same job upserts by a stable id derived from the row's JSON, so rows are
    /// never duplicated across re-ingests.
    #[derive(Debug, Clone)]
    pub struct PostgresSink {
        conn_str: String,
        table: String,
    }

    impl PostgresSink {
        /// `table` must be a plain SQL identifier (`[A-Za-z_][A-Za-z0-9_]*`) — it
        /// is interpolated into DDL/DML, so it cannot be parameterised.
        pub fn new(conn_str: impl Into<String>, table: impl Into<String>) -> anyhow::Result<Self> {
            let table = table.into();
            if !is_valid_ident(&table) {
                anyhow::bail!("invalid table name {table:?}: expected [A-Za-z_][A-Za-z0-9_]*");
            }
            Ok(Self {
                conn_str: conn_str.into(),
                table,
            })
        }
    }

    fn is_valid_ident(s: &str) -> bool {
        let mut chars = s.chars();
        matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Deterministic id from the row's JSON, so re-ingesting the same row
    /// upserts instead of inserting a duplicate.
    fn stable_id(row: &Value) -> String {
        let mut hasher = DefaultHasher::new();
        row.to_string().hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    #[async_trait]
    impl Sink for PostgresSink {
        fn name(&self) -> &str {
            "postgres"
        }

        async fn write(&self, ctx: &SinkContext<'_>) -> anyhow::Result<()> {
            let (client, connection) = tokio_postgres::connect(&self.conn_str, NoTls)
                .await
                .context("connecting to PostgreSQL sink")?;
            let handle = tokio::spawn(async move {
                if let Err(e) = connection.await {
                    eprintln!("postgres sink connection error: {e}");
                }
            });

            let result = self.upsert_all(&client, ctx).await;

            drop(client);
            let _ = handle.await;
            result
        }
    }

    impl PostgresSink {
        async fn upsert_all(
            &self,
            client: &tokio_postgres::Client,
            ctx: &SinkContext<'_>,
        ) -> anyhow::Result<()> {
            let create = format!(
                "CREATE TABLE IF NOT EXISTS {} \
                 (id text PRIMARY KEY, collection text NOT NULL, payload jsonb NOT NULL)",
                self.table
            );
            client
                .batch_execute(&create)
                .await
                .context("ensuring sink table exists")?;

            let upsert = format!(
                "INSERT INTO {} (id, collection, payload) VALUES ($1, $2, $3) \
                 ON CONFLICT (id) DO UPDATE SET collection = EXCLUDED.collection, \
                 payload = EXCLUDED.payload",
                self.table
            );
            let stmt = client.prepare(&upsert).await.context("preparing upsert")?;
            let collection = ctx.collection_name;
            for row in ctx.rows {
                let id = stable_id(row);
                client
                    .execute(&stmt, &[&id, &collection, row])
                    .await
                    .context("upserting row into sink table")?;
            }
            Ok(())
        }
    }
}
