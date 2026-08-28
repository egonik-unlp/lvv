#![warn(missing_docs)]

//! Build pipelines that turn structured datasets into vector embeddings.
//!
//! `lvv` provides the pieces needed to load records, generate embeddings with
//! Ollama or OpenAI, cache those embeddings, and write the result to Qdrant or
//! another [`db::Sink`]. Jobs can be collected in a [`jobs::job_queue::JobQueue`]
//! and sent to more than one sink in registration order.
//!
//! # Pipeline
//!
//! 1. Create an [`intake::dataset::DataSet`] directly or read records through
//!    an [`intake::source::Source`].
//! 2. Describe the embedding request with [`jobs::job::JobBuilder`].
//! 3. Add one or more [`db::Sink`] implementations to a
//!    [`jobs::job_queue::JobQueue`] and call its `run` method.
//!
//! # Backends and environment variables
//!
//! Ollama uses `OLLAMA_URL`, defaulting to `http://127.0.0.1:11434`. OpenAI
//! reads `OPENAI_API_KEY`. Remote Qdrant configuration reads `QDRANT_API_KEY`.
//! The optional `postgres` Cargo feature exposes PostgreSQL source and sink
//! connectors.
//!
//! # Example
//!
//! ```no_run
//! use lvv::{
//!     db::{Distance, QdrantSink},
//!     db::vector_database::{DatabaseParams, Location},
//!     intake::dataset::DataSet,
//!     jobs::{JobBuilder, Provider},
//!     jobs::job_queue::JobQueue,
//! };
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let dataset = DataSet::new("articles.json", "articles", vec![
//!     serde_json::json!({"title": "Rust documentation"}),
//! ]);
//! let job = JobBuilder::default()
//!     .dataset(dataset)
//!     .provider(Provider::Ollama("nomic-embed-text".into()))
//!     .dims(768_u64)
//!     .extends(false)
//!     .distance(Distance::Cosine)
//!     .collection_name()
//!     .build()?;
//!
//! let params = DatabaseParams::new(
//!     Location::new_local("http://localhost:6334"),
//!     "articles".into(),
//!     Distance::Cosine,
//!     768,
//! );
//! let mut queue = JobQueue::from_vec(vec![job]);
//! queue.with_sink(Arc::new(QdrantSink::new(params)));
//! queue.run().await?;
//! # Ok(())
//! # }
//! ```

/// Reusable embedding cache support.
pub mod cache;
/// Vector database connections and pipeline sinks.
pub mod db;
/// Completion and embedding model clients.
pub mod inference;
/// Dataset types and source connectors.
pub mod intake;
/// Embedding jobs and queue execution.
pub mod jobs;
