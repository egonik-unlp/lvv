#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
//! # Typed records
//!
//! Records that already exist as Rust types can describe their own points.
//! Implement [`points::VectorDatabaseItem`], or derive it with the `derive`
//! feature, and each value becomes a [`points::VectorPointDraft`]: a category,
//! the text to embed, and the payload to store with the vector. Embed the
//! descriptions with [`inference::EmbeddingProvider::embed_texts`]. See
//! [`points`].
//!
//! # Transforming records with LLMs
//!
//! Records can be rewritten by a chat model before they are embedded: to
//! summarize long text, extract keywords, translate, or clean up inconsistent
//! fields. A [`transform::Transform`] holds the instructions, what to send for
//! each record and where the reply goes; [`transform::Llm::run`] applies it to
//! a slice of records against Ollama or any OpenAI-compatible server. Replies
//! can be plain text or typed JSON, checked against a schema.
//!
//! Every record gets an outcome in the returned [`transform::Report`], in
//! record order. Rate limits and server errors are retried, a wrong model name
//! stops the run after a few requests, and with a cache an interrupted run
//! resumes where it stopped. With the `derive` feature, marking the field that
//! holds the reply `#[lvv(description)]` makes it part of the embedded text.
//! See [`transform`].
//!
//! # Backends and environment variables
//!
//! Models are reached through the [`backend`] clients. Ollama uses
//! `OLLAMA_URL`, defaulting to `http://127.0.0.1:11434`. OpenAI reads
//! `OPENAI_API_KEY` from the environment, or from a `.env` file when the
//! variable isn't set. Remote Qdrant configuration reads `QDRANT_API_KEY`
//! after loading a `.env` file, and fails if there isn't one.
//!
//! # Cargo features
//!
//! None are enabled by default.
//!
//! | Feature    | Enables |
//! |------------|---------|
//! | `derive`   | `#[derive(VectorDatabaseItem)]` and `#[derive(VectorDatabase)]` from [`lvv-macros`](https://docs.rs/lvv-macros), re-exported in [`points`] |
//! | `postgres` | `intake::PostgresSource` and `db::PostgresSink` |
//! | `sql`      | `intake::SqlSource` for SQLite and MySQL |
//! | `http`     | `intake::HttpSource` for paginated JSON APIs |
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
//!
//! # Full example
//!
//! [`examples/full_pipeline.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/full_pipeline.rs)
//! uses every stage in one program, with the `derive` feature:
//!
//! 1. [`intake::FileSource`] loads records from JSON Lines, CSV and JSON files
//!    into structs that derive `VectorDatabaseItem`.
//! 2. A [`transform::Transform`] run by [`transform::Llm`] writes a summary into
//!    each position, in a field marked `#[lvv(description)]`, caching the
//!    summaries between runs.
//! 3. `#[derive(VectorDatabase)]` turns the records into points.
//! 4. [`inference::EmbeddingProvider::embed_texts`] embeds each category's descriptions,
//!    reusing vectors from a [`cache::cache_embeddings::Cache`].
//! 5. Each category becomes a [`jobs::job::Job`] with precomputed embeddings.
//! 6. A [`jobs::job_queue::JobQueue`] writes the jobs to Qdrant.
//!
//! ```text
//! cargo run --example full_pipeline --features derive
//! QDRANT_URL=http://localhost:6334 cargo run --example full_pipeline --features derive
//! ```

/// HTTP clients for chat and embedding models.
pub mod backend;
/// Reusable embedding cache support.
pub mod cache;
/// Vector database connections and pipeline sinks.
pub mod db;
/// Embedding clients.
pub mod inference;
/// Dataset types and source connectors.
pub mod intake;
/// Embedding jobs and queue execution.
pub mod jobs;
/// Typed records as vector points, with optional derive macros.
pub mod points;
/// Rewriting records with a chat model before they are embedded.
pub mod transform;

/// Re-exports for code generated by `lvv-macros`, so users don't need these as
/// direct dependencies. Not public API.
#[doc(hidden)]
pub mod __private {
    pub use qdrant_client::Payload;
    pub use serde;
    pub use serde_json;
}
