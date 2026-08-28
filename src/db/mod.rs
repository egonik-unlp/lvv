//! Persistence backends for embeddings and their source records.

/// Sink abstractions and built-in Qdrant and PostgreSQL sinks.
pub mod sink;
/// Low-level Qdrant connection and collection management.
pub mod vector_database;
/// Qdrant vector-distance metric.
pub use qdrant_client::qdrant::Distance;
pub use sink::{QdrantSink, Sink, SinkContext};
pub use vector_database::QdrantDatabase;

#[cfg(feature = "postgres")]
pub use sink::PostgresSink;
