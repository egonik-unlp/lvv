//! Embedding clients.

/// Embedding generation through Ollama or OpenAI-compatible servers.
pub mod embedding_model;
pub use embedding_model::{EmbedError, EmbeddingProvider};
