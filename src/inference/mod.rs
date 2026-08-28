//! Clients for text completion and vector embedding providers.

/// Embedding generation through Ollama or OpenAI.
pub mod embedding_model;
pub use embedding_model::EmbeddingProvider;
/// Text completion through Ollama or OpenAI.
pub mod completion_model;
pub use completion_model::CompletionModel;
