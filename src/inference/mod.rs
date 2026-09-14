//! Clients for text completion and vector embedding providers.

/// Embedding generation through Ollama or OpenAI.
pub mod embedding_model;
pub use embedding_model::EmbeddingProvider;
/// Chat completion through Ollama or OpenAI, for transforming records with an
/// LLM before they are embedded.
pub mod completion_model;
pub use completion_model::CompletionModel;
