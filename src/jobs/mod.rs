//! Job descriptions and sequential pipeline execution.

/// An embedding job and its builder.
pub mod job;
pub use job::{JobBuilder, Provider};
/// Queue orchestration, caching, and sink dispatch.
pub mod job_queue;

use qdrant_client::QdrantError;

use crate::{backend::BackendError, db::SinkError, inference::EmbedError, intake::IntakeError};

/// Why a job or a queue run failed.
#[derive(Debug, thiserror::Error)]
pub enum JobError {
    /// The job's dataset has no data.
    #[error("the job's dataset has no data")]
    NoData,
    /// A row couldn't be serialized as JSON.
    #[error("serializing a row: {0}")]
    Serialize(#[from] serde_json::Error),
    /// A row isn't a valid Qdrant payload (it must be a JSON object).
    #[error("building a Qdrant payload: {0}")]
    Payload(Box<QdrantError>),
    /// The job's embedder couldn't be created, for example without an OpenAI key.
    #[error("creating the embedder: {0}")]
    Embedder(#[from] BackendError),
    /// Embedding the job's rows failed.
    #[error("embedding collection '{collection}': {source}")]
    Embed {
        /// The job's collection.
        collection: String,
        /// The embedding error.
        #[source]
        source: EmbedError,
    },
    /// Reading the job's rows failed.
    #[error(transparent)]
    Intake(#[from] IntakeError),
    /// A sink failed. Sinks run in order, so the ones in `written` succeeded
    /// and the rest were not attempted.
    #[error(
        "sink '{sink}' failed for collection '{collection}'. Written: [{}]; not written: '{sink}' \
         and any sink after it. Re-run to reconcile (writes are idempotent).",
        written.join(", ")
    )]
    Sink {
        /// The failing sink's name.
        sink: String,
        /// The job's collection.
        collection: String,
        /// Sinks written before the failure.
        written: Vec<String>,
        /// The sink's error.
        #[source]
        source: SinkError,
    },
}
