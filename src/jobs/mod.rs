//! Job descriptions and sequential pipeline execution.

/// An embedding job and its builder.
pub mod job;
pub use job::{JobBuilder, Provider};
/// Queue orchestration, caching, and sink dispatch.
pub mod job_queue;
