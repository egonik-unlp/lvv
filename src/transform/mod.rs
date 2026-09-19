//! Rewrite records with a chat model before they are embedded: summaries,
//! keywords, translations, structured extraction.
//!
//! Three pieces:
//!
//! - an [`Llm`] says where the model is (Ollama, OpenAI, or any
//!   OpenAI-compatible server);
//! - a [`Transform`] says what to do: the system prompt, what to send for each
//!   record, and where the reply goes;
//! - [`Llm::run`] applies a transform to a slice of records, and
//!   [`Llm::complete`] returns the outputs instead.
//!
//! Every record gets an outcome, in record order. A request that fails with a
//! rate limit, a server error or a timeout is retried; a record whose reply is
//! empty or doesn't parse is reported and left unchanged; and a run whose
//! first requests all fail with fatal errors, such as an unknown model, stops
//! early instead of failing every record. With a [`cache`](Run::cache), an
//! interrupted run picks up where it stopped.
//!
//! # Example
//!
//! Summarize each position, then pull out search keywords as typed JSON:
//!
//! ```no_run
//! use lvv::transform::{Llm, Transform};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize)]
//! struct Position {
//!     title: String,
//!     highlights: Vec<String>,
//!     summary: Option<String>,
//!     keywords: Vec<String>,
//! }
//!
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! struct Tags {
//!     keywords: Vec<String>,
//! }
//!
//! # async fn example(mut positions: Vec<Position>) -> Result<(), Box<dyn std::error::Error>> {
//! let llm = Llm::ollama("llama3.2");
//!
//! let summarize = Transform::text("Summarize the job position in one sentence.")
//!     .apply(|p: &mut Position, summary| p.summary = Some(summary));
//! let tag = Transform::structured("List search keywords for the job position.")
//!     .apply(|p: &mut Position, tags: Tags| p.keywords = tags.keywords);
//!
//! llm.run(&summarize, &mut positions).cache("summaries.jsonl").await?.ensure_all()?;
//! let report = llm.run(&tag, &mut positions).concurrency(4).await?;
//! println!("{} tagged, {} failed", report.counts().applied, report.counts().failed);
//! # Ok(())
//! # }
//! ```

mod cache;
mod definition;
mod error;
mod llm;
mod report;
mod run;

#[cfg(test)]
mod tests;

pub use definition::Transform;
pub use error::{
    CompletionCacheError, CompletionError, ErrorClass, FatalCause, InputError, RunError,
};
pub use llm::Llm;
pub use report::{Counts, Outcome, Progress, Report};
pub use run::{Complete, Run};
