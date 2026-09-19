use std::{io, path::PathBuf};

use crate::backend::{BackendError, ErrorKind};

/// What kind of failure a [`CompletionError`] is, which decides what a run
/// does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Rate limits, server errors, timeouts, connection failures. Retried.
    Retryable,
    /// Authentication failures, unknown models, rejected requests. Not
    /// retried, and enough of them at the start of a run open the circuit
    /// breaker.
    Fatal,
    /// A problem with this record only: an empty reply, a reply that doesn't
    /// parse, or an input that can't be rendered. Not retried, and it never
    /// opens the circuit breaker.
    Record,
}

/// Why one record didn't get an output.
#[derive(Debug, thiserror::Error)]
pub enum CompletionError {
    /// The backend failed, after any retries.
    #[error("backend request failed: {0}")]
    Backend(#[from] BackendError),
    /// The model replied with nothing but whitespace.
    #[error("the model returned an empty response")]
    EmptyResponse,
    /// The reply isn't JSON of the transform's output type.
    #[error("could not parse the response as the output type: {source}")]
    Parse {
        /// The model's reply, as received.
        raw: String,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
    /// The record couldn't be rendered as the transform's input.
    #[error("could not render the record as input: {0}")]
    Input(#[from] InputError),
}

impl CompletionError {
    /// Whether the failure is retryable, fatal, or specific to this record.
    pub fn class(&self) -> ErrorClass {
        match self {
            CompletionError::Backend(err) => match err.kind {
                ErrorKind::Retryable => ErrorClass::Retryable,
                ErrorKind::Fatal => ErrorClass::Fatal,
            },
            CompletionError::EmptyResponse
            | CompletionError::Parse { .. }
            | CompletionError::Input(_) => ErrorClass::Record,
        }
    }
}

/// A record that couldn't be serialized as JSON for the default input.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct InputError(#[from] serde_json::Error);

/// The completion cache file couldn't be read or written.
#[derive(Debug, thiserror::Error)]
#[error("completion cache {path}: {source}")]
pub struct CompletionCacheError {
    /// The cache file.
    pub path: PathBuf,
    /// The I/O error.
    #[source]
    pub source: io::Error,
}

/// Why a run stopped before every record had an outcome.
#[derive(Debug, thiserror::Error)]
pub enum FatalCause {
    /// The first requests of the run all failed with fatal errors, so the
    /// rest weren't sent.
    #[error(
        "{failures} requests failed with fatal errors before any succeeded; last error: {last}"
    )]
    CircuitOpen {
        /// How many fatal failures in a row opened the circuit.
        failures: usize,
        /// The last of them.
        last: BackendError,
    },
    /// The completion cache couldn't be read or written.
    #[error(transparent)]
    Cache(#[from] CompletionCacheError),
    /// The run is misconfigured, for example a transform without an apply
    /// step passed to [`Llm::run`](super::Llm::run).
    #[error("invalid configuration: {0}")]
    Config(String),
}

/// A run that failed as a whole.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// The run stopped early. Records completed before the error keep their
    /// outputs, and completed outputs are in the cache when one is set.
    #[error("run stopped after {completed} completed records: {source}")]
    Fatal {
        /// Records that got an output before the run stopped.
        completed: usize,
        /// Why the run stopped.
        #[source]
        source: FatalCause,
    },
    /// Returned by [`Report::ensure_all`](super::Report::ensure_all) when some
    /// records failed.
    #[error(
        "{failed} of {total} records failed; first failure (record {first_index}): {first_error}"
    )]
    Incomplete {
        /// Records that failed.
        failed: usize,
        /// Records in the run.
        total: usize,
        /// The index of the first failed record.
        first_index: usize,
        /// The first failure's message.
        first_error: String,
    },
}
