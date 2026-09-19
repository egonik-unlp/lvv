//! Dataset containers and connectors for loading source records.

use std::path::PathBuf;

/// Generic dataset container.
pub mod dataset;
/// File and optional PostgreSQL source connectors.
pub mod source;

pub use source::{FileFormat, FileSource, Source};

#[cfg(feature = "postgres")]
pub use source::PostgresSource;

#[cfg(feature = "sql")]
pub use source::{SqlEngine, SqlSource};

#[cfg(feature = "http")]
pub use source::{HttpSource, Pagination};

/// Why reading records from a source failed.
#[derive(Debug, thiserror::Error)]
pub enum IntakeError {
    /// The file's extension doesn't say which format it is.
    #[error(
        "cannot infer file format from extension {extension:?}; \
         set it explicitly with FileSource::with_format"
    )]
    UnknownFormat {
        /// The extension, lowercased, if the path has one.
        extension: Option<String>,
    },
    /// A file couldn't be read.
    #[error("reading {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A CSV file couldn't be parsed.
    #[error("parsing CSV {path}{}: {source}", record.map(|r| format!(" record {r}")).unwrap_or_default())]
    Csv {
        /// The file.
        path: PathBuf,
        /// The 1-based record that failed, when a record failed.
        record: Option<usize>,
        /// The CSV error.
        #[source]
        source: csv::Error,
    },
    /// A JSON or JSON Lines file couldn't be parsed.
    #[error("parsing JSON {path}{}: {source}", line.map(|l| format!(" line {l}")).unwrap_or_default())]
    Json {
        /// The file.
        path: PathBuf,
        /// The 1-based line that failed, for JSON Lines.
        line: Option<usize>,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
    /// A dataset has no data.
    #[error("the dataset has no data")]
    NoData,
    /// A record couldn't be serialized as JSON.
    #[error("could not serialize a record: {0}")]
    Serialize(#[source] serde_json::Error),
    /// A PostgreSQL call failed.
    #[cfg(feature = "postgres")]
    #[error("PostgreSQL source, {context}: {source}")]
    Postgres {
        /// What was being done.
        context: &'static str,
        /// The driver's error.
        #[source]
        source: tokio_postgres::Error,
    },
    /// A SQLite call failed.
    #[cfg(feature = "sql")]
    #[error("SQLite source, {context}: {source}")]
    Sqlite {
        /// What was being done.
        context: String,
        /// The driver's error.
        #[source]
        source: rusqlite::Error,
    },
    /// A MySQL call failed.
    #[cfg(feature = "sql")]
    #[error("MySQL source, {context}: {source}")]
    MySql {
        /// What was being done.
        context: &'static str,
        /// The driver's error.
        #[source]
        source: mysql_async::Error,
    },
    /// The blocking SQLite worker panicked or was cancelled.
    #[cfg(feature = "sql")]
    #[error("SQLite worker failed: {0}")]
    Worker(#[source] tokio::task::JoinError),
    /// An HTTP request failed or returned an error status.
    #[cfg(feature = "http")]
    #[error("HTTP source {url}: {source}")]
    Http {
        /// The URL requested.
        url: String,
        /// The request error.
        #[source]
        source: reqwest::Error,
    },
    /// The JSON pointer matched nothing in the response.
    #[cfg(feature = "http")]
    #[error("JSON pointer {pointer:?} not found in the response")]
    PointerNotFound {
        /// The pointer.
        pointer: String,
    },
}
