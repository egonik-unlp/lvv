//! Dataset containers and connectors for loading source records.

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
