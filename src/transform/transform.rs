//! Turn typed records into vector points.
//!
//! A [`VectorDatabaseItem`] is a record that becomes one point: a category, a
//! text description to embed, and a Qdrant [`Payload`] to store next to the
//! vector. A [`VectorDatabase`] holds items and lists all of their points at
//! once. Both produce [`VectorPointDraft`]s, which you embed with
//! [`EmbeddingProvider`](crate::inference::EmbeddingProvider) and write with a
//! [`Sink`](crate::db::Sink).
//!
//! # Deriving
//!
//! The `derive` feature re-exports the `VectorDatabaseItem` and
//! `VectorDatabase` derive macros from
//! [`lvv-macros`](https://docs.rs/lvv-macros) in this module, so one import
//! brings in both a trait and its derive:
//!
//! ```toml
//! [dependencies]
//! lvv = { version = "0.5", features = ["derive"] }
//! ```
//!
//! Mark the fields to embed with `#[lvv(description)]`, hide fields with
//! `#[lvv(skip)]`, and change payload keys with `#[lvv(rename = "...")]`:
//!
#![cfg_attr(feature = "derive", doc = "```")]
#![cfg_attr(not(feature = "derive"), doc = "```ignore")]
//! use lvv::transform::transform::{VectorDatabase, VectorDatabaseItem};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, VectorDatabaseItem)]
//! #[serde(rename_all = "camelCase")]
//! struct Position {
//!     // Embedded, and stored under `role` instead of `jobTitle`.
//!     #[lvv(description, rename = "role")]
//!     job_title: String,
//!     // Embedded when present; `None` adds nothing to the description.
//!     #[lvv(description)]
//!     organization: Option<String>,
//!     // Stored as `startedOn`, following serde.
//!     started_on: String,
//!     // Neither embedded nor stored.
//!     #[lvv(skip)]
//!     internal_id: u64,
//! }
//!
//! #[derive(Serialize, Deserialize, VectorDatabaseItem)]
//! struct Interest(#[lvv(description)] String);
//!
//! #[derive(VectorDatabase)]
//! struct Portfolio {
//!     positions: Vec<Position>,
//!     interests: Option<Vec<Interest>>,
//!     #[lvv(skip)]
//!     owner: String,
//! }
//!
//! # fn main() -> anyhow::Result<()> {
//! let portfolio = Portfolio {
//!     positions: vec![Position {
//!         job_title: "Researcher".into(),
//!         organization: Some("UNLP".into()),
//!         started_on: "2020-03-01".into(),
//!         internal_id: 42,
//!     }],
//!     interests: Some(vec![Interest("Vector search".into())]),
//!     owner: "Ada".into(),
//! };
//!
//! let drafts = portfolio.point_drafts()?;
//! assert_eq!(drafts.len(), 2);
//!
//! let position = &drafts[0];
//! assert_eq!(position.category, "Position");
//! assert_eq!(position.description, "Researcher\nUNLP");
//! assert_eq!(
//!     serde_json::to_value(&position.payload)?,
//!     serde_json::json!({
//!         "role": "Researcher",
//!         "organization": "UNLP",
//!         "startedOn": "2020-03-01",
//!         "category": "Position",
//!     }),
//! );
//! assert_eq!(drafts[1].category, "Interest");
//! # Ok(())
//! # }
//! ```
//!
//! See the [`lvv-macros` documentation](https://docs.rs/lvv-macros) for every
//! attribute and the field types `VectorDatabase` accepts, and
//! [`examples/derive.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/derive.rs)
//! for a complete pipeline that loads derived records from files, embeds them
//! and runs a `JobQueue` into Qdrant.
//!
//! # Implementing by hand
//!
//! ```
//! use lvv::transform::transform::VectorDatabaseItem;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Skill {
//!     name: String,
//!     level: String,
//! }
//!
//! impl VectorDatabaseItem for Skill {
//!     fn category(&self) -> String {
//!         "skill".into()
//!     }
//!
//!     fn into_description(&self) -> String {
//!         format!("{} ({})", self.name, self.level)
//!     }
//! }
//!
//! # fn main() -> anyhow::Result<()> {
//! let skill = Skill { name: "Rust".into(), level: "advanced".into() };
//! let draft = skill.try_into_database_item()?;
//! assert_eq!(draft.description, "Rust (advanced)");
//! # Ok(())
//! # }
//! ```
//!
//! # Embedding and storing drafts
//!
//! Embed the descriptions, then write the payloads as rows in the same order:
//!
//! ```no_run
//! use lvv::{
//!     db::{Distance, QdrantSink, Sink, SinkContext},
//!     db::vector_database::{DatabaseParams, Location},
//!     inference::EmbeddingProvider,
//!     intake::dataset::DataSet,
//!     transform::transform::VectorPointDraft,
//! };
//!
//! # async fn store(drafts: Vec<VectorPointDraft>) -> anyhow::Result<()> {
//! let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
//! let embeddings = EmbeddingProvider::new("nomic-embed-text")?
//!     .embed_properties(DataSet::new("portfolio", "points", descriptions))
//!     .await?;
//! let rows = drafts
//!     .iter()
//!     .map(|d| serde_json::to_value(&d.payload))
//!     .collect::<Result<Vec<_>, _>>()?;
//!
//! let sink = QdrantSink::new(DatabaseParams::new(
//!     Location::new_local("http://localhost:6334"),
//!     "portfolio".into(),
//!     Distance::Cosine,
//!     768,
//! ));
//! sink.write(&SinkContext {
//!     collection_name: "portfolio",
//!     dims: 768,
//!     extends: true,
//!     embeddings: &embeddings,
//!     rows: &rows,
//! })
//! .await?;
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;

use qdrant_client::Payload;
use serde::{Serialize, de::DeserializeOwned};

#[cfg(feature = "derive")]
pub use lvv_macros::{VectorDatabase, VectorDatabaseItem};

/// A record that becomes one vector point.
///
/// Implement [`category`](Self::category) and
/// [`into_description`](Self::into_description); the payload defaults to the
/// record's JSON serialization. With the `derive` feature,
/// `#[derive(VectorDatabaseItem)]` generates all three from field attributes.
/// See the [module documentation](crate::transform::transform).
pub trait VectorDatabaseItem: DeserializeOwned + Serialize {
    /// The kind of record, used to group or filter points. The derive uses the
    /// struct name.
    fn category(&self) -> String;
    /// The text to embed for this record.
    fn into_description(&self) -> String;
    /// The data stored next to the vector.
    ///
    /// Defaults to the record serialized with `serde_json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the record doesn't serialize to a JSON object.
    fn into_payload(&self) -> anyhow::Result<Payload> {
        let payload: Payload = serde_json::to_value(self)
            .map_err(|err| anyhow::anyhow!(err))?
            .try_into()?;
        Ok(payload)
    }
    /// Builds this record's [`VectorPointDraft`] from the methods above.
    ///
    /// # Errors
    ///
    /// Returns the error from [`into_payload`](Self::into_payload).
    fn try_into_database_item(&self) -> anyhow::Result<VectorPointDraft> {
        let payload = self.into_payload()?;
        Ok(VectorPointDraft {
            category: self.category(),
            description: self.into_description(),
            payload: payload,
        })
    }
}

/// Renders a field as text for a derived [`VectorDatabaseItem`] description.
///
/// Every field marked `#[lvv(description)]` must implement this trait. It is
/// implemented for `String`, integer and float primitives, `Option<T>` (empty
/// for `None`) and `Vec<T>` (one line per non-empty element). Implement it for your own
/// types:
///
/// ```
/// use lvv::transform::transform::IntoDescriptionValue;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// enum Level {
///     Beginner,
///     Expert,
/// }
///
/// impl IntoDescriptionValue for Level {
///     fn into_description_value(&self) -> String {
///         match self {
///             Level::Beginner => "beginner".into(),
///             Level::Expert => "expert".into(),
///         }
///     }
/// }
///
/// assert_eq!(Some(Level::Expert).into_description_value(), "expert");
/// ```
pub trait IntoDescriptionValue: Serialize {
    /// The text for this value. An empty string leaves the field out of the
    /// description.
    fn into_description_value(&self) -> String;
}

impl IntoDescriptionValue for String {
    fn into_description_value(&self) -> String {
        self.to_string()
    }
}

impl<T> IntoDescriptionValue for Option<T>
where
    T: IntoDescriptionValue,
{
    fn into_description_value(&self) -> String {
        match self {
            Some(inner_value) => inner_value.into_description_value(),
            None => "".to_string(),
        }
    }
}
impl<T> IntoDescriptionValue for Vec<T>
where
    T: IntoDescriptionValue,
{
    fn into_description_value(&self) -> String {
        self.iter()
            .map(IntoDescriptionValue::into_description_value)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

macro_rules! impl_into_description_value_for_num {
    ($($t:ty),* $(,)?) => {
        $(
            impl IntoDescriptionValue for $t {
                fn into_description_value(&self) -> String {
                    format!("{:?}", self)
                }
            }
        )*
    };
}

impl_into_description_value_for_num!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64,
);
/// A container of [`VectorDatabaseItem`]s that lists all of their points.
///
/// With the `derive` feature, `#[derive(VectorDatabase)]` implements this for a
/// struct whose fields hold items directly or in a `Vec`, `BTreeMap`, `HashMap`
/// or `Option`.
pub trait VectorDatabase {
    /// One draft per item, in field order.
    ///
    /// # Errors
    ///
    /// Returns an error if any item's payload can't be built.
    fn point_drafts(&self) -> anyhow::Result<Vec<VectorPointDraft>>;
}

/// A vector point before embedding: what to embed and what to store with it.
#[derive(Debug)]
pub struct VectorPointDraft {
    /// The kind of record, from [`VectorDatabaseItem::category`].
    pub category: String,
    /// The text to embed.
    pub description: String,
    /// The data stored next to the vector.
    pub payload: Payload,
}

impl VectorPointDraft {
    /// Creates a draft from its parts.
    pub fn new(category: String, description: String, payload: Payload) -> Self {
        Self {
            category,
            description,
            payload,
        }
    }
}
