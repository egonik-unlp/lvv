// TODO: reemplazar anyhow con thiserror
use anyhow::Context;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
/// A named collection of records to embed.
///
/// `data` is optional to support serialized job descriptions in which the
/// records may be populated later.
///
/// # Example
///
/// ```
/// use lvv::intake::dataset::DataSet;
/// let dataset = DataSet::new(
///     "memory",
///     "articles",
///     vec![serde_json::json!({"title": "A guide to vectors"})],
/// );
/// assert_eq!(dataset.identifier, "articles");
/// assert_eq!(dataset.data.as_ref().map(Vec::len), Some(1));
/// ```
pub struct DataSet<T> {
    /// Name or path of the dataset's origin.
    pub filename: String,
    /// Stable logical name used when deriving collection names.
    pub identifier: String,
    /// Records contained in the dataset.
    pub data: Option<Vec<T>>,
}

impl<T> DataSet<T>
where
    T: Serialize + Clone,
{
    /// Creates a populated dataset.
    pub fn new(filename: impl Into<String>, identifier: impl Into<String>, data: Vec<T>) -> Self {
        let data = Some(data);
        Self {
            filename: filename.into(),
            identifier: identifier.into(),
            data,
        }
    }
    /// Consumes the dataset and serializes each record as JSON.
    ///
    /// Returns an error if the dataset has no data or a record cannot be
    /// serialized.
    pub fn serialize_to_vec(self) -> anyhow::Result<Vec<String>> {
        let mut vec = vec![];
        let data = self.data.context("No data")?;
        for datum in data {
            let string = serde_json::to_string(&datum).context("Couldn't serialize a node")?;
            vec.push(string);
        }
        Ok(vec)
    }
}
