// TODO: reemplazar anyhow con thiserror
use anyhow::Context;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSet<T> {
    pub filename: String,
    pub identifier: String,
    pub data: Option<Vec<T>>,
}

impl<T> DataSet<T>
where
    T: Serialize + Clone,
{
    pub fn new(filename: impl Into<String>, identifier: impl Into<String>, data: Vec<T>) -> Self {
        let data = Some(data);
        Self {
            filename: filename.into(),
            identifier: identifier.into(),
            data,
        }
    }
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
