use serde::{Deserialize, Serialize};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSet {
    pub filename: String,
    pub identfier: String,
    pub data: Option<Vec<String>>,
}
