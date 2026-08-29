use qdrant_client::Payload;
use serde::{Serialize, de::DeserializeOwned};

pub trait VectorDatabaseItem: DeserializeOwned + Serialize {
    fn category(&self) -> &'static str;
    fn into_description(&self) -> String;
    fn into_payload(&self) -> anyhow::Result<Payload> {
        let payload: Payload = serde_json::to_value(self)
            .map_err(|err| anyhow::anyhow!(err))?
            .try_into()?;
        Ok(payload)
    }
    fn try_into_database_item(&self) -> anyhow::Result<VectorPointDraft> {
        let payload = self.into_payload()?;
        Ok(VectorPointDraft {
            category: self.category(),
            description: self.into_description(),
            payload: payload,
        })
    }
}
pub trait IntoDescriptionValue: Serialize {
    fn into_description_value(&self) -> String;
}
pub trait VectorDatabase {
    fn point_drafts(&self) -> anyhow::Result<Vec<VectorPointDraft>>;
}
#[derive(Debug)]
pub struct VectorPointDraft {
    pub category: &'static str,
    pub description: String,
    pub payload: Payload,
}

impl VectorPointDraft {
    pub fn new(category: &'static str, description: String, payload: Payload) -> Self {
        Self {
            category,
            description,
            payload,
        }
    }
}
