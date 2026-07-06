// TODO: reemplazar anyhow con thiserror
use std::fmt::Debug;
use std::fmt::Display;

use crate::intake::dataset::DataSet;
use anyhow::Context;
use derive_builder::Builder;
use qdrant_client::{Payload, qdrant::Distance};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Provider {
    Ollama(String),
    OpenAI(String),
}

// T must be JSON-serializable (Serialize) and clonable, because we serialize
// each item into payloads and clone data in several places.
// Adding these bounds here aligns with DataSet<T: Serialize + Clone> and fixes
// derive errors where serde required `T: Serialize`.
#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
#[builder(setter(into))]
pub struct Job<T: Serialize + Clone> {
    pub dataset: DataSet<T>,
    pub provider: Provider,
    pub dims: u64,
    pub extends: bool,
    #[serde(skip_deserializing, skip_serializing)]
    pub distance: Distance,
    #[builder(setter(custom))]
    pub collection_name: String,
    // #[serde(default)]
    #[builder(setter(into, strip_option), default)]
    pub embedding: Option<Vec<Vec<f32>>>,
}
fn escapechars(model: String) -> String {
    model.replace(":", "_")
}
impl<T: Serialize + Clone> JobBuilder<T> {
    // TODO: Esto se tiene que integrar de alguna manera con build() si es posible.
    pub fn collection_name(&mut self) -> &mut JobBuilder<T> {
        // TODO: Y entonces esto se va!
        let limitaciones = "Por limitaciones del escritor de esta libreria, collection_name es la ultima instruccion que se debe usar en el constructor";
        let provider = match self.provider.clone().expect(limitaciones) {
            Provider::Ollama(model) => format!("ollama_{}", escapechars(model)),
            Provider::OpenAI(model) => format!("openai_{}", escapechars(model)),
        };
        let distance = match self.distance.expect(limitaciones) {
            Distance::Cosine => "Cosine",
            Distance::Euclid => "Euclid",
            Distance::Manhattan => "Manhattan",
            Distance::Dot => "Dot",
            Distance::UnknownDistance => "UnknownDistance",
        };
        let dataset = self.dataset.clone().expect(limitaciones).identifier;
        self.collection_name = Some(format!("{}_{}_{}", provider, distance, dataset));
        self
    }
}

// TODO: Verificar si se puede remover Serialize como trait bound.
impl<T: Serialize + Clone> Display for Job<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Fixed field name typo here as well.
        let dataset = self.clone().dataset.identifier;
        write!(
            f,
            "Job {{\n\tdataset: {},\n\tprovider: {:?},\n\tcollection_name: {},\n\tdims: {},\n\textends: {}\n}}\nDataset size: {} entries",
            dataset,
            self.provider,
            self.collection_name,
            self.dims,
            self.extends,
            self.clone().dataset.data.unwrap().len()
        )
    }
}

impl<T: Serialize + Clone> Job<T> {
    pub fn get_model(&self) -> String {
        match self.clone().provider {
            Provider::Ollama(model) => escapechars(model),
            Provider::OpenAI(model) => escapechars(model),
        }
    }
    pub fn get_payloads(&self) -> anyhow::Result<Vec<Payload>> {
        let mut payloads = vec![];
        for datum in self.dataset.data.clone().context("Couldn't acquire data")? {
            let value = serde_json::to_value(datum).context("Could not create value from data")?;
            let payload = Payload::try_from(value).context("Could't create payload from data")?;
            payloads.push(payload);
        }
        Ok(payloads)
    }

    /// Same rows as [`Job::get_payloads`], but as raw JSON values. Sinks that
    /// persist metadata (e.g. PostgreSQL) consume these; each sink turns them
    /// into its own representation (Qdrant re-derives `Payload`, Postgres stores
    /// them as `jsonb`).
    pub fn get_payload_values(&self) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut values = vec![];
        for datum in self.dataset.data.clone().context("Couldn't acquire data")? {
            values.push(
                serde_json::to_value(datum).context("Could not create value from data")?,
            );
        }
        Ok(values)
    }
}
