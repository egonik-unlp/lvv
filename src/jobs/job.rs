use std::fmt::Display;

use derive_builder::Builder;
use llm::embedding;
use qdrant_client::{Payload, qdrant::Distance};
use serde::{Deserialize, Serialize};

use crate::{cache::cache_embeddings::Cache, intake::dataset::DataSet};
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Provider {
    Ollama(String),
    OpenAI(String),
}

#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
#[builder(setter(into))]
pub struct Job {
    pub dataset: DataSet,
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
impl JobBuilder {
    pub fn collection_name(&mut self) -> &mut JobBuilder {
        let limitaciones = "Por limitaciones del escritor de esta libreria, collection_name es la ultima instruccion que se debe usar en el constructor";
        let provider = match self.provider.clone().expect(limitaciones) {
            Provider::Ollama(model) => format!("ollama_{}", escapechars(model)),
            Provider::OpenAI(model) => format!("ollama_{}", escapechars(model)),
        };
        let distance = match self.distance.expect(limitaciones) {
            Distance::Cosine => "Cosine",
            Distance::Euclid => "Euclid",
            Distance::Manhattan => "Manhattan",
            Distance::Dot => "Dot",
            Distance::UnknownDistance => "UnknownDistance",
        };
        let dataset = self.dataset.clone().expect(limitaciones).identfier;
        self.collection_name = Some(format!("{}_{}_{}", provider, distance, dataset));
        self
    }
}

impl Display for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dataset = self.clone().dataset.identfier;
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
impl Job {
    pub fn get_model(&self) -> String {
        match self.clone().provider {
            Provider::Ollama(model) => escapechars(model),
            Provider::OpenAI(model) => escapechars(model),
        }
    }
    pub fn get_payload(&self) -> Vec<Payload> {
        self.clone()
            .dataset
            .data
            .unwrap()
            .into_iter()
            .map(|prop| {
                let prs = serde_json::to_value(prop).unwrap();
                Payload::try_from(prs).unwrap()
            })
            .collect()
    }

    // pub fn get_string(&self) -> Vec<String> {
    //     match self.to_owned().dataset {
    //         DataSet::Nico(data) => data
    //             .into_iter()
    //             .map(|prop| serde_json::to_string(&prop).unwrap())
    //             .collect(),
    //         DataSet::Taca(data) => data
    //             .into_iter()
    //             .map(|prop| serde_json::to_string(&prop).unwrap())
    //             .collect(),
    //     }
    // }
}
