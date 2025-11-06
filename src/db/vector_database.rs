#![allow(clippy::result_large_err)]
use std::{env::VarError, fmt::Debug};

use qdrant_client::{
    Payload, Qdrant, QdrantError,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder,
    },
};
use rand::Rng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiConnectionCreationError {}

#[derive(Debug, Error)]
pub enum VectorDatabaseError {
    #[error("Connection to the database has already been established")]
    DatabaseIsConnected,
    #[error("Couldn't create connection to vectorDB: {0}")]
    ConnectionCreationError(#[from] QdrantError),
    #[error("Issues acquiring env vars")]
    EnvVariableError(#[from] dotenvy::Error),
    #[error("Issues acquiring env vars")]
    ApiKeyMissing(#[from] VarError),
}
type VDBResult<T> = Result<T, VectorDatabaseError>;

#[derive(Debug, Clone)]
pub enum Location {
    Local { url: String },
    Remote { url: String, api_key: String },
}

impl Location {
    pub fn new_remote(url: &'static str) -> VDBResult<Self> {
        dotenvy::dotenv()?;
        let api_key = std::env::var("QDRANT_API_KEY")?;
        let url = url.to_string();
        Ok(Location::Remote { url, api_key })
    }
    pub fn new_local(url: &'static str) -> Self {
        let url = url.to_string();
        Location::Local { url }
    }
    fn get_url(&self) -> String {
        match self {
            Location::Local { url } => url.to_string(),
            Location::Remote { url, .. } => url.to_string(),
        }
    }
}
#[derive(Debug, Clone)]
pub struct DatabaseParams {
    pub location: Location,
    pub collection: String,
    pub distance: Distance,
    pub dims: u16,
}
#[derive(Clone)]
pub struct ConnectedDB {
    pub params: DatabaseParams,
    pub client: Qdrant,
}
impl Debug for ConnectedDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ConnectedDB{{\n\tparams: {:?}\n\tclient: QdrantClient\n}}",
            self.params
        )
    }
}
impl ConnectedDB {
    pub async fn collection_exists_and_is_not_empty(
        &self,
        collection_name: &str,
        extend: bool,
    ) -> bool {
        if self
            .client
            .collection_exists(collection_name)
            .await
            .unwrap()
        {
            self.client
                .collection_info(collection_name)
                .await
                .unwrap()
                .result
                .unwrap()
                .points_count
                .unwrap()
                > 0
                && !extend
        } else {
            false
        }
    }
    pub async fn get_collection(&self, collection_name: &str, dims: u64) -> VDBResult<()> {
        let evaluates_to = !self
            .client
            .collection_exists(collection_name)
            .await
            .unwrap();
        println!("evaluates_to = {}", evaluates_to);
        if evaluates_to {
            let result = self
                .client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name)
                        .vectors_config(VectorParamsBuilder::new(dims, self.params.distance)),
                )
                .await?;
            println!("{}", result.result);
        }
        Ok(())
    }
    pub async fn upload_embedddings(
        &self,
        collection_name: &str,
        dims: u64,
        embeddings: Vec<Vec<f32>>,
        payloads: Vec<Payload>,
    ) -> VDBResult<()> {
        self.get_collection(collection_name, dims).await.unwrap();
        let mut points = vec![];
        for (embedding, payload) in embeddings.into_iter().zip(payloads) {
            let random_bytes = rand::rng().random();
            let uuid = uuid::Builder::from_random_bytes(random_bytes)
                .into_uuid()
                .to_string();
            let point = PointStruct::new(uuid, embedding, payload);
            points.push(point);
        }
        println!("Array of points pre upload = {}", points.len());
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection_name, points))
            .await
            .unwrap();
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub enum QdrantDatabase {
    Disconnected(DatabaseParams),
    Connected(ConnectedDB),
}
impl QdrantDatabase {
    pub fn new(location: Location, collection: String, distance: Distance, dims: u16) -> Self {
        let params = DatabaseParams::new(location, collection, distance, dims);
        Self::Disconnected(params)
    }
    pub fn connect(self) -> VDBResult<Self> {
        if let QdrantDatabase::Disconnected(params) = self {
            let location = Location::new_local("http://localhost:6334");
            let client = Qdrant::from_url(location.get_url().as_str()).build()?;
            let connected_db = ConnectedDB { params, client };
            Ok(Self::Connected(connected_db))
        } else {
            Err(VectorDatabaseError::DatabaseIsConnected)
        }
    }
}
impl DatabaseParams {
    pub fn new(location: Location, collection: String, distance: Distance, dims: u16) -> Self {
        DatabaseParams {
            location,
            collection,
            distance,
            dims,
        }
    }
}
