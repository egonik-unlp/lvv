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
/// Reserved error type for API-backed connection creation.
///
/// This enum currently has no variants.
pub enum ApiConnectionCreationError {}

#[derive(Debug, Error)]
/// Errors raised while configuring or connecting to Qdrant.
pub enum VectorDatabaseError {
    /// [`QdrantDatabase::connect`] was called on an already connected value.
    #[error("Connection to the database has already been established")]
    DatabaseIsConnected,
    /// The Qdrant client could not be constructed or contacted.
    #[error("Couldn't create connection to vectorDB: {0}")]
    ConnectionCreationError(#[from] QdrantError),
    /// A `.env` file could not be loaded.
    #[error("Issues acquiring env vars")]
    EnvVariableError(#[from] dotenvy::Error),
    /// The `QDRANT_API_KEY` environment variable is missing.
    #[error("Issues acquiring env vars")]
    ApiKeyMissing(#[from] VarError),
}
type VDBResult<T> = Result<T, VectorDatabaseError>;

#[derive(Debug, Clone)]
/// Network location and credentials for a Qdrant deployment.
pub enum Location {
    /// A Qdrant endpoint that requires no API key.
    Local {
        /// Base URL of the Qdrant gRPC endpoint.
        url: String,
    },
    /// A Qdrant endpoint authenticated by an API key.
    Remote {
        /// Base URL of the hosted Qdrant endpoint.
        url: String,
        /// API key sent when authenticating requests.
        api_key: String,
    },
}

impl Location {
    /// Creates a remote location using `QDRANT_API_KEY` from the environment.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example() -> Result<(), lvv::db::vector_database::VectorDatabaseError> {
    /// use lvv::db::vector_database::Location;
    /// let location = Location::new_remote("https://example.qdrant.io:6334")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_remote(url: &'static str) -> VDBResult<Self> {
        dotenvy::dotenv()?;
        let api_key = std::env::var("QDRANT_API_KEY")?;
        let url = url.to_string();
        Ok(Location::Remote { url, api_key })
    }
    /// Creates an unauthenticated location, typically for a local Qdrant server.
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
/// Qdrant collection configuration shared by connections and sinks.
pub struct DatabaseParams {
    /// Qdrant endpoint and credentials.
    pub location: Location,
    /// Default collection name retained for compatibility.
    pub collection: String,
    /// Similarity metric for newly created collections.
    pub distance: Distance,
    /// Expected vector dimensions.
    pub dims: u16,
}
#[derive(Clone)]
/// An established Qdrant client and its collection configuration.
pub struct ConnectedDB {
    /// Configuration used to establish the client.
    pub params: DatabaseParams,
    /// Underlying Qdrant client.
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
    /// Returns `true` when a collection is populated and extension is disabled.
    ///
    /// This is used by the pipeline to skip an existing target. Qdrant request
    /// failures currently panic inside this method.
    pub async fn collection_exists_and_is_not_empty(
        &self,
        collection_name: &str,
        extend: bool,
    ) -> VDBResult<bool> {
        if self.client.collection_exists(collection_name).await? {
            let points = self
                .client
                .collection_info(collection_name)
                .await?
                .result
                .and_then(|info| info.points_count)
                .unwrap_or(0);
            Ok(points > 0 && !extend)
        } else {
            Ok(false)
        }
    }
    /// Creates `collection_name` with `dims` if it does not already exist.
    pub async fn get_collection(&self, collection_name: &str, dims: u64) -> VDBResult<()> {
        let evaluates_to = !self.client.collection_exists(collection_name).await?;
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
    /// Uploads embeddings and payloads as new Qdrant points.
    ///
    /// Embeddings and payloads are paired positionally; excess values in the
    /// longer input are ignored by the underlying `zip` operation.
    pub async fn upload_embedddings(
        &self,
        collection_name: &str,
        dims: u64,
        embeddings: Vec<Vec<f32>>,
        payloads: Vec<Payload>,
    ) -> VDBResult<()> {
        self.get_collection(collection_name, dims).await?;
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
            .await?;
        Ok(())
    }
}
#[derive(Clone, Debug)]
/// State machine for a Qdrant database connection.
pub enum QdrantDatabase {
    /// Configuration has been supplied but no client has been created.
    Disconnected(DatabaseParams),
    /// A Qdrant client is ready for collection operations.
    Connected(ConnectedDB),
}
impl QdrantDatabase {
    /// Creates a disconnected database from individual parameters.
    pub fn new(location: Location, collection: String, distance: Distance, dims: u16) -> Self {
        let params = DatabaseParams::new(location, collection, distance, dims);
        Self::Disconnected(params)
    }
    /// Creates a disconnected database from a reusable configuration value.
    pub fn new_with_database_params(params: DatabaseParams) -> Self {
        Self::Disconnected(params)
    }
    /// Creates the Qdrant client and transitions to [`Self::Connected`].
    ///
    /// Calling this on an already connected value returns
    /// [`VectorDatabaseError::DatabaseIsConnected`].
    pub fn connect(self) -> VDBResult<Self> {
        if let QdrantDatabase::Disconnected(params) = self {
            // Honour the configured location so instances can be
            // port-partitioned (e.g. one Qdrant per project on a distinct
            // port). A remote location also carries its API key.
            let mut builder = Qdrant::from_url(params.location.get_url().as_str());
            if let Location::Remote { api_key, .. } = &params.location {
                builder = builder.api_key(api_key.clone());
            }
            let client = builder.build()?;
            let connected_db = ConnectedDB { params, client };
            Ok(Self::Connected(connected_db))
        } else {
            Err(VectorDatabaseError::DatabaseIsConnected)
        }
    }
}
impl DatabaseParams {
    /// Creates Qdrant collection parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use lvv::db::{Distance, vector_database::{DatabaseParams, Location}};
    /// let params = DatabaseParams::new(
    ///     Location::new_local("http://localhost:6334"),
    ///     "documents".into(),
    ///     Distance::Cosine,
    ///     768,
    /// );
    /// assert_eq!(params.collection, "documents");
    /// ```
    pub fn new(location: Location, collection: String, distance: Distance, dims: u16) -> Self {
        DatabaseParams {
            location,
            collection,
            distance,
            dims,
        }
    }
}
