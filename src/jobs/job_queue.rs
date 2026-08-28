// TODO: reemplazar anyhow con thiserror
use std::sync::Arc;

use anyhow::Context;
use indicatif::ProgressIterator;
use serde::Serialize;

use crate::{
    cache::cache_embeddings::Cache,
    db::{
        QdrantDatabase,
        sink::{QdrantSink, Sink, SinkContext},
        vector_database::DatabaseParams,
    },
    inference::embedding_model::EmbeddingProvider, // Needed for `run()` when creating embedders
    jobs::{Provider, job::Job},
};

#[derive(Debug, Clone)]
/// Executes embedding jobs sequentially and writes each result to configured sinks.
///
/// The queue checks its optional [`Cache`] before contacting an embedding
/// provider. Explicit sinks execute in registration order; the first error
/// aborts processing and includes the names of sinks already written. Use
/// [`Self::with_sink`] for custom or multi-destination pipelines.
///
/// # Example
///
/// ```no_run
/// use lvv::{db::Sink, jobs::job::Job, jobs::job_queue::JobQueue};
/// use serde::Serialize;
/// use std::sync::Arc;
///
/// fn prepare<T>(jobs: Vec<Job<T>>, sink: Arc<dyn Sink>) -> JobQueue<T>
/// where T: Serialize + Clone {
///     let mut queue = JobQueue::from_vec(jobs);
///     queue.with_sink(sink);
///     queue
/// }
/// ```
pub struct JobQueue<T: Serialize + Clone> {
    queue: Vec<Job<T>>,
    cache: Option<Cache>,
    /// Legacy single-Qdrant target (via [`JobQueue::with_database_params`]).
    /// Kept for back-compat: at `run` time it is appended as a trailing
    /// [`QdrantSink`].
    connection: Option<QdrantDatabase>,
    /// Explicit sinks, written in registration order (e.g. Postgres, then
    /// Qdrant). Prefer these over `connection` for multi-sink pipelines.
    sinks: Vec<Arc<dyn Sink>>,
}
impl<T> JobQueue<T>
where
    T: Serialize + Clone,
{
    /// Creates a queue from jobs, without a cache or sinks.
    pub fn from_vec(vec: Vec<Job<T>>) -> Self {
        // Build without using `Default` to avoid requiring `T: Default`.
        JobQueue {
            queue: vec,
            cache: None,
            connection: None,
            sinks: Vec::new(),
        }
    }
    /// Adds an embedding cache used while building and running the queue.
    pub fn with_cache(&mut self, cache: Cache) -> &mut Self {
        self.cache = Some(cache);
        self
    }
    /// Legacy: register a single Qdrant target. Equivalent to
    /// `with_sink(Arc::new(QdrantSink::new(params)))`, kept for back-compat.
    pub fn with_database_params(&mut self, params: DatabaseParams) -> &mut Self {
        let database = QdrantDatabase::new_with_database_params(params);
        self.connection = Some(database);
        self
    }
    /// Register a sink. Sinks run in registration order; for a dual write put
    /// the authoritative store first (e.g. Postgres before Qdrant) so a partial
    /// failure leaves the durable store written and the derived index behind.
    pub fn with_sink(&mut self, sink: Arc<dyn Sink>) -> &mut Self {
        self.sinks.push(sink);
        self
    }
    /// Convenience for `with_sink(Arc::new(QdrantSink::new(params)))`.
    pub fn with_qdrant_sink(&mut self, params: DatabaseParams) -> &mut Self {
        self.sinks.push(Arc::new(QdrantSink::new(params)));
        self
    }
    /// Resolves cached embeddings and returns a cloned, runnable queue.
    pub fn build(&mut self) -> Self {
        if let Some(cache) = self.clone().cache {
            self.queue.iter_mut().for_each(|job| {
                // `Cache::get_embedding` expects `Vec<String>`. Convert `Vec<T>` to JSON strings.
                let data_as_strings: Vec<String> = job
                    .clone()
                    .dataset
                    .data
                    .unwrap()
                    .into_iter()
                    .map(|item| serde_json::to_string(&item).expect("Couldn't serialize item"))
                    .collect();
                job.embedding = cache
                    .get_embedding(job.get_model(), data_as_strings)
                    .map(|embedding| embedding.to_owned());
            })
        }
        self.to_owned()
    }

    // TODO: Rehacer
    /// Embeds every queued dataset and writes it to each sink in order.
    ///
    /// Processing stops at the first sink failure and the error reports which
    /// earlier sinks completed.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        // Use each job inside the loop; previous code referenced `job` before it existed.
        for job in self.clone().queue.into_iter().progress() {
            println!("Job begun: {:#?}", job.collection_name);
            let embedder = match job.provider.clone() {
                Provider::Ollama(ollama_model) => EmbeddingProvider::new(&ollama_model)
                    .expect("Couldn't create embedding provider"),
                Provider::OpenAI(openai_model) => {
                    EmbeddingProvider::new_openai(&openai_model).expect("Couldnt create embedder")
                }
            };
            let embeddings = match job.embedding.clone() {
                Some(embeddings) => anyhow::Ok(embeddings),
                None => {
                    let temp = embedder
                        .embed_properties(job.dataset.clone())
                        .await
                        .context("Embedding failed")?;
                    if let Some(mut cache) = self.cache.clone() {
                        let inner = &job.dataset;
                        let data = inner
                            .to_owned()
                            .serialize_to_vec()
                            .context("Could not serialize")?;
                        cache.add_embedding(job.get_model(), data, temp.clone());
                    }
                    Ok(temp)
                }
            }?;
            // Raw JSON rows, one per embedding; each sink derives its own
            // representation from these (Qdrant -> Payload, Postgres -> jsonb).
            let rows = job.get_payload_values()?;

            // Assemble the sinks for this job: explicit sinks first (in
            // registration order, e.g. Postgres), then the legacy `connection`
            // Qdrant target appended for back-compat.
            let mut sinks: Vec<Arc<dyn Sink>> = self.sinks.clone();
            if let Some(connection) = self.connection.clone() {
                let params = match connection {
                    QdrantDatabase::Disconnected(params) => params,
                    QdrantDatabase::Connected(db) => db.params,
                };
                sinks.push(Arc::new(QdrantSink::new(params)));
            }

            let ctx = SinkContext {
                collection_name: &job.collection_name,
                dims: job.dims,
                extends: job.extends,
                embeddings: &embeddings,
                rows: &rows,
            };

            // Write to each sink in order. On failure, report which sinks were
            // already written and which were not, and abort the job — a re-run
            // is idempotent (Postgres upserts by stable id; Qdrant skips a
            // populated collection when `extends` is false) and reconciles.
            let mut written: Vec<String> = Vec::new();
            for sink in &sinks {
                if let Err(e) = sink.write(&ctx).await {
                    return Err(e).with_context(|| {
                        format!(
                            "sink '{}' failed for collection '{}'. Written: [{}]; \
                             not written: '{}' and any sink after it. Re-run to reconcile \
                             (writes are idempotent).",
                            sink.name(),
                            job.collection_name,
                            written.join(", "),
                            sink.name(),
                        )
                    });
                }
                written.push(sink.name().to_string());
            }
            println!(
                "Done. Collection: {}  Provider: {:?}  Sinks: [{}]",
                job.collection_name,
                job.provider,
                written.join(", ")
            );
        }
        Ok(())
    }
}
