// TODO: reemplazar anyhow con thiserror
use std::{fs::OpenOptions, io::Write};

use anyhow::Context;
use indicatif::ProgressIterator;
use serde::Serialize;

use crate::{
    cache::cache_embeddings::Cache,
    db::{QdrantDatabase, vector_database::DatabaseParams},
    inference::embedding_model::EmbeddingProvider, // Needed for `run()` when creating embedders
    jobs::{Provider, job::Job},
};

#[derive(Debug, Clone)]
pub struct JobQueue<T: Serialize + Clone> {
    queue: Vec<Job<T>>,
    cache: Option<Cache>,
    connection: Option<QdrantDatabase>,
}
impl<T> JobQueue<T>
where
    T: Serialize + Clone,
{
    pub fn from_vec(vec: Vec<Job<T>>) -> Self {
        // Build without using `Default` to avoid requiring `T: Default`.
        JobQueue {
            queue: vec,
            cache: None,
            connection: None,
        }
    }
    pub fn with_cache(&mut self, cache: Cache) -> &mut Self {
        self.cache = Some(cache);
        self
    }
    pub fn with_database_params(&mut self, params: DatabaseParams) -> &mut Self {
        let database = QdrantDatabase::new_with_database_params(params);
        self.connection = Some(database);
        self
    }
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
            let payloads = job.get_payloads()?;
            if let Some(cache) = self.cache.clone() {
                cache
                    .to_json_file("database_joined2.json")
                    .context("Couldn't save cache")?;
                println!("Starting upload");
                println!(
                    "Just pre upload.\n{}\n{}\n{}\n{}",
                    &job.collection_name,
                    job.dims,
                    embeddings.len(),
                    payloads.len()
                );
            }

            let embeddings_string =
                serde_json::to_string(&embeddings).context("Could not stringify embeddings")?;
            let embedding_dump_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("embeddings_dump_changemyname.json");
            match embedding_dump_file {
                Ok(mut file) => file
                    .write_all(embeddings_string.as_bytes())
                    .context("Could not write to file")?,
                Err(error) => println!("Dump file not created, err:\n{}", error),
            }
            if let Some(connection) = self.connection.clone() {
                let db = connection
                    .connect()
                    .context("Could not connect to database")?;
                if let QdrantDatabase::Connected(db) = db {
                    if db
                        .collection_exists_and_is_not_empty(&job.collection_name, job.extends)
                        .await
                    {
                        println!("EXISTE Y NO SE TOCA");
                        println!(
                            "Done\nCollection Name: {}\nProvider: {:?}",
                            job.collection_name, job.provider
                        );
                        continue;
                    }
                    db.upload_embedddings(&job.collection_name, job.dims, embeddings, payloads)
                        .await
                        .unwrap();
                    println!(
                        "Done\nCollection Name: {}\nProvider: {:?}",
                        job.collection_name, job.provider
                    );
                }
            }
        }
        Ok(())
    }
}
