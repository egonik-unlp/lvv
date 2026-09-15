//! The full lvv stack in one program: records loaded from files into structs
//! that use the lvv-macros derives, rewritten by an LLM, embedded, and written
//! to Qdrant by a job queue.
//!
//! Needs Ollama with a chat model and an embedding model (by default
//! `ollama pull llama3.2` and `ollama pull embeddinggemma`; override them with
//! `CHAT_MODEL` and `EMBEDDING_MODEL`). Set `QDRANT_URL` to a Qdrant gRPC
//! endpoint to write the points; without it, the program stops after building
//! the jobs.
//!
//! ```text
//! cargo run --example full_pipeline --features derive
//! QDRANT_URL=http://localhost:6334 cargo run --example full_pipeline --features derive
//! ```

use std::{collections::BTreeMap, path::Path};

use anyhow::Context;
use lvv::{
    cache::cache_embeddings::Cache,
    db::{
        Distance,
        vector_database::{DatabaseParams, Location},
    },
    inference::{CompletionModel, EmbeddingProvider},
    intake::{FileSource, Source, dataset::DataSet},
    jobs::{JobBuilder, Provider, job_queue::JobQueue},
    transform::transform::{VectorDatabase, VectorDatabaseItem, VectorPointDraft},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A job, from `positions.jsonl`. The file's `salary` isn't a field, so it
/// never reaches the model or the payload.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    /// Embedded, and stored under `role`.
    #[lvv(description, rename = "role")]
    job_title: String,
    organization: Option<String>,
    highlights: Vec<String>,
    started_on: String,
    /// Written by the LLM, then embedded with the title.
    #[lvv(description)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

/// A skill, from `skills.csv`.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Skill {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    level: String,
    /// Neither embedded nor stored.
    #[lvv(skip)]
    internal_id: String,
}

/// A project, from `projects.json`.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Project {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    summary: String,
    /// Stored under `stack`, but not embedded.
    #[lvv(rename = "stack")]
    technologies: Vec<String>,
}

/// Every record, one point each.
#[derive(VectorDatabase)]
struct Portfolio {
    positions: Vec<Position>,
    skills: Vec<Skill>,
    projects: Vec<Project>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let chat_model = env_or("CHAT_MODEL", "llama3.2");
    let embedding_model = env_or("EMBEDDING_MODEL", "embeddinggemma");

    // 1. Load: rows from three files, deserialized into the derived structs.
    let mut portfolio = Portfolio {
        positions: load("positions.jsonl").await?,
        skills: load("skills.csv").await?,
        projects: load("projects.json").await?,
    };

    // 2. Transform: the LLM writes a one-sentence summary of each position.
    let model = CompletionModel::new(
        chat_model.as_str(),
        "You write search summaries of job positions. The last user message is one \
         job position as JSON. Reply with a single sentence of at most 25 words that \
         describes the role, the organization and what the person did. Do not add \
         quotes, greetings or any other text.",
    )?;
    let summaries = model
        .perform_completion(portfolio.positions.iter().collect::<Vec<_>>())
        .await?;
    // Failed records are left out of the responses, so pairing responses with
    // records is only safe when every record got one.
    anyhow::ensure!(
        summaries.len() == portfolio.positions.len(),
        "{} of {} summaries failed; check that `{chat_model}` is a chat model \
         available in Ollama",
        portfolio.positions.len() - summaries.len(),
        portfolio.positions.len()
    );
    for (position, summary) in portfolio.positions.iter_mut().zip(summaries) {
        // Small models sometimes wrap the sentence in quotes anyway.
        position.summary = Some(summary.trim().trim_matches('"').to_string());
    }

    // 3. Derive: one point per record, grouped by category.
    let mut by_category: BTreeMap<String, Vec<VectorPointDraft>> = BTreeMap::new();
    for draft in portfolio.point_drafts()? {
        by_category
            .entry(draft.category.clone())
            .or_default()
            .push(draft);
    }

    // 4. Embed: each category's descriptions, reusing vectors from earlier runs.
    let cache_path = std::env::temp_dir().join("lvv-full-pipeline-cache.json");
    let cache_path = cache_path.to_str().context("cache path is not UTF-8")?;
    let mut cache = if Path::new(cache_path).exists() {
        Cache::from_json_file(cache_path)?
    } else {
        Cache::new()
    };
    let embedder = EmbeddingProvider::new(&embedding_model)?;

    let mut jobs = Vec::new();
    for (category, drafts) in &by_category {
        let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
        let embeddings = match cache.get_embedding(embedding_model.clone(), descriptions.clone()) {
            Some(cached) => cached.clone(),
            None => {
                let dataset = DataSet::new("portfolio", category.as_str(), descriptions.clone());
                let fresh = embedder.embed_properties(dataset).await?;
                cache.add_embedding(embedding_model.clone(), descriptions, fresh.clone());
                fresh
            }
        };

        // 5. Jobs: the payloads are the rows. The vectors are precomputed
        //    because a queue would embed each row's JSON, not the description.
        let rows = drafts
            .iter()
            .map(|d| serde_json::to_value(&d.payload))
            .collect::<Result<Vec<_>, _>>()?;
        let job = JobBuilder::default()
            .dataset(DataSet::new(
                "portfolio",
                format!("portfolio_{}", category.to_lowercase()),
                rows,
            ))
            .provider(Provider::Ollama(embedding_model.clone()))
            .dims(embeddings.first().map_or(0, Vec::len) as u64)
            // Points get random IDs, so leave a collection that already has
            // points untouched instead of adding duplicates on every run.
            .extends(false)
            .distance(Distance::Cosine)
            .embedding(embeddings)
            // Must come after the dataset, provider and distance.
            .collection_name()
            .build()?;
        println!("\n{} points -> `{}`", drafts.len(), job.collection_name);
        for draft in drafts {
            println!("  {:?}", draft.description);
        }
        jobs.push(job);
    }
    cache.to_json_file(cache_path)?;

    // 6. Write: the queue creates each job's collection in Qdrant.
    let Ok(url) = std::env::var("QDRANT_URL") else {
        println!("\nSet QDRANT_URL to write the points to Qdrant.");
        return Ok(());
    };
    // `Location::new_local` takes a `&'static str`.
    let url: &'static str = Box::leak(url.into_boxed_str());
    let dims = jobs.first().map_or(0, |job| job.dims) as u16;
    let mut queue = JobQueue::from_vec(jobs);
    queue.with_qdrant_sink(DatabaseParams::new(
        Location::new_local(url),
        "portfolio".into(),
        Distance::Cosine,
        dims,
    ));
    queue.run().await
}

/// Reads every row of `examples/data/<file>` with a `FileSource`, which picks
/// the format from the extension, and deserializes each row into `T`.
async fn load<T: DeserializeOwned>(file: &str) -> anyhow::Result<Vec<T>> {
    let path = format!("{}/examples/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let mut records = Vec::new();
    for dataset in FileSource::new(&path, file)?.fetch().await? {
        for row in dataset.data.unwrap_or_default() {
            let record =
                serde_json::from_value(row).with_context(|| format!("parsing a row of {file}"))?;
            records.push(record);
        }
    }
    Ok(records)
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
