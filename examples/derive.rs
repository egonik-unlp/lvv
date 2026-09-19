//! A complete lvv pipeline whose records are Rust structs using the
//! `lvv-macros` derives.
//!
//! 1. **Intake**: `FileSource` reads positions from JSON Lines, skills from CSV
//!    and projects from JSON, and each row is deserialized into a struct that
//!    derives `VectorDatabaseItem`.
//! 2. **Transform**: `#[derive(VectorDatabase)]` turns every record into a
//!    `VectorPointDraft`: the text to embed and the payload to store.
//! 3. **Embed**: each category's descriptions are embedded with Ollama, and the
//!    vectors are kept in a `Cache` between runs.
//! 4. **Load**: one `Job` per category goes into a `JobQueue`, which writes
//!    each job to its own Qdrant collection.
//!
//! A `JobQueue` embeds the JSON of each row by itself. To embed the text chosen
//! with `#[lvv(description)]` instead, the example embeds the descriptions first
//! and passes the vectors to each job with `JobBuilder::embedding`.
//!
//! ```text
//! cargo run --example derive --features derive                  # load and print the points
//! cargo run --example derive --features derive -- --embed       # embed and build the jobs
//! cargo run --example derive --features derive -- --embed --qdrant http://localhost:6334
//! ```

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use anyhow::Context;
use clap::Parser;
use lvv::{
    cache::cache_embeddings::Cache,
    db::{
        Distance,
        vector_database::{DatabaseParams, Location},
    },
    inference::EmbeddingProvider,
    intake::{FileSource, Source, dataset::DataSet},
    jobs::{JobBuilder, Provider, job_queue::JobQueue},
    points::{IntoDescriptionValue, VectorDatabase, VectorDatabaseItem, VectorPointDraft},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A job, read from `positions.jsonl`. A `VectorDatabaseItem` needs
/// `Serialize`, `Deserialize` and at least one `#[lvv(description)]` field.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    /// Embedded, and stored under `role` instead of `jobTitle`.
    #[lvv(description, rename = "role")]
    job_title: String,
    /// Embedded when present; `null` adds nothing to the description.
    #[lvv(description)]
    organization: Option<String>,
    /// Embedded one line per highlight.
    #[lvv(description)]
    highlights: Vec<String>,
    /// Stored as `startedOn`, but not embedded.
    started_on: String,
    /// Private: neither embedded nor stored.
    #[lvv(skip)]
    salary: u32,
}

/// A skill level, written as `"expert"` and the like.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Level {
    Familiar,
    Proficient,
    Expert,
}

/// Lets `Level` be a description field. `String`, numbers, `Option` and `Vec`
/// already implement `IntoDescriptionValue`.
impl IntoDescriptionValue for Level {
    fn into_description_value(&self) -> String {
        match self {
            Level::Familiar => "familiar",
            Level::Proficient => "proficient",
            Level::Expert => "expert",
        }
        .into()
    }
}

/// A skill, read from `skills.csv`. CSV fields are text, so the ID is a
/// `String`.
#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Skill {
    #[lvv(description)]
    name: String,
    #[lvv(description)]
    level: Level,
    /// Neither embedded nor stored.
    #[lvv(skip)]
    internal_id: String,
}

/// A project, read from `projects.json`.
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

/// All the records. Each field yields one point per record, and the container
/// itself needs no serde derives.
#[derive(VectorDatabase)]
struct Portfolio {
    positions: Vec<Position>,
    skills: Vec<Skill>,
    /// Keyed by project name. One point per value, in key order.
    projects: BTreeMap<String, Project>,
    /// Not a record, so it must be skipped.
    #[lvv(skip)]
    owner: String,
}

#[derive(Parser)]
#[command(about = "A complete lvv pipeline over structs that use the lvv-macros derives")]
struct Args {
    /// Embed the descriptions with Ollama and build the jobs.
    #[arg(long)]
    embed: bool,
    /// Ollama embedding model.
    #[arg(long, default_value = "embeddinggemma")]
    model: String,
    /// Embedding cache file. Defaults to one in the system temp directory.
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Run the jobs into the Qdrant instance at this gRPC URL.
    #[arg(long, requires = "embed")]
    qdrant: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Intake: rows from three files, deserialized into the derived structs.
    let portfolio = Portfolio {
        positions: load("positions.jsonl").await?,
        skills: load("skills.csv").await?,
        projects: load::<Project>("projects.json")
            .await?
            .into_iter()
            .map(|project| (project.name.clone(), project))
            .collect(),
        owner: "Ana Example".into(),
    };

    // 2. Transform: one draft per record, grouped by category.
    let mut by_category: BTreeMap<String, Vec<VectorPointDraft>> = BTreeMap::new();
    for draft in portfolio.point_drafts()? {
        by_category
            .entry(draft.category.clone())
            .or_default()
            .push(draft);
    }
    let total: usize = by_category.values().map(Vec::len).sum();
    println!("{total} points for {}", portfolio.owner);
    for (category, drafts) in &by_category {
        println!("\n[{category}]");
        for draft in drafts {
            println!("  description: {:?}", draft.description);
            println!("  payload:     {}", serde_json::to_string(&draft.payload)?);
        }
    }

    if !args.embed {
        println!("\nDry run: pass --embed to embed the descriptions and build the jobs.");
        return Ok(());
    }

    // 3. Embed: each category's descriptions, reusing vectors from earlier runs.
    let cache_path = args
        .cache
        .unwrap_or_else(|| std::env::temp_dir().join("lvv-derive-example-cache.json"));
    let cache_path = cache_path.to_str().context("cache path is not UTF-8")?;
    let mut cache = if Path::new(cache_path).exists() {
        Cache::from_json_file(cache_path)?
    } else {
        Cache::new()
    };
    let embedder = EmbeddingProvider::new(&args.model);

    let mut jobs = Vec::new();
    for (category, drafts) in &by_category {
        let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
        let embeddings = match cache.get_embedding(args.model.clone(), descriptions.clone()) {
            Some(cached) => cached.clone(),
            None => {
                let fresh = embedder.embed_texts(&descriptions).await?;
                cache.add_embedding(args.model.clone(), descriptions, fresh.clone());
                fresh
            }
        };

        // 4. Jobs: the payloads are the rows, and the vectors are precomputed.
        let rows = drafts
            .iter()
            .map(|d| serde_json::to_value(&d.payload))
            .collect::<Result<Vec<_>, _>>()?;
        let job = JobBuilder::default()
            .dataset(DataSet::new("portfolio", category.to_lowercase(), rows))
            .provider(Provider::Ollama(args.model.clone()))
            .dims(embeddings.first().map_or(0, Vec::len) as u64)
            // Points get random IDs, so leave a collection that already has
            // points untouched instead of adding duplicates on every run.
            .extends(false)
            .distance(Distance::Cosine)
            .embedding(embeddings)
            // Must come after the dataset, provider and distance.
            .collection_name()
            .build()?;
        println!(
            "\n{} {category} points -> collection `{}` ({} dimensions)",
            drafts.len(),
            job.collection_name,
            job.dims
        );
        jobs.push(job);
    }
    cache.to_json_file(cache_path)?;

    // 5. Load: run the queue. The sink creates each job's collection, and the
    //    params supply the location and distance.
    let Some(url) = args.qdrant else {
        println!("\nPass --qdrant <url> to run the jobs into Qdrant.");
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
    Ok(queue.run().await?)
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
