# lvv

`lvv` is a Rust library for turning structured datasets into vector embeddings
and loading them into one or more storage backends. It supports Ollama and
OpenAI-compatible models, reusable embedding caches, sequential job queues, Qdrant, flat
files, optional SQL, HTTP and PostgreSQL connectors, and derive macros that let
your own Rust types describe what gets embedded and what gets stored. Records
can also be transformed by an LLM, for example summarized, before they are
embedded.

## Pipeline

1. Load records with a file, SQL, HTTP or PostgreSQL source, or create a
   `DataSet` directly.
2. Optionally transform the records with an LLM; see
   [Transforming records with LLMs](#transforming-records-with-llms).
3. Configure an embedding `Job` with an Ollama or OpenAI model.
4. Add the job to a `JobQueue`.
5. Register one or more sinks and run the queue.

For every stage in one program, with the derive macros and an LLM transform,
see the [full example](#full-example) at the end of this README.

```rust,no_run
use lvv::{
    db::{Distance, QdrantSink},
    db::vector_database::{DatabaseParams, Location},
    intake::dataset::DataSet,
    jobs::{JobBuilder, Provider},
    jobs::job_queue::JobQueue,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset = DataSet::new(
        "articles.json",
        "articles",
        vec![serde_json::json!({"title": "Rust documentation"})],
    );

    let job = JobBuilder::default()
        .dataset(dataset)
        .provider(Provider::Ollama("nomic-embed-text".into()))
        .dims(768_u64)
        .extends(false)
        .distance(Distance::Cosine)
        .collection_name()
        .build()?;

    let params = DatabaseParams::new(
        Location::new_local("http://localhost:6334"),
        "articles".into(),
        Distance::Cosine,
        768,
    );
    let mut queue = JobQueue::from_vec(vec![job]);
    queue.with_sink(Arc::new(QdrantSink::new(params)));
    Ok(queue.run().await?)
}
```

## Typed records with derive macros

The `derive` feature adds `#[derive(VectorDatabaseItem)]` and
`#[derive(VectorDatabase)]` from
[`lvv-macros`](https://crates.io/crates/lvv-macros). They are re-exported in
`lvv::points`, next to the traits they implement.

```toml
[dependencies]
lvv = { version = "0.6", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
```

```rust,ignore
use lvv::points::{VectorDatabase, VectorDatabaseItem};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, VectorDatabaseItem)]
#[serde(rename_all = "camelCase")]
struct Position {
    #[lvv(description, rename = "role")]
    job_title: String,
    #[lvv(description)]
    organization: Option<String>,
    started_on: String,
    #[lvv(skip)]
    internal_id: u64,
}

#[derive(VectorDatabase)]
struct Portfolio {
    positions: Vec<Position>,
    #[lvv(skip)]
    owner: String,
}

let drafts = portfolio.point_drafts()?;
```

Each item becomes a `VectorPointDraft`:

- `category`: the struct name, `"Position"`.
- `description`: the `#[lvv(description)]` fields joined with newlines, such
  as `"Researcher\nUNLP"`. This is the text you embed.
- `payload`: the struct serialized with serde, without `#[lvv(skip)]` fields,
  with `#[lvv(rename)]` applied and a `category` key added, such as
  `{"role": "Researcher", "organization": "UNLP", "startedOn": "2020-03-01", "category": "Position"}`.

`#[derive(VectorDatabase)]` collects the points of fields typed `T`, `Vec<T>`,
`BTreeMap<K, T>` or `HashMap<K, T>`, each optionally wrapped in `Option`. See the
[`lvv-macros` documentation](https://docs.rs/lvv-macros) for every attribute,
and the
[`points` module](https://docs.rs/lvv/latest/lvv/points/)
for embedding and storing drafts. Embed descriptions with
`EmbeddingProvider::embed_texts`, which sends the text as is;
`embed_properties` embeds each record's JSON instead.

[`examples/derive.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/derive.rs)
is a complete lvv pipeline over derived structs. It reads records from JSON
Lines, CSV and JSON files with `FileSource`, turns them into points with the
derives, embeds the descriptions with Ollama using a `Cache`, and runs a
`JobQueue` that writes one Qdrant collection per category.

```sh
cargo run --example derive --features derive                  # load and print the points
cargo run --example derive --features derive -- --embed       # embed and build the jobs
cargo run --example derive --features derive -- --embed --qdrant http://localhost:6334
```

## Transforming records with LLMs

`lvv::transform` rewrites records with a chat model before they are embedded:
to summarize long text, extract keywords, translate, or clean up inconsistent
fields.

- An `Llm` says where the model is: `Llm::ollama`, `Llm::openai`, or
  `Llm::openai_compatible` for vLLM, LM Studio, llama.cpp's server, OpenRouter
  and the like.
- A `Transform` says what to do: the system prompt, what to send for each
  record (its JSON by default, or `.input(...)`), and where the reply goes
  (`.apply(...)`). `Transform::text` takes the reply as text;
  `Transform::structured` sends the JSON schema of a type and parses the reply
  into it. A record type can have any number of transforms.
- `llm.run(&transform, &mut records)` applies the outputs in place and returns
  a `Report`. `llm.complete(&transform, &records)` returns the outputs instead.

```rust,ignore
use lvv::points::VectorDatabaseItem;
use lvv::transform::{Llm, Transform};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, VectorDatabaseItem)]
struct Article {
    #[lvv(description)]
    title: String,
    body: String,
    // Filled in by the model, then embedded with the title.
    #[lvv(description)]
    summary: Option<String>,
    keywords: Vec<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct Tags {
    keywords: Vec<String>,
}

let llm = Llm::ollama("llama3.2");

let summarize = Transform::text("Summarize the article in one sentence.")
    .input(|a: &Article| a.body.clone())
    .apply(|a: &mut Article, summary| a.summary = Some(summary));
let tag = Transform::structured("List search keywords for the article.")
    .apply(|a: &mut Article, tags: Tags| a.keywords = tags.keywords);

let report = llm
    .run(&summarize, &mut articles)
    .concurrency(4)
    .cache("summaries.jsonl")
    .await?;
for (index, error) in report.failures() {
    eprintln!("article {index}: {error}");
}
llm.run(&tag, &mut articles).await?.ensure_all()?;

let drafts = articles
    .iter()
    .map(Article::try_into_database_item)
    .collect::<Result<Vec<_>, _>>()?;
// drafts[0].description is "<title>\n<summary>".
```

Things to know:

- Every record gets an outcome, in record order. A record whose request fails
  is left unchanged and listed in `report.failures()`; the others are not
  affected.
- Rate limits, server errors and timeouts are retried (`.retries(n)`, 2 by
  default, with exponential backoff). Authentication errors and unknown models
  are not.
- If the first requests of a run all fail with such fatal errors, the run
  stops with an error instead of failing every record
  (`.circuit_breaker(n)`, 3 by default).
- A reply that is empty or doesn't parse fails only its record.
- Requests go one at a time unless you set `.concurrency(n)`.
- With `.cache(path)`, completed outputs are appended to a JSON Lines file and
  reused. An output is reused only for the same backend, model, prompt, schema
  and input, so an interrupted run resumes where it stopped and a changed
  prompt runs again.
- The only prompt sent is yours. The library prints nothing: follow progress
  with `.on_progress(...)` or `tracing`.
- If a run stops early, records completed before that keep their outputs, and
  the error says how many there were.

See the
[`transform` documentation](https://docs.rs/lvv/latest/lvv/transform/)
for details.

## Cargo features

No features are enabled by default.

| Feature    | Enables |
|------------|---------|
| `derive`   | `VectorDatabaseItem` and `VectorDatabase` derive macros from `lvv-macros` |
| `postgres` | `PostgresSource` and `PostgresSink` |
| `sql`      | `SqlSource` for SQLite and MySQL |
| `http`     | `HttpSource` for paginated JSON APIs |

```toml
[dependencies]
lvv = { version = "0.6", features = ["derive", "postgres"] }
```

## Configuration

- `OLLAMA_URL` selects the Ollama endpoint. It defaults to
  `http://127.0.0.1:11434`.
- `OPENAI_API_KEY` authenticates OpenAI requests, for both completions and
  embeddings. When it isn't set, lvv looks for it in a `.env` file.
- `QDRANT_API_KEY` authenticates remote Qdrant requests. lvv loads a `.env`
  file before reading it, and fails if there isn't one.

See the [API documentation](https://docs.rs/lvv) for detailed descriptions and
examples for each public API.

## Full example

[`examples/full_pipeline.rs`](https://github.com/egonik-unlp/lvv/blob/main/examples/full_pipeline.rs)
puts every stage together, with the `derive` feature:

1. `FileSource` loads positions, skills and projects from JSON Lines, CSV and
   JSON files into structs that derive `VectorDatabaseItem`.
2. A `Transform` run by an `Llm` writes a summary into each position, in a
   field marked `#[lvv(description)]`. Summaries are cached, so a rerun only
   sends the positions that don't have one yet.
3. `#[derive(VectorDatabase)]` turns the records into points.
4. `EmbeddingProvider::embed_texts` embeds each category's descriptions,
   reusing vectors from a `Cache`.
5. Each category becomes a `Job` with precomputed embeddings, because a
   `JobQueue` would otherwise embed each row's JSON instead of its description.
6. A `JobQueue` writes the jobs to Qdrant, one collection per category.

It needs Ollama with a chat model and an embedding model: `llama3.2` and
`embeddinggemma` by default, or the models named by `CHAT_MODEL` and
`EMBEDDING_MODEL`. Set `QDRANT_URL` to write the points; without it, the
program stops after building the jobs. The data files are in
[`examples/data`](https://github.com/egonik-unlp/lvv/tree/main/examples/data).

```sh
cargo run --example full_pipeline --features derive
QDRANT_URL=http://localhost:6334 cargo run --example full_pipeline --features derive
```

```rust,ignore
use std::{collections::BTreeMap, path::Path};

use anyhow::Context;
use lvv::{
    cache::cache_embeddings::Cache,
    db::{
        Distance,
        vector_database::{DatabaseParams, Location},
    },
    inference::EmbeddingProvider,
    intake::{FileSource, Source, dataset::DataSet},
    jobs::{JobBuilder, Provider, job_queue::JobQueue},
    points::{VectorDatabase, VectorDatabaseItem, VectorPointDraft},
    transform::{Llm, Transform},
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
    //    Each position is sent as JSON, and the reply goes into `summary`.
    let summarize = Transform::text(
        "You write search summaries of job positions. The user message is one job \
         position as JSON. Reply with a single sentence of at most 25 words that \
         describes the role, the organization and what the person did. Do not add \
         quotes, greetings or any other text.",
    )
    // Small models sometimes wrap the sentence in quotes anyway.
    .apply(|position: &mut Position, summary| {
        position.summary = Some(summary.trim_matches('"').to_string())
    });
    let summary_cache = std::env::temp_dir().join("lvv-full-pipeline-summaries.jsonl");
    let report = Llm::ollama(&chat_model)
        .run(&summarize, &mut portfolio.positions)
        .concurrency(2)
        .cache(summary_cache)
        .on_progress(|p| eprint!("\rsummaries: {}/{}", p.done, p.total))
        .await
        .with_context(|| format!("check that `{chat_model}` is a chat model available in Ollama"))?;
    eprintln!();
    let counts = report.counts();
    println!(
        "{} summaries written, {} reused from the cache, {} failed",
        counts.applied, counts.cached, counts.failed
    );
    // A position without a summary is still embedded, by its title alone.
    for (index, error) in report.failures() {
        eprintln!("position {index}: {error}");
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
    let embedder = EmbeddingProvider::new(&embedding_model);

    let mut jobs = Vec::new();
    for (category, drafts) in &by_category {
        let descriptions: Vec<String> = drafts.iter().map(|d| d.description.clone()).collect();
        let embeddings = match cache.get_embedding(embedding_model.clone(), descriptions.clone()) {
            Some(cached) => cached.clone(),
            None => {
                let fresh = embedder.embed_texts(&descriptions).await?;
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

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}
```

## Migrating from 0.5

- `lvv::transform::transform` is now `lvv::points`. `lvv::transform` is the
  new LLM API, and `lvv-macros` 0.2 is required for the derives.
- `CompletionModel`, `FieldEnhanceable` and the `perform_completion*` methods
  are gone. Use `Llm` and `Transform`; see
  [Transforming records with LLMs](#transforming-records-with-llms).
- `EmbeddingProvider::new` no longer returns a `Result`, and `new_openai` is
  now `openai`. To embed text such as `VectorPointDraft` descriptions, use
  `embed_texts`. `embed_properties` serializes each item as JSON, so in 0.5 a
  `DataSet<String>` of descriptions was embedded with quotes and escapes
  around the text.
- Vectors created from descriptions with 0.5 were computed on that quoted
  text. Re-index those Qdrant collections so stored vectors match the ones
  your queries produce. Embedding cache files from 0.5 load as empty caches.
- Errors are typed (`thiserror`) instead of `anyhow`: `IntakeError`,
  `EmbedError`, `SinkError`, `JobError`, `PointError`, `CacheError`,
  `BackendError`, `RunError`. They all implement `std::error::Error`, so `?`
  into `anyhow::Result` keeps working. Custom sinks return
  `SinkError::Other`.
- The `llm` crate is no longer a dependency. Models are reached through
  `lvv::backend`, which you can implement for other providers.

## License

Licensed under the Apache License, Version 2.0.
